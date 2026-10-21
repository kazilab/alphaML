"""
Uploads in pieces. A public proxy such as Cloudflare limits one request to 100 MB, so the browser sends big files
as a series of small chunks, which are appended to a file here. While the chunks arrive, the rows, columns and
sample names of a table are counted, so a data set can be summarised without reading the file again.
"""
import csv
import os
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .jobs import read_json, write_json

HEADER_LIMIT = 64 << 20  # longest header line kept for counting columns
ID_LIMIT = 1000          # longest sample name kept


class UploadError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class TableStats:
    """Rows, header and first-column values of a delimited text file, fed chunk by chunk."""

    def __init__(self, sep):
        self.sep = sep.encode()
        self.header = bytearray()
        self.header_done = False
        self.newlines = 0
        self.ids = []
        self.field = bytearray()
        self.field_open = False  # collecting the first field of the current line
        self.line_open = False   # the current line has content after the last newline

    def feed(self, data):
        pos, n = 0, len(data)
        while pos < n:
            if not self.header_done:
                nl = data.find(b'\n', pos)
                end = n if nl < 0 else nl
                room = HEADER_LIMIT - len(self.header)
                if room > 0:
                    self.header += data[pos:min(end, pos + room)]
                if nl < 0:
                    return
                self.header_done, self.newlines, pos = True, 1, nl + 1
                self.field_open = True
                continue
            if self.field_open:
                ends = [i for i in (data.find(self.sep, pos), data.find(b'\n', pos)) if i >= 0]
                end = min(ends) if ends else n
                if len(self.field) < ID_LIMIT:
                    self.field += data[pos:min(end, pos + ID_LIMIT - len(self.field))]
                self.line_open = self.line_open or end > pos
                if end == n:
                    return
                self._close_field()
                pos = end
            nl = data.find(b'\n', pos)
            if nl < 0:
                self.line_open = self.line_open or pos < n
                return
            self.newlines += 1
            self.line_open = False
            self.field_open = True
            pos = nl + 1

    def _close_field(self):
        value = bytes(self.field).strip(b'\r').strip().strip(b'"')
        if value:
            self.ids.append(value.decode('utf-8', errors='replace'))
        self.field = bytearray()
        self.field_open = False

    def result(self):
        """{'rows', 'columns', 'ids'}: data rows, columns after the index column, first-column values."""
        if self.field_open and self.field:
            self._close_field()
        rows = max(0, self.newlines - 1) + (1 if self.line_open else 0)
        header = bytes(self.header).rstrip(b'\r').decode('utf-8', errors='replace')
        columns = next(csv.reader([header], delimiter=self.sep.decode()), []) if header else []
        return {'rows': rows, 'columns': max(0, len(columns) - 1), 'ids': self.ids}


class UploadStore:
    def __init__(self, root, max_bytes, chunk_bytes, max_open=20):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes, self.chunk_bytes, self.max_open = max_bytes, chunk_bytes, max_open
        self._locks = {}
        self._stats = {}
        self._guard = threading.Lock()

    def _folder(self, upload_id):
        if not upload_id or not all(c in '0123456789abcdef' for c in upload_id):
            raise KeyError(upload_id)
        return self.root / upload_id

    def _lock(self, upload_id):
        with self._guard:
            return self._locks.setdefault(upload_id, threading.Lock())

    def get(self, upload_id):
        try:
            return read_json(self._folder(upload_id) / 'upload.json')
        except KeyError:
            return None

    def open_count(self, owner):
        return sum(1 for r in self.list() if r.get('owner') == owner and r['received'] < r['size'])

    def list(self):
        records = (read_json(d / 'upload.json') for d in self.root.iterdir() if d.is_dir())
        return [r for r in records if r]

    def start(self, owner, filename, size):
        if size > self.max_bytes:
            raise UploadError(413, f'{filename} is larger than {self.max_bytes >> 20} MB.')
        if self.open_count(owner) >= self.max_open:
            raise UploadError(429, 'Too many unfinished uploads. Finish or cancel some first.')
        upload_id = uuid.uuid4().hex[:16]
        folder = self.root / upload_id
        folder.mkdir()
        (folder / 'data').touch()
        suffix = Path(filename).suffix.lower()
        record = {'id': upload_id, 'owner': owner, 'filename': Path(filename).name, 'suffix': suffix, 'size': size,
                  'received': 0, 'created': datetime.now(timezone.utc).isoformat(timespec='seconds')}
        write_json(folder / 'upload.json', record)
        if suffix in ('.csv', '.tsv', '.txt'):
            self._stats[upload_id] = TableStats('\t' if suffix in ('.tsv', '.txt') else ',')
        return record

    def write(self, upload_id, offset, data):
        """Append a chunk at `offset`. A repeated chunk (a retry) is accepted without writing it twice."""
        with self._lock(upload_id):
            record = self.get(upload_id)
            if record is None:
                raise UploadError(404, 'Upload not found.')
            received = record['received']
            if offset < received and offset + len(data) <= received:
                return received  # already stored
            if offset != received:
                raise UploadError(409, f'Expected the chunk at byte {received}.')
            if received + len(data) > record['size']:
                raise UploadError(400, 'The upload is larger than announced.')
            with open(self._folder(upload_id) / 'data', 'ab') as fh:
                fh.write(data)
            stats = self._stats.get(upload_id)
            if stats is not None:
                stats.feed(data)
            record['received'] = received + len(data)
            write_json(self._folder(upload_id) / 'upload.json', record)
            return record['received']

    def take(self, upload_id, dest):
        """Move a complete upload to dest. Returns its table statistics, or None if they were not collected."""
        with self._lock(upload_id):
            record = self.get(upload_id)
            if record is None:
                raise UploadError(404, 'Upload not found.')
            if record['received'] != record['size']:
                raise UploadError(400, f"{record['filename']} has not been uploaded completely.")
            os.replace(self._folder(upload_id) / 'data', dest)
            stats = self._stats.pop(upload_id, None)
            self._remove(upload_id)
        return stats.result() if stats is not None else None

    def delete(self, upload_id):
        with self._lock(upload_id):
            self._stats.pop(upload_id, None)
            self._remove(upload_id)

    def _remove(self, upload_id):
        shutil.rmtree(self._folder(upload_id), ignore_errors=True)
        with self._guard:
            self._locks.pop(upload_id, None)

    def purge(self, max_age_hours=24):
        """Delete uploads that were started more than max_age_hours ago and never used."""
        cutoff = time.time() - max_age_hours * 3600
        for record in self.list():
            if datetime.fromisoformat(record['created']).timestamp() < cutoff:
                self.delete(record['id'])
