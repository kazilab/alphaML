"""
Background job queue. Each job runs in its own process so it can be cancelled cleanly and cannot take the
web server down with it. Job state lives in <root>/<job id>/job.json, so it survives server restarts.
"""
import json
import logging
import multiprocessing
import os
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger('alphaml.web')

ACTIVE = ('queued', 'running')


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path, data, attempts=40):
    """Write atomically so readers never see a half-written file."""
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows refuses while another process is reading the file
            if attempt == attempts - 1:
                raise
            time.sleep(0.05)


def kill_tree(pid):
    """Kill a process and its children (joblib workers)."""
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is None:
        return False
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass
    return True


class JobManager:
    def __init__(self, root, max_running=1, poll_interval=0.5):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_running = max(1, int(max_running))
        self.poll_interval = poll_interval
        self._ctx = multiprocessing.get_context('spawn')
        self._procs = {}
        self._queue = []
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._recover()
        self._thread = threading.Thread(target=self._loop, name='alphaml-jobs', daemon=True)
        self._thread.start()

    # ------------------------------ storage ------------------------------ #

    def job_dir(self, job_id):
        if not job_id or not all(c in '0123456789abcdef' for c in job_id):
            raise KeyError(job_id)
        return self.root / job_id

    def get(self, job_id):
        try:
            return read_json(self.job_dir(job_id) / 'job.json')
        except KeyError:
            return None

    def list(self):
        jobs = [read_json(d / 'job.json') for d in self.root.iterdir() if d.is_dir()]
        return sorted((j for j in jobs if j), key=lambda j: j['created'], reverse=True)

    def _update(self, job_id, **changes):
        with self._lock:
            job = self.get(job_id)
            if job is None:
                return None
            job.update(changes)
            write_json(self.job_dir(job_id) / 'job.json', job)
            return job

    def _recover(self):
        for job in sorted(self.list(), key=lambda j: j['created']):
            if job['status'] == 'running':
                self._update(job['id'], status='interrupted', finished=now(),
                             error='The server stopped while this job was running.')
            elif job['status'] == 'queued':
                self._queue.append(job['id'])

    # ------------------------------ actions ------------------------------ #

    def create_dir(self):
        """Reserve a job folder so uploads can be saved before the job is submitted."""
        job_id = uuid.uuid4().hex[:12]
        (self.root / job_id / 'results').mkdir(parents=True)
        return job_id

    def submit(self, job_id, kind, title, params, inputs, owner=None):
        job = {'id': job_id, 'kind': kind, 'title': title, 'status': 'queued', 'created': now(),
               'started': None, 'finished': None, 'params': params, 'inputs': inputs, 'error': None,
               'summary': None, 'owner': owner}
        with self._lock:
            write_json(self.job_dir(job_id) / 'job.json', job)
            self._queue.append(job_id)
        return job

    def cancel(self, job_id):
        with self._lock:
            job = self.get(job_id)
            if job is None or job['status'] not in ACTIVE:
                return job
            if job_id in self._queue:
                self._queue.remove(job_id)
            proc = self._procs.pop(job_id, None)
            if proc is not None and proc.is_alive():
                if not kill_tree(proc.pid):
                    proc.kill()
                proc.join(timeout=10)
            return self._update(job_id, status='cancelled', finished=now())

    def delete(self, job_id):
        self.cancel(job_id)
        with self._lock:
            shutil.rmtree(self.job_dir(job_id), ignore_errors=True)

    def active_inputs(self):
        """Input file paths referenced by queued or running jobs."""
        with self._lock:
            ids = list(self._queue) + list(self._procs)
        paths = set()
        for job_id in ids:
            job = self.get(job_id) or {}
            paths.update(str(p) for p in (job.get('inputs') or {}).values() if p)
        return paths

    def shutdown(self):
        self._stop.set()
        with self._lock:
            for job_id in list(self._procs):
                proc = self._procs.pop(job_id)
                if proc.is_alive():
                    if not kill_tree(proc.pid):
                        proc.kill()
                self._update(job_id, status='interrupted', finished=now(), error='The server was stopped.')

    # ----------------------------- dispatcher ---------------------------- #

    def _loop(self):
        while not self._stop.wait(self.poll_interval):
            try:
                self._tick()
            except Exception:
                logger.exception('Job dispatcher error')

    def _tick(self):
        from .worker import run_job

        with self._lock:
            for job_id, proc in list(self._procs.items()):
                if not proc.is_alive():
                    proc.join()
                    del self._procs[job_id]
                    self._finish(job_id, proc.exitcode)
            while self._queue and len(self._procs) < self.max_running:
                job_id = self._queue.pop(0)
                job = self.get(job_id)
                if job is None or job['status'] != 'queued':
                    continue
                self._update(job_id, status='running', started=now())
                proc = self._ctx.Process(target=run_job, args=(str(self.job_dir(job_id)),),
                                         name=f'alphaml-job-{job_id}')
                proc.start()
                self._procs[job_id] = proc

    def _finish(self, job_id, exitcode):
        job = self.get(job_id)
        if job is None or job['status'] != 'running':
            return
        result = read_json(self.job_dir(job_id) / 'result.json', {})
        if exitcode == 0 and result.get('ok'):
            self._update(job_id, status='succeeded', finished=now(), summary=result.get('summary'))
        else:
            error = result.get('error') or f'The job process exited unexpectedly (exit code {exitcode}).'
            self._update(job_id, status='failed', finished=now(), error=error)
