"""Entry point of a job process."""
import logging
import os
import sys
import traceback
from pathlib import Path

from .jobs import now, read_json, write_json


def run_job(job_dir):
    job_dir = Path(job_dir)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    # Cap OpenMP threads before any model library loads (see models.folds.MAX_THREADS)
    os.environ.setdefault('OMP_NUM_THREADS', os.environ.get('ALPHAML_THREADS') or str(min(os.cpu_count() or 1, 16)))
    log = open(job_dir / 'run.log', 'a', buffering=1, encoding='utf-8')
    # Library output (Python and native) goes to the job log
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
            os.dup2(log.fileno(), stream.fileno())
        except (OSError, ValueError, AttributeError):
            pass
    sys.stdout = sys.stderr = log
    logging.basicConfig(stream=log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        force=True)
    logger = logging.getLogger('alphaml')

    def progress(stage, detail=''):
        logger.info(f'{stage}: {detail}' if detail else stage)
        try:
            write_json(job_dir / 'progress.json', {'stage': stage, 'detail': detail, 'time': now()})
        except OSError:
            pass  # progress display is best effort, never a reason to fail the job

    job = read_json(job_dir / 'job.json')
    exit_code = 0
    try:
        if job['kind'] == 'train':
            from ..run import train
            summary = train(**job['inputs'], **job['params']['train'], result_path=job_dir / 'results',
                            progress=progress)
        else:
            from ..prediction import predict
            summary = predict(**job['inputs'], **job['params']['predict'], result_path=job_dir / 'results',
                              progress=progress)
        progress('Finished')
        write_json(job_dir / 'result.json', {'ok': True, 'summary': summary})
    except Exception as exc:
        traceback.print_exc(file=log)
        write_json(job_dir / 'result.json', {'ok': False, 'error': f'{type(exc).__name__}: {exc}'})
        exit_code = 1
    finally:
        log.flush()
        # A multiprocessing child joins its child processes before joblib's exit hook stops its idle worker
        # pool, so a normal exit would wait forever: stop the workers and exit directly.
        _kill_children()
        os._exit(exit_code)


def _kill_children():
    try:
        import psutil
        for child in psutil.Process().children(recursive=True):
            child.kill()
    except Exception:
        pass
