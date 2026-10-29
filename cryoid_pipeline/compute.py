"""Process-local CPU controls; Auto deliberately leaves library defaults alone."""
from contextlib import contextmanager
import os


def available_cpu_threads() -> int:
    process_count = getattr(os, "process_cpu_count", None)
    if process_count is not None:
        return process_count() or 1
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return os.cpu_count() or 1


@contextmanager
def cpu_thread_limit(count: int | None):
    if count is None:
        yield
        return
    # Load the supported numerical pools before taking their original limits,
    # so a later Auto run in the same Python process restores normal behavior.
    import torch
    import numpy
    import scipy.linalg
    import sklearn
    from threadpoolctl import threadpool_limits

    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS")
    previous_env = {name: os.environ.get(name) for name in names}
    previous_torch = torch.get_num_threads()
    try:
        for name in names:
            os.environ[name] = str(count)
        with threadpool_limits(limits=count):
            torch.set_num_threads(count)
            yield
    finally:
        try:
            torch.set_num_threads(previous_torch)
        finally:
            for name, value in previous_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
