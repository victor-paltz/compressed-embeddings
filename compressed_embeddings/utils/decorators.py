from contextlib import contextmanager
from functools import wraps
import time


def timing(f):
    """
    I you want to time a function and save the time and the output, use this decorator.
    
    Usage:
    -----
    def my_func(...):
        return 42

    elapsed_time, output = time_func(my_func)
    
    """

    @wraps(f)
    def wrap(*args, **kw):
        ts = time.perf_counter()
        result = f(*args, **kw)
        te = time.perf_counter()
        return te - ts, result

    return wrap


@contextmanager
def TimeIt(name):
    st = time.time()
    yield
    elappsed_time = time.time() - st
    print(f"{name} done in {elappsed_time:.5f}s")
