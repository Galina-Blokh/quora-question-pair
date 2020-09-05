import random
import numpy as np
import torch

try:
    import dill as pickle
except:
    import pickle
import time
from functools import wraps
from decorator import decorator

def setup_seed(seed_value: int = 42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def prepare_device(use_cuda=True):
    return (
        torch.device('cuda')
        if use_cuda and torch.cuda.is_available()
        else torch.device('cpu')
    )


def to_pickle(in_object, filename: str):
    with open(filename, 'wb') as f:
        pickle.dump(in_object, f, pickle.HIGHEST_PROTOCOL)


def from_pickle(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def exception(logger, reraise=False):
    """
    A decorator that wraps the passed in function and logs
    exceptions should one occur

    @param logger: The logging object
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"There was an exception in {func.__name__}")
            if reraise:
                raise

        return wrapper

    return decorator


@decorator
def warn_slow(func, logging, timelimit=60, *args, **kw):
    t0 = time.time()
    result = func(*args, **kw)
    dt = time.time() - t0
    if dt > timelimit:
        logging.warn('%s took %d seconds', func.__name__, dt)
    else:
        logging.info('%s took %d seconds', func.__name__, dt)
    return result

def download_file(url, filename):
    if "drive.google.com" in url:
        import gdown
        gdown.download(url, filename, quiet=False)
    else:
        import subprocess
        subprocess.run(["wget", url, "-O", filename])