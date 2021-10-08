# base.py
"""Base utility functions for accurity project."""

import logging
from time import time, sleep
from datetime import datetime
from contextlib import contextmanager
from collections import defaultdict
from functools import wraps


# Pandas dataframe formatting
def float_format(x):
    """Defines default float display format for pandas dataframes"""
    if x * 10 % 10 == 0:
        return "%.0f" % x
    elif x * 100 % 10 == 0:
        return "%.1f" % x
    else:
        return "%.2f" % x


# Logging
@contextmanager
def log_level(level, name):
    logger = logging.getLogger(name)
    old_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    try:
        yield logger
    finally:
        logger.setLevel(old_level)


def log_info(name: str, msg: str):
    with log_level(logging.INFO, name) as logger:
        sleep(0.5)
        logger.info(msg)


# Timing context manager
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"Function {f.__name__} took: {te-ts:.4f} sec.")
        return result
    return wrap


# Dictionary mappings
def get_setvalued_dict(keys, values):
    res = defaultdict(set)
    for i, j in zip(keys, values):
        res[i].add(j)
    return res


# Datetime operations
def get_timestamp():
    return datetime.now().strftime(("%Y-%m-%d %H:%M:%S"))
