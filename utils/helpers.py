import functools
import logging
import os
import re
import time

import utils.file as uf

# import pprint as pp

logger = logging.getLogger(__name__)


def timer(func):
    @functools.wraps(func)
    def clocked(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.info(f"Function '{func.__name__}' executed in {format_time(elapsed)}.")
        print(f"Function '{func.__name__}' executed in {format_time(elapsed)}.")
        return result

    return clocked


def format_time(elapsed: float) -> str:
    """Takes a time in seconds and returns a string hh:mm:ss"""
    if elapsed < 60:
        return f"{elapsed:.4f} seconds"
    elif elapsed < 3600:
        return time.strftime("%Mm:%Ss", time.gmtime(elapsed))
    else:
        return time.strftime("%Hh:%Mm:%Ss", time.gmtime(elapsed))


def get_key(filename: str, keyname: str | None = None):
    # Determine the user's home directory
    home_dir = os.path.expanduser("~")
    files = uf.grab_files(os.path.join(home_dir, ".pw"), filename=filename)
    if not files:
        raise FileNotFoundError(f"File {filename} not found in {home_dir}/.pw")
    elif len(files) > 1:
        raise ValueError(f"Multiple files found with name {filename}: {files}")
    else:
        keys = uf.File(files[0]).load()
        return keys[keyname] if keyname else keys


def get_timestamp(fmt: str = "yy-mm-dd"):
    fmt_map = {
        "yy": "%y",
        "yyyy": "%Y",
        "mm": "%m",
        "mmm": "%b",
        "dd": "%d",
        "H": "%H",
        "M": "%M",
        "S": "%S",
    }
    pattern = re.compile(r"[y|m|d|H|M|S]+")
    matches = pattern.findall(fmt)
    if not matches:
        raise ValueError(f"Invalid format string: {fmt}")
    for match in matches:
        if match not in fmt_map:
            raise ValueError(f"Invalid format specifier: {match}")
        fmt = fmt.replace(match, fmt_map[match])
    return time.strftime(fmt, time.localtime())


if __name__ == "__main__":
    get_timestamp()
