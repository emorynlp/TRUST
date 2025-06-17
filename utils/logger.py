import datetime
import functools
import logging
import os
import pathlib as pl
import re
import sys
import textwrap

from utils.file import File
from utils.helpers import format_time, timer
from utils.send_email import send_email

logger = logging.getLogger(__name__)


class LevelFilter(logging.Filter):
    def __init__(self, level):
        self.level = level

    def filter(self, record):
        if record.levelno == self.level:
            return True
        else:
            return False


class PlainFormatter(logging.Formatter):
    ln_fmt = "%(levelname)-8s %(filename)s:%(lineno)d %(message)s"
    ln_fmt_date = "[%(asctime)-16s] %(levelname)-8s %(filename)s:%(lineno)d %(message)s"
    date_fmt = "%m/%d/%y %H:%M:%S"

    def __init__(
        self,
        info_fmt: str | None = None,
        error_fmt: str | None = None,
        warning_fmt: str | None = None,
        debug_fmt: str | None = None,
        critical_fmt: str | None = None,
        date: bool = False,
        width: int = 100,
    ) -> None:
        super().__init__()

        # default formats
        self.date = date
        self.width = width
        self.ln_fmt = self.ln_fmt_date if date else self.ln_fmt
        self.debug_fmt = debug_fmt if debug_fmt else self.ln_fmt
        self.info_fmt = info_fmt if info_fmt else self.ln_fmt
        self.warning_fmt = warning_fmt if warning_fmt else self.ln_fmt
        self.error_fmt = error_fmt if error_fmt else self.ln_fmt
        self.critical_fmt = critical_fmt if critical_fmt else self.ln_fmt

        self.FORMATS = {
            logging.DEBUG: self.debug_fmt,
            logging.INFO: self.info_fmt,
            logging.WARNING: self.warning_fmt,
            logging.ERROR: self.error_fmt,
            logging.CRITICAL: self.critical_fmt,
        }

    @staticmethod
    def _get_header_len(header):
        hlen = len(header)
        gap = 4 - hlen % 4
        hlen = hlen if gap == 4 else hlen + gap
        spaces = "" if gap == 4 else " " * gap
        return hlen, spaces

    def wrap_message(self, message, header, lstrip=True):
        header_len, gap = self._get_header_len(header.split("{message}")[0])
        wrapper = textwrap.TextWrapper(
            width=self.width,
            initial_indent=" " * header_len,
            subsequent_indent=" " * header_len,
        )
        wrapped_msg = "\n".join([wrapper.fill(m) for m in message.splitlines()])
        if lstrip:
            return gap + wrapped_msg.lstrip()
        return gap + wrapped_msg

    def split_header(self, header):
        # header_splits = header.split("{message}")
        header_splits = re.split(r"\n?{message}", header)
        # seg_wrap = re.finditer(r"\n+|-+|=+|#+", header_splits[0])
        seg_wrap = re.finditer(r"\s?(-+|=+|#+)\s?", header_splits[0])
        seg_map = {
            # f"match_{i}": (match.start(0), match.group(0))
            f"match_{i}": match.group(0)
            for i, match in enumerate(seg_wrap)
        }
        for key, value in seg_map.items():
            header_splits[0] = header_splits[0].replace(value, f"{{{key}}}")
        return header_splits, seg_map

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(
            log_fmt, datefmt=self.date_fmt if self.date else None
        )
        message = record.getMessage()
        record.msg = "{message}"
        if record.args:
            record.args = tuple()
        header = formatter.format(record)
        record.msg = message

        header_splits, seg_map = self.split_header(header)
        start = re.sub(r"{match_\d+}", "", header_splits[0])
        if re.findall(r"\n+{message}", header):
            wrapped_msg = self.wrap_message(message, start, lstrip=False)
        else:
            wrapped_msg = self.wrap_message(message, start)

        message = header.format(message=wrapped_msg, **seg_map)
        # if len(header_splits) > 1 and not (header_splits[1]).isspace():
        #     ending = self.wrap_message(header_splits[1], header, lstrip=False)
        #     message += ending
        return message


class ColorFormatter(PlainFormatter):
    """Custom logging formatter with colors."""

    # colors
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    # ln_fmt = "%(levelname)-8s %(name)-12s %(message)s"

    def __init__(
        self,
        info_fmt: str | None = None,
        error_fmt: str | None = None,
        warning_fmt: str | None = None,
        debug_fmt: str | None = None,
        critical_fmt: str | None = None,
        date: bool = False,
        width: int = 100,
    ) -> None:
        super().__init__()

        # default formats
        self.date = date
        self.width = width
        self.ln_fmt = self.ln_fmt_date if date else self.ln_fmt
        self.debug_fmt = debug_fmt if debug_fmt else self.ln_fmt
        self.info_fmt = info_fmt if info_fmt else self.ln_fmt
        self.warning_fmt = warning_fmt if warning_fmt else self.ln_fmt
        self.error_fmt = error_fmt if error_fmt else self.ln_fmt
        self.critical_fmt = critical_fmt if critical_fmt else self.ln_fmt

        self.FORMATS = {
            logging.DEBUG: self.blue + self.debug_fmt + self.reset,
            logging.INFO: self.grey + self.info_fmt + self.reset,
            logging.WARNING: self.yellow + self.warning_fmt + self.reset,
            logging.ERROR: self.red + self.error_fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.critical_fmt + self.reset,
        }


def set_logger(
    logger: logging.Logger,
    path: str | None = None,
    error_log: bool = False,
    console: bool = True,
    flevel: int = logging.INFO,
    uncaught: bool = True,
    **formats,
) -> None:
    """Set up logger, including file logging, error logging, and console logging.

    Args:
        logger (logging.Logger): Logger object
        path (str | None, optional): file path to store logs. Defaults to None.
        error_log (bool, optional): save errors to seperate file. Defaults to False.
        console (bool, optional): print log in console. Defaults to True.
        flevel (int, optional): logging level for file handler. Defaults to logging.INFO.
        uncaught (bool, optional): log unexpected exceptions. Defaults to True.
        **formats: custom formats for different log levels, including info_fmt, err_fmt, wrn_fmt, and dbg_fmt.
    """

    logger.setLevel(logging.DEBUG)
    if path:
        pformatter = PlainFormatter(**formats)
        file_handler = logging.FileHandler(path, "a")
        file_handler.setLevel(flevel)
        file_handler.setFormatter(pformatter)
        logger.addHandler(file_handler)

        if error_log:
            err_path = path.replace(".log", "_error.log")
            error_handler = logging.FileHandler(err_path, "a")
            error_handler.setLevel(logging.WARNING)
            error_handler.setFormatter(pformatter)
            logger.addHandler(error_handler)

    if console:
        cformatter = ColorFormatter(**formats)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(cformatter)
        logger.addHandler(stream_handler)

    if uncaught:
        # Assign the excepthook to the handler
        sys.excepthook = lambda x, y, z: handle_unhandled_exception(logger, x, y, z)


class Logger:
    def __init__(
        self,
        logger: logging.Logger | None = None,
        log_level: int = logging.INFO,
        file_log_level: int | None = None,
        console_log_level: int | None = None,
        **fmt_kwargs,
    ):
        self.logger = logger if logger else self.get_default_logger()
        self.logger.setLevel(log_level)
        self.log_level = log_level
        self.file_log_level = file_log_level if file_log_level else log_level
        self.console_log_level = console_log_level if console_log_level else log_level
        self.log_name: str | None = None
        self.log_path: str | None = None
        self.name_suffix = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        self.fmt_kwargs = fmt_kwargs

    @staticmethod
    def get_default_logger(name: str | None = None):
        return logging.getLogger(name)

    def config(
        self,
        log_name: str | None = None,
        log_dir: str | None = None,
        time_suffix: bool = True,
        error_log: bool = False,
        console_log: bool = True,
        uncaught_log: bool = True,
    ):
        # set file handler is there are log_name and log_dir passed in
        if log_name and log_dir:
            self.log_name = (
                log_name if not time_suffix else f"{log_name}_{self.name_suffix}"
            )
            self.log_path = os.path.join(log_dir, f"{self.log_name}.log")
            self._set_file_handler(self.log_path, error_log=error_log)

        if console_log:
            self._set_console_handler()

        if uncaught_log:
            sys.excepthook = lambda x, y, z: handle_unhandled_exception(
                self.logger, x, y, z
            )
        # return self, self.logger
        return self

    def set_handler(
        self,
        handler_name: str,
        level: int = logging.INFO,
        formatter: logging.Formatter | None = None,
        filter: logging.Filter | None = None,
        **kwargs,
    ):
        handler = getattr(logging, handler_name, None)
        if not handler:
            raise ValueError(f"Handler {handler} not found.")

        if "filename" in kwargs:
            pl.Path(kwargs["filename"]).parent.mkdir(parents=True, exist_ok=True)

        new_handler = handler(**kwargs)
        new_handler.setLevel(level)
        if formatter:
            new_handler.setFormatter(formatter)
        if filter:
            new_handler.addFilter(filter)
        self.logger.addHandler(new_handler)

    def _set_file_handler(self, path: str, error_log: bool = False):
        pformatter = PlainFormatter(**self.fmt_kwargs)
        self.set_handler(
            "FileHandler",
            level=self.file_log_level,
            formatter=pformatter,
            filename=path,
            mode="a",
        )

        if error_log:
            err_path = path.replace(".log", "_error.log")
            self.set_handler(
                "FileHandler",
                level=logging.WARNING,
                formatter=pformatter,
                filename=err_path,
                mode="a",
            )

    def _set_console_handler(self):
        try:
            from rich.logging import RichHandler

            handler = RichHandler(rich_tracebacks=True, level=self.console_log_level)
            self.logger.addHandler(handler)
        except ImportError:
            cformatter = ColorFormatter(**self.fmt_kwargs)
            self.set_handler(
                "StreamHandler",
                level=self.console_log_level,
                formatter=cformatter,
            )


def handle_unhandled_exception(logger, exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # Will call default excepthook
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    # Create a critical level log message with info from the except hook.
    logger.critical(
        "Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback)
    )


def log_file(log_dir: str, config_name: str, name_suffix: str | None = None) -> str:
    if not name_suffix:
        name_suffix = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join(log_dir, f"{config_name}_{name_suffix}.log")


def get_log_name(config_name: str):
    name_suffix = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    return f"{config_name}_{name_suffix}"


def get_run_time(log_name: str, config_name: str) -> str:
    elapsed = datetime.datetime.now() - datetime.datetime.strptime(
        log_name.replace(config_name + "_", ""), "%b%d_%H-%M-%S"
    ).replace(year=datetime.datetime.now().year)
    return format_time(elapsed.seconds)


def email_logger(
    title: str,
    log_file: str | None = None,
    *,
    err: str | None = None,
) -> None:
    if not err:
        full_title = f"Success: {title}"
        message = f"{File(log_file).load()}" if log_file else ""
        send_email("code.run.log@gmail.com", full_title, message)
    else:
        full_title = f"Error: {title}"
        message = f"{err}\n\n\n\n\n{File(log_file).load()}" if log_file else f"{err}"
        send_email("code.run.log@gmail.com", full_title, message)


def _emlogger(
    func=None,
    *,
    title: str | None = None,
    log_path: str | None = None,
):
    def wrapper(func, *args, **kwargs):
        lname = title if title else func.__name__
        try:
            results = func(*args, **kwargs)
            email_logger(lname, log_path)
            return results
        except Exception as e:
            logger.error(f"Error: {e}", stack_info=True, exc_info=True)
            email_logger(lname, log_path, err=f"{e}")
            return None

    if func is not None:
        return functools.wraps(func)(functools.partial(wrapper, func))

    def decorator(func):
        return functools.wraps(func)(functools.partial(wrapper, func))

    return decorator


def emlogger(
    func=None,
    *,
    title: str | None = None,
    log_path: str | None = None,
):
    if func is None:
        return functools.partial(emlogger, title=title, log_path=log_path)

    @functools.wraps(func)
    def wrapper(func, *args, **kwargs):
        lname = title if title else func.__name__
        try:
            results = func(*args, **kwargs)
            email_logger(lname, log_path)
            return results
        except Exception as e:
            logger.error(f"Error: {e}", stack_info=True, exc_info=True)
            email_logger(lname, log_path, err=f"{e}")
            return None

    return wrapper
