import logging
from datetime import datetime, timedelta, timezone
from logging import Formatter, Logger, StreamHandler, getLogger
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_json_liner(name: str, logfile: str = "") -> Logger:
    """Generate Logger instance

    Args:
        name: (str) name of the logger
        logfile: (str) logfile name
    Returns:
        Logger
    """

    # --------------------------------
    # 0. mkdir
    # --------------------------------
    if logfile == "":
        JST = timezone(timedelta(hours=+9), "JST")
        now = datetime.now(JST)
        now = datetime(now.year, now.month, now.day, now.hour, now.minute, 0, tzinfo=JST)
        log_dir = Path("./logs") / now.strftime("%Y%m%d%H%M%S")
        log_dir.mkdir(parents=True, exist_ok=True)
        logfile = name
    else:
        log_dir = Path(logfile).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        logfile = Path(logfile).name

    # --------------------------------
    # 1. logger configuration
    # --------------------------------
    logger = getLogger(name)
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        handler_format = Formatter("")
        # --------------------------------
        # 3. log file configuration
        # --------------------------------
        fh = RotatingFileHandler(str(log_dir / name), maxBytes=2**30, backupCount=3000)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(handler_format)
        logger.addHandler(fh)

        # --------------------------------
        # 4. error log file configuration
        # --------------------------------
        er_fh = RotatingFileHandler(str(log_dir / name), maxBytes=2**30, backupCount=3000)
        er_fh.setLevel(logging.ERROR)
        er_fh.setFormatter(handler_format)
        logger.addHandler(er_fh)

    return logger


def get_logger(name, logfile: str = "", silent: bool = False) -> Logger:
    """Generate Logger instance

    Args:
        name (str): name of the logger
        logfile (str): logfile name
        silent (bool): if True, not log into stream
    Returns:
        Logger
    """

    # --------------------------------
    # 0. mkdir
    # --------------------------------
    if logfile == "":
        JST = timezone(timedelta(hours=+9), "JST")
        now = datetime.now(JST)
        now = datetime(now.year, now.month, now.day, now.hour, now.minute, 0, tzinfo=JST)
        log_dir = Path("./logs") / now.strftime("%Y%m%d%H%M%S")
        log_dir.mkdir(parents=True, exist_ok=True)
        logfile = name
    else:
        log_dir = Path(logfile).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        logfile = Path(logfile).name

    # --------------------------------
    # 1. logger configuration
    # --------------------------------
    logger = getLogger(name)
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.setLevel(logging.DEBUG)

    if not logger.hasHandlers():
        handler_format = Formatter("%(asctime)s [%(levelname)8s] %(name)15s - %(message)s")

        # --------------------------------
        # 2. handler configuration
        # --------------------------------
        if not silent:
            stream_handler = StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(handler_format)
            logger.addHandler(stream_handler)

        # --------------------------------
        # 3. log file configuration
        # --------------------------------
        fh = RotatingFileHandler(str(log_dir / logfile), maxBytes=3145728, backupCount=3000)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(handler_format)
        logger.addHandler(fh)

        # --------------------------------
        # 4. error log file configuration
        # --------------------------------
        er_fh = RotatingFileHandler(str(log_dir / logfile), maxBytes=3145728, backupCount=3000)
        er_fh.setLevel(logging.ERROR)
        er_fh.setFormatter(handler_format)
        logger.addHandler(er_fh)

    return logger


def kill_logger(logger: Logger):
    name = logger.name
    if name in logging.Logger.manager.loggerDict:
        del logging.Logger.manager.loggerDict[name]
