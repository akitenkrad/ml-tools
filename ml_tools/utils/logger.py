import logging
from datetime import datetime, timedelta, timezone
from logging import Formatter, Logger, StreamHandler, basicConfig, getLogger
from logging.handlers import RotatingFileHandler
from pathlib import Path

HANDLER_FORMAT = "%(asctime)s [%(levelname)8s] %(name)15s - %(message)s"


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
        log_dir = Path("./logs") / now.strftime("%Y%m%d%H%M%S%f")
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
        # --------------------------------
        # 3. log file configuration
        # --------------------------------
        fh = RotatingFileHandler(str(log_dir / name), maxBytes=2**30, backupCount=3000)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(Formatter(""))
        logger.addHandler(fh)

        # --------------------------------
        # 4. error log file configuration
        # --------------------------------
        er_fh = RotatingFileHandler(str(log_dir / name), maxBytes=2**30, backupCount=3000)
        er_fh.setLevel(logging.ERROR)
        er_fh.setFormatter(Formatter(""))
        logger.addHandler(er_fh)

    return logger


def get_logger(name, logfile: str = "", silent: bool = False, loglevel: int = logging.DEBUG) -> Logger:
    """Generate Logger instance

    Args:
        name (str): name of the logger
        logfile (str): logfile name
        silent (bool): if True, not log into stream
    Returns:
        Logger
    """

    logger = getLogger(name)

    # --------------------------------
    # 1. mkdir
    # --------------------------------
    if logfile == "":
        JST = timezone(timedelta(hours=+9), "JST")
        now = datetime.now(JST)
        now = datetime(now.year, now.month, now.day, now.hour, now.minute, 0, tzinfo=JST)
        log_dir = Path("./logs") / now.strftime("%Y%m%d%H%M%S%f")
        log_dir.mkdir(parents=True, exist_ok=True)
        logfile = name
    else:
        log_dir = Path(logfile).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        logfile = Path(logfile).name

    # --------------------------------
    # 2. remove handlers
    # --------------------------------
    while len(logger.handlers) > 0:
        for h in logger.handlers:
            logger.removeHandler(h)
            h.close()
    logger.setLevel(loglevel)
    logger.propagate = False

    # --------------------------------
    # 3. handler configuration
    # --------------------------------
    if not silent:
        stream_handler = StreamHandler()
        stream_handler.setLevel(loglevel)
        stream_handler.setFormatter(Formatter(HANDLER_FORMAT))
        logger.addHandler(stream_handler)

    # --------------------------------
    # 4. log file configuration
    # --------------------------------
    fh = RotatingFileHandler(str(log_dir / logfile), maxBytes=3145728, backupCount=3000, encoding="utf-8")
    fh.setLevel(loglevel)
    fh.setFormatter(Formatter(HANDLER_FORMAT))
    logger.addHandler(fh)

    return logger


def kill_logger(logger: Logger):
    name = logger.name
    if name in logging.Logger.manager.loggerDict:
        del logging.Logger.manager.loggerDict[name]
