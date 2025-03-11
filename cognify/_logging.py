import logging
import warnings
import optuna
import threading


def _create_default_formatter(log_level) -> logging.Formatter:
    """Create a default formatter of log messages.

    This function is not supposed to be directly accessed by library users.
    """
    if log_level == "DEBUG":
        header = "[%(levelname)1s %(asctime)s - %(pathname)s:%(lineno)d]"
    else:
        header = "[%(levelname)1s %(asctime)s]"
    message = "%(message)s"
    formatter = logging.Formatter(
        fmt=f"{header} {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return formatter


_handler = None
_handler_lock = threading.Lock()


def _configure_logger(log_level):
    global _handler
    with _handler_lock:
        # config root logger
        _handler = logging.StreamHandler()
        _handler.setFormatter(_create_default_formatter(log_level))
        root_logger = logging.getLogger()
        root_logger.addHandler(_handler)
        root_logger.setLevel(log_level)

        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("absl").setLevel(logging.WARNING)
        logging.getLogger("datasets").setLevel(logging.WARNING)

        warnings.filterwarnings("ignore", module="pydantic")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        warnings.filterwarnings(
            "ignore", category=optuna.exceptions.ExperimentalWarning, module="optuna"
        )
        warnings.filterwarnings("ignore", category=FutureWarning)

