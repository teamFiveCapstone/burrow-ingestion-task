import logging
import sys
import json
import datetime
import traceback

SERVICE_NAME = "ingestion-task"
ENV = "production"

logger = logging.getLogger(SERVICE_NAME)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)
logger.propagate = False


def log_info(message, **fields):
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat(timespec="milliseconds")
        + "Z",
        "level": "info",
        "message": message,
        "service": SERVICE_NAME,
        "env": ENV,
        **fields,
    }
    logger.info(json.dumps(record))


def log_error(message, **fields):
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat(timespec="milliseconds")
        + "Z",
        "level": "error",
        "message": message,
        "service": SERVICE_NAME,
        "env": ENV,
        **fields,
    }
    logger.error(json.dumps(record))


def log_exception(message, **fields):
    fields["stack"] = traceback.format_exc()
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat(timespec="milliseconds")
        + "Z",
        "level": "error",
        "message": message,
        "service": SERVICE_NAME,
        "env": ENV,
        **fields,
    }
    logger.error(json.dumps(record))
