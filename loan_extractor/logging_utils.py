import logging

from .normalizers import mask_sensitive


class MaskingFilter(logging.Filter):
    """Masks SSNs and account numbers in log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg, record.args = mask_sensitive(record.getMessage()), None
        return True


def setup_logging(level=logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
        root.addHandler(handler)
    for handler in root.handlers:
        if not any(isinstance(f, MaskingFilter) for f in handler.filters):
            handler.addFilter(MaskingFilter())