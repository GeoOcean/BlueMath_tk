import contextlib
import io
import logging
import unittest

from bluemath_tk.core import models
from bluemath_tk.core.models import BlueMathModel, set_models_console_level


class _DummyModel(BlueMathModel):
    def __init__(self, logger: logging.Logger | None = None) -> None:
        super().__init__()
        if logger is not None:
            self._logger = logger


class TestModelsConsoleLevel(unittest.TestCase):
    def setUp(self):
        self.addCleanup(set_models_console_level, models.MODELS_CONSOLE_LEVEL)
        set_models_console_level("ERROR")

    def test_new_model_is_quiet_on_console(self):
        with contextlib.redirect_stderr(io.StringIO()) as buf:
            _DummyModel().logger.warning("routine message")
        self.assertEqual(buf.getvalue(), "")

    def test_unpickled_logger_is_repaired(self):
        # a model loaded from disk refers to its logger by name; in a new
        # process that logger has no handlers and propagates to the root
        bare = logging.getLogger("_BareUnpickledModel")
        bare.handlers = []
        bare.propagate = True
        model = _DummyModel(bare)
        with contextlib.redirect_stderr(io.StringIO()) as buf:
            model.logger.warning("must not reach the console")
        self.assertEqual(buf.getvalue(), "")
        self.assertFalse(model.logger.propagate)
        self.assertEqual(model.logger.name, "_BareUnpickledModel")

    def test_existing_model_loggers_follow_the_level(self):
        logger = _DummyModel().logger
        set_models_console_level("CRITICAL")
        consoles = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]
        self.assertTrue(consoles)
        self.assertTrue(all(h.level == logging.CRITICAL for h in consoles))


if __name__ == "__main__":
    unittest.main()
