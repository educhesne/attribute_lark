import logging
from contextlib import contextmanager
import pytest
from io import StringIO

from attribute_lark import AttributeLark, logger

try:
    import interegular
except ImportError:
    interegular = None


@contextmanager
def capture_log():
    stream = StringIO()
    orig_handler = logger.handlers[0]
    del logger.handlers[:]
    logger.addHandler(logging.StreamHandler(stream))
    yield stream
    del logger.handlers[:]
    logger.addHandler(orig_handler)


def test_debug():
    logger.setLevel(logging.DEBUG)
    collision_grammar = """
    start: as as { stack[-1] = stack[-1] }
    as: a* { stack[-1] = list(stack[-1]) }
    a: "a" { stack[-1] = stack[-1] }
    """
    with capture_log() as log:
        AttributeLark.from_string(collision_grammar, debug=True)

    log = log.getvalue()
    # since there are conflicts about A
    # symbol A should appear in the log message for hint
    assert "A" in log

def test_non_debug():
    logger.setLevel(logging.WARNING)
    collision_grammar = """
    start: as as { stack[-1] = stack[-1] }
    as: a* { stack[-1] = list(stack[-1]) }
    a: "a" { stack[-1] = stack[-1] }
    """
    with capture_log() as log:
        AttributeLark.from_string(collision_grammar, debug=False)
    log = log.getvalue()
    # no log message
    assert log == ""

def test_loglevel_higher():
    logger.setLevel(logging.ERROR)
    collision_grammar = """
    start: as as { stack[-1] = stack[-1] }
    as: a* { stack[-1] = list(stack[-1]) }
    a: "a" { stack[-1] = stack[-1] }
    """
    with capture_log() as log:
        AttributeLark.from_string(collision_grammar, debug=True)
    log = log.getvalue()
    # no log message
    assert len(log) == 0

@pytest.mark.skipif(interegular is None, reason="interegular is not installed, can't test regex collisions")
def test_regex_collision():
    logger.setLevel(logging.WARNING)
    collision_grammar = """
    start: A | B { stack[-1] = stack[-1] }
    A: /a+/ { stack[-1] = stack[-1] }
    B: /(a|b)+/ { stack[-1] = stack[-1] }
    """
    with capture_log() as log:
        AttributeLark.from_string(collision_grammar)

    log = log.getvalue()
    # since there are conflicts between A and B
    # symbols A and B should appear in the log message
    assert "A" in log
    assert "B" in log

@pytest.mark.skipif(interegular is None, reason="interegular is not installed, can't test regex collisions")
def test_no_regex_collision():
    logger.setLevel(logging.WARNING)
    collision_grammar = """
    start: A " " B { stack[-1] = stack[-1] }
    A: /a+/ { stack[-1] = stack[-1] }
    B: /(a|b)+/ { stack[-1] = stack[-1] }
    """
    with capture_log() as log:
        AttributeLark.from_string(collision_grammar)

    log = log.getvalue()
    assert log == ""
