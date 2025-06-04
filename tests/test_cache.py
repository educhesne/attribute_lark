from __future__ import absolute_import

import pytest
from io import BytesIO

from attribute_lark import AttributeLark, Tree, Transformer, UnexpectedInput
from attribute_lark.lexer import Token
from attribute_lark.common import Lexer
from attribute_lark import utils as lark_module

try:
    import regex
except ImportError:
    regex = None


class MockFile(BytesIO):
    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MockFS:
    def __init__(self):
        self.files = {}

    def open(self, name, mode="r", **kwargs):
        if name not in self.files:
            if "r" in mode:
                raise FileNotFoundError(name)
            f = self.files[name] = MockFile()
        else:
            f = self.files[name]
            f.seek(0)
        return f

    def exists(self, name):
        return name in self.files


class CustomLexer(Lexer):
    def __init__(self, lexer_conf):
        pass

    def lex(self, data):
        for obj in data:
            yield Token("A", obj)


class InlineTestT(Transformer):
    def add(self, children):
        return sum(children if isinstance(children, list) else children.children)

    def NUM(self, token):
        return int(token)

    def __reduce__(self):
        raise TypeError("This Transformer should not be pickled.")


def append_zero(t):
    return t.update(value=t.value + "0")


@pytest.fixture
def mock_fs(monkeypatch):
    fs = getattr(lark_module, 'mock_fs', None)
    mock_fs = MockFS()
    monkeypatch.setattr(lark_module, 'mock_fs', mock_fs)
    yield mock_fs
    mock_fs.files = {}
    if fs is not None:
        monkeypatch.setattr(lark_module, 'mock_fs', fs)


TEST_GRAMMAR = '''start: "a" { stack[-1] = stack[-1] }'''


def test_simple(mock_fs):
    fn = "bla"

    AttributeLark(TEST_GRAMMAR, cache=fn)
    assert fn in mock_fs.files
    parser = AttributeLark(TEST_GRAMMAR, cache=fn)
    tree, attr = parser.parse("a")
    assert tree == Tree("start", [])


def test_automatic_naming(mock_fs):
    assert len(mock_fs.files) == 0
    AttributeLark(TEST_GRAMMAR, cache=True)
    assert len(mock_fs.files) == 1
    parser = AttributeLark(TEST_GRAMMAR, cache=True)
    tree, attr = parser.parse("a")
    assert tree == Tree("start", [])

    parser = AttributeLark(TEST_GRAMMAR + ' "b" { stack.append(stack.pop(-2) + stack.pop()) }', cache=True)
    assert len(mock_fs.files) == 2
    tree, attr = parser.parse("ab")
    assert tree == Tree("start", [])


def test_custom_lexer(mock_fs):
    parser = AttributeLark(TEST_GRAMMAR, lexer=CustomLexer, cache=True)
    parser = AttributeLark(TEST_GRAMMAR, lexer=CustomLexer, cache=True)
    assert len(mock_fs.files) == 1
    tree, attr = parser.parse("a")
    assert tree == Tree("start", [])


def test_options(mock_fs):
    # Test options persistence
    AttributeLark(TEST_GRAMMAR, debug=True, cache=True)
    parser = AttributeLark(TEST_GRAMMAR, debug=True, cache=True)
    assert parser.options.debug is True


def test_inline(mock_fs):
    # Test inline transformer (tree-less) & lexer_callbacks
    g = r"""
    start: add+ { stack[-1] = list(stack[-1]) }
    add: NUM "+" NUM { stack[-1] = int(stack[-3]) + int(stack[-1]) }
    NUM: /\d+/
    %ignore " "
    """
    text = "1+2 3+4"
    expected = [3, 7]

    parser = AttributeLark(
        g,
        transformer=InlineTestT(),
        cache=True,
        lexer_callbacks={"NUM": append_zero}
    )
    tree, attr = parser.parse(text)
    res0 = attr

    parser = AttributeLark(
        g,
        transformer=InlineTestT(),
        cache=True,
        lexer_callbacks={"NUM": append_zero}
    )
    assert len(mock_fs.files) == 1
    tree, attr = parser.parse(text)
    res1 = attr

    tree, attr = AttributeLark(g, cache=True, lexer_callbacks={"NUM": append_zero}).parse(text)
    res2 = InlineTestT().transform(tree)
    assert res0 == res1 == expected
    assert res2 == Tree("start", [30, 70])


def test_imports(mock_fs):
    g = """
    %import .grammars.ab (startab, expr)
    """
    parser = AttributeLark(g, start="startab", cache=True, source_path=__file__)
    assert len(mock_fs.files) == 1
    parser = AttributeLark(g, start="startab", cache=True, source_path=__file__)
    assert len(mock_fs.files) == 1
    tree, _ = parser.parse("ab")
    assert tree == Tree("startab", [Tree("expr", ["a", "b"])])


@pytest.mark.skipif(regex is None, reason="'regex' lib not installed")
def test_recursive_pattern(mock_fs):
    g = """
    start: recursive+ { stack[-1] = list(stack[-1]) }
    recursive: /\\w{3}\\d{3}(?R)?/ { stack[-1] = stack[-1] }
    """

    assert len(mock_fs.files) == 0
    AttributeLark(g, regex=True, cache=True)
    assert len(mock_fs.files) == 1

    with pytest.warns(None) as record:
        AttributeLark(g, regex=True, cache=True)
        assert len(mock_fs.files) == 1
    assert not record  # No warnings should be recorded


def test_error_message(mock_fs):
    g = r"""
    start: add+
    add: /\d+/ "+" /\d+/
    %ignore " "
    """
    texts = ("1+", "+1", "", "1 1+1")

    parser1 = AttributeLark(g, cache=True)
    parser2 = AttributeLark(g, cache=True)
    assert len(mock_fs.files) == 1

    for text in texts:
        with pytest.raises(UnexpectedInput) as cm1:
            parser1.parse(text)
        with pytest.raises(UnexpectedInput) as cm2:
            parser2.parse(text)
        assert str(cm1.value) == str(cm2.value)
