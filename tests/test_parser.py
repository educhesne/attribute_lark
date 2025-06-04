# -*- coding: utf-8 -*-
from __future__ import absolute_import

import re
import os
import sys
import pytest
import unittest
from copy import deepcopy
from io import StringIO as uStringIO, open
from typing import (
    Any, Tuple, TypeVar, Generic, Union, Optional,
    List, Iterator
)
from abc import ABC, abstractmethod

from attribute_lark import UnexpectedCharacters, ParseError, UnexpectedInput
from attribute_lark.lexer import Lexer
from attribute_lark.common import LexerConf
from attribute_lark.parse_tree_builder import ParseTreeBuilder
from attribute_lark.tree import Tree, Token
from attribute_lark.visitors import Transformer
from dataclasses import dataclass

from attribute_lark import (
    AttributeLark as Lark,  # Main parser class
    Tree, ParseTree, Token,  # Core data structures
    GrammarError, UnexpectedToken, v_args
)

from attribute_lark.visitors import (
    Transformer_NonRecursive
)

T = TypeVar('T')
Branch = Union[ParseTree, Token]

@dataclass
class ParseResult(Generic[T]):
    """Result from parsing, combining tree and attribute values."""
    _result: Union[ParseTree, Tuple[ParseTree, T]]

    @property
    def tree(self) -> ParseTree:
        """Get the parse tree."""
        if isinstance(self._result, tuple):
            return self._result[0]
        return self._result

    @property
    def attrs(self) -> Optional[T]:
        """Get the computed attributes, if any."""
        if isinstance(self._result, tuple):
            return self._result[1]
        return None

from attribute_lark.tree import Tree

# Base abstract lexer class
class LexerBase(ABC):
    """Abstract base class for lexers"""
    @abstractmethod
    def __init__(self, lexer_conf: LexerConf):
        """Initialize the lexer with configuration."""
        pass

    @abstractmethod
    def lex(self, lexer_state: str, parser_state: Optional['ParseTreeBuilder'] = None) -> Iterator[Token]:
        """Lex the input and return an iterator of tokens."""
        pass

# Basic lexer implementation
class CustomLexerNew(LexerBase):
    """New-style lexer implementation"""
    def __init__(self, lexer_conf):
        self.lexer_conf = lexer_conf

    def lex(self, lexer_state, parser_state) -> Iterator[Token]:
        # Basic implementation that returns no tokens
        return iter(())

    @property
    def __future_interface__(self):
        return 2

class CustomLexerOld1(LexerBase):
    """Old v1 lexer implementation"""
    def __init__(self, lexer_conf):
        self.lexer_conf = lexer_conf

    def lex(self, lexer_state, parser_state) -> Iterator[Token]:
        return iter(())

    @property
    def __future_interface__(self):
        return 1

class CustomLexerOld0(LexerBase):
    """Old v0 lexer implementation"""
    def __init__(self, lexer_conf):
        self.lexer_conf = lexer_conf

    def lex(self, lexer_state) -> Iterator[Token]:
        parser_state = None
        return iter(())

    def make_lexer_state(self, text):
        return text

    @property
    def __future_interface__(self):
        return 0

LEXER_CLASSES = {
    'custom_new': CustomLexerNew,
    'custom_old1': CustomLexerOld1,
    'custom_old0': CustomLexerOld0,
}

@dataclass
class ParseResult(Generic[T]):
    """Result from parsing that combines tree and attribute values"""
    _result: Union[ParseTree, Tuple[ParseTree, T]]

    @property
    def tree(self) -> ParseTree:
        """Get the parse tree"""
        if isinstance(self._result, tuple):
            return self._result[0]
        return self._result

    @property
    def attrs(self) -> Optional[T]:
        """Get attribute values if present"""
        if isinstance(self._result, tuple):
            return self._result[1]
        return None

    @property
    def children(self) -> List[Branch]:
        """Get children of parse tree"""
        return self.tree.children

    @property
    def data(self) -> str:
        """Get tree node data"""
        return self.tree.data

    @property
    def meta(self) -> Any:
        """Get tree node metadata"""
        return self.tree.meta

    def unwrap(self) -> Union[ParseTree, Tuple[ParseTree, T]]:
        """Get the raw parse result"""
        return self._result

def _wrap_tree(tree: Tree) -> ParseResult:
    """Wrap a tree in a ParseResult object"""
    return ParseResult((tree, None))

def _wrap_parse_result(result: Union[ParseTree, Tuple[ParseTree, T]]) -> ParseResult[T]:
    """Wrap a parse result in a ParseResult object"""
    return ParseResult(result)

def tree_equal_including_attrs(a: Tree, b: Tree) -> bool:
    """Compare two trees for equality, including attributes and metadata"""
    if not isinstance(a, type(b)):
        return False

    if a.data != b.data:
        return False
    if len(a.children) != len(b.children):
        return False

    for ca, cb in zip(a.children, b.children):
        if not isinstance(ca, type(cb)):
            return False
        if isinstance(ca, Tree) and isinstance(cb, Tree):
            if not tree_equal_including_attrs(ca, cb):
                return False
        elif isinstance(ca, Token) and isinstance(cb, Token):
            if ca.type != cb.type or ca.value != cb.value:
                return False
        else:
            if ca != cb:
                return False
    return True

# Test fixtures and configuration
@pytest.fixture(params=['basic', 'contextual', 'custom_new'])
def lexer_type(request):
    return request.param

@pytest.fixture
def make_parser(lexer_type):
    def _make(grammar, **kwargs):
        lexer = LEXER_CLASSES.get(lexer_type, lexer_type)
        return Lark(grammar, lexer=lexer, propagate_positions=True, **kwargs)
    return _make

@pytest.fixture
def parser(make_parser):
    grammar = """
        start: "a"+ "b" { stack[-1] = stack[-1] }
        %ignore " "
    """
    return make_parser(grammar)

@pytest.fixture
def make_attr_parser():
    def _make(grammar_text: str, **kwargs):
        parser = Lark(grammar_text, propagate_positions=True, attribute_eval=True, **kwargs)
        def parse(text: str) -> ParseResult:
            result = parser.parse(text)
            return ParseResult(result)
        return parse
    return _make

@pytest.fixture
def basic_attr_parser(make_attr_parser):
    grammar = """
        start: expr { stack[-1] = stack[-1] }
        expr: term "+" term { stack[-1] = stack[-3] + stack[-1] }
            | term { stack[-1] = stack[-1] }
        term: NUMBER { stack[-1] = int(stack[-1]) }
        NUMBER: /[0-9]+/
        %ignore " "
    """
    return make_attr_parser(grammar)

@pytest.fixture
def list_attr_parser(make_attr_parser):
    grammar = """
        start: list { stack[-1] = stack[-1] }
        list: "[" items "]" { stack[-1] = stack[-2] }
        items: item ("," item)* { stack[-1] = [x for x in stack[-len(stack):]] }
             | { stack.append([]) }
        item: NUMBER { stack[-1] = int(stack[-1]) }
        NUMBER: /[0-9]+/
        %ignore " "
    """
    return make_attr_parser(grammar)

def test_big_list():
    """Test parsing with large number of alternatives"""
    grammar = """
        start: {}
    """.format("|".join([f'"{i}"' for i in range(250)]))

    Lark(grammar)

def test_infinite_recurse():
    """Test handling of infinite recursion in grammar"""
    g = """start: a
           a: a | "a"
        """
    with pytest.raises(GrammarError):
        Lark(g)

def test_propagate_positions():
    g = Lark(
        """start: a
                a: "a"
             """,
        propagate_positions=True,
    )

    r = g.parse("a")
    assert r.children[0].meta.line == 1

    g = Lark(
        """start: x
                x: a
                a: "a"
             """,
        propagate_positions=True,
    )

    r = g.parse("a")
    assert r.children[0].meta.line == 1

def test_propagate_positions2():
    g = Lark(
        """start: a
                a: b
                ?b: "(" t ")"
                !t: "t"
             """,
        propagate_positions=True,
    )

    start = g.parse("(t)")
    (a,) = start.children
    (t,) = a.children
    assert t.children[0] == "t"

    assert t.meta.column == 2
    assert t.meta.end_column == 3

    assert start.meta.column == a.meta.column == 1
    assert start.meta.end_column == a.meta.end_column == 4

def test_expand1():
    g = Lark("""start: a
                ?a: b
                b: "x"
             """)

    r = g.parse("x")
    assert r.children[0].data == "b"

    g = Lark("""start: a
                ?a: b -> c
                b: "x"
             """)

    r = g.parse("x")
    assert r.children[0].data == "c"

    g = Lark("""start: a
                ?a: B -> c
                B: "x"
             """)
    assert r.children[0].data == "c"

    g = Lark("""start: a
                ?a: b b -> c
                b: "x"
             """)
    r = g.parse("xx")
    assert r.children[0].data == "c"

def test_comment_in_rule_definition():
    g = Lark("""start: a
           a: "a"
            // A comment
            // Another comment
            | "b"
            // Still more

           c: "unrelated"
        """)
    r = g.parse("b")
    assert r.children[0].data == "a"

def test_visit_tokens():
    class T(Transformer):
        def a(self, children):
            return children[0] + "!"

        def A(self, tok):
            return tok.update(value=tok.upper())

    # Test regular
    g = """start: a
        a : A
        A: "x"
        """
    p = Lark(g)
    r = T(False).transform(p.parse("x"))
    assert r.children == ["x!"]
    r = T().transform(p.parse("x"))
    assert r.children == ["X!"]

    # Test internal transformer
    p = Lark(g, transformer=T())
    r = p.parse("x")
    assert r.children == ["X!"]

def test_visit_tokens2():
    g = """
    start: add+
    add: NUM "+" NUM
    NUM: /\\d+/
    %ignore " "
    """
    text = "1+2 3+4"
    expected = Tree("start", [3, 7])
    for base in (
        Transformer,
        Transformer_NonRecursive,
    ):

        class T(base):
            def add(self, children):
                return sum(
                    children if isinstance(children, list) else children.children
                )

            def NUM(self, token):
                return int(token)

        parser = Lark(g, transformer=T())
        result = parser.parse(text)
        assert result == expected

def test_vargs_meta():
    @v_args(meta=True)
    class T1(Transformer):
        def a(self, meta, children):
            assert not children
            return meta.line

        def start(self, meta, children):
            return children

    @v_args(meta=True, inline=True)
    class T2(Transformer):
        def a(self, meta):
            return meta.line

        def start(self, meta, *res):
            return list(res)

    for T in (T1, T2):
        for internal in [False, True]:
            try:
                g = Lark(
                    r"""start: a+
                            a : "x" _NL?
                            _NL: /\n/+
                        """,
                    transformer=T() if internal else None,
                    propagate_positions=True,
                )
            except NotImplementedError:
                assert internal
                continue

            res = g.parse("xx\nx\nxxx\n\n\nxx")
            assert not internal
            res = T().transform(res)

            assert res == [1, 1, 2, 3, 3, 3, 6, 6]

def test_vargs_tree():
    tree = Lark("""
        start: a a a
        !a: "A"
    """).parse("AAA")
    tree_copy = deepcopy(tree)

    @v_args(tree=True)
    class T(Transformer):
        def a(self, tree):
            return 1

        def start(self, tree):
            return tree.children

    res = T().transform(tree)
    assert res == [1, 1, 1]
    assert tree == tree_copy

def test_embedded_transformer():
    class T(Transformer):
        def a(self, children):
            return "<a>"

        def b(self, children):
            return "<b>"

        def c(self, children):
            return "<c>"

    # Test regular
    g = Lark("""start: a
                a : "x"
             """)
    r = T().transform(g.parse("x"))
    assert r.children == ["<a>"]

    g = Lark(
        """start: a
                a : "x"
             """,
        transformer=T(),
    )
    r = g.parse("x")
    assert r.children == ["<a>"]

    # Test Expand1
    g = Lark("""start: a
                ?a : b
                b : "x"
             """)
    r = T().transform(g.parse("x"))
    assert r.children == ["<b>"]

    g = Lark(
        """start: a
                ?a : b
                b : "x"
             """,
        transformer=T(),
    )
    r = g.parse("x")
    assert r.children == ["<b>"]

    # Test Expand1 -> Alias
    g = Lark("""start: a
                ?a : b b -> c
                b : "x"
             """)
    r = T().transform(g.parse("xx"))
    assert r.children == ["<c>"]

    g = Lark(
        """start: a
                ?a : b b -> c
                b : "x"
             """,
        transformer=T(),
    )
    r = g.parse("xx")
    assert r.children == ["<c>"]

def test_embedded_transformer_inplace():
    @v_args(tree=True)
    class T1(Transformer_In_Place):
        def a(self, tree):
            assert isinstance(tree, Tree), tree
            tree.children.append("tested")
            return tree

        def b(self, tree):
            return Tree(tree.data, tree.children + ["tested2"])

    @v_args(tree=True)
    class T2(Transformer):
        def a(self, tree):
            assert isinstance(tree, Tree), tree
            tree.children.append("tested")
            return tree

        def b(self, tree):
            return Tree(tree.data, tree.children + ["tested2"])

    class T3(Transformer):
        @v_args(tree=True)
        def a(self, tree):
            assert isinstance(tree, Tree)
            tree.children.append("tested")
            return tree

        @v_args(tree=True)
        def b(self, tree):
            return Tree(tree.data, tree.children + ["tested2"])

    for t in [T1(), T2(), T3()]:
        for internal in [False, True]:
            g = Lark(
                """start: a b
                        a : "x"
                        b : "y"
                    """,
                transformer=t if internal else None,
            )
            r = g.parse("xy")
            if not internal:
                r = t.transform(r)

            a, b = r.children
            assert a.children == ["tested"]
            assert b.children == ["tested2"]

def test_alias():
    Lark("""start: ["a"] "b" ["c"] "e" ["f"] ["g"] ["h"] "x" -> d """)

def test_backwards_custom_lexer():
    class OldCustomLexer(Lexer):
        def __init__(self, lexer_conf):
            pass

        def lex(self, text):
            yield Token("A", "A")

    p = Lark(
        """
    start: A
    %declare A
    """,
        lexer=OldCustomLexer,
    )

    r = p.parse("")
    assert r == Tree("start", [Token("A", "A")])

def test_lexer_token_limit():
    "Python has a stupid limit of 100 groups in a regular expression. Test that we handle this limitation"
    tokens = {"A%d" % i: '"%d"' % i for i in range(300)}
    g = """start: %s
              %s""" % (
        " ".join(tokens),
        "\n".join("%s: %s" % x for x in tokens.items()),
    )

    p = Lark(g)

def _tree_structure_check(a, b):
    """
    Checks that both Tree objects have the same structure, without checking their values.
    """
    assert a.data == b.data and len(a.children) == len(b.children)
    for ca, cb in zip(a.children, b.children):
        assert type(ca) == type(cb)
        if isinstance(ca, Tree):
            _tree_structure_check(ca, cb)
        elif isinstance(ca, Token):
            assert ca.type == cb.type
        else:
            assert ca == cb

class DualBytesLark:
    """
    A helper class that wraps both a normal parser, and a parser for bytes.
    It automatically transforms `.parse` calls for both lexer, returning the value from the text lexer
    It always checks that both produce the same output/error

    NOTE: Not currently used, but left here for future debugging.
    """

    def __init__(self, g, *args, **kwargs):
        self.text_lexer = Lark(g, *args, use_bytes=False, **kwargs)
        g = self.text_lexer.grammar_source.lower()
        if "\\u" in g or not g.isascii():
            # Bytes re can't deal with uniode escapes
            self.bytes_lark = None
        else:
            # Everything here should work, so use `use_bytes='force'`
            self.bytes_lark = Lark(
                self.text_lexer.grammar_source, *args, use_bytes="force", **kwargs
            )

    def parse(self, text, start=None):
        # TODO: Easy workaround, more complex checks would be beneficial
        if not text.isascii() or self.bytes_lark is None:
            return self.text_lexer.parse(text, start)
        try:
            rv = self.text_lexer.parse(text, start)
        except Exception as e:
            try:
                self.bytes_lark.parse(text.encode(), start)
            except Exception as be:
                assert type(e) == type(be), (
                    "Parser with and without `use_bytes` raise different exceptions"
                )
                raise e
            assert False, "Parser without `use_bytes` raises exception, with doesn't"
        try:
            bv = self.bytes_lark.parse(text.encode(), start)
        except Exception:
            assert False, (
                "Parser without `use_bytes` doesn't raise an exception, with does"
            )
        _tree_structure_check(rv, bv)
        return rv

    @classmethod
    def open(cls, grammar_filename, rel_to=None, **options):
        if rel_to:
            basepath = os.path.dirname(rel_to)
            grammar_filename = os.path.join(basepath, grammar_filename)
        with open(grammar_filename, encoding="utf8") as f:
            return cls(f, **options)

    def save(self, f):
        self.text_lexer.save(f)
        if self.bytes_lark is not None:
            self.bytes_lark.save(f)

    def load(self, f):
        self.text_lexer = self.text_lexer.load(f)
        if self.bytes_lark is not None:
            self.bytes_lark.load(f)

def _make_parser_test(LEXER):
    lexer_class_or_name = {
        "custom_new": CustomLexerNew,
        "custom_old1": CustomLexerOld1,
        "custom_old0": CustomLexerOld0,
    }.get(LEXER, LEXER)

    def _Lark(grammar, **kwargs):
        return Lark(
            grammar, lexer=lexer_class_or_name, propagate_positions=True, **kwargs
        )

    def _Lark_open(gfilename, **kwargs):
        return Lark.open(
            gfilename, lexer=lexer_class_or_name, propagate_positions=True, **kwargs
        )

    class _TestParser(unittest.TestCase):
        def test_basic1(self):
            g = _Lark("""start: a+ b a* "b" a*
                        b: "b"
                        a: "a"
                     """)

            r = g.parse("aaabaab")
            self.assertEqual("".join(x.data for x in r.children), "aaabaa")
            r = g.parse("aaabaaba")
            self.assertEqual("".join(x.data for x in r.children), "aaabaaa")

            self.assertRaises(ParseError, g.parse, "aaabaa")

        def test_basic2(self):
            # Multiple parsers and colliding tokens
            g = _Lark("""start: B A
                         B: "12"
                         A: "1" """)
            g2 = _Lark("""start: B A
                         B: "12"
                         A: "2" """)
            x = g.parse("121")
            assert x.data == "start" and x.children == ["12", "1"], x
            x = g2.parse("122")
            assert x.data == "start" and x.children == ["12", "2"], x

        def test_stringio_unicode(self):
            """Verify that a Lark can be created from file-like objects other than Python's standard 'file' object"""
            _Lark(uStringIO('start: a+ b a* "b" a*\n b: "b"\n a: "a" '))

        def test_unicode(self):
            g = _Lark("""start: UNIA UNIB UNIA
                        UNIA: /\xa3/
                        UNIB: /\u0101/
                        """)
            g.parse("\xa3\u0101\u00a3")

        def test_unicode2(self):
            g = _Lark(r"""start: UNIA UNIB UNIA UNIC
                        UNIA: /\xa3/
                        UNIB: "a\u0101b\ "
                        UNIC: /a?\u0101c\n/
                        """)
            g.parse("\xa3a\u0101b\\ \u00a3\u0101c\n")

        def test_unicode3(self):
            g = _Lark(r"""start: UNIA UNIB UNIA UNIC
                        UNIA: /\xa3/
                        UNIB: "\u0101"
                        UNIC: /\u0203/ /\n/
                        """)
            g.parse("\xa3\u0101\u00a3\u0203\n")

        def test_unicode4(self):
            g = _Lark(r"""start: UNIA UNIB UNIA UNIC
                        UNIA: /\xa3/
                        UNIB: "\U0010FFFF"
                        UNIC: /\U00100000/ /\n/
                        """)
            g.parse("\xa3\U0010ffff\u00a3\U00100000\n")

        def test_hex_escape(self):
            g = _Lark(r"""start: A B C
                          A: "\x01"
                          B: /\x02/
                          C: "\xABCD"
                          """)
            g.parse("\x01\x02\xabCD")

        def test_unicode_literal_range_escape(self):
            g = _Lark(r"""start: A+
                          A: "\u0061".."\u0063"
                          """)
            g.parse("abc")

        def test_unicode_literal_range_escape2(self):
            g = _Lark(r"""start: A+
                          A: "\U0000FFFF".."\U00010002"
                          """)
            g.parse("\U0000ffff\U00010000\U00010001\U00010002")

        def test_hex_literal_range_escape(self):
            g = _Lark(r"""start: A+
                          A: "\x01".."\x03"
                          """)
            g.parse("\x01\x02\x03")

        def test_bytes_utf8(self):
            g = r"""
            start: BOM? char+
            BOM: "\xef\xbb\xbf"
            char: CHAR1 | CHAR2 | CHAR3 | CHAR4
            CONTINUATION_BYTE: "\x80" .. "\xbf"
            CHAR1: "\x00" .. "\x7f"
            CHAR2: "\xc0" .. "\xdf" CONTINUATION_BYTE
            CHAR3: "\xe0" .. "\xef" CONTINUATION_BYTE CONTINUATION_BYTE
            CHAR4: "\xf0" .. "\xf7" CONTINUATION_BYTE CONTINUATION_BYTE CONTINUATION_BYTE
            """
            g = _Lark(g, use_bytes=True)
            s = "🔣 地? gurīn".encode("utf-8")
            self.assertEqual(len(g.parse(s).children), 10)

            for enc, j in [
                (
                    "sjis",
                    "地球の絵はグリーンでグッド?  Chikyuu no e wa guriin de guddo",
                ),
                ("sjis", "売春婦"),
                ("euc-jp", "乂鵬鵠"),
            ]:
                s = j.encode(enc)
                self.assertRaises(UnexpectedCharacters, g.parse, s)

        def test_stack_for_ebnf(self):
            """Verify that stack depth isn't an issue for EBNF grammars"""
            g = _Lark(r"""start: a+
                         a : "a" """)

            g.parse("a" * (sys.getrecursionlimit() * 2))

        def test_expand1_lists_with_one_item(self):
            g = _Lark(r"""start: list
                            ?list: item+
                            item : A
                            A: "a"
                        """)
            r = g.parse("a")

            # because 'list' is an expand-if-contains-one rule and we only provided one element it should have expanded to 'item'
            self.assertSequenceEqual(
                [subtree.data for subtree in r.children], ("item",)
            )

            # regardless of the amount of items: there should be only *one* child in 'start' because 'list' isn't an expand-all rule
            self.assertEqual(len(r.children), 1)

        def test_expand1_lists_with_one_item_2(self):
            g = _Lark(r"""start: list
                            ?list: item+ "!"
                            item : A
                            A: "a"
                        """)
            r = g.parse("a!")

            # because 'list' is an expand-if-contains-one rule and we only provided one element it should have expanded to 'item'
            self.assertSequenceEqual(
                [subtree.data for subtree in r.children], ("item",)
            )

            # regardless of the amount of items: there should be only *one* child in 'start' because 'list' isn't an expand-all rule
            self.assertEqual(len(r.children), 1)

        def test_dont_expand1_lists_with_multiple_items(self):
            g = _Lark(r"""start: list
                            ?list: item+
                            item : A
                            A: "a"
                        """)
            r = g.parse("aa")

            # because 'list' is an expand-if-contains-one rule and we've provided more than one element it should *not* have expanded
            self.assertSequenceEqual(
                [subtree.data for subtree in r.children], ("list",)
            )

            # regardless of the amount of items: there should be only *one* child in 'start' because 'list' isn't an expand-all rule
            self.assertEqual(len(r.children), 1)

            # Sanity check: verify that 'list' contains the two 'item's we've given it
            [list] = r.children
            self.assertSequenceEqual(
                [item.data for item in list.children], ("item", "item")
            )

        def test_dont_expand1_lists_with_multiple_items_2(self):
            g = _Lark(r"""start: list
                            ?list: item+ "!"
                            item : A
                            A: "a"
                        """)
            r = g.parse("aa!")

            # because 'list' is an expand-if-contains-one rule and we've provided more than one element it should *not* have expanded
            self.assertSequenceEqual(
                [subtree.data for subtree in r.children], ("list",)
            )

            # regardless of the amount of items: there should be only *one* child in 'start' because 'list' isn't an expand-all rule
            self.assertEqual(len(r.children), 1)

            # Sanity check: verify that 'list' contains the two 'item's we've given it
            [list] = r.children
            self.assertSequenceEqual(
                [item.data for item in list.children], ("item", "item")
            )

        def test_empty_expand1_list(self):
            g = _Lark(r"""start: list
                            ?list: item*
                            item : A
                            A: "a"
                         """)
            r = g.parse("")

            # because 'list' is an expand-if-contains-one rule and we've provided less than one element (i.e. none) it should *not* have expanded
            self.assertSequenceEqual(
                [subtree.data for subtree in r.children], ("list",)
            )

            # regardless of the amount of items: there should be only *one* child in 'start' because 'list' isn't an expand-all rule
            self.assertEqual(len(r.children), 1)

            # Sanity check: verify that 'list' contains no 'item's as we've given it none
            [list] = r.children
            self.assertSequenceEqual([item.data for item in list.children], ())

        def test_empty_expand1_list_2(self):
            g = _Lark(r"""start: list
                            ?list: item* "!"?
                            item : A
                            A: "a"
                         """)
            r = g.parse("")

            # because 'list' is an expand-if-contains-one rule and we've provided less than one element (i.e. none) it should *not* have expanded
            self.assertSequenceEqual(
                [subtree.data for subtree in r.children], ("list",)
            )

            # regardless of the amount of items: there should be only *one* child in 'start' because 'list' isn't an expand-all rule
            self.assertEqual(len(r.children), 1)

            # Sanity check: verify that 'list' contains no 'item's as we've given it none
            [list] = r.children
            self.assertSequenceEqual([item.data for item in list.children], ())

        def test_empty_flatten_list(self):
            g = _Lark(r"""start: list
                            list: | item "," list
                            item : A
                            A: "a"
                         """)
            r = g.parse("")

            # Because 'list' is a flatten rule it's top-level element should *never* be expanded
            self.assertSequenceEqual(
                [subtree.data for subtree in r.children], ("list",)
            )

            # Sanity check: verify that 'list' contains no 'item's as we've given it none
            [list] = r.children
            self.assertSequenceEqual([item.data for item in list.children], ())

        def test_token_collision(self):
            g = _Lark(r"""start: "Hello" NAME
                        NAME: /\w/+
                        %ignore " "
                    """)
            x = g.parse("Hello World")
            self.assertSequenceEqual(x.children, ["World"])
            x = g.parse("Hello HelloWorld")
            self.assertSequenceEqual(x.children, ["HelloWorld"])

        def test_token_collision_WS(self):
            g = _Lark(r"""start: "Hello" NAME
                        NAME: /\w/+
                        %import common.WS
                        %ignore WS
                    """)
            x = g.parse("Hello World")
            self.assertSequenceEqual(x.children, ["World"])
            x = g.parse("Hello HelloWorld")
            self.assertSequenceEqual(x.children, ["HelloWorld"])

        def test_token_collision2(self):
            g = _Lark("""
                    !start: "starts"

                    %import common.LCASE_LETTER
                    """)

            x = g.parse("starts")
            self.assertSequenceEqual(x.children, ["starts"])

        def test_templates(self):
            g = _Lark(r"""
                       start: "[" sep{NUMBER, ","} "]"
                       sep{item, delim}: item (delim item)*
                       NUMBER: /\d+/
                       %ignore " "
                       """)
            x = g.parse("[1, 2, 3, 4]")
            self.assertSequenceEqual(x.children, [Tree("sep", ["1", "2", "3", "4"])])
            x = g.parse("[1]")
            self.assertSequenceEqual(x.children, [Tree("sep", ["1"])])

        def test_templates_recursion(self):
            g = _Lark(r"""
                       start: "[" _sep{NUMBER, ","} "]"
                       _sep{item, delim}: item | _sep{item, delim} delim item
                       NUMBER: /\d+/
                       %ignore " "
                       """)
            x = g.parse("[1, 2, 3, 4]")
            self.assertSequenceEqual(x.children, ["1", "2", "3", "4"])
            x = g.parse("[1]")
            self.assertSequenceEqual(x.children, ["1"])

        def test_templates_import(self):
            g = _Lark_open("test_templates_import.lark", rel_to=__file__)
            x = g.parse("[1, 2, 3, 4]")
            self.assertSequenceEqual(x.children, [Tree("sep", ["1", "2", "3", "4"])])
            x = g.parse("[1]")
            self.assertSequenceEqual(x.children, [Tree("sep", ["1"])])

        def test_templates_alias(self):
            g = _Lark(r"""
                       start: expr{"C"}
                       expr{t}: "A" t
                              | "B" t -> b
                       """)
            x = g.parse("AC")
            self.assertSequenceEqual(x.children, [Tree("expr", [])])
            x = g.parse("BC")
            self.assertSequenceEqual(x.children, [Tree("b", [])])

        def test_templates_modifiers(self):
            g = _Lark(r"""
                       start: expr{"B"}
                       !expr{t}: "A" t
                       """)
            x = g.parse("AB")
            self.assertSequenceEqual(x.children, [Tree("expr", ["A", "B"])])
            g = _Lark(r"""
                       start: _expr{"B"}
                       !_expr{t}: "A" t
                       """)
            x = g.parse("AB")
            self.assertSequenceEqual(x.children, ["A", "B"])
            g = _Lark(r"""
                       start: expr{b}
                       b: "B"
                       ?expr{t}: "A" t
                       """)
            x = g.parse("AB")
            self.assertSequenceEqual(x.children, [Tree("b", [])])

        def test_templates_templates(self):
            g = _Lark("""start: a{b}
                         a{t}: t{"a"}
                         b{x}: x""")
            x = g.parse("a")
            self.assertSequenceEqual(x.children, [Tree("a", [Tree("b", [])])])

        def test_g_regex_flags(self):
            g = _Lark(
                """
                    start: "a" /b+/ C
                    C: "C" | D
                    D: "D" E
                    E: "e"
                    """,
                g_regex_flags=re.I,
            )
            x1 = g.parse("ABBc")
            x2 = g.parse("abdE")

        def test_rule_collision(self):
            g = _Lark("""start: "a"+ "b"
                             | "a"+ """)
            x = g.parse("aaaa")
            x = g.parse("aaaab")

        def test_rule_collision2(self):
            g = _Lark("""start: "a"* "b"
                             | "a"+ """)
            x = g.parse("aaaa")
            x = g.parse("aaaab")
            x = g.parse("b")

        def test_token_not_anon(self):
            """Tests that "a" is matched as an anonymous token, and not A."""

            g = _Lark("""start: "a"
                        A: "a" """)
            x = g.parse("a")
            self.assertEqual(len(x.children), 0, '"a" should be considered anonymous')

            g = _Lark("""start: "a" A
                        A: "a" """)
            x = g.parse("aa")
            self.assertEqual(
                len(x.children), 1, 'only "a" should be considered anonymous'
            )
            self.assertEqual(x.children[0].type, "A")

            g = _Lark("""start: /a/
                        A: /a/ """)
            x = g.parse("a")
            self.assertEqual(len(x.children), 1)
            self.assertEqual(x.children[0].type, "A", "A isn't associated with /a/")

        def test_maybe(self):
            g = _Lark("""start: ["a"] """)
            x = g.parse("a")
            x = g.parse("")

        def test_start(self):
            g = _Lark("""a: "a" a? """, start="a")
            x = g.parse("a")
            x = g.parse("aa")
            x = g.parse("aaa")

        def test_alias(self):
            g = _Lark("""start: "a" -> b """)
            x = g.parse("a")
            self.assertEqual(x.data, "b")

        def test_token_ebnf(self):
            g = _Lark("""start: A
                      A: "a"* ("b"? "c".."e")+
                      """)
            x = g.parse("abcde")
            x = g.parse("dd")

        def test_backslash(self):
            g = _Lark(r"""start: "\\" "a"
                      """)
            x = g.parse(r"\a")

            g = Lark(r"""start: /\\/ /a/
                      """)
            x = g.parse(r"\a")

        def test_backslash2(self):
            g = _Lark(r"""start: "\"" "-"
                      """)
            x = g.parse('"-')

            g = Lark(r"""start: /\// /-/
                      """)
            x = g.parse("/-")

        def test_special_chars(self):
            g = _Lark(r"""start: "\n"
                      """)
            x = g.parse("\n")

            g = _Lark(r"""start: /\n/
                      """)
            x = g.parse("\n")

        def test_empty(self):
            # Fails an Earley implementation without special handling for empty rules,
            # or re-processing of already completed rules.
            g = _Lark(r"""start: _empty a "B"
                          a: _empty "A"
                          _empty:
                            """)
            x = g.parse("AB")

        def test_regex_quote(self):
            g = r"""
            start: SINGLE_QUOTED_STRING | DOUBLE_QUOTED_STRING
            SINGLE_QUOTED_STRING  : /'[^']*'/
            DOUBLE_QUOTED_STRING  : /"[^"]*"/
            """

            g = _Lark(g)
            self.assertEqual(g.parse('"hello"').children, ['"hello"'])
            self.assertEqual(g.parse("'hello'").children, ["'hello'"])

        def test_join_regex_flags(self):
            g = r"""
                start: A
                A: B C
                B: /./s
                C: /./
            """
            g = _Lark(g)
            self.assertEqual(g.parse("  ").children, ["  "])
            self.assertEqual(g.parse("\n ").children, ["\n "])
            self.assertRaises(UnexpectedCharacters, g.parse, "\n\n")

            g = r"""
                start: A
                A: B | C
                B: "b"i
                C: "c"
            """
            g = _Lark(g)
            self.assertEqual(g.parse("b").children, ["b"])
            self.assertEqual(g.parse("B").children, ["B"])
            self.assertEqual(g.parse("c").children, ["c"])
            self.assertRaises(UnexpectedCharacters, g.parse, "C")

        def test_float_without_lexer(self):
            expected_error = (
                UnexpectedCharacters if "dynamic" in LEXER else UnexpectedToken
            )

            g = _Lark("""start: ["+"|"-"] float
                         float: digit* "." digit+ exp?
                              | digit+ exp
                         exp: ("e"|"E") ["+"|"-"] digit+
                         digit: "0"|"1"|"2"|"3"|"4"|"5"|"6"|"7"|"8"|"9"
                      """)
            g.parse("1.2")
            g.parse("-.2e9")
            g.parse("+2e-9")
            self.assertRaises(expected_error, g.parse, "+2e-9e")

        def test_keep_all_tokens(self):
            l = _Lark("""start: "a"+ """, keep_all_tokens=True)
            tree = l.parse("aaa")
            self.assertEqual(tree.children, ["a", "a", "a"])

        def test_token_flags(self):
            l = _Lark("""!start: "a"i+
                      """)
            tree = l.parse("aA")
            self.assertEqual(tree.children, ["a", "A"])

            l = _Lark("""!start: /a/i+
                      """)
            tree = l.parse("aA")
            self.assertEqual(tree.children, ["a", "A"])

            g = """start: NAME "," "a"
                   NAME: /[a-z_]/i /[a-z0-9_]/i*
                """
            l = _Lark(g)
            tree = l.parse("ab,a")
            self.assertEqual(tree.children, ["ab"])
            tree = l.parse("AB,a")
            self.assertEqual(tree.children, ["AB"])

        @unittest.skipIf(
            LEXER in ("basic", "custom_old0", "custom_old1", "custom_new"),
            "Requires context sensitive terminal selection",
        )
        def test_token_flags_collision(self):
            g = """!start: "a"i "a"
                """
            l = _Lark(g)
            self.assertEqual(l.parse("aa").children, ["a", "a"])
            self.assertEqual(l.parse("Aa").children, ["A", "a"])
            self.assertRaises(UnexpectedInput, l.parse, "aA")
            self.assertRaises(UnexpectedInput, l.parse, "AA")

            g = """!start: /a/i /a/
                """
            l = _Lark(g)
            self.assertEqual(l.parse("aa").children, ["a", "a"])
            self.assertEqual(l.parse("Aa").children, ["A", "a"])
            self.assertRaises(UnexpectedInput, l.parse, "aA")
            self.assertRaises(UnexpectedInput, l.parse, "AA")

        def test_token_flags3(self):
            l = _Lark("""!start: ABC+
                      ABC: "abc"i
                      """)
            tree = l.parse("aBcAbC")
            self.assertEqual(tree.children, ["aBc", "AbC"])

        def test_token_flags2(self):
            g = """!start: ("a"i | /a/ /b/?)+
                """
            l = _Lark(g)
            tree = l.parse("aA")
            self.assertEqual(tree.children, ["a", "A"])

        def test_token_flags_verbose(self):
            g = _Lark(r"""start: NL | ABC
                          ABC: / [a-z] /x
                          NL: /\n/
                      """)
            x = g.parse("a")
            self.assertEqual(x.children, ["a"])

        def test_token_flags_verbose_multiline(self):
            g = _Lark(r"""start: ABC
                          ABC: /  a      b c
                               d
                                e f
                           /x
                       """)
            x = g.parse("abcdef")
            self.assertEqual(x.children, ["abcdef"])

        def test_twice_empty(self):
            g = """!start: ("A"?)?
                """
            l = _Lark(g)
            tree = l.parse("A")
            self.assertEqual(tree.children, ["A"])

            tree = l.parse("")
            self.assertEqual(tree.children, [])

        def test_line_and_column(self):
            g = r"""!start: "A" bc "D"
                !bc: "B\nC"
                """
            l = _Lark(g)
            a, bc, d = l.parse("AB\nCD").children
            self.assertEqual(a.line, 1)
            self.assertEqual(a.column, 1)

            (bc,) = bc.children
            self.assertEqual(bc.line, 1)
            self.assertEqual(bc.column, 2)

            self.assertEqual(d.line, 2)
            self.assertEqual(d.column, 2)

            # if LEXER != 'dynamic':
            self.assertEqual(a.end_line, 1)
            self.assertEqual(a.end_column, 2)
            self.assertEqual(bc.end_line, 2)
            self.assertEqual(bc.end_column, 2)
            self.assertEqual(d.end_line, 2)
            self.assertEqual(d.end_column, 3)

        def test_basic_parser(self):
            """Test basic parsing with attribute evaluation"""
            g = Lark("""
                start: expr { stack[-1] = stack[-1] }
                expr: term "+" term { stack[-1] = stack[-3] + stack[-1] }
                    | term { stack[-1] = stack[-1] }
                term: NUMBER { stack[-1] = int(stack[-1]) }
                NUMBER: /[0-9]+/
                %ignore " "
            """)

            tree, attrs = g.parse("1 + 2")
            assert attrs == 3

            tree, attrs = g.parse("42")
            assert attrs == 42

        def test_nested_attributes(self):
            """Test nested attribute evaluation"""
            g = Lark("""
                start: list { stack[-1] = stack[-1] }
                list: "[" items "]" { stack[-1] = stack[-2] }
                items: item ("," item)* { stack[-1] = [x for x in stack[-len(stack):]] }
                     | { stack.append([]) }
                item: NUMBER { stack[-1] = int(stack[-1]) }
                NUMBER: /[0-9]+/
                %ignore " "
            """)

            tree, attrs = g.parse("[1, 2, 3]")
            assert attrs == [1, 2, 3]

            tree, attrs = g.parse("[]")
            assert attrs == []

        def test_attribute_parsing(self):
            """Test parsing with attribute evaluation"""
            g = _Lark("""
                start: expr { stack[-1] = stack[-1] }
                expr: term "+" term { stack[-1] = stack[-3] + stack[-1] }
                    | term { stack[-1] = stack[-1] }
                term: NUMBER { stack[-1] = int(stack[-1]) }
                NUMBER: /[0-9]+/
                %ignore " "
            """)

            result = g.parse("1 + 2")
            assert isinstance(result, ParseResult)
            assert isinstance(result.tree, Tree)
            assert result.tree.data == "start"
            assert result.attrs == 3

            result = g.parse("42")
            assert result.attrs == 42

        def test_nested_attribute_eval(self):
            """Test nested attribute evaluation in parse trees"""
            g = _Lark("""
                start: list { stack[-1] = stack[-1] }
                list: "[" items "]" { stack[-1] = stack[-2] }
                items: item ("," item)* { stack[-1] = [x for x in stack[-len(stack):]] }
                      | { stack.append([]) }
                item: NUMBER { stack[-1] = int(stack[-1]) }
                NUMBER: /[0-9]+/
                %ignore " "
            """)

            result = g.parse("[1, 2, 3]")
            assert result.attrs == [1, 2, 3]

            result = g.parse("[]")
            assert result.attrs == []

        def test_attribute_transformer(self):
            """Test using transformers with attribute grammar"""
            g = _Lark("""
                start: NUMBER+ { stack[-1] = stack[-len(stack):] }
                NUMBER: /[0-9]+/ { stack[-1] = int(stack[-1]) }
                %ignore " "
            """)

            @v_args(tree=True)
            class SumTransformer(Transformer):
                def start(self, tree):
                    return sum(tree.attrs)

            result = g.parse("1 2 3")
            assert result.attrs == [1, 2, 3]

            transformed = SumTransformer().transform(result)
            assert transformed == 6

        def test_attribute_error_handling(self):
            """Test error handling in attribute evaluation"""
            g = _Lark("""
                start: "a"+ { stack[-1] = len(stack[-1]) }  // Will fail - stack[-1] is not a list
                """)

            with pytest.raises(TypeError):
                g.parse("aaa")

        def test_attribute_state_isolation(self):
            """Test that attribute evaluation state is properly isolated between parses"""
            g = _Lark("""
                start: NUMBER+ { stack[-1] = sum(stack[-len(stack):]) }
                NUMBER: /[0-9]+/ { stack[-1] = int(stack[-1]) }
                %ignore " "
            """)

            result1 = g.parse("1 2")
            assert result1.attrs == 3

            result2 = g.parse("3 4")
            assert result2.attrs == 7

            # Verify first result wasn't affected
            assert result1.attrs == 3
_TO_TEST = [
    "basic",
    "contextual",
    "custom_new",
]

for _LEXER in _TO_TEST:
    _make_parser_test(_LEXER)

if __name__ == "__main__":
    unittest.main()
