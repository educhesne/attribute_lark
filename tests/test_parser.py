# -*- coding: utf-8 -*-
from __future__ import absolute_import

import re
import unittest
import os
import sys
from copy import copy, deepcopy

from lark import Token, Transformer_NonRecursive, LexError

from io import (
    StringIO as uStringIO,
    BytesIO,
    open,
)


try:
    import regex
except ImportError:
    regex = None


import lark
from lark.attribute_lark import AttributeLark as Lark
from lark.utils import TextSlice
from lark.exceptions import (
    GrammarError,
    ParseError,
    UnexpectedToken,
    UnexpectedInput,
    UnexpectedCharacters,
)
from lark.tree import Tree
from lark.visitors import (
    Transformer,
    Transformer_InPlace,
    v_args,
    Transformer_InPlaceRecursive,
)
from lark.lexer import Lexer, BasicLexer
from lark.indenter import Indenter

__all__ = ["TestParsers"]


class TestParsers(unittest.TestCase):
    def test_big_list(self):
        Lark(
            r"""
            start: {}
        """.format("|".join(['"%s"' % i for i in range(250)]))
        )

    def test_infinite_recurse(self):
        g = """start: a
               a: a | "a"
            """

        self.assertRaises(GrammarError, Lark, g)

        # TODO: should it? shouldn't it?
        # l = Lark(g, parser='earley', lexer='dynamic')
        # self.assertRaises(ParseError, l.parse, 'a')

    def test_propagate_positions(self):
        g = Lark(
            """start: a
                    a: "a"
                 """,
            propagate_positions=True,
        )

        r = g.parse("a")
        self.assertEqual(r.children[0].meta.line, 1)

        g = Lark(
            """start: x
                    x: a
                    a: "a"
                 """,
            propagate_positions=True,
        )

        r = g.parse("a")
        self.assertEqual(r.children[0].meta.line, 1)

    def test_propagate_positions2(self):
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

    def test_expand1(self):
        g = Lark("""start: a
                    ?a: b
                    b: "x"
                 """)

        r = g.parse("x")
        self.assertEqual(r.children[0].data, "b")

        g = Lark("""start: a
                    ?a: b -> c
                    b: "x"
                 """)

        r = g.parse("x")
        self.assertEqual(r.children[0].data, "c")

        g = Lark("""start: a
                    ?a: B -> c
                    B: "x"
                 """)
        self.assertEqual(r.children[0].data, "c")

        g = Lark("""start: a
                    ?a: b b -> c
                    b: "x"
                 """)
        r = g.parse("xx")
        self.assertEqual(r.children[0].data, "c")

    def test_comment_in_rule_definition(self):
        g = Lark("""start: a
               a: "a"
                // A comment
                // Another comment
                | "b"
                // Still more

               c: "unrelated"
            """)
        r = g.parse("b")
        self.assertEqual(r.children[0].data, "a")

    def test_visit_tokens(self):
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
        self.assertEqual(r.children, ["x!"])
        r = T().transform(p.parse("x"))
        self.assertEqual(r.children, ["X!"])

        # Test internal transformer
        p = Lark(g, transformer=T())
        r = p.parse("x")
        self.assertEqual(r.children, ["X!"])

    def test_visit_tokens2(self):
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
            Transformer_InPlace,
            Transformer_NonRecursive,
            Transformer_InPlaceRecursive,
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
            self.assertEqual(result, expected)

    def test_vargs_meta(self):
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

                self.assertEqual(res, [1, 1, 2, 3, 3, 3, 6, 6])

    def test_vargs_tree(self):
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
        self.assertEqual(res, [1, 1, 1])
        self.assertEqual(tree, tree_copy)

    def test_embedded_transformer(self):
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
        self.assertEqual(r.children, ["<a>"])

        g = Lark(
            """start: a
                    a : "x"
                 """,
            transformer=T(),
        )
        r = g.parse("x")
        self.assertEqual(r.children, ["<a>"])

        # Test Expand1
        g = Lark("""start: a
                    ?a : b
                    b : "x"
                 """)
        r = T().transform(g.parse("x"))
        self.assertEqual(r.children, ["<b>"])

        g = Lark(
            """start: a
                    ?a : b
                    b : "x"
                 """,
            transformer=T(),
        )
        r = g.parse("x")
        self.assertEqual(r.children, ["<b>"])

        # Test Expand1 -> Alias
        g = Lark("""start: a
                    ?a : b b -> c
                    b : "x"
                 """)
        r = T().transform(g.parse("xx"))
        self.assertEqual(r.children, ["<c>"])

        g = Lark(
            """start: a
                    ?a : b b -> c
                    b : "x"
                 """,
            transformer=T(),
        )
        r = g.parse("xx")
        self.assertEqual(r.children, ["<c>"])

    def test_embedded_transformer_inplace(self):
        @v_args(tree=True)
        class T1(Transformer_InPlace):
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
                self.assertEqual(a.children, ["tested"])
                self.assertEqual(b.children, ["tested2"])

    def test_alias(self):
        Lark("""start: ["a"] "b" ["c"] "e" ["f"] ["g"] ["h"] "x" -> d """)

    def test_backwards_custom_lexer(self):
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
        self.assertEqual(r, Tree("start", [Token("A", "A")]))

    def test_lexer_token_limit(self):
        "Python has a stupid limit of 100 groups in a regular expression. Test that we handle this limitation"
        tokens = {"A%d" % i: '"%d"' % i for i in range(300)}
        g = """start: %s
                  %s""" % (
            " ".join(tokens),
            "\n".join("%s: %s" % x for x in tokens.items()),
        )

        p = Lark(g)


class CustomLexerNew(Lexer):
    """
    Purpose of this custom lexer is to test the integration,
    so it uses the traditionalparser as implementation without custom lexing behaviour.
    """

    def __init__(self, lexer_conf):
        self.lexer = BasicLexer(copy(lexer_conf))

    def lex(self, lexer_state, parser_state):
        return self.lexer.lex(lexer_state, parser_state)

    __future_interface__ = 2


class CustomLexerOld1(Lexer):
    """
    Purpose of this custom lexer is to test the integration,
    so it uses the traditionalparser as implementation without custom lexing behaviour.
    """

    def __init__(self, lexer_conf):
        self.lexer = BasicLexer(copy(lexer_conf))

    def lex(self, lexer_state, parser_state):
        return self.lexer.lex(lexer_state, parser_state)

    __future_interface__ = 1


class CustomLexerOld0(Lexer):
    """
    Purpose of this custom lexer is to test the integration,
    so it uses the traditionalparser as implementation without custom lexing behaviour.
    """

    def __init__(self, lexer_conf):
        self.lexer = BasicLexer(copy(lexer_conf))

    def lex(self, text):
        ls = self.lexer.make_lexer_state(text)
        return self.lexer.lex(ls, None)

    __future_interface__ = 0


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

            g = _Lark(r"""start: /\\/ /a/
                      """)
            x = g.parse(r"\a")

        def test_backslash2(self):
            g = _Lark(r"""start: "\"" "-"
                      """)
            x = g.parse('"-')

            g = _Lark(r"""start: /\// /-/
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

        def test_reduce_cycle(self):
            """Tests an edge-condition in the LALR parser, in which a transition state looks exactly like the end state.
            It seems that the correct solution is to explicitly distinguish finalization in the reduce() function.
            """

            l = _Lark(
                """
                term: A
                    | term term

                A: "a"

            """,
                start="term",
            )

            tree = l.parse("aa")
            self.assertEqual(len(tree.children), 2)

        @unittest.skipIf(
            "dynamic" in LEXER,
            "basic lexer prioritization differs from dynamic lexer prioritization",
        )
        def test_lexer_prioritization(self):
            "Tests effect of priority on result"

            grammar = """
            start: A B | AB
            A.2: "a"
            B: "b"
            AB: "ab"
            """
            l = _Lark(grammar)
            res = l.parse("ab")

            self.assertEqual(res.children, ["a", "b"])
            self.assertNotEqual(res.children, ["ab"])

            grammar = """
            start: A B | AB
            A: "a"
            B: "b"
            AB.3: "ab"
            """
            l = _Lark(grammar)
            res = l.parse("ab")

            self.assertNotEqual(res.children, ["a", "b"])
            self.assertEqual(res.children, ["ab"])

            grammar = """
            start: A B | AB
            A: "a"
            B.-20: "b"
            AB.-10: "ab"
            """
            l = _Lark(grammar)
            res = l.parse("ab")
            self.assertEqual(res.children, ["a", "b"])

            grammar = """
            start: A B | AB
            A.-99999999999999999999999: "a"
            B: "b"
            AB: "ab"
            """
            l = _Lark(grammar)
            res = l.parse("ab")

            self.assertEqual(res.children, ["ab"])

        @unittest.skipIf(
            "dynamic" not in LEXER,
            "dynamic lexer prioritization differs from basic lexer prioritization",
        )
        def test_dynamic_lexer_prioritization(self):
            "Tests effect of priority on result"

            grammar = """
            start: A B | AB
            A.2: "a"
            B: "b"
            AB: "ab"
            """
            l = _Lark(grammar)
            res = l.parse("ab")

            self.assertEqual(res.children, ["a", "b"])
            self.assertNotEqual(res.children, ["ab"])

            grammar = """
            start: A B | AB
            A: "a"
            B: "b"
            AB.3: "ab"
            """
            l = _Lark(grammar)
            res = l.parse("ab")

            self.assertNotEqual(res.children, ["a", "b"])
            self.assertEqual(res.children, ["ab"])

            # this case differs from prioritization with a basic lexer
            grammar = """
            start: A B | AB
            A: "a"
            B.-20: "b"
            AB.-10: "ab"
            """
            l = _Lark(grammar)
            res = l.parse("ab")
            self.assertEqual(res.children, ["ab"])

            grammar = """
            start: A B | AB
            A.-99999999999999999999999: "a"
            B: "b"
            AB: "ab"
            """
            l = _Lark(grammar)
            res = l.parse("ab")

            self.assertEqual(res.children, ["ab"])

        def test_import(self):
            grammar = """
            start: NUMBER WORD

            %import common.NUMBER
            %import common.WORD
            %import common.WS
            %ignore WS

            """
            l = _Lark(grammar)
            x = l.parse("12 elephants")
            self.assertEqual(x.children, ["12", "elephants"])

        def test_import_rename(self):
            grammar = """
            start: N W

            %import common.NUMBER -> N
            %import common.WORD -> W
            %import common.WS
            %ignore WS

            """
            l = _Lark(grammar)
            x = l.parse("12 elephants")
            self.assertEqual(x.children, ["12", "elephants"])

        def test_relative_import(self):
            l = _Lark_open("test_relative_import.lark", rel_to=__file__)
            x = l.parse("12 lions")
            self.assertEqual(x.children, ["12", "lions"])

        def test_relative_import_unicode(self):
            l = _Lark_open("test_relative_import_unicode.lark", rel_to=__file__)
            x = l.parse("Ø")
            self.assertEqual(x.children, ["Ø"])

        def test_relative_import_rename(self):
            l = _Lark_open("test_relative_import_rename.lark", rel_to=__file__)
            x = l.parse("12 lions")
            self.assertEqual(x.children, ["12", "lions"])

        def test_relative_rule_import(self):
            l = _Lark_open("test_relative_rule_import.lark", rel_to=__file__)
            x = l.parse("xaabby")
            self.assertEqual(
                x.children,
                ["x", Tree("expr", ["a", Tree("expr", ["a", "b"]), "b"]), "y"],
            )

        def test_relative_rule_import_drop_ignore(self):
            # %ignore rules are dropped on import
            l = _Lark_open(
                "test_relative_rule_import_drop_ignore.lark", rel_to=__file__
            )
            self.assertRaises((ParseError, UnexpectedInput), l.parse, "xa abby")

        def test_relative_rule_import_subrule(self):
            l = _Lark_open("test_relative_rule_import_subrule.lark", rel_to=__file__)
            x = l.parse("xaabby")
            self.assertEqual(
                x.children,
                [
                    "x",
                    Tree(
                        "startab",
                        [
                            Tree(
                                "grammars__ab__expr",
                                [
                                    "a",
                                    Tree("grammars__ab__expr", ["a", "b"]),
                                    "b",
                                ],
                            ),
                        ],
                    ),
                    "y",
                ],
            )

        def test_relative_rule_import_subrule_no_conflict(self):
            l = _Lark_open(
                "test_relative_rule_import_subrule_no_conflict.lark", rel_to=__file__
            )
            x = l.parse("xaby")
            self.assertEqual(
                x.children,
                [
                    Tree(
                        "expr",
                        [
                            "x",
                            Tree(
                                "startab",
                                [
                                    Tree("grammars__ab__expr", ["a", "b"]),
                                ],
                            ),
                            "y",
                        ],
                    )
                ],
            )
            self.assertRaises((ParseError, UnexpectedInput), l.parse, "xaxabyby")

        def test_relative_rule_import_rename(self):
            l = _Lark_open("test_relative_rule_import_rename.lark", rel_to=__file__)
            x = l.parse("xaabby")
            self.assertEqual(
                x.children, ["x", Tree("ab", ["a", Tree("ab", ["a", "b"]), "b"]), "y"]
            )

        def test_multi_import(self):
            grammar = """
            start: NUMBER WORD

            %import common (NUMBER, WORD, WS)
            %ignore WS

            """
            l = _Lark(grammar)
            x = l.parse("12 toucans")
            self.assertEqual(x.children, ["12", "toucans"])

        def test_relative_multi_import(self):
            l = _Lark_open("test_relative_multi_import.lark", rel_to=__file__)
            x = l.parse("12 capybaras")
            self.assertEqual(x.children, ["12", "capybaras"])

        def test_relative_import_preserves_leading_underscore(self):
            l = _Lark_open(
                "test_relative_import_preserves_leading_underscore.lark",
                rel_to=__file__,
            )
            x = l.parse("Ax")
            self.assertEqual(next(x.find_data("c")).children, ["A"])

        def test_relative_import_of_nested_grammar(self):
            l = _Lark_open(
                "grammars/test_relative_import_of_nested_grammar.lark", rel_to=__file__
            )
            x = l.parse("N")
            self.assertEqual(next(x.find_data("rule_to_import")).children, ["N"])

        def test_relative_import_rules_dependencies_imported_only_once(self):
            l = _Lark_open(
                "test_relative_import_rules_dependencies_imported_only_once.lark",
                rel_to=__file__,
            )
            x = l.parse("AAA")
            self.assertEqual(next(x.find_data("a")).children, ["A"])
            self.assertEqual(next(x.find_data("b")).children, ["A"])
            self.assertEqual(next(x.find_data("d")).children, ["A"])

        def test_import_errors(self):
            grammar = """
            start: NUMBER WORD

            %import .grammars.bad_test.NUMBER
            """
            self.assertRaises(IOError, _Lark, grammar)

            grammar = """
            start: NUMBER WORD

            %import bad_test.NUMBER
            """
            self.assertRaises(IOError, _Lark, grammar)

        @unittest.skipIf(
            "dynamic" in LEXER, "%declare/postlex doesn't work with dynamic"
        )
        def test_postlex_declare(
            self,
        ):  # Note: this test does a lot. maybe split it up?
            class TestPostLexer:
                def process(self, stream):
                    for t in stream:
                        if t.type == "A":
                            t.type = "B"
                            yield t
                        else:
                            yield t

                always_accept = ("A",)

            parser = _Lark(
                """
            start: B
            A: "A"
            %declare B
            """,
                postlex=TestPostLexer(),
            )

            test_file = "A"
            tree = parser.parse(test_file)
            self.assertEqual(tree.children, [Token("B", "A")])

        @unittest.skipIf(
            "dynamic" in LEXER, "%declare/postlex doesn't work with dynamic"
        )
        def test_postlex_indenter(self):
            class CustomIndenter(Indenter):
                NL_type = "NEWLINE"
                OPEN_PAREN_types = []
                CLOSE_PAREN_types = []
                INDENT_type = "INDENT"
                DEDENT_type = "DEDENT"
                tab_len = 8

            grammar = r"""
            start: "a" NEWLINE INDENT "b" NEWLINE DEDENT

            NEWLINE: ( /\r?\n */  )+

            %ignore " "+
            %declare INDENT DEDENT
            """

            parser = _Lark(grammar, postlex=CustomIndenter())
            parser.parse("a\n    b\n")

        def test_prioritization(self):
            "Tests effect of priority on result"

            grammar = """
            start: a | b
            a.1: "a"
            b.2: "a"
            """

            l = _Lark(grammar)
            res = l.parse("a")
            self.assertEqual(res.children[0].data, "b")

            grammar = """
            start: a | b
            a.2: "a"
            b.1: "a"
            """

            l = _Lark(grammar)
            res = l.parse("a")
            self.assertEqual(res.children[0].data, "a")

            grammar = """
            start: a | b
            a.2: "A"+
            b.1: "A"+ "B"?
            """

            l = _Lark(grammar)
            res = l.parse("AAAA")
            self.assertEqual(res.children[0].data, "a")

            l = _Lark(grammar)
            res = l.parse("AAAB")
            self.assertEqual(res.children[0].data, "b")

            l = _Lark(grammar, priority="invert")
            res = l.parse("AAAA")
            self.assertEqual(res.children[0].data, "b")

        def test_utf8(self):
            g = """start: a
                   a: "±a"
                """
            l = _Lark(g)
            self.assertEqual(l.parse("±a"), Tree("start", [Tree("a", [])]))

            g = """start: A
                   A: "±a"
                """
            l = _Lark(g)
            self.assertEqual(l.parse("±a"), Tree("start", ["\xb1a"]))

        def test_ignore(self):
            grammar = r"""
            COMMENT: /(!|(\/\/))[^\n]*/
            %ignore COMMENT
            %import common.WS -> _WS
            %import common.INT
            start: "INT"i _WS+ INT _WS*
            """

            parser = _Lark(grammar)

            tree = parser.parse("int 1 ! This is a comment\n")
            self.assertEqual(tree.children, ["1"])

            tree = parser.parse(
                "int 1 ! This is a comment"
            )  # A trailing ignore token can be tricky!
            self.assertEqual(tree.children, ["1"])

            parser = _Lark(r"""
                start : "a"*
                %ignore "b"
            """)
            tree = parser.parse("bb")
            self.assertEqual(tree.children, [])

        def test_regex_escaping(self):
            g = _Lark("start: /[ab]/")
            g.parse("a")
            g.parse("b")

            self.assertRaises(UnexpectedInput, g.parse, "c")

            _Lark(r"start: /\w/").parse("a")

            g = _Lark(r"start: /\\w/")
            self.assertRaises(UnexpectedInput, g.parse, "a")
            g.parse(r"\w")

            _Lark(r"start: /\[/").parse("[")

            _Lark(r"start: /\//").parse("/")

            _Lark(r"start: /\\/").parse("\\")

            _Lark(r"start: /\[ab]/").parse("[ab]")

            _Lark(r"start: /\\[ab]/").parse("\\a")

            _Lark(r"start: /\t/").parse("\t")

            _Lark(r"start: /\\t/").parse("\\t")

            _Lark(r"start: /\\\t/").parse("\\\t")

            _Lark(r'start: "\t"').parse("\t")

            _Lark(r'start: "\\t"').parse("\\t")

            _Lark(r'start: "\\\t"').parse("\\\t")

        def test_ranged_repeat_rules(self):
            g = """!start: "A"~3
                """
            l = _Lark(g)
            self.assertEqual(l.parse("AAA"), Tree("start", ["A", "A", "A"]))
            self.assertRaises(ParseError, l.parse, "AA")
            self.assertRaises((ParseError, UnexpectedInput), l.parse, "AAAA")

            g = """!start: "A"~0..2
                """

            l = _Lark(g)
            self.assertEqual(l.parse(""), Tree("start", []))
            self.assertEqual(l.parse("A"), Tree("start", ["A"]))
            self.assertEqual(l.parse("AA"), Tree("start", ["A", "A"]))
            self.assertRaises((UnexpectedToken, UnexpectedInput), l.parse, "AAA")

            g = """!start: "A"~3..2
                """
            self.assertRaises(GrammarError, _Lark, g)

            g = """!start: "A"~2..3 "B"~2
                """
            l = _Lark(g)
            self.assertEqual(l.parse("AABB"), Tree("start", ["A", "A", "B", "B"]))
            self.assertEqual(l.parse("AAABB"), Tree("start", ["A", "A", "A", "B", "B"]))
            self.assertRaises(ParseError, l.parse, "AAAB")
            self.assertRaises((ParseError, UnexpectedInput), l.parse, "AAABBB")
            self.assertRaises((ParseError, UnexpectedInput), l.parse, "ABB")
            self.assertRaises((ParseError, UnexpectedInput), l.parse, "AAAABB")

        def test_priority_vs_embedded(self):
            g = """
            A.2: "a"
            WORD: ("a".."z")+

            start: (A | WORD)+
            """
            l = _Lark(g)
            t = l.parse("abc")
            self.assertEqual(t.children, ["a", "bc"])
            self.assertEqual(t.children[0].type, "A")

        def test_line_counting(self):
            p = _Lark("start: /[^x]+/")

            text = "hello\nworld"
            t = p.parse(text)
            tok = t.children[0]
            self.assertEqual(tok, text)
            self.assertEqual(tok.line, 1)
            self.assertEqual(tok.column, 1)
            # if _LEXER != 'dynamic':
            self.assertEqual(tok.end_line, 2)
            self.assertEqual(tok.end_column, 6)

        def test_empty_end(self):
            p = _Lark("""
                start: b c d
                b: "B"
                c: | "C"
                d: | "D"
            """)
            res = p.parse("B")
            self.assertEqual(len(res.children), 3)

        def test_maybe_placeholders(self):
            # Anonymous tokens shouldn't count
            p = _Lark("""start: ["a"] ["b"] ["c"] """, maybe_placeholders=True)
            self.assertEqual(p.parse("").children, [])

            # Unless keep_all_tokens=True
            p = _Lark(
                """start: ["a"] ["b"] ["c"] """,
                maybe_placeholders=True,
                keep_all_tokens=True,
            )
            self.assertEqual(p.parse("").children, [None, None, None])

            # All invisible constructs shouldn't count
            p = _Lark(
                """start: [A] ["b"] [_c] ["e" "f" _c]
                        A: "a"
                        _c: "c" """,
                maybe_placeholders=True,
            )
            self.assertEqual(p.parse("").children, [None])
            self.assertEqual(p.parse("c").children, [None])
            self.assertEqual(p.parse("aefc").children, ["a"])

            # ? shouldn't apply
            p = _Lark("""!start: ["a"] "b"? ["c"] """, maybe_placeholders=True)
            self.assertEqual(p.parse("").children, [None, None])
            self.assertEqual(p.parse("b").children, [None, "b", None])

            p = _Lark("""!start: ["a"] ["b"] ["c"] """, maybe_placeholders=True)
            self.assertEqual(p.parse("").children, [None, None, None])
            self.assertEqual(p.parse("a").children, ["a", None, None])
            self.assertEqual(p.parse("b").children, [None, "b", None])
            self.assertEqual(p.parse("c").children, [None, None, "c"])
            self.assertEqual(p.parse("ab").children, ["a", "b", None])
            self.assertEqual(p.parse("ac").children, ["a", None, "c"])
            self.assertEqual(p.parse("bc").children, [None, "b", "c"])
            self.assertEqual(p.parse("abc").children, ["a", "b", "c"])

            p = _Lark("""!start: (["a"] "b" ["c"])+ """, maybe_placeholders=True)
            self.assertEqual(p.parse("b").children, [None, "b", None])
            self.assertEqual(p.parse("bb").children, [None, "b", None, None, "b", None])
            self.assertEqual(p.parse("abbc").children, ["a", "b", None, None, "b", "c"])
            self.assertEqual(
                p.parse("babbcabcb").children,
                [
                    None,
                    "b",
                    None,
                    "a",
                    "b",
                    None,
                    None,
                    "b",
                    "c",
                    "a",
                    "b",
                    "c",
                    None,
                    "b",
                    None,
                ],
            )

            p = _Lark(
                """!start: ["a"] ["c"] "b"+ ["a"] ["d"] """, maybe_placeholders=True
            )
            self.assertEqual(p.parse("bb").children, [None, None, "b", "b", None, None])
            self.assertEqual(p.parse("bd").children, [None, None, "b", None, "d"])
            self.assertEqual(p.parse("abba").children, ["a", None, "b", "b", "a", None])
            self.assertEqual(
                p.parse("cbbbb").children, [None, "c", "b", "b", "b", "b", None, None]
            )

            p = _Lark("""!start: ["a" "b" "c"] """, maybe_placeholders=True)
            self.assertEqual(p.parse("").children, [None, None, None])
            self.assertEqual(p.parse("abc").children, ["a", "b", "c"])

            p = _Lark("""!start: ["a" ["b" "c"]] """, maybe_placeholders=True)
            self.assertEqual(p.parse("").children, [None, None, None])
            self.assertEqual(p.parse("a").children, ["a", None, None])
            self.assertEqual(p.parse("abc").children, ["a", "b", "c"])

            p = _Lark(r"""!start: "a" ["b" | "c"] """, maybe_placeholders=True)
            self.assertEqual(p.parse("a").children, ["a", None])
            self.assertEqual(p.parse("ab").children, ["a", "b"])

            p = _Lark(r"""!start: "a" ["b" | "c" "d"] """, maybe_placeholders=True)
            self.assertEqual(p.parse("a").children, ["a", None, None])
            # self.assertEqual(p.parse("ab").children, ['a', 'b', None])        # Not implemented; current behavior is incorrect
            self.assertEqual(p.parse("acd").children, ["a", "c", "d"])

        def test_escaped_string(self):
            "Tests common.ESCAPED_STRING"
            grammar = r"""
            start: ESCAPED_STRING+

            %import common (WS_INLINE, ESCAPED_STRING)
            %ignore WS_INLINE
            """

            parser = _Lark(grammar)
            parser.parse(r'"\\" "b" "c"')

            parser.parse(r'"That" "And a \"b"')

        def test_meddling_unused(self):
            "Unless 'unused' is removed, LALR analysis will fail on reduce-reduce collision"

            grammar = """
                start: EKS* x
                x: EKS
                unused: x*
                EKS: "x"
            """
            parser = _Lark(grammar)

        @unittest.skipIf(
            LEXER == "custom_old0",
            "Serialize currently only works for LALR parsers without custom lexers (though it should be easy to extend)",
        )
        def test_serialize(self):
            grammar = """
                start: _ANY b "C"
                _ANY: /./
                b: "B"
            """
            parser = _Lark(grammar)
            s = BytesIO()
            parser.save(s)
            s.seek(0)
            parser2 = Lark.load(s)
            self.assertEqual(parser2.parse("ABC"), Tree("start", [Tree("b", [])]))

        def test_multi_start(self):
            parser = _Lark(
                """
                a: "x" "a"?
                b: "x" "b"?
            """,
                start=["a", "b"],
            )

            self.assertEqual(parser.parse("xa", "a"), Tree("a", []))
            self.assertEqual(parser.parse("xb", "b"), Tree("b", []))

        def test_lexer_detect_newline_tokens(self):
            # Detect newlines in regular tokens
            g = _Lark(r"""start: "go" tail*
            !tail : SA "@" | SB "@" | SC "@" | SD "@"
            SA : "a" /\n/
            SB : /b./s
            SC : "c" /[^a-z]/
            SD : "d" /\s/
            """)
            a, b, c, d = [x.children[1] for x in g.parse("goa\n@b\n@c\n@d\n@").children]
            self.assertEqual(a.line, 2)
            self.assertEqual(b.line, 3)
            self.assertEqual(c.line, 4)
            self.assertEqual(d.line, 5)

            # Detect newlines in ignored tokens
            for re in ["/\\n/", "/[^a-z]/", "/\\s/"]:
                g = _Lark(
                    """!start: "a" "a"
                             %ignore {}""".format(re)
                )
                a, b = g.parse("a\na").children
                self.assertEqual(a.line, 1)
                self.assertEqual(b.line, 2)

        @unittest.skipIf(
            LEXER == "custom_old0",
            "match_examples() not supported for old custom lexer",
        )
        def test_match_examples(self):
            p = _Lark(r"""
                start: "a" "b" "c"
            """)

            def match_error(s):
                try:
                    _ = p.parse(s)
                except UnexpectedInput as u:
                    return u.match_examples(
                        p.parse,
                        {
                            0: ["abe"],
                            1: ["ab"],
                            2: ["cbc", "dbc"],
                        },
                    )
                assert False

            assert match_error("abe") == 0
            assert match_error("ab") == 1
            assert match_error("bbc") == 2
            assert match_error("cbc") == 2
            self.assertEqual(match_error("dbc"), 2)
            self.assertEqual(match_error("ebc"), 2)

        @unittest.skipIf(not regex, "regex not installed")
        def test_unicode_class(self):
            "Tests that character classes from the `regex` module work correctly."
            g = _Lark(
                r"""?start: NAME
                           NAME: ID_START ID_CONTINUE*
                           ID_START: /[\p{Lu}\p{Ll}\p{Lt}\p{Lm}\p{Lo}\p{Nl}_]+/
                           ID_CONTINUE: ID_START | /[\p{Mn}\p{Mc}\p{Nd}\p{Pc}]+/""",
                regex=True,
            )

            self.assertEqual(g.parse("வணக்கம்"), "வணக்கம்")

        @unittest.skipIf(not regex, "regex not installed")
        def test_unicode_word(self):
            "Tests that a persistent bug in the `re` module works when `regex` is enabled."
            g = _Lark(
                r"""?start: NAME
                           NAME: /[\w]+/
                        """,
                regex=True,
            )
            self.assertEqual(g.parse("வணக்கம்"), "வணக்கம்")

        @unittest.skipIf(not regex, "regex not installed")
        def test_regex_width_fallback(self):
            g = r"""
                start: NAME NAME?
                NAME: /(?(?=\d)\d+|\w+)/
            """
            self.assertRaises((GrammarError, LexError, re.error), _Lark, g)
            p = _Lark(g, regex=True)
            self.assertEqual(p.parse("123abc"), Tree("start", ["123", "abc"]))

            g = r"""
                start: NAME NAME?
                NAME: /(?(?=\d)\d+|\w*)/
            """
            self.assertRaises((GrammarError, LexError, re.error), _Lark, g, regex=True)

        def test_parser_interactive_parser(self):
            g = _Lark(r"""
                start: A+ B*
                A: "a"
                B: "b"
            """)

            ip = g.parse_interactive()

            self.assertRaises(UnexpectedToken, ip.feed_eof)
            self.assertRaises(TypeError, ip.exhaust_lexer)
            ip.feed_token(Token("A", "a"))
            res = ip.feed_eof()
            self.assertEqual(res, Tree("start", ["a"]))

            ip = g.parse_interactive("ab")

            ip.exhaust_lexer()

            ip_copy = ip.copy()
            self.assertEqual(ip_copy.parser_state, ip.parser_state)
            self.assertEqual(ip_copy.lexer_thread.state, ip.lexer_thread.state)
            self.assertIsNot(ip_copy.parser_state, ip.parser_state)
            self.assertIsNot(ip_copy.lexer_thread.state, ip.lexer_thread.state)
            self.assertIsNot(
                ip_copy.lexer_thread.state.line_ctr, ip.lexer_thread.state.line_ctr
            )

            res = ip.feed_eof(ip.lexer_thread.state.last_token)
            self.assertEqual(res, Tree("start", ["a", "b"]))
            self.assertRaises(UnexpectedToken, ip.feed_eof)

            self.assertRaises(UnexpectedToken, ip_copy.feed_token, Token("A", "a"))
            ip_copy.feed_token(Token("B", "b"))
            res = ip_copy.feed_eof()
            self.assertEqual(res, Tree("start", ["a", "b", "b"]))

        def test_error_with_interactive_parser(self):
            def ignore_errors(e):
                if isinstance(e, UnexpectedCharacters):
                    # Skip bad character
                    return True

                # Must be UnexpectedToken
                if e.token.type == "COMMA":
                    # Skip comma
                    return True
                elif e.token.type == "SIGNED_NUMBER":
                    # Try to feed a comma and retry the number
                    e.interactive_parser.feed_token(Token("COMMA", ","))
                    e.interactive_parser.feed_token(e.token)

                    return True

                # Unhandled error. Will stop parse and raise exception
                return False

            g = _Lark(r"""
                start: "[" num ("," num)* "]"
                ?num: SIGNED_NUMBER
                %import common.SIGNED_NUMBER
                %ignore " "
            """)
            s = "[0 1, 2,, 3,,, 4, 5 6 ]"
            tree = g.parse(s, on_error=ignore_errors)
            res = [int(x) for x in tree.children]
            assert res == list(range(7))

            s = "[0 1, 2,@, 3,,, 4, 5 6 ]$"
            tree = g.parse(s, on_error=ignore_errors)

        def test_iter_parse(self):
            ab_grammar = '!start: "a"* "b"*'
            parser = _Lark(ab_grammar)
            ip = parser.parse_interactive("aaabb")
            i = ip.iter_parse()
            assert next(i) == "a"
            assert next(i) == "a"
            assert next(i) == "a"
            assert next(i) == "b"

        def test_interactive_treeless_transformer(self):
            grammar = r"""
                start: SYM+

                SYM: "a" | "b"
            """

            class SYMTransformer(lark.visitors.Transformer):
                def SYM(self, token):
                    return {"a": 1, "b": 2}[str(token)]

            parser = _Lark(grammar, transformer=SYMTransformer())
            res = parser.parse("aba")
            self.assertEqual(res.children, [1, 2, 1])
            ip = parser.parse_interactive("aba")
            ip.exhaust_lexer()
            # Previously `accepts` would call `SYMTransformer.SYM` with `Token('SYM', '')`, which would cause an error.
            self.assertEqual(
                ip.accepts(), {Token("$END", "(.*?)"), Token("SYM", "(.*?)")}
            )
            res = ip.feed_eof()
            self.assertEqual(res.children, [1, 2, 1])

        def test_default_in_treeless_mode(self):
            grammar = r"""
                start: expr

                expr: A B
                    | A expr B

                A: "a"
                B: "b"

                %import common.WS
                %ignore WS
            """
            s = "a a a b b b"

            class AbTransformer(Transformer):
                def __default__(self, data, children, meta):
                    return "@", data, children

            parser = _Lark(grammar)
            a = AbTransformer().transform(parser.parse(s))
            parser = _Lark(grammar, transformer=AbTransformer())
            b = parser.parse(s)
            assert a == b

        def test_strict(self):
            # Test regex collision
            grammar = r"""
            start: A | B

            A: /e?rez/
            B: /erez?/
            """

            self.assertRaises(LexError, _Lark, grammar, strict=True)

            # Test shift-reduce collision
            grammar = r"""
            start: a "."
            a: "."+
            """
            self.assertRaises(GrammarError, _Lark, grammar, strict=True)

        @unittest.skipIf(
            LEXER in ("dynamic", "dynamic_complete", "custom_old0", "custom_old1"),
            "start_pos and end_pos not compatible with old style custom/dynamic lexer ",
        )
        def test_parse_textslice(self):
            grammar = r"""
            start: (WORD|FRAG_END|FRAG_START)+
            WORD: /\b\w+\b/ # match full word
            FRAG_END: /\B\w+/ # end of a word, i.e. start is not at a word boundary
            FRAG_START: /\w+\B/ # start of a word, i.e. end is not at a word boundary
            %ignore /\s+/
            """

            parser = _Lark(grammar)
            self.assertEqual(
                parser.parse(TextSlice(" abc def ", 1, -1)),
                Tree("start", [Token("WORD", "abc"), Token("WORD", "def")]),
            )
            self.assertEqual(
                parser.parse(TextSlice(" abc def ", 1 - 9, -1 + 9)),
                Tree("start", [Token("WORD", "abc"), Token("WORD", "def")]),
            )
            self.assertEqual(
                parser.parse(TextSlice("xabc def ", 1, -1)),
                Tree("start", [Token("FRAG_END", "abc"), Token("WORD", "def")]),
            )

            # We match the behavior of python's re module here: It doesn't look ahead beyond `end_pos`,
            # despite looking behind before `start_pos`
            self.assertEqual(
                parser.parse(TextSlice(" abc defx", 1, -1)),
                Tree("start", [Token("WORD", "abc"), Token("WORD", "def")]),
            )

            grammar = r"""
            start: (_NL | ANY)+
            _NL: "\n"
            ANY: /[^\n]/
            """
            parser = _Lark(grammar)
            digits = "\n".join("123456789")
            tree = parser.parse(TextSlice(digits, 2, 3))
            self.assertEqual(tree.children, ["2"])
            t: Token = tree.children[0]
            assert t.start_pos == (2 - 1) * 2
            assert t.line == 2

            tree = parser.parse(TextSlice(digits, -1, None))
            self.assertEqual(tree.children, ["9"])
            t: Token = tree.children[0]
            assert t.start_pos == (9 - 1) * 2
            assert t.line == 9

        @unittest.skipIf(
            LEXER not in ("dynamic", "dynamic_complete", "custom_old0", "custom_old1"),
            "start_pos and end_pos not compatible with old style custom/dynamic lexer ",
        )
        def test_parse_textslice_fails(self):
            parser = _Lark("start: ")
            s = TextSlice("hello", 2, 3)
            self.assertRaises(TypeError, parser.parse, s)

    _NAME = "Test LALR" + LEXER.capitalize()
    _TestParser.__name__ = _NAME
    _TestParser.__qualname__ = "tests.test_parser." + _NAME
    globals()[_NAME] = _TestParser
    __all__.append(_NAME)


_TO_TEST = [
    "basic",
    "contextual",
    "custom_new",
]

for _LEXER in _TO_TEST:
    _make_parser_test(_LEXER)

if __name__ == "__main__":
    unittest.main()
