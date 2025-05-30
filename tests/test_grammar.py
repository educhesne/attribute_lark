from __future__ import absolute_import

import os
from unittest import TestCase, main

from attribute_lark import AttributeLark, Token, Tree, ParseError, UnexpectedInput
from attribute_lark.load_grammar import (
    GrammarError,
    GRAMMAR_ERRORS,
    find_grammar_errors,
    list_grammar_imports,
)
from attribute_lark.load_grammar import FromPackageLoader
from attribute_lark.grammar import Symbol


class TestGrammar(TestCase):
    def setUp(self):
        pass

    def test_errors(self):
        for msg, examples in GRAMMAR_ERRORS:
            for example in examples:
                try:
                    p = AttributeLark.from_string(example)
                except GrammarError as e:
                    assert msg in str(e)
                else:
                    assert False, "example did not raise an error"

    def test_empty_literal(self):
        # Issues #888
        self.assertRaises(GrammarError, AttributeLark, 'start: ""')

    def test_ignore_name(self):
        spaces = []
        p = AttributeLark.from_string(
            """
            start: "a" "b"
            WS: " "
            %ignore WS
        """,
            lexer_callbacks={"WS": spaces.append},
        )
        assert p.parse("a b")[0] == p.parse("a    b")[0]
        assert len(spaces) == 5

    def test_override_rule(self):
        # Overrides the 'sep' template in existing grammar to add an optional terminating delimiter
        # Thus extending it beyond its original capacity
        p = AttributeLark.from_string(
            """
            %import .test_templates_import (start, sep)

            %override sep{item, delim}: item (delim item)* delim?
            %ignore " "
        """,
            source_path=__file__,
        )

        a = p.parse("[1, 2, 3]")[0]
        b = p.parse("[1, 2, 3, ]")[0]
        assert a == b

        self.assertRaises(
            GrammarError,
            AttributeLark,
            """
            %import .test_templates_import (start, sep)

            %override sep{item}: item (delim item)* delim?
        """,
            source_path=__file__,
        )

        self.assertRaises(
            GrammarError,
            AttributeLark,
            """
            %override sep{item}: item (delim item)* delim?
        """,
            source_path=__file__,
        )

    def test_override_terminal(self):
        p = AttributeLark.from_string(
            """

            %import .grammars.ab (startab, A, B)

            %override A: "c"
            %override B: "d"
        """,
            start="startab",
            source_path=__file__,
        )

        a = p.parse("cd")[0]
        self.assertEqual(a.children[0].children, [Token("A", "c"), Token("B", "d")])

    def test_extend_rule(self):
        p = AttributeLark.from_string(
            """
            %import .grammars.ab (startab, A, B, expr)

            %extend expr: B A
        """,
            start="startab",
            source_path=__file__,
        )
        a = p.parse("abab")[0]
        self.assertEqual(a.children[0].children, ["a", Tree("expr", ["b", "a"]), "b"])

        self.assertRaises(
            GrammarError,
            AttributeLark,
            """
            %extend expr: B A
        """,
        )

    def test_extend_term(self):
        p = AttributeLark.from_string(
            """
            %import .grammars.ab (startab, A, B, expr)

            %extend A: "c"
        """,
            start="startab",
            source_path=__file__,
        )
        a = p.parse("acbb")[0]
        self.assertEqual(a.children[0].children, ["a", Tree("expr", ["c", "b"]), "b"])

    def test_extend_twice(self):
        p = AttributeLark.from_string("""
            start: x+

            x: "a"
            %extend x: "b"
            %extend x: "c"
        """)

        assert p.parse("abccbba")[0] == p.parse("cbabbbb")[0]

    def test_undefined_ignore(self):
        g = """!start: "A"

            %ignore B
            """
        self.assertRaises(GrammarError, AttributeLark, g)

        g = """!start: "A"

            %ignore start
            """
        self.assertRaises(GrammarError, AttributeLark, g)

    def test_alias_in_terminal(self):
        g = """start: TERM
            TERM: "a" -> alias
            """
        self.assertRaises(GrammarError, AttributeLark, g)

    def test_undefined_rule(self):
        self.assertRaises(GrammarError, AttributeLark, """start: a""")

    def test_undefined_term(self):
        self.assertRaises(GrammarError, AttributeLark, """start: A""")

    def test_token_multiline_only_works_with_x_flag(self):
        g = r"""start: ABC
                ABC: /  a      b c
                            d
                            e f
                        /i
                    """
        self.assertRaises(GrammarError, AttributeLark, g)

    def test_import_custom_sources(self):
        custom_loader = FromPackageLoader(__name__, ("grammars",))

        grammar = """
        start: startab

        %import ab.startab
        """

        p = AttributeLark.from_string(grammar, import_paths=[custom_loader])
        self.assertEqual(
            p.parse("ab")[0],
            Tree(
                "start",
                [
                    Tree(
                        "startab",
                        [Tree("ab__expr", [Token("ab__A", "a"), Token("ab__B", "b")])],
                    )
                ],
            ),
        )

    def test_import_custom_sources2(self):
        custom_loader = FromPackageLoader(__name__, ("grammars",))

        grammar = """
        start: rule_to_import

        %import test_relative_import_of_nested_grammar__grammar_to_import.rule_to_import
        """
        p = AttributeLark.from_string(grammar, import_paths=[custom_loader])
        x = p.parse("N")[0]
        self.assertEqual(next(x.find_data("rule_to_import")).children, ["N"])

    def test_import_custom_sources3(self):
        custom_loader2 = FromPackageLoader(__name__)
        grammar = """
        %import .test_relative_import (start, WS)
        %ignore WS
        """
        p = AttributeLark.from_string(
            grammar, import_paths=[custom_loader2], source_path=__file__
        )  # import relative to current file
        x = p.parse("12 capybaras")[0]
        self.assertEqual(x.children, ["12", "capybaras"])

    def test_find_grammar_errors(self):
        text = """
        a: rule
        b rule
        c: rule
        B.: "hello" f
        D: "okay"
        """

        assert [e.line for e, _s in find_grammar_errors(text)] == [3, 5]

        text = """
        a: rule
        b rule
        | ok
        c: rule
        B.: "hello" f
        D: "okay"
        """

        assert [e.line for e, _s in find_grammar_errors(text)] == [3, 4, 6]

        text = """
        a: rule @#$#@$@&&
        b: rule
        | ok
        c: rule
        B: "hello" f @
        D: "okay"
        """

        x = find_grammar_errors(text)
        assert [e.line for e, _s in find_grammar_errors(text)] == [2, 6]

    def test_ranged_repeat_terms(self):
        g = """!start: AAA
                AAA: "A"~3
            """
        l = AttributeLark.from_string(g)
        self.assertEqual(l.parse("AAA")[0], Tree("start", ["AAA"]))
        self.assertRaises((ParseError, UnexpectedInput), l.parse, "AA")
        self.assertRaises((ParseError, UnexpectedInput), l.parse, "AAAA")

        g = """!start: AABB CC
                AABB: "A"~0..2 "B"~2
                CC: "C"~1..2
            """
        l = AttributeLark.from_string(g)
        self.assertEqual(l.parse("AABBCC")[0], Tree("start", ["AABB", "CC"]))
        self.assertEqual(l.parse("BBC")[0], Tree("start", ["BB", "C"]))
        self.assertEqual(l.parse("ABBCC")[0], Tree("start", ["ABB", "CC"]))
        self.assertRaises((ParseError, UnexpectedInput), l.parse, "AAAB")
        self.assertRaises((ParseError, UnexpectedInput), l.parse, "AAABBB")
        self.assertRaises((ParseError, UnexpectedInput), l.parse, "ABB")
        self.assertRaises((ParseError, UnexpectedInput), l.parse, "AAAABB")

    def test_ranged_repeat_large(self):
        g = """!start: "A"~60
            """
        l = AttributeLark.from_string(g)
        self.assertGreater(
            len(l.rules), 1, "Expected that more than one rule will be generated"
        )
        self.assertEqual(l.parse("A" * 60)[0], Tree("start", ["A"] * 60))
        self.assertRaises(ParseError, l.parse, "A" * 59)
        self.assertRaises((ParseError, UnexpectedInput), l.parse, "A" * 61)

        g = """!start: "A"~15..100
            """
        l = AttributeLark.from_string(g)
        for i in range(0, 110):
            if 15 <= i <= 100:
                self.assertEqual(l.parse("A" * i)[0], Tree("start", ["A"] * i))
            else:
                self.assertRaises(UnexpectedInput, l.parse, "A" * i)

        # 8191 is a Mersenne prime
        g = """start: "A"~8191
            """
        l = AttributeLark.from_string(g)
        self.assertEqual(l.parse("A" * 8191)[0], Tree("start", []))
        self.assertRaises(UnexpectedInput, l.parse, "A" * 8190)
        self.assertRaises(UnexpectedInput, l.parse, "A" * 8192)

    def test_large_terminal(self):
        g = "start: NUMBERS\n"
        g += "NUMBERS: " + "|".join('"%s"' % i for i in range(0, 1000))

        l = AttributeLark.from_string(g)
        for i in (0, 9, 99, 999):
            self.assertEqual(l.parse(str(i))[0], Tree("start", [str(i)]))
        for i in (-1, 1000):
            self.assertRaises(UnexpectedInput, l.parse, str(i))

    def test_list_grammar_imports(self):
        grammar = """
            %import .test_templates_import (start, sep)

            %override sep{item, delim}: item (delim item)* delim?
            %ignore " "
            """

        imports = list_grammar_imports(grammar, [os.path.dirname(__file__)])
        self.assertEqual(
            {os.path.split(i)[-1] for i in imports},
            {"test_templates_import.lark", "templates.lark"},
        )

        imports = list_grammar_imports("%import common.WS", [])
        assert len(imports) == 1 and imports[0].pkg_name == "lark"

    def test_inline_with_expand_single(self):
        grammar = r"""
        start: _a
        !?_a: "A"
        """
        self.assertRaises(GrammarError, AttributeLark, grammar)

    def test_line_breaks(self):
        p = AttributeLark.from_string(r"""start: "a" \
                       "b"
                """)
        p.parse("ab")

    def test_symbol_eq(self):
        a = None
        b = Symbol("abc")

        self.assertNotEqual(a, b)


if __name__ == "__main__":
    main()
