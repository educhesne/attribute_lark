# coding=utf-8

import json
import pytest
from itertools import product

from attribute_lark import AttributeLark
from attribute_lark.reconstruct import Reconstructor

common = """
%import common (WS_INLINE, NUMBER, WORD)
%ignore WS_INLINE
"""


def _remove_ws(s):
    return s.replace(" ", "").replace("\n", "")


def assert_reconstruct(grammar, code, **options):
    parser = AttributeLark(grammar, maybe_placeholders=False, **options)
    tree, _ = parser.parse(code)
    new = Reconstructor(parser).reconstruct(tree)
    assert _remove_ws(code) == _remove_ws(new)

def test_starred_rule():
    g = (
        """
    start: item* { stack[-1] = list(stack[-1]) }
    item: NL { stack[-1] = stack[-1] }
        | rule { stack[-1] = stack[-1] }
    rule: WORD ":" NUMBER { stack[-1] = stack[-2] + ": " + stack[-1] }
    NL: /(\\r?\\n)+\\s*/
    """
        + common
    )

    code = """
    Elephants: 12
    """

    assert_reconstruct(g, code)

def test_starred_group():
    g = (
        """
    start: (rule | NL)* { stack[-1] = list(stack[-1]) }
    rule: WORD ":" NUMBER { stack[-1] = stack[-3] + ": " + stack[-1] }
    NL: /(\\r?\\n)+\\s*/
    """
        + common
    )

    code = """
    Elephants: 12
    """

    assert_reconstruct(g, code)

def test_alias():
    g = (
        """
    start: line* { stack[-1] = list(stack[-1]) }
    line: NL { stack[-1] = stack[-1] }
        | rule { stack[-1] = stack[-1] }
        | "hello" -> hi { stack[-1] = "hello" }
    rule: WORD ":" NUMBER { stack[-1] = stack[-3] + ": " + stack[-1] }
    NL: /(\\r?\\n)+\\s*/
    """
        + common
    )

    code = """
    Elephants: 12
    hello
    """

    assert_reconstruct(g, code)

def test_keep_tokens():
    g = (
        """
    start: (NL | stmt)* { stack[-1] = list(stack[-1]) }
    stmt: var op var { stack[-1] = stack[-3] + stack[-2] + stack[-1] }
    !op: ("+" | "-" | "*" | "/") { stack[-1] = stack[-1] }
    var: WORD { stack[-1] = stack[-1] }
    NL: /(\\r?\\n)+\\s*/
    """
        + common
    )

    code = """
    a+b
    """

    assert_reconstruct(g, code)

@pytest.mark.parametrize("test_code", ["a", "a*b", "a+b", "a*b+c", "a+b*c", "a+b*c+d"])
def test_expand_rule(test_code):
    g = (
        """
    ?start: (NL | mult_stmt)* { stack[-1] = list(stack[-1]) }
    ?mult_stmt: sum_stmt ["*" sum_stmt] { stack[-1] = stack[-3] + "*" + stack[-1] if len(stack) > 1 else stack[-1] }
    ?sum_stmt: var ["+" var] { stack[-1] = stack[-3] + "+" + stack[-1] if len(stack) > 1 else stack[-1] }
    var: WORD { stack[-1] = stack[-1] }
    NL: /(\\r?\\n)+\\s*/
    """
        + common
    )

    assert_reconstruct(g, test_code)

def test_json_example():
    test_json = """
        {
            "empty_object" : {},
            "empty_array"  : [],
            "booleans"     : { "YES" : true, "NO" : false },
            "numbers"      : [ 0, 1, -2, 3.3, 4.4e5, 6.6e-7 ],
            "strings"      : [ "This", [ "And" , "That", "And a \\"b" ] ],
            "nothing"      : null
        }
    """

    json_grammar = r"""
        ?start: value { stack[-1] = stack[-1] }

        ?value: object { stack[-1] = stack[-1] }
              | array { stack[-1] = stack[-1] }
              | string { stack[-1] = stack[-1] }
              | SIGNED_NUMBER      -> number { stack[-1] = stack[-1] }
              | "true"             -> true { stack[-1] = True }
              | "false"            -> false { stack[-1] = False }
              | "null"             -> null { stack[-1] = None }

        array  : "[" [value ("," value)*] "]" { stack[-1] = [x for x in stack[-2:-1]] if len(stack) > 2 else [] }
        object : "{" [pair ("," pair)*] "}" { stack[-1] = dict(stack[-2:-1]) if len(stack) > 2 else {} }
        pair   : string ":" value { stack[-1] = (stack[-3], stack[-1]) }

        string : ESCAPED_STRING { stack[-1] = stack[-1][1:-1] }

        %import common.ESCAPED_STRING
        %import common.SIGNED_NUMBER
        %import common.WS

        %ignore WS
    """

    json_parser = AttributeLark(json_grammar, maybe_placeholders=False)
    tree, _ = json_parser.parse(test_json)

    new_json = Reconstructor(json_parser).reconstruct(tree)
    assert json.loads(new_json) == json.loads(test_json)

@pytest.mark.parametrize("test_code",
    ["".join(p) for p in product(("", "a"), ("", "b"), ("", "c"), ("", "d"))]
)
def test_keep_all_tokens(test_code):
    g = """
    start: "a"? _B? c? _d? { stack[-1] = list(stack[-1]) if stack else [] }
    _B: "b" { stack[-1] = stack[-1] }
    c: "c" { stack[-1] = stack[-1] }
    _d: "d" { stack[-1] = stack[-1] }
    """
    assert_reconstruct(g, test_code, keep_all_tokens=True)

def test_switch_grammar_unicode_terminal():
    """
    This test checks that a parse tree built with a grammar containing only ascii characters can be reconstructed
    with a grammar that has unicode rules (or vice versa). The original bug assigned ANON terminals to unicode
    keywords, which offsets the ANON terminal count in the unicode grammar and causes subsequent identical ANON
    tokens (e.g., `+=`) to mismatch between the two grammars.
    """

    g1 = (
        """
    start: (NL | stmt)* { stack[-1] = list(stack[-1]) }
    stmt: "keyword" var op var { stack[-1] = " ".join([stack[-4], stack[-3], stack[-2], stack[-1]]) }
    !op: ("+=" | "-=" | "*=" | "/=") { stack[-1] = stack[-1] }
    var: WORD { stack[-1] = stack[-1] }
    NL: /(\\r?\\n)+\\s*/
    """
        + common
    )

    g2 = (
        """
    start: (NL | stmt)* { stack[-1] = list(stack[-1]) }
    stmt: "குறிப்பு" var op var { stack[-1] = " ".join([stack[-4], stack[-3], stack[-2], stack[-1]]) }
    !op: ("+=" | "-=" | "*=" | "/=") { stack[-1] = stack[-1] }
    var: WORD { stack[-1] = stack[-1] }
    NL: /(\\r?\\n)+\\s*/
    """
        + common
    )

    code = """
    keyword x += y
    """

    p1 = AttributeLark(g1, maybe_placeholders=False)
    p2 = AttributeLark(g2, maybe_placeholders=False)
    r = Reconstructor(p2)

    tree, _ = p1.parse(code)
    code2 = r.reconstruct(tree)
    tree2, _ = p2.parse(code2)
    assert tree == tree2
