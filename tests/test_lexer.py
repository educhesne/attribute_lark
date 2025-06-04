"""Test the lexer component."""
import pytest
from attribute_lark import AttributeLark

def test_basic_lexing():
    """Test basic lexing functionality."""
    p = AttributeLark.from_string("""
        start: "a" "b" "c" "d"
        %ignore " "
    """)

    # Test basic lexing
    res = list(p.lexer.lex("abc cba dd"))
    assert len(res) == 7
    assert [t.value for t in res] == list("abccbadd")

    # Test lexing with ignored tokens
    res = list(p.lexer.lex("abc cba dd", dont_ignore=True))
    assert len(res) == 8
    assert "".join(t.value for t in res) == "abc cba dd"

def test_attribute_lexing():
    """Test lexing with attribute grammar syntax."""
    p = AttributeLark.from_string("""
        start: item {{ return $1 }}
        item: "a" {{ return int($0) }}
    """)

    res = list(p.lexer.lex("a"))
    assert len(res) == 1
    assert res[0].type == '"a"'
    assert res[0].value == 'a'

def test_context_sensitive_lexing():
    """Test the contextual lexer capabilities."""
    # This grammar has an ambiguity that the contextual lexer should resolve
    p = AttributeLark.from_string("""
        start: expr+
        ?expr: NAME "=" NUMBER   -> assign
             | NAME             -> var
             | NUMBER          -> number
        NAME: /[a-z]+/
        NUMBER: /[0-9]+/
        %ignore " "
    """)

    res = list(p.lexer.lex("abc = 123 def = 456"))
    expected_types = ['NAME', '=', 'NUMBER', 'NAME', '=', 'NUMBER']
    expected_values = ['abc', '=', '123', 'def', '=', '456']

    assert len(res) == len(expected_types)
    assert [t.type for t in res] == expected_types
    assert [t.value for t in res] == expected_values

def test_unicode_lexing():
    """Test lexing of Unicode text."""
    p = AttributeLark.from_string(r"""
        start: WORD+
        WORD: /[\p{L}]+/u
        %ignore " "
    """)

    text = "Hello 你好 Γειά सौम"
    res = list(p.lexer.lex(text))
    expected = ['Hello', '你好', 'Γειά', 'सौम']
    assert len(res) == len(expected)
    assert [t.value for t in res] == expected

def test_regex_priority():
    """Test that regex patterns are matched with correct priority."""
    p = AttributeLark.from_string("""
        start: (TOKEN1 | TOKEN2)+
        TOKEN1: "aa"
        TOKEN2: "a"
        %ignore " "
    """)

    # Should match 'aa' before 'a'
    res = list(p.lexer.lex("aa a aa"))
    assert len(res) == 3
    assert [t.value for t in res] == ['aa', 'a', 'aa']
    assert [t.type for t in res] == ['TOKEN1', 'TOKEN2', 'TOKEN1']

def test_lexer_error():
    """Test lexer error handling."""
    from attribute_lark.exceptions import UnexpectedCharacters

    p = AttributeLark.from_string("""
        start: NAME+
        NAME: /[a-z]+/
        %ignore " "
    """)

    with pytest.raises(UnexpectedCharacters):
        list(p.lexer.lex("abc123"))
