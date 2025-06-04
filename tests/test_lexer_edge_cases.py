import pytest
from attribute_lark import AttributeLark
from attribute_lark.lexer import LineCounter
from attribute_lark.exceptions import UnexpectedCharacters

def test_lexer_empty_input():
    """Test lexer behavior with empty input"""
    p = AttributeLark.from_string("""
        start: NUMBER* { stack[-1] = len([x for x in stack[-1] if x.type == 'NUMBER']) }
        NUMBER: /[0-9]+/
        WS: /\\s+/
        %ignore WS
    """)

    tokens = list(p.lexer.lex(""))
    assert len(tokens) == 0

    tree, result = p.parse("")
    assert result == 0

def test_lexer_overlapping_tokens():
    """Test lexer with overlapping token definitions"""
    p = AttributeLark.from_string("""
        start: (NUM | DECIMAL)+ { stack[-1] = len(stack[-1]) }
        NUM: /[0-9]+/
        DECIMAL: /[0-9]+\\.[0-9]+/
        WS: /\\s+/
        %ignore WS
    """)

    tokens = list(p.lexer.lex("123 45.67 89"))
    assert len(tokens) == 3
    assert [t.type for t in tokens if t.type != 'WS'] == ['NUM', 'DECIMAL', 'NUM']

def test_lexer_unicode():
    """Test lexer with Unicode characters"""
    p = AttributeLark.from_string("""
        start: WORD+ { stack[-1] = len(stack[-1]) }
        WORD: /\\w+/u
        WS: /\\s+/
        %ignore WS
    """)

    input_text = "Hello 你好 Καλημέρα"
    tokens = list(p.lexer.lex(input_text))
    assert len([t for t in tokens if t.type == 'WORD']) == 3

def test_lexer_line_counting():
    """Test line and column counting in lexer"""
    p = AttributeLark.from_string("""
        start: WORD+ { stack[-1] = len(stack[-1]) }
        WORD: /\\w+/
        NEWLINE: /\\n/
        WS: /[ \\t]+/
        %ignore WS
        %ignore NEWLINE
    """)

    input_text = "hello\\n  world\\n  test"
    tokens = list(p.lexer.lex(input_text))
    assert tokens[0].line == 1
    assert tokens[1].line == 2
    assert tokens[2].line == 3

def test_lexer_invalid_chars():
    """Test lexer error handling with invalid characters"""
    p = AttributeLark.from_string("""
        start: WORD+ { stack[-1] = len(stack[-1]) }
        WORD: /[a-zA-Z]+/
        WS: /\\s+/
        %ignore WS
    """)

    with pytest.raises(UnexpectedCharacters):
        list(p.lexer.lex("hello123world"))

def test_line_counter():
    """Test LineCounter functionality"""
    lc = LineCounter("\\n")

    # Test single line
    lc.feed("hello world")
    assert lc.line == 1
    assert lc.column == 12

    # Test multiple lines
    lc.feed("\\ntest\\nmore")
    assert lc.line == 3
    assert lc.column == 5
