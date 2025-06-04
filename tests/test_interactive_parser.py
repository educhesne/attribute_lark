import pytest
from attribute_lark import AttributeLark

def test_interactive_parser_basic():
    """Test basic interactive parser functionality"""
    p = AttributeLark.from_string("""
        start: expr { stack[-1] = stack[-1] }
        expr: NUMBER { stack[-1] = int(stack[-1]) }
            | expr "+" expr { stack[-1] = stack[-3] + stack[-1] }
        NUMBER: /[0-9]+/
        WS: /\\s+/
        %ignore WS
    """)

    states = p.parse_interactive("42")
    assert len(states) > 0

    final_state = p.interactive_parser.feed_eos(states)
    assert final_state.attribute_stack[-1] == 42

def test_interactive_parser_incremental():
    """Test incremental parsing"""
    p = AttributeLark.from_string("""
        start: expr { stack[-1] = stack[-1] }
        expr: NUMBER { stack[-1] = int(stack[-1]) }
            | expr "+" expr { stack[-1] = stack[-3] + stack[-1] }
        NUMBER: /[0-9]+/
        WS: /\\s+/
        %ignore WS
    """)

    # Parse in multiple steps
    states = p.parse_interactive("42")
    states = p.interactive_parser.resume_parser(states, " + ")
    states = p.interactive_parser.resume_parser(states, "10")

    final_state = p.interactive_parser.feed_eos(states)
    assert final_state.attribute_stack[-1] == 52

def test_interactive_parser_ambiguous_input():
    """Test interactive parser with ambiguous input"""
    p = AttributeLark.from_string("""
        start: expr { stack[-1] = stack[-1] }
        expr: NUMBER { stack[-1] = int(stack[-1]) }
            | ID { stack[-1] = stack[-1] }
        NUMBER: /[0-9]+/
        ID: /[a-z][a-z0-9]*/
        WS: /\\s+/
        %ignore WS
    """)

    # Both NUMBER and ID could match at first
    states = p.parse_interactive("1")
    assert len(states) > 0

    final_state = p.interactive_parser.feed_eos(states)
    assert final_state.attribute_stack[-1] == 1

def test_interactive_parser_error_recovery():
    """Test interactive parser error handling and recovery"""
    p = AttributeLark.from_string("""
        start: expr { stack[-1] = stack[-1] }
        expr: NUMBER { stack[-1] = int(stack[-1]) }
            | expr "+" expr { stack[-1] = stack[-3] + stack[-1] }
        NUMBER: /[0-9]+/
        WS: /\\s+/
        %ignore WS
    """)

    states = p.parse_interactive("42 + ")
    with pytest.raises(Exception):  # Should handle incomplete input gracefully
        p.interactive_parser.feed_eos(states)

def test_interactive_parser_state_copy():
    """Test InteractiveParserState copy functionality"""
    p = AttributeLark.from_string("""
        start: NUMBER { stack[-1] = int(stack[-1]) }
        NUMBER: /[0-9]+/
    """)

    states = p.parse_interactive("42")
    state = states[0]
    state_copy = state.__copy__()

    assert state.lexer_state.text == state_copy.lexer_state.text
    assert state.lookahead_PDA_state.keys() == state_copy.lookahead_PDA_state.keys()

def test_interactive_parser_whitespace():
    """Test interactive parser handling of whitespace"""
    p = AttributeLark.from_string("""
        start: expr { stack[-1] = stack[-1] }
        expr: NUMBER { stack[-1] = int(stack[-1]) }
            | expr "+" expr { stack[-1] = stack[-3] + stack[-1] }
        NUMBER: /[0-9]+/
        WS: /\\s+/
        %ignore WS
    """)

    # Test with various whitespace patterns
    states = p.parse_interactive("42    +\\n\\n    10")
    final_state = p.interactive_parser.feed_eos(states)
    assert final_state.attribute_stack[-1] == 52
