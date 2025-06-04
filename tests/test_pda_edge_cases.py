import pytest
from attribute_lark import AttributeLark
from attribute_lark.exceptions import UnexpectedToken

def test_pda_empty_rule():
    """Test PDA behavior with empty rules"""
    p = AttributeLark.from_string("""
        start: empty { stack[-1] = "empty" }
        empty: { stack[-1] = None }
        WS: /\\s+/
        %ignore WS
    """)

    tree, result = p.parse("")
    assert result == "empty"

def test_pda_deep_recursion():
    """Test PDA with deeply nested expressions"""
    p = AttributeLark.from_string("""
        start: expr { stack[-1] = stack[-1] }
        expr: NUMBER { stack[-1] = int(stack[-1]) }
            | "(" expr ")" { stack[-1] = stack[-2] }
        NUMBER: /[0-9]+/
        WS: /\\s+/
        %ignore WS
    """)

    # Test nested parentheses
    tree, result = p.parse("(((42)))")
    assert result == 42

def test_pda_error_recovery():
    """Test PDA error handling and recovery"""
    p = AttributeLark.from_string("""
        start: expr { stack[-1] = stack[-1] }
        expr: NUMBER { stack[-1] = int(stack[-1]) }
            | expr "+" expr { stack[-1] = stack[-3] + stack[-1] }
        NUMBER: /[0-9]+/
        WS: /\\s+/
        %ignore WS
    """)

    with pytest.raises(UnexpectedToken):
        p.parse("42 + + 10")

def test_pda_state_copy():
    """Test PDAState copy functionality"""
    p = AttributeLark.from_string("""
        start: NUMBER { stack[-1] = int(stack[-1]) }
        NUMBER: /[0-9]+/
    """)

    state = p.PDA.get_initial_state()
    state_copy = state.__copy__()

    assert state.state_stack == state_copy.state_stack
    assert state.value_stack == state_copy.value_stack
    assert state.attribute_stack == state_copy.attribute_stack
    assert state != state_copy  # Different objects

def test_pda_global_vars():
    """Test PDA global variables"""
    p = AttributeLark.from_string("""
        %header { total = 0 }
        start: NUMBER+ { stack[-1] = GLOBAL.total }
        NUMBER: /[0-9]+/ { GLOBAL.total += int(stack[-1]) }
        WS: /\\s+/
        %ignore WS
    """)

    tree, result = p.parse("1 2 3")
    assert result == 6  # 1 + 2 + 3

def test_pda_multiple_starts():
    """Test PDA with multiple start symbols"""
    p = AttributeLark.from_string("""
        start1: NUMBER { stack[-1] = int(stack[-1]) }
        start2: NUMBER "+" NUMBER { stack[-1] = int(stack[-3]) + int(stack[-1]) }
        NUMBER: /[0-9]+/
        WS: /\\s+/
        %ignore WS
    """, start=['start1', 'start2'])

    tree1, result1 = p.parse("42", start='start1')
    assert result1 == 42

    tree2, result2 = p.parse("10 + 20", start='start2')
    assert result2 == 30
