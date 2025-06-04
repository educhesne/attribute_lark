"""Test attribute grammar functionality."""
import pytest
from attribute_lark import AttributeLark

def test_basic_attribute():
    """Test basic attribute synthesis."""
    parser = AttributeLark.from_string("""
        start: expr {{ return $1 }}
        expr: NUMBER {{ return int($0) }}
        NUMBER: /[0-9]+/
    """)

    tree, attr = parser.parse("42")
    assert attr == 42

def test_synthesized_attribute():
    """Test synthesized attribute computation."""
    parser = AttributeLark.from_string("""
        start: expr {{ return $1 }}
        ?expr: term {{ return $1 }}
             | expr "+" term {{ return $1 + $3 }}
        ?term: factor {{ return $1 }}
              | term "*" factor {{ return $1 * $3 }}
        ?factor: NUMBER {{ return int($0) }}
               | "(" expr ")" {{ return $2 }}
        NUMBER: /[0-9]+/
        %ignore " "
    """)

    tree, attr = parser.parse("2 + 3 * 4")
    assert attr == 14  # Should follow operator precedence

def test_inherited_attribute():
    """Test propagation of inherited attributes."""
    parser = AttributeLark.from_string("""
        start: decl {{ return $1 }}
        decl: type NAME {{
            env = {}  # Create symbol table
            env[$2] = $1['type']
            return env
        }}
        type: "int" {{ return {'type': 'integer'} }}
            | "str" {{ return {'type': 'string'} }}
        NAME: /[a-z]+/
        %ignore " "
    """)

    tree, attr = parser.parse("int foo")
    assert attr == {'foo': 'integer'}

def test_attribute_error():
    """Test error handling in attribute evaluation."""
    parser = AttributeLark.from_string("""
        start: expr {{ return $1 }}
        expr: NUMBER {{ return 1/int($0) }}  # Potential division by zero
        NUMBER: /[0-9]+/
    """)

    with pytest.raises(ZeroDivisionError):
        parser.parse("0")

def test_multiple_attributes():
    """Test handling of multiple attributes."""
    parser = AttributeLark.from_string("""
        start: pair {{ return {'sum': $1[0], 'product': $1[1]} }}
        pair: NUMBER NUMBER {{
            x, y = int($1), int($2)
            return (x + y, x * y)
        }}
        NUMBER: /[0-9]+/
        %ignore " "
    """)

    tree, attr = parser.parse("3 4")
    assert attr['sum'] == 7
    assert attr['product'] == 12

def test_contextual_attribute():
    """Test contextual attribute usage."""
    parser = AttributeLark.from_string("""
        start: statements {{ return $1 }}
        statements: statement+ {{
            env = {}
            for stmt in $1:
                env.update(stmt)
            return env
        }}
        statement: NAME "=" expr {{ return {$1: $3} }}
        expr: NUMBER {{ return int($0) }}
            | NAME {{
                # Look up variable in current environment
                env = GLOBAL['env']
                if $0 not in env:
                    raise NameError(f"Variable {$0} not defined")
                return env[$0]
            }}
        NAME: /[a-z]+/
        NUMBER: /[0-9]+/
        %ignore " "
    """)

    tree, attr = parser.parse("x = 42\ny = x")
    assert attr['x'] == 42
    assert attr['y'] == 42

    with pytest.raises(NameError):
        parser.parse("x = y")
