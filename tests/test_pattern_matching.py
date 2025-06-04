import pytest
from attribute_lark import Token, Tree, AttributeLark

# Create a token fixture for reuse
@pytest.fixture
def simple_token():
    return Token("A", "a")

def test_matches_with_string(simple_token):
    match simple_token:
        case "a":
            pass
        case _:
            assert False

def test_matches_with_str_positional_arg(simple_token):
    match simple_token:
        case str("a"):
            pass
        case _:
            assert False

def test_matches_with_token_positional_arg(simple_token):
    match simple_token:
        case Token("a"):
            assert False
        case Token("A"):
            pass
        case _:
            assert False

def test_matches_with_token_kwarg_type(simple_token):
    match simple_token:
        case Token(type="A"):
            pass
        case _:
            assert False

def test_matches_with_bad_token_type(simple_token):
    match simple_token:
        case Token(type="B"):
            assert False
        case _:
            pass

@pytest.fixture
def complex_trees():
    tree1 = Tree("a", [
        Tree("b", [Token("DATA", "x")]),
        Tree("c", [Token("DATA", "y")]),
        Tree("d", [Token("DATA", "z")])
    ])
    tree2 = Tree(
        "a",
        [
            Tree("b", [Token("T", "x")]),
            Tree("c", [Token("T", "y")]),
            Tree("d", [Tree("z", [Token("T", "zz"), Tree("zzz", [Token("DATA", "zzz")])])]),
        ],
    )
    return tree1, tree2

def test_match_basic_tree_patterns(complex_trees):
    tree1, _ = complex_trees

    # Test various tree pattern matching scenarios
    match tree1:
        case Tree("X", []):
            assert False
        case Tree("a", []):
            assert False
        case Tree("b", _):
            assert False
        case Tree("X", _):
            assert False

def test_match_simple_tree():
    tree = Tree("q", [Token("T", "x")])
    match tree:
        case Tree("q", [Token("T", "x")]):
            pass
        case _:
            assert False

def test_match_nested_simple_tree():
    tr = Tree("a", [Tree("b", [Token("T", "a")])])
    match tr:
        case Tree("a", [Tree("b", [Token("T", "a")])]):
            pass
        case _:
            assert False

def test_match_complex_nested_tree(complex_trees):
    _, tree2 = complex_trees
    # Test deeply nested tree structures
    match tree2:
        case Tree(
            "a",
            [
                Tree("b", [Token("T", "x")]),
                Tree("c", [Token("T", "y")]),
                Tree("d", [Tree("z", [Token("T", "zz"), Tree("zzz", "zzz")])]),
            ],
        ):
            pass
        case _:
            assert False

# Add new test for attribute grammar specific features
def test_match_tree_with_attributes():
    """Test pattern matching with attribute grammar parse trees"""
    p = AttributeLark.from_string("""
        start: NUMBER "+" NUMBER { stack[-1] = int(stack[-3]) + int(stack[-1]) }
        NUMBER: /[0-9]+/
    """)
    tree, attr = p.parse("42 + 123")

    match tree:
        case Tree("start", [Token("NUMBER", "42"), Token("+", "+"), Token("NUMBER", "123")]):
            pass
        case _:
            assert False

    assert attr == 165  # Verify attribute evaluation
