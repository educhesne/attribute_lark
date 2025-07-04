"""
Custom lexer
============

Demonstrates using a custom lexer to parse a non-textual stream of data

You can use a custom lexer to tokenize text when the lexers offered by Lark
are too slow, or not flexible enough.

You can also use it (as shown in this example) to tokenize streams of objects.
"""

from attribute_lark import AttributeLark, Transformer, v_args
from attribute_lark.lexer import FSMLexer, Token


class TypeLexer(FSMLexer):
    def __init__(self, lexer_conf):
        pass

    def lex(self, data):
        for obj in data:
            if isinstance(obj, int):
                yield Token("INT", obj)
            elif isinstance(obj, (type(""), type(""))):
                yield Token("STR", obj)
            else:
                raise TypeError(obj)


parser = AttributeLark.from_string(
    """
        start: data_item+
        data_item: STR INT*

        %declare STR INT
        """,
    lexer=TypeLexer,
)


class ParseToDict(Transformer):
    @v_args(inline=True)
    def data_item(self, name, *numbers):
        return name.value, [n.value for n in numbers]

    start = dict


def test():
    data = ["alice", 1, 27, 3, "bob", 4, "carrie", "dan", 8, 6]

    print(data)

    tree = parser.parse(data)
    res = ParseToDict().transform(tree)

    print("-->")
    print(res)  # prints {'alice': [1, 27, 3], 'bob': [4], 'carrie': [], 'dan': [8, 6]}


if __name__ == "__main__":
    test()
