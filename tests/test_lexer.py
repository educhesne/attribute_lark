from unittest import TestCase, main

from lark import AttributeLark


class TestLexer(TestCase):
    def setUp(self):
        pass

    def test_basic(self):
        p = AttributeLark.from_string("""
            start: "a" "b" "c" "d"
            %ignore " "
        """)

        res = list(p.lex("abc cba dd"))
        assert res == list("abccbadd")

        res = list(p.lex("abc cba dd", dont_ignore=True))
        assert res == list("abc cba dd")


if __name__ == "__main__":
    main()
