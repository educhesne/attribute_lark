from typing import (
    List,
    Dict,
    Callable,
    Union,
    Optional,
    Tuple,
    Any,
    TYPE_CHECKING,
    Collection,
    Literal,
    Sequence,
)
from ast import Module as AstModule

if TYPE_CHECKING:
    from .tree import ParseTree

from .parsers.pushdown_automata import ParseConf, PushDownAutomata
from .parsers.lalr_analysis import LALR_Analyzer

from .exceptions import ConfigurationError, assert_config
from .utils import Serialize
from .load_grammar import (
    load_grammar,
    Grammar,
    PackageResource,
    FromPackageLoader,
)
from .tree import Tree
from .common import LexerConf, ParserConf, _LexerArgType, ParserCallbacks

from .lexer import TerminalDef, FSMLexer, PostLex
from .parse_tree_builder import ParseTreeBuilder
from .grammar import Rule
from .parsers.parser import Parser, InteractiveParser, InteractiveParserState


class AttributeLarkOptions(Serialize):
    """Specifies the options for the attribute grammar parser"""

    start: List[str]
    debug: bool
    strict: bool
    propagate_positions: bool
    lexer: _LexerArgType
    postlex: Optional[PostLex]
    priority: 'Optional[Literal["auto", "normal", "invert"]]'
    g_regex_flags: int
    ordered_sets: bool
    edit_terminals: Optional[Callable[[TerminalDef], TerminalDef]]
    import_paths: "List[Union[str, Callable[[Union[None, str, PackageResource], str], Tuple[str, str]]]]"
    source_path: Optional[str]

    OPTIONS_DOC = r"""
    **===  General Options  ===**

    start
            The start symbol. Either a string, or a list of strings for multiple possible starts (Default: "start")
    debug
            Display debug information and extra warnings. Use only when debugging (Default: ``False``)
    strict
            Throw an exception on any potential ambiguity, including shift/reduce conflicts, and regex collisions
    propagate_positions
            When True, propagates positional information through the parse tree
    g_regex_flags
            Flags that are applied to all terminals (both regex and strings)
    ordered_sets
            Use ordered sets for more deterministic parsing (Default: ``True``)

    **=== Advanced Options ===**

    postlex
            Lexer post-processing (Default: ``None``)
    edit_terminals
            A callback for editing the terminals before parse
    import_paths
            A List of either paths or loader functions to specify from where grammars are imported
    source_path
            Override the source path for grammar imports
    """
    if __doc__:
        __doc__ += OPTIONS_DOC

    _defaults: Dict[str, Any] = {
        "debug": False,
        "strict": False,
        "postlex": None,
        "lexer": "contextual",
        "start": "start",
        "priority": "normal",
        "propagate_positions": False,
        "edit_terminals": None,
        "g_regex_flags": 0,
        "ordered_sets": True,
        "import_paths": [],
        "source_path": None,
    }

    def __init__(self, options_dict: Dict[str, Any]) -> None:
        o = dict(options_dict)

        options = {}
        for name, default in self._defaults.items():
            if name in o:
                value = o.pop(name)
                if isinstance(default, bool) and name not in (
                    "cache",
                    "use_bytes",
                    "propagate_positions",
                ):
                    value = bool(value)
            else:
                value = default

            options[name] = value

        if isinstance(options["start"], str):
            options["start"] = [options["start"]]

        self.__dict__["options"] = options

        if o:
            raise ConfigurationError("Unknown options: %s" % o.keys())

    def __getattr__(self, name: str) -> Any:
        try:
            return self.__dict__["options"][name]
        except KeyError as e:
            raise AttributeError(e)

    def __setattr__(self, name: str, value: str) -> None:
        assert_config(
            name, self.options.keys(), "%r isn't a valid option. Expected one of: %s"
        )
        self.options[name] = value

    def serialize(self, memo=None) -> Dict[str, Any]:
        return self.options

    @classmethod
    def deserialize(
        cls, data: Dict[str, Any], memo: Dict[int, Union[TerminalDef, Rule]]
    ) -> "AttributeLarkOptions":
        return cls(data)


class AttributeLark(Serialize):
    """A parser that supports attribute grammars.

    Parameters:
        grammar: Grammar specification using attribute grammar syntax
        options: Configuration options for the parser
    """

    source_path: str
    source_grammar: str
    grammar: Grammar
    options: AttributeLarkOptions
    lexer: FSMLexer
    terminals: Collection[TerminalDef]
    python_header: Optional[AstModule]

    def __init__(self, grammar_str: str, grammar_name: str = "<?>", **options):
        self.grammar = self._load_grammar(grammar_str, grammar_name, **options)
        self.source_grammar = grammar_str

        self.options = AttributeLarkOptions(options)

        # Compile grammar and prepare terminals
        terminals_to_keep = (
            set(self.options.postlex.always_accept) if self.options.postlex else set()
        )

        self.terminals, self.rules, self.ignore_tokens, self.python_header = (
            self.grammar.compile(self.options.start, terminals_to_keep)
        )

        if self.options.edit_terminals:
            for t in self.terminals:
                self.options.edit_terminals(t)

        # Build parser components
        self.PDA = self._build_pda()

        collisions_to_check = [
            {name for name in self.PDA.transitions[state].keys() if name.isupper()}
            for state in self.PDA.transitions
        ]

        self.lexer_conf = LexerConf(
            self.terminals,
            self.ignore_tokens,
            self.options.postlex.always_accept if self.options.postlex else (),
            self.options.g_regex_flags,
            strict=self.options.strict,
        )
        self.lexer = FSMLexer.from_conf(
            self.lexer_conf, check_collisions=collisions_to_check
        )
        self.parser = Parser(self.PDA, self.lexer, self.options.postlex)
        self.interactive_parser = InteractiveParser(
            self.PDA, self.lexer, self.options.postlex
        )

    def _load_grammar(
        self, grammar_str: str, grammar_name: str = "<?>", **options
    ) -> Grammar:
        """Create a Grammar instance from a grammar string."""
        keep_all_tokens = options.get("keep_all_tokens", False)
        import_paths = options.get("import_paths", None)
        grammar, _ = load_grammar(
            grammar_str,
            grammar_name,
            import_paths,
            global_keep_all_tokens=keep_all_tokens,
        )
        return grammar

    @classmethod
    def open_from_package(
        cls,
        package: str,
        grammar_path: str,
        search_paths: "Sequence[str]" = [""],
        **options,
    ):
        """Create an instance of Lark with the grammar loaded from within the package `package`.
        This allows grammar loading from zipapps.

        Imports in the grammar will use the `package` and `search_paths` provided, through `FromPackageLoader`

        Example:

            Lark.open_from_package(__name__, "example.lark", ("grammars",), parser=...)
        """
        package_loader = FromPackageLoader(package, search_paths)
        full_path, text = package_loader(None, grammar_path)
        options.setdefault("source_path", full_path)
        options.setdefault("import_paths", [])
        options["import_paths"].append(package_loader)
        return cls(text, **options)

    def _prepare_callbacks(self) -> ParserCallbacks:
        callbacks = ParseTreeBuilder(
            self.rules,
            Tree,
            self.options.propagate_positions,
            True,  # maybe_placeholders
        ).create_callback(None)
        return callbacks

    def _build_pda(self) -> PushDownAutomata:
        callbacks = self._prepare_callbacks()
        parser_conf = ParserConf(
            self.rules, callbacks, self.options.start, self.python_header
        )
        analysis = LALR_Analyzer(
            parser_conf, debug=self.options.debug, strict=self.options.strict
        )
        analysis.compute_lalr()
        callbacks = parser_conf.callbacks

        parse_conf = ParseConf(
            analysis.parse_table, callbacks, self.options.start, self.python_header
        )
        return PushDownAutomata(parse_conf, self.options.debug)

    def parse(self, text: str, start: Optional[str] = None) -> Tuple["ParseTree", Any]:
        """Parse input text and return (parse_tree, attributes)."""
        return self.parser.parse(text, start=start)

    def parse_interactive(
        self, text: str, start: Optional[str] = None
    ) -> List[InteractiveParserState]:
        """Parse text in interactive mode, returning parser states."""
        return self.interactive_parser.parse_interactive(text, start=start)

    def __repr__(self) -> str:
        return f"AttributeLark(grammar={self.source_path!r})"
