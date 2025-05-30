from abc import ABC, abstractmethod
import os
import pickle
import re
from typing import (
    TypeVar,
    Type,
    List,
    Dict,
    Iterator,
    Callable,
    Union,
    Optional,
    Sequence,
    Tuple,
    Iterable,
    Any,
    TYPE_CHECKING,
    Collection,
    Literal,
)
from ast import Module as AstModule

if TYPE_CHECKING:
    from .tree import ParseTree
    from .visitors import Transformer

from .parsers.pushdown_automata import PDAState, ParseConf, PushDownAutomata
from .parsers.lalr_analysis import LALR_Analyzer

from .exceptions import ConfigurationError, assert_config, UnexpectedInput
from .utils import Serialize, SerializeMemoizer, FS, logger, TextOrSlice
from .load_grammar import (
    load_grammar,
    FromPackageLoader,
    Grammar,
    PackageResource,
)
from .tree import Tree
from .common import LexerConf, ParserConf, _LexerArgType, ParserCallbacks

from .lexer import TerminalDef, FSMLexer, Token, PostLex
from .parse_tree_builder import ParseTreeBuilder
from .grammar import Rule
from .parsers.parser import Parser, InteractiveParser


class LarkOptions(Serialize):
    """Specifies the options for Lark"""

    start: List[str]
    debug: bool
    strict: bool
    transformer: "Optional[Transformer]"
    propagate_positions: Union[bool, Callable]
    maybe_placeholders: bool
    cache: Union[bool, str]
    # regex: bool
    g_regex_flags: int
    keep_all_tokens: bool
    tree_class: Optional[Callable[[str, List], Any]]
    lexer: _LexerArgType
    postlex: Optional[PostLex]
    priority: 'Optional[Literal["auto", "normal", "invert"]]'
    lexer_callbacks: Dict[str, Callable[[Token], Token]]
    use_bytes: bool
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
            Throw an exception on any potential ambiguity, including shift/reduce conflicts, and regex collisions.
    transformer
            Applies the transformer to every parse tree (equivalent to applying it after the parse, but faster)
    propagate_positions
            Propagates positional attributes into the 'meta' attribute of all tree branches.
            Sets attributes: (line, column, end_line, end_column, start_pos, end_pos,
                              container_line, container_column, container_end_line, container_end_column)
            Accepts ``False``, ``True``, or a callable, which will filter which nodes to ignore when propagating.
    maybe_placeholders
            When ``True``, the ``[]`` operator returns ``None`` when not matched.
            When ``False``,  ``[]`` behaves like the ``?`` operator, and returns no value at all.
            (default= ``True``)
    cache
            Cache the results of the Lark grammar analysis, for x2 to x3 faster loading.

            - When ``False``, does nothing (default)
            - When ``True``, caches to a temporary file in the local directory
            - When given a string, caches to the path pointed by the string
    # regex
    #         When True, uses the ``regex`` module instead of the stdlib ``re``.
    g_regex_flags
            Flags that are applied to all terminals (both regex and strings)
    keep_all_tokens
            Prevent the tree builder from automagically removing "punctuation" tokens (Default: ``False``)
    tree_class
            Lark will produce trees comprised of instances of this class instead of the default ``lark.Tree``.

    **=== Algorithm Options ===**

    lexer
            Decides whether or not to use a lexer stage

            - "basic": Use a basic lexer
            - "contextual" (default): Stronger lexer (only works with parser="lalr")

    **=== Misc. / Domain Specific Options ===**

    postlex
            Lexer post-processing (Default: ``None``) Only works with the basic and contextual lexers.
    priority
            How priorities should be evaluated - "auto", ``None``, "normal", "invert" (Default: "auto")
    lexer_callbacks
            Dictionary of callbacks for the lexer. May alter tokens during lexing. Use with caution.
    use_bytes
            Accept an input of type ``bytes`` instead of ``str``.
    ordered_sets
            Should Earley use ordered-sets to achieve stable output (~10% slower than regular sets. Default: True)
    edit_terminals
            A callback for editing the terminals before parse.
    import_paths
            A List of either paths or loader functions to specify from where grammars are imported
    source_path
            Override the source of from where the grammar was loaded. Useful for relative imports and unconventional grammar loading
    **=== End of Options ===**
    """
    if __doc__:
        __doc__ += OPTIONS_DOC

    # Adding a new option needs to be done in multiple places:
    # - In the dictionary below. This is the primary truth of which options `Lark.__init__` accepts
    # - In the docstring above. It is used both for the docstring of `LarkOptions` and `Lark`, and in readthedocs
    # - As an attribute of `LarkOptions` above
    # - Potentially in `_LOAD_ALLOWED_OPTIONS` below this class, when the option doesn't change how the grammar is loaded
    # - Potentially in `lark.tools.__init__`, if it makes sense, and it can easily be passed as a cmd argument
    _defaults: Dict[str, Any] = {
        "debug": False,
        "strict": False,
        "keep_all_tokens": False,
        "tree_class": None,
        "cache": False,
        "postlex": None,
        "lexer": "contextual",
        "transformer": None,
        "start": "start",
        "priority": "normal",
        # 'regex': False,
        "propagate_positions": False,
        "lexer_callbacks": {},
        "maybe_placeholders": True,
        "edit_terminals": None,
        "g_regex_flags": 0,
        "use_bytes": False,
        "ordered_sets": True,
        "import_paths": [],
        "source_path": None,
        "_plugins": {},
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
    ) -> "LarkOptions":
        return cls(data)


# Options that can be passed to the Lark parser, even when it was loaded from cache/standalone.
# These options are only used outside of `load_grammar`.
_LOAD_ALLOWED_OPTIONS = {
    "postlex",
    "transformer",
    "lexer_callbacks",
    "use_bytes",
    "debug",
    "g_regex_flags",
    "regex",
    "propagate_positions",
    "tree_class",
    "_plugins",
}

_VALID_PRIORITY_OPTIONS = ("auto", "normal", "invert", None)


_T = TypeVar("_T", bound="AttributeLark")


class AttributeLark(Serialize):
    """Main interface for the library.

    It's mostly a thin wrapper for the many different parsers, and for the tree constructor.

    Parameters:
        grammar: a string or file-object containing the grammar spec (using Lark's ebnf syntax)
        options: a dictionary controlling various aspects of AttributeLark.

    Example:
        >>> AttributeLark(r'''start: "foo" ''')
        AttributeLark(...)
    """

    source_path: str
    source_grammar: str
    grammar: "Grammar"
    options: LarkOptions
    lexer: FSMLexer
    terminals: Collection[TerminalDef]
    python_header: Optional[AstModule]

    def __init__(self, grammar: Grammar, **options):
        self.options = LarkOptions(options)

        self.grammar = grammar

        # Compile the EBNF grammar into BNF
        if self.options.postlex is not None:
            terminals_to_keep = set(self.options.postlex.always_accept)
        else:
            terminals_to_keep = set()
        self.terminals, self.rules, self.ignore_tokens, self.python_header = (
            self.grammar.compile(self.options.start, terminals_to_keep)
        )

        if self.options.edit_terminals:
            for t in self.terminals:
                self.options.edit_terminals(t)

        # self.parser = self._build_parser()

        self.PDA = self._build_pda()

        collisions_to_check = [
            set([name for name in self.PDA.transitions[state].keys() if name.isupper()])
            for state in self.PDA.transitions
        ]
        self.lexer_conf = LexerConf(
            self.terminals,
            self.ignore_tokens,
            self.options.postlex,
            self.options.g_regex_flags,
            use_bytes=self.options.use_bytes,
            strict=self.options.strict,
        )
        self.lexer = FSMLexer.from_conf(
            self.lexer_conf, check_collisions=collisions_to_check
        )
        self.parser = Parser(self.PDA, self.lexer)
        self.interactive_parser = InteractiveParser(self.PDA, self.lexer)
        # if cache_fn:
        #     logger.debug('Saving grammar to cache: %s', cache_fn)
        #     try:
        #         with FS.open(cache_fn, 'wb') as f:
        #             assert cache_sha256 is not None
        #             f.write(cache_sha256.encode('utf8') + b'\n')
        #             pickle.dump(used_files, f)
        #             self.save(f, _LOAD_ALLOWED_OPTIONS)
        #     except IOError as e:
        #         logger.exception("Failed to save Lark to cache: %r.", cache_fn, e)

    @classmethod
    def from_string(cls, grammar_str: str, grammar_name: str = "<?>", **options):
        keep_all_tokens = options.get("keep_all_tokens", False)
        import_paths = options.get("import_paths", None)
        grammar, _ = load_grammar(
            grammar_str,
            grammar_name,
            import_paths,
            global_keep_all_tokens=keep_all_tokens,
        )
        inst = cls(grammar, **options)
        inst.source_grammar = grammar_str
        return inst

    @classmethod
    def from_cache_file(cls, cache_file, **options):
        with FS.open(cache_file, "rb") as f:
            logger.debug("Loading grammar from cache: %s", cache_file)
            cached_parser_data = pickle.load(f)
            return cls.load(cached_parser_data, **options)

    if __doc__:
        __doc__ += "\n\n" + LarkOptions.OPTIONS_DOC

    __serialize_fields__ = "parser", "rules", "options"

    def _prepare_callbacks(self) -> ParserCallbacks:
        # we don't need these callbacks if we aren't building a tree
        callbacks = ParseTreeBuilder(
            self.rules,
            self.options.tree_class or Tree,
            self.options.propagate_positions,
            self.options.maybe_placeholders,
        ).create_callback(self.options.transformer)
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

    def save(self, f, exclude_options: Collection[str] = ()) -> None:
        """Saves the instance into the given file object

        Useful for caching and multiprocessing.
        """
        data, m = self.memo_serialize([TerminalDef, Rule])
        if exclude_options:
            data["options"] = {
                n: v for n, v in data["options"].items() if n not in exclude_options
            }
        pickle.dump({"data": data, "memo": m}, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls: Type[_T], f) -> _T:
        """Loads an instance from the given file object

        Useful for caching and multiprocessing.
        """
        inst = cls.__new__(cls)
        return inst._load(f)

    def _deserialize_lexer_conf(
        self,
        data: Dict[str, Any],
        memo: Dict[int, Union[TerminalDef, Rule]],
        options: LarkOptions,
    ) -> LexerConf:
        lexer_conf = LexerConf.deserialize(data["lexer_conf"], memo)
        lexer_conf.callbacks = options.lexer_callbacks or {}
        lexer_conf.re_module = regex if options.regex else re
        lexer_conf.use_bytes = options.use_bytes
        lexer_conf.g_regex_flags = options.g_regex_flags
        lexer_conf.skip_validation = True
        lexer_conf.postlex = options.postlex
        return lexer_conf

    def _load(self: _T, f: Any, **kwargs) -> _T:
        if isinstance(f, dict):
            d = f
        else:
            d = pickle.load(f)
        memo_json = d["memo"]
        data = d["data"]

        assert memo_json
        memo = SerializeMemoizer.deserialize(
            memo_json, {"Rule": Rule, "TerminalDef": TerminalDef}, {}
        )
        options = dict(data["options"])
        if (set(kwargs) - _LOAD_ALLOWED_OPTIONS) & set(LarkOptions._defaults):
            raise ConfigurationError(
                "Some options are not allowed when loading a Parser: {}".format(
                    set(kwargs) - _LOAD_ALLOWED_OPTIONS
                )
            )
        options.update(kwargs)
        self.options = LarkOptions.deserialize(options, memo)
        self.rules = [Rule.deserialize(r, memo) for r in data["rules"]]
        self.source_path = "<deserialized>"
        _validate_frontend_args(self.options.lexer)
        self.lexer_conf = self._deserialize_lexer_conf(
            data["parser"], memo, self.options
        )
        self.terminals = self.lexer_conf.terminals
        self._prepare_callbacks()
        self._terminals_dict = {t.name: t for t in self.terminals}
        self.parser = _deserialize_parsing_frontend(
            data["parser"],
            memo,
            self.lexer_conf,
            self._callbacks,
            self.options,  # Not all, but multiple attributes are used
        )
        return self

    @classmethod
    def _load_from_dict(cls, data, memo, **kwargs):
        inst = cls.__new__(cls)
        return inst._load({"data": data, "memo": memo}, **kwargs)

    @classmethod
    def open(
        cls: Type[_T], grammar_filename: str, rel_to: Optional[str] = None, **options
    ) -> _T:
        """Create an instance of Lark with the grammar given by its filename

        If ``rel_to`` is provided, the function will find the grammar filename in relation to it.

        Example:

            >>> Lark.open("grammar_file.lark", rel_to=__file__)
            Lark(...)

        """
        if rel_to:
            basepath = os.path.dirname(rel_to)
            grammar_filename = os.path.join(basepath, grammar_filename)

        with open(grammar_filename, encoding="utf8") as f:
            grammar_str = f.read()

        return cls.from_string(grammar_str, **options)

    @classmethod
    def open_from_package(
        cls: Type[_T],
        package: str,
        grammar_path: str,
        search_paths: "Sequence[str]" = [""],
        **options,
    ) -> _T:
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

    def __repr__(self):
        return "Lark(open(%r), lexer=%r, ...)" % (self.source_path, self.options.lexer)

    def lex(self, text: str) -> Iterator[Token]:
        """Only lex (and postlex) the text, without parsing it. Only relevant when lexer='basic'

        :raises UnexpectedCharacters: In case the lexer cannot find a suitable match.
        """
        stream = self.lexer.lex(None)
        if self.options.postlex:
            return self.options.postlex.process(stream)
        return stream

    def get_terminal(self, name: str) -> TerminalDef:
        """Get information about a terminal"""
        return self._terminals_dict[name]

    def parse_interactive(
        self, text: Optional[TextOrSlice] = None, start: Optional[str] = None
    ) -> "InteractiveParser":
        """Start an interactive parsing session.

        Parameters:
            text (TextOrSlice, optional): Text to be parsed. Required for ``resume_parse()``.
            start (str, optional): Start symbol

        Returns:
            A new InteractiveParser instance.

        See Also: ``Lark.parse()``
        """
        return self.parser.parse_interactive(text, start=start)

    def parse(
        self,
        text: str,
        start: Optional[str] = None,
    ) -> Tuple[ParseTree, Any]:
        """Parse the given text, according to the options provided.

        Parameters:
            text (TextOrSlice): Text to be parsed, as `str` or `bytes`.
                TextSlice may also be used, but only when lexer='basic' or 'contextual'.
            start (str, optional): Required if Lark was given multiple possible start symbols (using the start option).
            on_error (function, optional): if provided, will be called on UnexpectedInput error,
                with the exception as its argument. Return true to resume parsing, or false to raise the exception.
                LALR only. See examples/advanced/error_handling.py for an example of how to use on_error.

        Returns:
            If a transformer is supplied to ``__init__``, returns whatever is the
            result of the transformation. Otherwise, returns a Tree instance.

        :raises UnexpectedInput: On a parse error, one of these sub-exceptions will rise:
                ``UnexpectedCharacters``, ``UnexpectedToken``, or ``UnexpectedEOF``.
                For convenience, these sub-exceptions also inherit from ``ParserError`` and ``LexerError``.

        """
        return self.parser.parse(text, start=start)


###}
