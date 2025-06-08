from copy import deepcopy
import sys
from typing import Callable, Collection, Dict, Optional, TYPE_CHECKING, List

if TYPE_CHECKING:
    from .grammar import Rule

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

from .utils import Serialize
from .lexer import TerminalDef

from ast import Module as AstModule

_LexerArgType: "TypeAlias" = 'Union[Literal["contextual"], Type[Lexer]]'
ParserCallbacks = Dict[str, Callable]


class LexerConf(Serialize):
    __serialize_fields__ = (
        "terminals",
        "ignore",
        "g_regex_flags",
        "use_bytes",
        "lexer_type",
    )
    __serialize_namespace__ = (TerminalDef,)

    terminals: Collection[TerminalDef]
    ignore: Collection[str]
    always_accept: Collection[str]
    g_regex_flags: int
    skip_validation: bool
    lexer_type: Optional[_LexerArgType]
    strict: bool

    def __init__(
        self,
        terminals: Collection[TerminalDef],
        ignore: Collection[str] = (),
        always_accept: Collection[str] = (),
        g_regex_flags: int = 0,
        skip_validation: bool = False,
        use_bytes: bool = False,
        strict: bool = False,
    ):
        self.terminals = terminals
        self.terminals_by_name = {t.name: t for t in self.terminals}
        assert len(self.terminals) == len(self.terminals_by_name)
        self.ignore = ignore
        self.always_accept = always_accept
        self.g_regex_flags = g_regex_flags
        self.skip_validation = skip_validation
        self.strict = strict
        self.lexer_type = None

    def _deserialize(self):
        self.terminals_by_name = {t.name: t for t in self.terminals}

    def __deepcopy__(self, memo=None):
        return type(self)(
            deepcopy(self.terminals, memo),
            deepcopy(self.ignore, memo),
            deepcopy(self.always_accept, memo),
            deepcopy(self.g_regex_flags, memo),
            deepcopy(self.skip_validation, memo),
        )


class ParserConf(Serialize):
    __serialize_fields__ = "rules", "start", "python_header"

    rules: List["Rule"]
    callbacks: ParserCallbacks
    start: List[str]
    python_header: Optional[AstModule]

    def __init__(
        self,
        rules: List["Rule"],
        callbacks: ParserCallbacks,
        start: List[str],
        python_header: Optional[AstModule] = None,
    ):
        assert isinstance(start, list)
        self.rules = rules
        self.callbacks = callbacks
        self.start = start
        self.python_header = python_header
