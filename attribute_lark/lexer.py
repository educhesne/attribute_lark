# Lexer Implementation

from abc import abstractmethod, ABC
import re
from typing import (
    TypeVar,
    Type,
    Dict,
    Iterator,
    Collection,
    Optional,
    Any,
    ClassVar,
    TYPE_CHECKING,
    overload,
    Tuple,
    List,
    Iterable,
)
import warnings
import interegular
from copy import copy, deepcopy
from interegular.fsm import FSM as InteregularFSM, State as FSMState
from dataclasses import dataclass

if TYPE_CHECKING:
    from .common import LexerConf

from .utils import classify, get_regexp_width, Serialize, logger, TextOrSlice
from .exceptions import LexError, UnexpectedCharacters
from .grammar import TOKEN_DEFAULT_PRIORITY


class Pattern(Serialize, ABC):
    "An abstraction over regular expressions."

    value: str
    flags: Collection[str]
    raw: Optional[str]
    type: ClassVar[str]

    def __init__(
        self, value: str, flags: Collection[str] = (), raw: Optional[str] = None
    ) -> None:
        self.value = value
        self.flags = frozenset(flags)
        self.raw = raw

    def __repr__(self):
        return repr(self.to_regexp())

    # Pattern Hashing assumes all subclasses have a different priority!
    def __hash__(self):
        return hash((type(self), self.value, self.flags))

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.value == other.value
            and self.flags == other.flags
        )

    @abstractmethod
    def to_regexp(self) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def min_width(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def max_width(self) -> int:
        raise NotImplementedError()

    def _get_flags(self, value):
        for f in self.flags:
            value = "(?%s:%s)" % (f, value)
        return value


class PatternStr(Pattern):
    __serialize_fields__ = "value", "flags", "raw"

    type: ClassVar[str] = "str"

    def to_regexp(self) -> str:
        return self._get_flags(re.escape(self.value))

    @property
    def min_width(self) -> int:
        return len(self.value)

    @property
    def max_width(self) -> int:
        return len(self.value)


class PatternRE(Pattern):
    __serialize_fields__ = "value", "flags", "raw", "_width"

    type: ClassVar[str] = "re"

    def to_regexp(self) -> str:
        return self._get_flags(self.value)

    _width = None

    def _get_width(self):
        if self._width is None:
            self._width = get_regexp_width(self.to_regexp())
        return self._width

    @property
    def min_width(self) -> int:
        return self._get_width()[0]

    @property
    def max_width(self) -> int:
        return self._get_width()[1]


class TerminalDef(Serialize):
    "A definition of a terminal"

    __serialize_fields__ = "name", "pattern", "priority"
    __serialize_namespace__ = PatternStr, PatternRE

    name: str
    pattern: Pattern
    priority: int

    def __init__(
        self, name: str, pattern: Pattern, priority: int = TOKEN_DEFAULT_PRIORITY
    ) -> None:
        assert isinstance(pattern, Pattern), pattern
        self.name = name
        self.pattern = pattern
        self.priority = priority

    def __repr__(self):
        return "%s(%r, %r)" % (type(self).__name__, self.name, self.pattern)

    def user_repr(self) -> str:
        if self.name.startswith("__"):  # We represent a generated terminal
            return self.pattern.raw or self.name
        else:
            return self.name


_T = TypeVar("_T", bound="Token")


class Token(str):
    """A string with meta-information, that is produced by the lexer.

    When parsing text, the resulting chunks of the input that haven't been discarded,
    will end up in the tree as Token instances. The Token class inherits from Python's ``str``,
    so normal string comparisons and operations will work as expected.

    Attributes:
        type: Name of the token (as specified in grammar)
        value: Value of the token (redundant, as ``token.value == token`` will always be true)
        start_pos: The index of the token in the text
        line: The line of the token in the text (starting with 1)
        column: The column of the token in the text (starting with 1)
        end_line: The line where the token ends
        end_column: The next column after the end of the token. For example,
            if the token is a single character with a column value of 4,
            end_column will be 5.
        end_pos: the index where the token ends (basically ``start_pos + len(token)``)
    """

    __slots__ = (
        "type",
        "start_pos",
        "value",
        "line",
        "column",
        "end_line",
        "end_column",
        "end_pos",
    )

    __match_args__ = ("type", "value")

    type: str
    start_pos: Optional[int]
    value: Any
    line: Optional[int]
    column: Optional[int]
    end_line: Optional[int]
    end_column: Optional[int]
    end_pos: Optional[int]

    @overload
    def __new__(
        cls,
        type: str,
        value: Any,
        start_pos: Optional[int] = None,
        line: Optional[int] = None,
        column: Optional[int] = None,
        end_line: Optional[int] = None,
        end_column: Optional[int] = None,
        end_pos: Optional[int] = None,
    ) -> "Token": ...

    @overload
    def __new__(
        cls,
        type_: str,
        value: Any,
        start_pos: Optional[int] = None,
        line: Optional[int] = None,
        column: Optional[int] = None,
        end_line: Optional[int] = None,
        end_column: Optional[int] = None,
        end_pos: Optional[int] = None,
    ) -> "Token": ...

    def __new__(cls, *args, **kwargs):
        if "type_" in kwargs:
            warnings.warn(
                "`type_` is deprecated use `type` instead", DeprecationWarning
            )

            if "type" in kwargs:
                raise TypeError(
                    "Error: using both 'type' and the deprecated 'type_' as arguments."
                )
            kwargs["type"] = kwargs.pop("type_")

        return cls._future_new(*args, **kwargs)

    @classmethod
    def _future_new(
        cls,
        type,
        value,
        start_pos=None,
        line=None,
        column=None,
        end_line=None,
        end_column=None,
        end_pos=None,
    ):
        inst = super(Token, cls).__new__(cls, value)

        inst.type = type
        inst.start_pos = start_pos
        inst.value = value
        inst.line = line
        inst.column = column
        inst.end_line = end_line
        inst.end_column = end_column
        inst.end_pos = end_pos
        return inst

    @overload
    def update(
        self, type: Optional[str] = None, value: Optional[Any] = None
    ) -> "Token": ...

    @overload
    def update(
        self, type_: Optional[str] = None, value: Optional[Any] = None
    ) -> "Token": ...

    def update(self, *args, **kwargs):
        if "type_" in kwargs:
            warnings.warn(
                "`type_` is deprecated use `type` instead", DeprecationWarning
            )

            if "type" in kwargs:
                raise TypeError(
                    "Error: using both 'type' and the deprecated 'type_' as arguments."
                )
            kwargs["type"] = kwargs.pop("type_")

        return self._future_update(*args, **kwargs)

    def _future_update(
        self, type: Optional[str] = None, value: Optional[Any] = None
    ) -> "Token":
        return Token.new_borrow_pos(
            type if type is not None else self.type,
            value if value is not None else self.value,
            self,
        )

    @classmethod
    def new_borrow_pos(cls: Type[_T], type_: str, value: Any, borrow_t: "Token") -> _T:
        return cls(
            type_,
            value,
            borrow_t.start_pos,
            borrow_t.line,
            borrow_t.column,
            borrow_t.end_line,
            borrow_t.end_column,
            borrow_t.end_pos,
        )

    def __reduce__(self):
        return (
            self.__class__,
            (self.type, self.value, self.start_pos, self.line, self.column),
        )

    def __repr__(self):
        return "Token(%r, %r)" % (self.type, self.value)

    def __deepcopy__(self, memo):
        return Token(self.type, self.value, self.start_pos, self.line, self.column)

    def __eq__(self, other):
        if isinstance(other, Token) and self.type != other.type:
            return False

        return str.__eq__(self, other)

    __hash__ = str.__hash__


class LineCounter:
    "A utility class for keeping track of line & column information"

    __slots__ = "char_pos", "line", "column", "line_start_pos", "newline_char"

    def __init__(self, newline_char):
        self.newline_char = newline_char
        self.char_pos = 0
        self.line = 1
        self.column = 1
        self.line_start_pos = 0

    def __eq__(self, other):
        if not isinstance(other, LineCounter):
            return NotImplemented

        return (
            self.char_pos == other.char_pos and self.newline_char == other.newline_char
        )

    def feed(self, token: TextOrSlice, test_newline=True):
        """Consume a token and calculate the new line & column.

        As an optional optimization, set test_newline=False if token doesn't contain a newline.
        """
        if test_newline:
            newlines = token.count(self.newline_char)
            if newlines:
                self.line += newlines
                self.line_start_pos = (
                    self.char_pos + token.rindex(self.newline_char) + 1
                )

        self.char_pos += len(token)
        self.column = self.char_pos - self.line_start_pos + 1


def _regexp_has_newline(r: str):
    r"""Expressions that may indicate newlines in a regexp:
    - newlines (\n)
    - escaped newline (\\n)
    - anything but ([^...])
    - any-char (.) when the flag (?s) exists
    - spaces (\s)
    """
    return (
        "\n" in r or "\\n" in r or "\\s" in r or "[^" in r or ("(?s" in r and "." in r)
    )


def _check_regex_collisions(
    terminal_to_regexp: Dict[TerminalDef, str],
    comparator,
    strict_mode,
    max_collisions_to_show=8,
):
    if not comparator:
        comparator = interegular.Comparator.from_regexes(terminal_to_regexp)

    # When in strict mode, we only ever try to provide one example, so taking
    # a long time for that should be fine
    max_time = 2 if strict_mode else 0.2

    # We don't want to show too many collisions.
    if comparator.count_marked_pairs() >= max_collisions_to_show:
        return
    for group in classify(terminal_to_regexp, lambda t: t.priority).values():
        for a, b in comparator.check(group, skip_marked=True):
            assert a.priority == b.priority
            # Mark this pair to not repeat warnings when multiple different BasicLexers see the same collision
            comparator.mark(a, b)

            # Notify the user
            message = f"Collision between Terminals {a.name} and {b.name}. "
            try:
                example = comparator.get_example_overlap(
                    a, b, max_time
                ).format_multiline()
            except ValueError:
                # Couldn't find an example within max_time steps.
                example = "No example could be found fast enough. However, the collision does still exists"
            if strict_mode:
                raise LexError(f"{message}\n{example}")
            logger.warning(
                "%s The lexer will choose between them arbitrarily.\n%s",
                message,
                example,
            )
            if comparator.count_marked_pairs() >= max_collisions_to_show:
                logger.warning("Found 8 regex collisions, will not check for more.")
                return


class FSM(InteregularFSM):
    """Finite State Machine implementation extending InteregularFSM.

    A class that implements a Finite State Machine (FSM) based on the InteregularFSM class,
    with additional functionality for regex pattern matching and state transitions.

    Parameters
    ----------
    *args
        Positional arguments passed to InteregularFSM
    **kwargs
        Keyword arguments passed to InteregularFSM

    See Also
    --------
    InteregularFSM : Base class for FSM implementation
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    @classmethod
    def from_interegular(cls, fsm: InteregularFSM):
        return cls(fsm.alphabet, fsm.states, fsm.initial, fsm.finals, fsm.map)

    @classmethod
    def from_regex(cls, regex_str: str):
        """
        Create a lexer from a regular expression string.
        Parameters
        ----------
        regex_str : str
            The regular expression pattern to parse into a lexer
        Returns
        -------
        FSM
            A new FSM instance constructed from the regex pattern
        Notes
        -----
        Uses the interegular library to parse the regex pattern into a finite state machine
        which is then reduced for optimization.
        """

        return cls.from_interegular(
            interegular.parse_pattern(regex_str).to_fsm().reduce()
        )

    def get_next_state(self, state: FSMState, symbol: Any) -> Optional[FSMState]:
        """
        Get the next state in the finite state machine given current state and symbol.
        Parameters
        ----------
        state : FSMState
            Current state in the finite state machine
        symbol : Any
            Symbol to process for state transition
        Returns
        -------
        Optional[FSMState]
            Next state if transition exists, None otherwise
        """

        from interegular.fsm import anything_else

        if symbol not in self.alphabet:
            if anything_else in self.alphabet and symbol not in self.alphabet:
                symbol = anything_else
            else:
                return None
        transition = self.alphabet[symbol]
        if transition is None:
            return None

        if state in self.map and transition in self.map[state]:
            return self.map[state][transition]
        else:
            return None

    def is_final_state(self, state: FSMState) -> bool:
        """Check if state is a final state.
        Parameters
        ----------
        state : FSMState
            The state to check.
        Returns
        -------
        bool
        """

        return state in self.finals

    def is_active_state(self, state: FSMState) -> bool:
        """Check if a FSM state is active.
        Parameters
        ----------
        state : FSMState
            The state to check
        Returns
        -------
        bool
            True if state is in the map and has active transitions,
            False otherwise
        """

        return (state in self.map) and (len(self.map[state]) > 0)

    def is_prefix(self, text: str) -> bool:
        """Check if the given text can be a prefix of a string accepted by the automaton.
        Parameters
        ----------
        text : str
            The text to check.
        Returns
        -------
        bool
            True if text can be a prefix of an accepted string, False otherwise.
        """

        state = self.initial
        for symbol in text:
            state = self.get_next_state(state, symbol)
            if state is None:
                return False

        return True


class PostLex(ABC):
    @abstractmethod
    def process(self, stream: Iterator[Token]) -> Iterator[Token]:
        raise NotImplementedError

    always_accept: Iterable[str] = ()


class IdPostLex(PostLex):
    def process(self, stream):
        return stream

    always_accept: Iterable[str] = ()


@dataclass
class FSMLexerState:
    text: str
    pos: int
    scan_pos: int
    dict_states: Dict[str, Tuple[int, FSMState]]
    line_ctr: LineCounter

    def __copy__(self):
        return self.__class__(
            copy(self.text),
            copy(self.pos),
            copy(self.scan_pos),
            copy(self.dict_states),
            deepcopy(self.line_ctr),
        )

    @property
    def current_scan(self):
        return self.text[self.pos : self.scan_pos]


class FSMScanner:
    def __init__(self, fsm_dict: Dict[str, FSM]):
        self.fsm_dict = fsm_dict

    def advance_lexer_state(self, lexer_state: FSMLexerState):
        new_dict_states = {}
        scan_length = -1
        prev_scan_length = lexer_state.scan_pos - lexer_state.pos
        for name, (match_length, state) in lexer_state.dict_states.items():
            fsm = self.fsm_dict[name]
            if state is not None:
                i = -1
                for i, symbol in enumerate(lexer_state.text[lexer_state.scan_pos :]):
                    state = fsm.get_next_state(state, symbol)
                    if state is None:
                        break
                    if fsm.is_final_state(state):
                        match_length = prev_scan_length + i + 1
                scan_length = max(scan_length, i)
            new_dict_states[name] = (match_length, state)

        lexer_state.scan_pos += scan_length + 1
        lexer_state.dict_states = new_dict_states

    def advance_and_match(
        self, lexer_state: FSMLexerState
    ) -> Optional[Tuple[str, str]]:
        prev_match_length = max(
            [match_length for match_length, _ in lexer_state.dict_states.values()]
        )
        self.advance_lexer_state(lexer_state)
        res_match = ""
        res_name = None

        for name, (match_length, _) in lexer_state.dict_states.items():
            if match_length > len(res_match) and match_length > prev_match_length:
                res_match = lexer_state.text[
                    lexer_state.pos : lexer_state.pos + match_length
                ]
                res_name = name

        if res_name is None:
            return None
        else:
            return res_name, res_match


class FSMLexer:
    def __init__(
        self,
        fsm_dict: Dict[str, FSM],
        newline_types: set[str] = set(),
        ignore_types: set[str] = set(),
        always_accept: set[str] = set(),
        postlexer: PostLex = IdPostLex(),
    ):
        self.fsm_dict = fsm_dict
        self.scanner = FSMScanner(fsm_dict)

        self.newline_types = frozenset(newline_types) & fsm_dict.keys()
        self.ignore_types = frozenset(ignore_types) & fsm_dict.keys()
        self.always_accept = frozenset(always_accept) & fsm_dict.keys()

        # Cache initial states
        self._cached_initial_states = {
            name: (0, fsm.initial) for name, fsm in fsm_dict.items()
        }

        self.postlexer = postlexer

    @classmethod
    def from_conf(cls, conf: "LexerConf", check_collisions: List[set[str]] = []):
        terminals = list(conf.terminals)
        assert all(isinstance(t, TerminalDef) for t in terminals), terminals

        for t in terminals:
            assert t.pattern.min_width > 0, (
                "Lexer does not allow zero-width terminals. (%s: %s)"
                % (t.name, t.pattern)
            )

        ignore_types = set(conf.ignore)
        newline_types = set(
            t.name for t in terminals if _regexp_has_newline(t.pattern.to_regexp())
        )
        always_accept = set(conf.postlex.always_accept) if conf.postlex else set()

        terminals.sort(
            key=lambda x: (
                -x.priority,
                (x.pattern.type == "re"),
                -x.pattern.max_width,
                -len(x.pattern.value),
                x.name,
            )
        )

        terminals_regex = {t: t.pattern.to_regexp() for t in terminals}

        if not conf.skip_validation:
            comparator = interegular.Comparator.from_regexes(terminals_regex)

            for names in check_collisions:
                accepts = set(names) | ignore_types | always_accept
                terminals_regex_to_check = {
                    k: v for k, v in terminals_regex.items() if k.name in accepts
                }
                _check_regex_collisions(
                    terminals_regex_to_check, comparator, conf.strict
                )

        fsm_dict = {
            t.name: FSM.from_regex(regexp) for t, regexp in terminals_regex.items()
        }
        # self.g_regex_flags = conf.g_regex_flags
        # self.use_bytes = conf.use_bytes
        return cls(
            fsm_dict,
            newline_types=newline_types,
            ignore_types=ignore_types,
            always_accept=always_accept,
            postlexer=conf.postlex if conf.postlex is not None else IdPostLex(),
        )

    def initial_dict_states(
        self, fsm_names: Optional[Collection[str]] = None
    ) -> Dict[str, Tuple[int, FSMState]]:
        # keep the key order of self.fsm_dict for priorities
        if fsm_names is None:
            return self._cached_initial_states.copy()
        return {
            name: self._cached_initial_states[name]
            for name in self.fsm_dict
            if name in fsm_names
        }

    def initial_lexer_state(
        self,
        text: str,
        fsm_names: Optional[Collection[str]] = None,
        line_ctr: Optional[LineCounter] = None,
    ) -> FSMLexerState:
        initial_dict_states = self.initial_dict_states(fsm_names)
        return FSMLexerState(
            text=text,
            pos=0,
            scan_pos=0,
            dict_states=initial_dict_states,
            line_ctr=line_ctr or LineCounter("\n"),
        )

    def next_token(self, lexer_state: FSMLexerState) -> Token:
        res = self.scanner.advance_and_match(lexer_state)
        if res is None:
            allowed = set(lexer_state.dict_states.keys()) - self.ignore_types
            if not allowed:
                allowed = {"<END-OF-FILE>"}
            raise UnexpectedCharacters(
                lexer_state.text,
                lexer_state.scan_pos - 1,
                lexer_state.line_ctr.line,
                lexer_state.line_ctr.column,
                allowed=allowed,
            )

        name, value = res
        return Token(
            name,
            value,
            start_pos=lexer_state.line_ctr.char_pos,
            line=lexer_state.line_ctr.line,
            column=lexer_state.line_ctr.column,
            end_pos=lexer_state.line_ctr.char_pos + len(value),
        )

    def is_active_state(self, lexer_state: FSMLexerState) -> bool:
        active = any(
            [
                state is not None and self.fsm_dict[name].is_active_state(state)
                for name, (_, state) in lexer_state.dict_states.items()
            ]
        )
        return active

    def keep_active_states(self, lexer_state: FSMLexerState):
        active_states = {
            name: (match_length, state)
            for name, (match_length, state) in lexer_state.dict_states.items()
            if state is not None and self.fsm_dict[name].is_active_state(state)
        }
        lexer_state.dict_states = active_states

    def lex(self, text: str) -> Iterator[Token]:
        lexer_state = self.initial_lexer_state(text)

        while lexer_state.pos < len(lexer_state.text):
            tok = self.next_token(lexer_state)

            lexer_state.line_ctr.feed(
                tok.value, test_newline=(tok.type in self.newline_types)
            )
            lexer_state.dict_states = self.initial_dict_states()
            lexer_state.pos += len(tok.value)
            lexer_state.scan_pos = lexer_state.pos

            tok.end_line, tok.end_column, tok.end_pos = (
                lexer_state.line_ctr.line,
                lexer_state.line_ctr.column,
                lexer_state.line_ctr.char_pos,
            )

            if tok.type not in self.ignore_types:
                yield tok
