from .pushdown_automata import PDAState, PushDownAutomata, Shift
from ..lexer import FSMLexer, FSMLexerState, FSMState, FSM, Token, Lexer
from .lalr_analysis import StateT
from ..exceptions import UnexpectedCharacters, UnexpectedToken
from ..tree import ParseTree

from typing import Generic, List, Tuple, Any, Optional, Dict, Iterable
from copy import copy, deepcopy


def _subdict(dict: Dict, keys: Iterable) -> Dict:
    return {k: dict[k] for k in dict.keys() if k in keys}


class Parser(Generic[StateT]):
    def __init__(self, PDA: PushDownAutomata[StateT], lexer: Lexer):
        self.PDA = PDA
        self.lexer = lexer

    def parse(self, text: str, start: Optional[str] = None) -> Tuple[ParseTree, Any]:
        state = self.PDA.get_initial_state(start=start)

        for token in self.lexer.postlexer.process(self.lexer.lex(text)):
            state = self.PDA.get_next_state(state, token)

        end_state = self.PDA.feed_eos(state)

        return end_state.value_stack[-1], end_state.attribute_stack[-1]


class InteractiveParserState(Generic[StateT]):
    lookahead_PDA_state: Dict[str, PDAState[StateT]]
    lexer_state: FSMLexerState
    lookahead_shifts: Dict[str, Shift]
    _lookahead_ctx_fsm: Dict[str, Optional[FSM]]

    def __init__(
        self,
        lookahead_PDA_state: Dict[str, PDAState[StateT]],
        lexer_state: FSMLexerState,
        lookahead_shifts: Dict[str, Shift],
        _lookahead_ctx_fsm: Dict[str, Optional[FSM]] = {},
    ):
        self.lookahead_PDA_state = lookahead_PDA_state
        self.lexer_state = lexer_state
        self.lookahead_shifts = lookahead_shifts
        self._lookahead_ctx_fsm = _lookahead_ctx_fsm

    def __getitem__(
        self, key
    ) -> Tuple[PDAState[StateT], Optional[Tuple[int, FSMState]], Optional[Shift]]:
        return (
            self.lookahead_PDA_state[key],
            self.lexer_state.dict_states.get(key, None),
            self.lookahead_shifts.get(key, None),
        )

    @property
    def lookahead_ctx_fsm(self):
        class LazyDictCtxFSM:
            def __init__(self, parent: InteractiveParserState):
                self._parent = parent

            def __getitem__(self, key) -> Optional[FSM]:
                if key not in self._parent.lookahead_shifts:
                    raise KeyError(key)

                if key not in self._parent._lookahead_ctx_fsm:
                    shift = self._parent.lookahead_shifts[key]
                    self._parent._lookahead_ctx_fsm[key] = (
                        FSM.from_regex(shift.pattern)
                        if shift and shift.pattern
                        else None
                    )

                return self._parent._lookahead_ctx_fsm[key]

            def __contains__(self, key):
                return key in self._parent.lookahead_shifts

            def keys(self):
                return set(self._parent.lookahead_shifts.keys())

        return LazyDictCtxFSM(self)

    def __copy__(self) -> "InteractiveParserState[StateT]":
        return self.__class__(
            deepcopy(self.lookahead_PDA_state),
            copy(self.lexer_state),
            copy(self.lookahead_shifts),
            copy(self._lookahead_ctx_fsm),
        )

    def __repr__(self) -> str:
        return (
            f"{{\n\tLexer state: {self.lexer_state};\n"
            f"\tLookahead PDA states: {self.lookahead_PDA_state};\n"
            f"\tLookahead shifts: {self.lookahead_shifts} \n}}"
        )


class InteractiveParser(Generic[StateT]):
    def __init__(self, PDA: PushDownAutomata[StateT], lexer: FSMLexer):
        self.PDA = PDA
        self.lexer = lexer

    def initial_interactive_state(
        self, text: str, start: Optional[str] = "start"
    ) -> InteractiveParserState[StateT]:
        PDA_state = self.PDA.get_initial_state(start)
        lookahead_PDA_state, lookahead_shifts = self.PDA.get_lookahead_states(PDA_state)
        lexer_state = self.lexer.initial_lexer_state(text)
        return self.filter_interactive_state(
            InteractiveParserState(lookahead_PDA_state, lexer_state, lookahead_shifts)
        )

    def parse_interactive(
        self, text: str, start: Optional[str] = "start"
    ) -> List[InteractiveParserState[StateT]]:
        state = self.initial_interactive_state(text, start=start)
        state = self.advance_interactive_state(state)
        return state

    def resume_parser(
        self, state_list: List[InteractiveParserState[StateT]], text: str
    ) -> List[InteractiveParserState[StateT]]:
        err = ValueError("Empty list of interactive parser states")
        for state in state_list:
            state.lexer_state.text += text
            try:
                return self.advance_interactive_state(state)
            except UnexpectedCharacters as e:
                err = e
                continue

        raise err

    def feed_eos(
        self, state_list: List[InteractiveParserState[StateT]]
    ) -> PDAState[StateT]:
        most_advanced_state = state_list[-1]

        pos = most_advanced_state.lexer_state.pos
        text = most_advanced_state.lexer_state.text
        if pos != len(text) or "$END" not in most_advanced_state.lookahead_PDA_state:
            raise UnexpectedToken(Token("$END", "", 0, 1, 1), [])

        end_state = self.PDA.feed_eos(most_advanced_state.lookahead_PDA_state["$END"])
        return end_state

    def filter_interactive_state(
        self, state: InteractiveParserState[StateT]
    ) -> InteractiveParserState[StateT]:
        """
        Filter interactive parser state based on lexer state and lookahead states.
        Parameters
        ----------
        state : InteractiveParserState[StateT]
            Current interactive parser state containing lookahead PDA state,
            shifts and lexer state.
        Returns
        -------
        InteractiveParserState[StateT]
            Filtered parser state containing only valid active states based on
            current scan prefix matching.
        Notes
        -----
        Filters lexer and lookahead states by keeping only those states whose context FSM
        matches the current scan as a prefix when it exists. Special $END state is preserved
        if present.
        """

        lookahead_PDA_state = state.lookahead_PDA_state
        lookahead_shifts = state.lookahead_shifts
        lexer_state = state.lexer_state
        self.lexer.keep_active_states(lexer_state)
        current_scan = lexer_state.current_scan

        dict_keys = (
            set(lexer_state.dict_states)
            & set(lookahead_shifts)
            & set(lookahead_PDA_state)
        )

        keys = []
        for name in dict_keys:
            ctx_fsm = state.lookahead_ctx_fsm[name]
            if ctx_fsm is not None:
                if ctx_fsm.is_prefix(current_scan):
                    keys.append(name)
            else:
                keys.append(name)

        is_end = ["$END"] if "$END" in lookahead_PDA_state.keys() else []
        active_lookahead_states = _subdict(lookahead_PDA_state, keys + is_end)
        lexer_state.dict_states = _subdict(
            lexer_state.dict_states,
            set(keys) | self.lexer.always_accept | self.lexer.ignore_types,
        )
        lookahead_shifts = _subdict(lookahead_shifts, keys)
        _lookahead_ctx_fsm = _subdict(state._lookahead_ctx_fsm, lookahead_shifts.keys())

        return InteractiveParserState(
            active_lookahead_states, lexer_state, lookahead_shifts, _lookahead_ctx_fsm
        )

    def advance_interactive_state(
        self, state: InteractiveParserState[StateT]
    ) -> List[InteractiveParserState[StateT]]:
        """Advances the interactive parser state by processing tokens and updating parser/lexer states.

        This method processes input text through the lexer and updates the parser state accordingly,
        handling token generation, state transitions, and lookahead calculations.

        Args:
            state (InteractiveParserState[StateT]): Current state of the interactive parser,
                containing lexer state and lookahead PDA state information.

        Returns:
            List[InteractiveParserState[StateT]]: List of possible next parser states after
                processing available tokens. Each state represents a valid parsing path.

        Raises:
            UnexpectedCharacters: If the lexer encounters invalid input that cannot be tokenized,
                except when at the end of input in an active lexer state.

        Notes:
            - Maintains both lexer and parser state while processing tokens
            - Handles ignored tokens and special end markers
            - Updates line/column position tracking
            - Manages state transitions in the underlying PDA (Push-Down Automaton)
            - Generates lookahead states for interactive parsing
        """

        lexer_state = state.lexer_state

        def token_iterator():
            while lexer_state.pos < len(lexer_state.text):
                try:
                    tok = self.lexer.next_token(lexer_state)
                except UnexpectedCharacters as e:
                    if (
                        lexer_state.scan_pos == len(lexer_state.text)
                    ) and self.lexer.is_active_state(lexer_state):
                        return
                    else:
                        raise e

                yield tok

        lookahead_PDA_state = state.lookahead_PDA_state
        active_interactive_states = []
        PDA_state = None

        for token in self.lexer.postlexer.process(token_iterator()):
            self.lexer.keep_active_states(lexer_state)
            if len(lexer_state.dict_states) > 0:
                if PDA_state is None:
                    lookahead_state = copy(
                        InteractiveParserState(
                            lookahead_PDA_state, lexer_state, state.lookahead_shifts
                        )
                    )
                else:
                    lookahead_PDA_state, lookahead_shifts = (
                        self.PDA.get_lookahead_states(copy(PDA_state))
                    )
                    lookahead_state = InteractiveParserState(
                        lookahead_PDA_state, copy(lexer_state), lookahead_shifts
                    )
                if (
                    self.lexer.ignore_types & set(lexer_state.dict_states.keys())
                    == set()
                ):
                    lookahead_state = self.filter_interactive_state(lookahead_state)

                active_interactive_states.append(lookahead_state)

            lexer_state.line_ctr.feed(
                token.value, test_newline=(token.type in self.lexer.newline_types)
            )
            lexer_state.pos += len(token.value)
            lexer_state.scan_pos = lexer_state.pos

            token.end_line, token.end_column, token.end_pos = (
                lexer_state.line_ctr.line,
                lexer_state.line_ctr.column,
                lexer_state.line_ctr.char_pos,
            )

            if token.type not in self.lexer.ignore_types:
                if PDA_state:
                    PDA_state = self.PDA.get_next_state(PDA_state, token)
                else:
                    PDA_state = lookahead_PDA_state[token.type]
                    shift = state.lookahead_shifts[token.type]
                    ctx_fsm = state.lookahead_ctx_fsm[token.type]
                    if ctx_fsm and not ctx_fsm.accepts(token.value):
                        raise UnexpectedToken(token, [])
                    self.PDA.shift_token(PDA_state, token, shift.state_id)

            lookahead_tokens = (
                self.PDA.get_lookahead_tokens(PDA_state)
                if PDA_state
                else list(lookahead_PDA_state.keys())
            )

            next_fsm_names = (
                set(lookahead_tokens)
                | self.lexer.ignore_types
                | self.lexer.always_accept
            )
            lexer_state.dict_states = self.lexer.initial_dict_states(next_fsm_names)

        if PDA_state is None:
            lookahead_state = copy(
                InteractiveParserState(
                    lookahead_PDA_state, lexer_state, state.lookahead_shifts
                )
            )
        else:
            lookahead_PDA_state, lookahead_shifts = self.PDA.get_lookahead_states(
                copy(PDA_state)
            )
            lookahead_state = InteractiveParserState(
                lookahead_PDA_state, copy(lexer_state), lookahead_shifts
            )

        lookahead_state = self.filter_interactive_state(lookahead_state)
        if len(lookahead_state.lookahead_PDA_state) > 0:
            active_interactive_states.append(lookahead_state)

        return active_interactive_states

    def parse(self, text: str, start: Optional[str] = None) -> Tuple[ParseTree, Any]:
        state = self.parse_interactive(text, start)
        end_state = self.feed_eos(state)

        return end_state.value_stack[-1], end_state.attribute_stack[-1]
