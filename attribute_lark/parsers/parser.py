from .pushdown_automata import PDAState, PushDownAutomata
from ..lexer import FSMLexer, FSMLexerState, Token
from .lalr_analysis import StateT
from ..exceptions import UnexpectedCharacters
from ..tree import ParseTree

from typing import Generic, List, Tuple, Any, Optional
from dataclasses import dataclass
from copy import copy


class Parser(Generic[StateT]):
    def __init__(self, PDA: PushDownAutomata[StateT], lexer: FSMLexer):
        self.PDA = PDA
        self.lexer = lexer

    def parse(self, text: str, start: Optional[str] = None) -> Tuple[ParseTree, Any]:
        state = self.PDA.get_initial_state(start=start)

        for token in self.lexer.postlexer.process(self.lexer.lex(text)):
            state = self.PDA.get_next_state(state, token)

        end_state = self.PDA.feed_eos(state)

        return end_state.value_stack[-1], end_state.attribute_stack[-1]


@dataclass
class InteractiveParserState(Generic[StateT]):
    PDA_state: PDAState[StateT]
    lexer_state: FSMLexerState

    def __copy__(self) -> "InteractiveParserState[StateT]":
        return self.__class__(copy(self.PDA_state), copy(self.lexer_state))


class InteractiveParser(Generic[StateT]):
    def __init__(self, PDA: PushDownAutomata[StateT], lexer: FSMLexer):
        self.PDA = PDA
        self.lexer = lexer

    def initial_interactive_state(
        self, text: str, start: str = "start"
    ) -> InteractiveParserState[StateT]:
        PDA_state = self.PDA.get_initial_state(start)
        initial_lookahead = self.PDA.get_lookahead_tokens(PDA_state)
        initial_fsm_names = (
            set(initial_lookahead) | self.lexer.ignore_types | self.lexer.always_accept
        )
        lexer_state = self.lexer.initial_lexer_state(text, initial_fsm_names)
        return InteractiveParserState(PDA_state, lexer_state)

    def parse_interactive(
        self, text: str, start: str = "start"
    ) -> List[InteractiveParserState[StateT]]:
        state = self.initial_interactive_state(text, start=start)
        state = self.advance_interactive_state(state)
        return state

    def resume_parser(
        self, state_list: List[InteractiveParserState[StateT]], text: str
    ) -> List[InteractiveParserState[StateT]]:
        err = ValueError("Empty list of interactive parser states")

        next_states = None
        for state in state_list:
            state.lexer_state.text += text
            try:
                next_states = self.advance_interactive_state(state)
                break
            except UnexpectedCharacters as e:
                err = e
                continue

        if next_states is None:
            raise err
        return next_states

    def feed_eos(
        self, state_list: List[InteractiveParserState[StateT]]
    ) -> InteractiveParserState[StateT]:
        most_advanced_state = state_list[-1]

        pos = most_advanced_state.lexer_state.pos
        text = most_advanced_state.lexer_state.text
        assert pos == len(text), f"lexer at position {pos}, expecting {len(text)}"

        end_state = self.PDA.feed_eos(most_advanced_state.PDA_state)
        return InteractiveParserState(end_state, most_advanced_state.lexer_state)

    def finish_parser(
        self, state_list: List[InteractiveParserState[StateT]]
    ) -> Tuple[ParseTree, Any]:
        end_state = self.feed_eos(state_list).PDA_state
        return end_state.value_stack[-1], end_state.attribute_stack[-1]

    def advance_interactive_state(
        self, state: InteractiveParserState[StateT]
    ) -> List[InteractiveParserState[StateT]]:
        PDA_state = state.PDA_state
        lexer_state = state.lexer_state

        def token_iterator():
            while lexer_state.pos < len(lexer_state.text):
                try:
                    tok = self.lexer.next_token(lexer_state)
                except UnexpectedCharacters as e:
                    if (lexer_state.scan_pos == len(lexer_state.text)) and self.lexer.is_active_state(lexer_state):
                        return
                    else:
                        raise e

                yield tok

        active_interactive_states = []
        for token in self.lexer.postlexer.process(token_iterator()):
            self.lexer.keep_active_states(lexer_state)
            if len(lexer_state.dict_states) > 0:
                active_interactive_states.append(
                    copy(InteractiveParserState(PDA_state, lexer_state))
                )

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
                PDA_state = self.PDA.get_next_state(PDA_state, token)
            next_tokens = self.PDA.get_lookahead_tokens(PDA_state)
            next_fsm_names = (
                set(next_tokens) | self.lexer.ignore_types | self.lexer.always_accept
            )

            lexer_state.dict_states = self.lexer.initial_dict_states(next_fsm_names)

        self.lexer.keep_active_states(lexer_state)
        active_interactive_states.append(InteractiveParserState(PDA_state, lexer_state))
        return active_interactive_states
