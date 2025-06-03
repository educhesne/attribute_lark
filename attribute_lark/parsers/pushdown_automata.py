"""
Implements a pushdown automata for parsing with attribute evaluation.

This module provides classes and functions for implementing a LALR parser with attribute
evaluation capabilities.
"""

from copy import copy, deepcopy
from typing import Dict, Any, Generic, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import re
from ast import Expression as AstExpression, Module as AstModule, fix_missing_locations
from ..lexer import Token
from ..common import ParserCallbacks, ParserConf
from ..grammar import Rule

from .lalr_analysis import (
    ParseTableBase,
    StateT,
    Shift as ShiftTransition,
    LALR_Analyzer,
)
from attribute_lark.exceptions import UnexpectedToken


class ParseConf(Generic[StateT]):
    """A configuration class for parsers utilizing pushdown automata.

    Parameters
    ----------
    parse_table : ParseTableBase
        Table containing parsing rules and transitions
    callbacks : ParserCallbacks
        Callback functions for parser events
    start : List[str]
        List of starting symbols/tokens
    python_header : Optional[AstModule]
        Optional Python AST module header

    Attributes
    ----------
    parse_table : ParseTableBase
        The parse table containing grammar rules
    callbacks : ParserCallbacks
        Parser event callbacks
    start : List[str]
        Starting symbols/tokens
    start_states : Dict[str, StateT]
        Mapping of start symbols to starting states
    end_states : Dict[str, StateT]
        Mapping of start symbols to accepting states
    states : Dict[StateT, Dict[str, tuple]]
        State transition mapping
    python_header : Optional[AstModule]
        Python AST module header if present
    """

    __slots__ = (
        "parse_table",
        "callbacks",
        "start",
        "start_states",
        "end_states",
        "states",
        "python_header",
    )

    parse_table: ParseTableBase
    callbacks: ParserCallbacks
    start: List[str]
    python_header: Optional[AstModule]

    start_state: Dict[str, StateT]
    end_state: Dict[str, StateT]
    states: Dict[StateT, Dict[str, tuple]]

    def __init__(
        self,
        parse_table: ParseTableBase,
        callbacks: ParserCallbacks,
        start: List[str],
        python_header: Optional[AstModule],
    ):
        self.parse_table = parse_table

        self.start_states = self.parse_table.start_states
        self.end_states = self.parse_table.end_states
        self.states = self.parse_table.states

        self.callbacks = callbacks
        self.start = start
        self.python_header = python_header


class GlobalVariables:
    "Basic class to collect global variables for the attribute ast evaluation"

    pass


@dataclass
class PDAState(Generic[StateT]):
    """Push-down automata state representation.

    A class representing the state of a push-down automata (PDA), maintaining stacks for
    states, values and attributes.

    Parameters
    ----------
    initial_state : StateT
        Starting state of the automata
    final_state : StateT
        Accepting/final state of the automata
    state_stack : List[StateT]
        Stack of states tracking the PDA's state history
    value_stack : List[Any]
        Stack storing values during PDA execution
    attribute_stack : List[Any]
        Stack storing attributes during PDA execution
    python_header : AstModule, optional
        Python AST module header information
    global_vars : GlobalVariables
        Container for global variable state
    """

    initial_state: StateT
    final_state: StateT
    state_stack: List[StateT] = field(default_factory=list)
    value_stack: List[Any] = field(default_factory=list)
    attribute_stack: List[Any] = field(default_factory=list)
    python_header: Optional[AstModule] = field(default=None)
    global_vars: GlobalVariables = field(default_factory=GlobalVariables)

    def push(self, state_id: StateT, value: Any, attribute: Any):
        """Push state, value and attribute onto their respective stacks.

        Parameters
        ----------
        state_id : StateT
            State identifier to push onto state stack
        value : Any
            Value to push onto value stack
        attribute : Any
            Attribute to push onto attribute stack
        """

        self.state_stack.append(state_id)
        self.value_stack.append(value)
        self.attribute_stack.append(attribute)

    def pop(self, n: int = 1) -> Tuple[List[StateT], List[Any], List[Any]]:
        """
        Pop n items from each stack (states, values, attributes).

        Parameters
        ----------
        n : int, default=1
            Number of items to pop from each stack

        Returns
        -------
        tuple
            Contains three lists:
            - states: List[StateT] - The popped states
            - values: List[Any] - The popped values
            - attributes: List[Any] - The popped attributes
        """

        states = self.state_stack[-n:]
        del self.state_stack[-n:]

        values = self.value_stack[-n:]
        del self.value_stack[-n:]

        attributes = self.attribute_stack[-n:]
        del self.attribute_stack[-n:]

        return states, values, attributes

    @property
    def state_id(self):
        return self.state_stack[-1]

    def __copy__(self):
        return self.__class__(
            copy(self.initial_state),
            copy(self.final_state),
            copy(self.state_stack),
            copy(self.value_stack),
            copy(self.attribute_stack),
            deepcopy(self.python_header),
            deepcopy(self.global_vars),
        )


def eval_attribute(ast: Optional[AstExpression], PDA_state: PDAState[StateT]) -> Any:
    """
    Evaluates an AST expression in a controlled environment with access to specific variables.

    Parameters
    ----------
    ast : Optional[AstExpression]
        The abstract syntax tree to evaluate. If None, returns None.
    PDA_state : PDAState[StateT]
        State object containing attribute stack, global variables, and Python header.

    Returns
    -------
    Any
        Result of the AST evaluation, or None if ast is None.

    Notes
    -----
    The function executes the AST with access to only three external variables:
    - PDA_state.python_header: instance of AST Module
    - PDA_state.global_vars: instance of GlobalVariables
    - PDA_state.attribute_stack: current attribute stack
    """
    stack = PDA_state.attribute_stack
    GLOBAL = PDA_state.global_vars
    header = PDA_state.python_header

    globals_dict = dict()
    locals_dict = dict()
    if ast:
        if header:
            exec(
                compile(header, filename="<ast>", mode="exec"),
                globals_dict,
                locals_dict,
            )
            assert "GLOBAL" not in locals_dict and "stack" not in locals_dict, (
                "GLOBAL and stack are reserved variables"
            )
        # the expression are evaluated in a local context where "stack" and "GLOBAL" are defined
        locals_dict.update({"GLOBAL": GLOBAL, "stack": stack})
        return eval(
            compile(fix_missing_locations(ast), filename="<ast>", mode="eval"),
            globals_dict,
            locals_dict,
        )
    else:
        return None


@dataclass
class Shift(Generic[StateT]):
    state_id: StateT
    pattern: Optional[str] = None


@dataclass
class Reduce:
    rule: Rule


Action = Union[Shift, Reduce]


class PushDownAutomata(Generic[StateT]):
    """Implementation of a Push Down Automata for parsing context-sensitive languages.

    A PDA is a state machine with a stack that can recognize context-free languages.
    This implementation adds support for attribute evaluation during parsing.

    Parameters
    ----------
    parse_conf : ParseConf[StateT]
        Configuration object containing parsing tables and callbacks
    debug : bool, optional
        Enable debug output, by default False
    ctx_term : bool, optional
        Enable context-sensitive terminal matching, by default True

    Attributes
    ----------
    parse_conf : ParseConf[StateT]
        Parsing configuration
    callbacks : ParserCallbacks
        Callback functions for semantic actions
    transitions : Dict[StateT, Dict[str, tuple]]
        State transition table
    start_states : Dict[str, StateT]
        Map of start symbols to their initial states
    end_states : Dict[str, StateT]
        Map of start symbols to their final states
    python_header : Optional[AstModule]
        Python code to be included in attribute evaluation
    """

    parse_conf: ParseConf[StateT]
    debug: bool
    ctx_term: bool

    callbacks: ParserCallbacks
    transitions: Dict[StateT, Dict[str, tuple]]
    start_states: Dict[str, StateT]
    end_states: Dict[str, StateT]
    python_header: Optional[AstModule]

    def __init__(
        self, parse_conf: ParseConf[StateT], debug: bool = False, ctx_term: bool = True
    ):
        self.parse_conf = parse_conf
        self.debug = debug
        self.ctx_term = ctx_term
        self.callbacks = parse_conf.callbacks
        self.transitions = parse_conf.states
        self.start_states = parse_conf.start_states
        self.end_states = parse_conf.end_states
        self.python_header = parse_conf.python_header

    def shift_token(
        self, state: PDAState[StateT], token: Token, state_id: StateT
    ) -> PDAState[StateT]:
        callbacks = self.parse_conf.callbacks

        value = token if token.type not in callbacks else callbacks[token.type](token)
        state.push(
            state_id, value, token.value
        )  # the attribute of a token is its value
        return state

    def reduce_shift(self, state: PDAState[StateT], rule: Rule) -> PDAState[StateT]:
        size = len(rule.expansion)

        # the synthesized attribute of a non-terminal symbol is the evaluation of the expression
        # attached to the rule deriving it
        attribute = eval_attribute(rule.ast, state)

        if size > 0:
            _, value_list, _ = state.pop(size)
        else:
            value_list = []

        value = self.callbacks[rule](value_list) if self.callbacks else value_list

        _action = self.get_next_action(state, Token(rule.origin.name, ""))
        assert isinstance(_action, Shift)
        state.push(_action.state_id, value, attribute)

        return state

    def get_next_action(self, state: PDAState[StateT], token: Token) -> Action:
        state_id = state.state_id
        try:
            action, arg, maybe_ast_pattern = self.transitions[state_id][token.type]
        except KeyError:
            expected = {s for s in self.transitions[state_id].keys() if s.isupper()}
            raise UnexpectedToken(token, expected, state=self, interactive_parser=None)

        if action == ShiftTransition:
            pattern = (
                eval_attribute(maybe_ast_pattern, state) if self.ctx_term else None
            )
            return Shift(state_id=arg, pattern=pattern)
        else:
            return Reduce(rule=arg)

    def get_initial_state(self, start: Optional[str] = None) -> PDAState[StateT]:
        if start is None:
            assert len(self.start_states) == 1
            (start,) = self.start_states
        state = PDAState(
            initial_state=self.start_states[start],
            final_state=self.end_states[start],
            python_header=self.python_header,
        )
        state.push(self.start_states[start], None, None)
        return state

    def get_next_state(self, state: PDAState[StateT], token: Token) -> PDAState[StateT]:
        while True:
            action = self.get_next_action(state, token)

            if isinstance(action, Reduce):
                state = self.reduce_shift(state, action.rule)
            else:
                pattern = action.pattern
                if (
                    (pattern is None)
                    or (not self.ctx_term)
                    or re.match(pattern, token.value)
                ):
                    return self.shift_token(state, token, action.state_id)
                else:
                    raise UnexpectedToken(token, pattern)

    def feed_eos(self, state: PDAState[StateT]) -> PDAState[StateT]:
        eos_token = Token("$END", "", 0, 1, 1)
        while True:
            action = self.get_next_action(state, eos_token)
            assert isinstance(action, Reduce)
            state = self.reduce_shift(state, action.rule)
            if state.state_id == state.final_state:
                return state

    def get_lookahead_tokens(self, state: PDAState[StateT]) -> List[str]:
        return [
            name for name in self.transitions[state.state_id].keys() if name.isupper()
        ]

    def get_lookahead_states(
        self, state: PDAState[StateT]
    ) -> Tuple[Dict[str, PDAState[StateT]], Dict[str, Shift]]:
        next_token_types = self.get_lookahead_tokens(state)
        lookahead_states = {}
        lookahead_shifts = {}
        for name in next_token_types:
            if name == "$END":
                continue
            tok = Token(name, "", 0, 1, 1)
            action = self.get_next_action(state, tok)
            if isinstance(action, Shift):
                lookahead_states[name] = state
                lookahead_shifts[name] = action
            else:
                reduce_state = copy(state)
                try:
                    while isinstance(action, Reduce):
                        reduce_state = self.reduce_shift(reduce_state, action.rule)
                        action = self.get_next_action(reduce_state, tok)
                except UnexpectedToken:
                    continue

                lookahead_states[name] = reduce_state
                lookahead_shifts[name] = action
        if "$END" in next_token_types:
            lookahead_states["$END"] = state
        return lookahead_states, lookahead_shifts

    def copy(self) -> "PushDownAutomata[StateT]":
        raise NotImplementedError

    @classmethod
    def from_parser_conf(
        cls, parser_conf: ParserConf, debug: bool = False, strict: bool = False
    ):
        analysis = LALR_Analyzer(parser_conf, debug=debug, strict=strict)
        analysis.compute_lalr()
        callbacks = parser_conf.callbacks

        parse_conf = ParseConf(
            analysis.parse_table,
            callbacks,
            parser_conf.start,
            parser_conf.python_header,
        )

        return cls(parse_conf, debug)
