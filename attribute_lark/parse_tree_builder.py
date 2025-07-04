"""Provides functions for the automatic building and shaping of the parse-tree."""

from typing import List, Callable, Union, Optional

from .exceptions import GrammarError, ConfigurationError
from .lexer import Token
from .tree import Tree
from .visitors import Transformer_InPlace
from .visitors import _vargs_meta, _vargs_meta_inline

from functools import partial, wraps


class ExpandSingleChild:
    def __init__(self, node_builder):
        self.node_builder = node_builder

    def __call__(self, children):
        if len(children) == 1:
            return children[0]
        else:
            return self.node_builder(children)


class PropagatePositions:
    def __init__(self, node_builder, node_filter=None):
        self.node_builder = node_builder
        self.node_filter = node_filter

    def __call__(self, children):
        res = self.node_builder(children)

        if isinstance(res, Tree):
            # Calculate positions while the tree is streaming, according to the rule:
            # - nodes start at the start of their first child's container,
            #   and end at the end of their last child's container.
            # Containers are nodes that take up space in text, but have been inlined in the tree.

            res_meta = res.meta

            first_meta = self._pp_get_meta(children)
            if first_meta is not None:
                if not hasattr(res_meta, "line"):
                    # meta was already set, probably because the rule has been inlined (e.g. `?rule`)
                    res_meta.line = getattr(
                        first_meta, "container_line", first_meta.line
                    )
                    res_meta.column = getattr(
                        first_meta, "container_column", first_meta.column
                    )
                    res_meta.start_pos = getattr(
                        first_meta, "container_start_pos", first_meta.start_pos
                    )
                    res_meta.empty = False

                res_meta.container_line = getattr(
                    first_meta, "container_line", first_meta.line
                )
                res_meta.container_column = getattr(
                    first_meta, "container_column", first_meta.column
                )
                res_meta.container_start_pos = getattr(
                    first_meta, "container_start_pos", first_meta.start_pos
                )

            last_meta = self._pp_get_meta(reversed(children))
            if last_meta is not None:
                if not hasattr(res_meta, "end_line"):
                    res_meta.end_line = getattr(
                        last_meta, "container_end_line", last_meta.end_line
                    )
                    res_meta.end_column = getattr(
                        last_meta, "container_end_column", last_meta.end_column
                    )
                    res_meta.end_pos = getattr(
                        last_meta, "container_end_pos", last_meta.end_pos
                    )
                    res_meta.empty = False

                res_meta.container_end_line = getattr(
                    last_meta, "container_end_line", last_meta.end_line
                )
                res_meta.container_end_column = getattr(
                    last_meta, "container_end_column", last_meta.end_column
                )
                res_meta.container_end_pos = getattr(
                    last_meta, "container_end_pos", last_meta.end_pos
                )

        return res

    def _pp_get_meta(self, children):
        for c in children:
            if self.node_filter is not None and not self.node_filter(c):
                continue
            if isinstance(c, Tree):
                if not c.meta.empty:
                    return c.meta
            elif isinstance(c, Token):
                return c
            elif hasattr(c, "__lark_meta__"):
                return c.__lark_meta__()


def make_propagate_positions(option):
    if callable(option):
        return partial(PropagatePositions, node_filter=option)
    elif option is True:
        return PropagatePositions
    elif option is False:
        return None

    raise ConfigurationError("Invalid option for propagate_positions: %r" % option)


class ChildFilter:
    def __init__(self, to_include, append_none, node_builder):
        self.node_builder = node_builder
        self.to_include = to_include
        self.append_none = append_none

    def __call__(self, children):
        filtered = []

        for i, to_expand, add_none in self.to_include:
            if add_none:
                filtered += [None] * add_none
            if to_expand:
                filtered += children[i].children
            else:
                filtered.append(children[i])

        if self.append_none:
            filtered += [None] * self.append_none

        return self.node_builder(filtered)


class ChildFilterLALR(ChildFilter):
    """Optimized childfilter for LALR (assumes no duplication in parse tree, so it's safe to change it)"""

    def __call__(self, children):
        filtered = []
        for i, to_expand, add_none in self.to_include:
            if add_none:
                filtered += [None] * add_none
            if to_expand:
                if filtered:
                    filtered += children[i].children
                else:  # Optimize for left-recursion
                    filtered = children[i].children
            else:
                filtered.append(children[i])

        if self.append_none:
            filtered += [None] * self.append_none

        return self.node_builder(filtered)


class ChildFilterLALR_NoPlaceholders(ChildFilter):
    "Optimized childfilter for LALR (assumes no duplication in parse tree, so it's safe to change it)"

    def __init__(self, to_include, node_builder):
        self.node_builder = node_builder
        self.to_include = to_include

    def __call__(self, children):
        filtered = []
        for i, to_expand in self.to_include:
            if to_expand:
                if filtered:
                    filtered += children[i].children
                else:  # Optimize for left-recursion
                    filtered = children[i].children
            else:
                filtered.append(children[i])
        return self.node_builder(filtered)


def _should_expand(sym):
    return not sym.is_term and sym.name.startswith("_")


def maybe_create_child_filter(
    expansion, keep_all_tokens, _empty_indices: Optional[List[bool]]
):
    # Prepare empty_indices as: How many Nones to insert at each index?
    if _empty_indices:
        assert _empty_indices.count(False) == len(expansion)
        s = "".join(str(int(b)) for b in _empty_indices)
        empty_indices = [len(ones) for ones in s.split("0")]
        assert len(empty_indices) == len(expansion) + 1, (empty_indices, len(expansion))
    else:
        empty_indices = [0] * (len(expansion) + 1)

    to_include = []
    nones_to_add = 0
    for i, sym in enumerate(expansion):
        nones_to_add += empty_indices[i]
        if keep_all_tokens or not (sym.is_term and sym.filter_out):
            to_include.append((i, _should_expand(sym), nones_to_add))
            nones_to_add = 0

    nones_to_add += empty_indices[len(expansion)]

    if (
        _empty_indices
        or len(to_include) < len(expansion)
        or any(to_expand for i, to_expand, _ in to_include)
    ):
        if _empty_indices:
            return partial(ChildFilterLALR, to_include, nones_to_add)
        else:
            # LALR without placeholders
            return partial(
                ChildFilterLALR_NoPlaceholders, [(i, x) for i, x, _ in to_include]
            )


def inplace_transformer(func):
    @wraps(func)
    def f(children):
        # function name in a Transformer is a rule name.
        tree = Tree(func.__name__, children)
        return func(tree)

    return f


def apply_visit_wrapper(func, name, wrapper):
    if wrapper is _vargs_meta or wrapper is _vargs_meta_inline:
        raise NotImplementedError(
            "Meta args not supported for internal transformer; use YourTransformer().transform(parser.parse()) instead"
        )

    @wraps(func)
    def f(children):
        return wrapper(func, name, children, None)

    return f


class ParseTreeBuilder:
    def __init__(
        self,
        rules,
        tree_class,
        propagate_positions: Union[bool, Callable] = False,
        maybe_placeholders=False,
    ):
        self.tree_class = tree_class
        self.propagate_positions = propagate_positions
        self.maybe_placeholders = maybe_placeholders

        self.rule_builders = list(self._init_builders(rules))

    def _init_builders(self, rules):
        propagate_positions = make_propagate_positions(self.propagate_positions)

        for rule in rules:
            options = rule.options
            keep_all_tokens = options.keep_all_tokens
            expand_single_child = options.expand1

            wrapper_chain = list(
                filter(
                    None,
                    [
                        (expand_single_child and not rule.alias) and ExpandSingleChild,
                        maybe_create_child_filter(
                            rule.expansion,
                            keep_all_tokens,
                            options.empty_indices if self.maybe_placeholders else None,
                        ),
                        propagate_positions,
                    ],
                )
            )

            yield rule, wrapper_chain

    def create_callback(self, transformer=None):
        callbacks = {}

        default_handler = getattr(transformer, "__default__", None)
        if default_handler:

            def default_callback(data, children):
                return default_handler(data, children, None)
        else:
            default_callback = self.tree_class

        for rule, wrapper_chain in self.rule_builders:
            user_callback_name = (
                rule.alias or rule.options.template_source or rule.origin.name
            )
            try:
                f = getattr(transformer, user_callback_name)
                wrapper = getattr(f, "visit_wrapper", None)
                if wrapper is not None:
                    f = apply_visit_wrapper(f, user_callback_name, wrapper)
                elif isinstance(transformer, Transformer_InPlace):
                    f = inplace_transformer(f)
            except AttributeError:
                f = partial(default_callback, user_callback_name)

            for w in wrapper_chain:
                f = w(f)

            if rule in callbacks:
                raise GrammarError("Rule '%s' already exists" % (rule,))

            callbacks[rule] = f

        return callbacks
