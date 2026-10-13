# alpha/__init__.py
# -*- coding: utf-8 -*-

"""
alpha：Alpha 表达式层。
"""

from alpha.ast import (
    Node,
    binary,
    clone_node,
    collect_nodes,
    field,
    node_count,
    node_depth,
    node_to_expr,
    replace_subtree,
    ts,
    unary,
)
from alpha.families import GRAMMAR_FAMILIES, detect_family
from alpha.fields import (
    FIELDS,
    FIELD_TYPES,
    PRICE_FIELDS,
    FieldCatalog,
    FieldDefinition,
    default_catalog as default_field_catalog,
)
from alpha.generator import (
    combine_nodes,
    generate_grammar_alpha,
    generate_initial_alpha,
    generate_smart_alpha,
)
from alpha.operators import (
    BINARY_OPERATORS,
    TS_OPERATORS,
    UNARY_OPERATORS,
    WINDOWS,
    OperatorCatalog,
    OperatorDefinition,
    catalog_from_api,
    default_catalog as default_operator_catalog,
    hardcoded_catalog as hardcoded_operator_catalog,
    load_from_api as load_operators_from_api,
    load_from_json as load_operators_from_json,
    set_default_catalog as set_default_operator_catalog,
)
from alpha.parser import parse_expression, split_top_level_binary
from alpha.types import FieldType, NodeKind
from alpha.validator import (
    infer_type,
    semantic_filter,
    structural_filter,
    validate_node,
)

__all__ = [
    # types
    "NodeKind", "FieldType",
    # fields
    "FieldCatalog", "FieldDefinition", "default_field_catalog",
    "FIELDS", "FIELD_TYPES", "PRICE_FIELDS",
    # operators
    "OperatorCatalog", "OperatorDefinition",
    "default_operator_catalog", "hardcoded_operator_catalog",
    "set_default_operator_catalog",
    "catalog_from_api",
    "load_operators_from_api", "load_operators_from_json",
    "UNARY_OPERATORS", "TS_OPERATORS", "BINARY_OPERATORS", "WINDOWS",
    # ast
    "Node", "clone_node", "node_depth", "node_count",
    "collect_nodes", "replace_subtree", "node_to_expr",
    "field", "unary", "ts", "binary",
    # parser
    "parse_expression", "split_top_level_binary",
    # validator
    "infer_type", "semantic_filter", "structural_filter", "validate_node",
    # families
    "GRAMMAR_FAMILIES", "detect_family",
    # generator
    "combine_nodes", "generate_grammar_alpha",
    "generate_smart_alpha", "generate_initial_alpha",
]