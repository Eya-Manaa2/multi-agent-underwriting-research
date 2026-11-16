"""Product-line registry for multi-line underwriting extension."""

from products.registry import (
    DEFAULT_PRODUCT_LINE,
    get_product_pack,
    list_product_lines,
    validate_product_line,
)

__all__ = [
    "DEFAULT_PRODUCT_LINE",
    "get_product_pack",
    "list_product_lines",
    "validate_product_line",
]
