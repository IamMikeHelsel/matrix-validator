"""Column normalization for KGX data files.

Normalizes column names by:
1. Stripping KGX type annotations (e.g., :string[], :int)
2. Prefixing non-Biolink internal columns with _
"""

import logging
import re

import polars as pl

logger = logging.getLogger("matrix-validator.normalize")

# Type annotation pattern: matches :string[], :int, :float[], :double, :boolean, :long, etc.
TYPE_ANNOTATION_REGEX = re.compile(r":(?:string|int|float|double|boolean|long)(?:\[\])?$")

# Biolink-standard node columns — these will NOT be prefixed with _
BIOLINK_NODE_COLUMNS = frozenset(
    {
        "id",
        "name",
        "category",
        "description",
        "iri",
        "international_resource_identifier",
        "publications",
        "equivalent_identifiers",
        "all_categories",
        "labels",
    }
)

# Biolink-standard edge columns — these will NOT be prefixed with _
BIOLINK_EDGE_COLUMNS = frozenset(
    {
        "subject",
        "predicate",
        "object",
        "primary_knowledge_source",
        "aggregator_knowledge_source",
        "knowledge_level",
        "agent_type",
        "qualified_predicate",
        "qualified_object_aspect",
        "qualified_object_direction",
        "domain_range_exclusion",
        "publications",
        "id",
        "original_subject",
        "original_object",
        "subject_aspect_qualifier",
        "subject_direction_qualifier",
        "object_aspect_qualifier",
        "object_direction_qualifier",
        "upstream_data_source",
    }
)


def strip_type_annotation(column_name: str) -> str:
    """Strip KGX type annotations (e.g., :string[]) from a column name."""
    return TYPE_ANNOTATION_REGEX.sub("", column_name)


def normalize_column_names(columns: list[str], file_type: str) -> tuple[dict[str, str], list[str]]:
    """Compute a rename mapping for column normalization.

    Steps:
    1. Strip type annotations (:string[], :int, etc.)
    2. Prefix non-Biolink columns with _
    3. Leave already _-prefixed columns as-is

    Args:
        columns: List of raw column names from the TSV header.
        file_type: Either "nodes" or "edges".

    Returns:
        Tuple of (rename_mapping, changelog):
        - rename_mapping: dict mapping original column name -> normalized name (only for changed columns)
        - changelog: list of human-readable change descriptions

    """
    biolink_columns = BIOLINK_NODE_COLUMNS if file_type == "nodes" else BIOLINK_EDGE_COLUMNS

    rename_mapping = {}
    changelog = []

    for col in columns:
        # Already prefixed — leave as-is
        if col.startswith("_"):
            continue

        # Strip type annotation
        base_name = strip_type_annotation(col)

        if base_name in biolink_columns:
            # Biolink-standard column: only strip type annotation if present
            if base_name != col:
                rename_mapping[col] = base_name
                changelog.append(f"stripped type annotation: '{col}' -> '{base_name}'")
        else:
            # Non-Biolink column: prefix with _
            normalized = f"_{base_name}"
            rename_mapping[col] = normalized
            changelog.append(f"prefixed internal column: '{col}' -> '{normalized}'")

    if changelog:
        logger.info(f"Column normalization ({file_type}): {len(changelog)} changes")
        for change in changelog:
            logger.debug(f"  {change}")

    return rename_mapping, changelog


def normalize_dataframe(df: pl.DataFrame, file_type: str) -> tuple[pl.DataFrame, list[str]]:
    """Normalize column names in a Polars DataFrame.

    Args:
        df: The DataFrame to normalize.
        file_type: Either "nodes" or "edges".

    Returns:
        Tuple of (normalized_df, changelog).

    """
    rename_mapping, changelog = normalize_column_names(df.columns, file_type)
    if rename_mapping:
        df = df.rename(rename_mapping)
    return df, changelog


def normalize_header(header: list[str], file_type: str) -> tuple[list[str], dict[str, str], list[str]]:
    """Normalize a list of column names (for pure Python validator).

    Args:
        header: List of raw column names.
        file_type: Either "nodes" or "edges".

    Returns:
        Tuple of (normalized_header, rename_mapping, changelog).

    """
    rename_mapping, changelog = normalize_column_names(header, file_type)
    normalized = [rename_mapping.get(col, col) for col in header]
    return normalized, rename_mapping, changelog
