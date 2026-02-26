"""Tests for column normalization."""

import unittest

import polars as pl

from matrix_validator.normalize import (
    BIOLINK_EDGE_COLUMNS,
    BIOLINK_NODE_COLUMNS,
    normalize_column_names,
    normalize_dataframe,
    normalize_header,
    strip_type_annotation,
)


class TestStripTypeAnnotation(unittest.TestCase):
    """Test stripping of KGX type annotations from column names."""

    def test_strip_string_array(self):
        self.assertEqual(strip_type_annotation("all_names:string[]"), "all_names")

    def test_strip_int(self):
        self.assertEqual(strip_type_annotation("score:int"), "score")

    def test_strip_float_array(self):
        self.assertEqual(strip_type_annotation("values:float[]"), "values")

    def test_strip_double(self):
        self.assertEqual(strip_type_annotation("weight:double"), "weight")

    def test_strip_boolean(self):
        self.assertEqual(strip_type_annotation("flag:boolean"), "flag")

    def test_strip_long(self):
        self.assertEqual(strip_type_annotation("count:long"), "count")

    def test_no_annotation(self):
        self.assertEqual(strip_type_annotation("subject"), "subject")

    def test_colon_in_curie_not_stripped(self):
        """Ensure CURIE-style colons are not stripped."""
        self.assertEqual(strip_type_annotation(":LABEL"), ":LABEL")
        self.assertEqual(strip_type_annotation(":TYPE"), ":TYPE")
        self.assertEqual(strip_type_annotation(":START_ID"), ":START_ID")


class TestNormalizeColumnNames(unittest.TestCase):
    """Test column name normalization logic."""

    def test_biolink_node_columns_unchanged(self):
        """Biolink-standard node columns should not be renamed."""
        columns = ["id", "name", "category", "description", "iri", "publications"]
        rename_mapping, changelog = normalize_column_names(columns, "nodes")
        self.assertEqual(rename_mapping, {})
        self.assertEqual(changelog, [])

    def test_biolink_edge_columns_unchanged(self):
        """Biolink-standard edge columns should not be renamed."""
        columns = ["subject", "predicate", "object", "primary_knowledge_source", "knowledge_level", "agent_type"]
        rename_mapping, changelog = normalize_column_names(columns, "edges")
        self.assertEqual(rename_mapping, {})
        self.assertEqual(changelog, [])

    def test_type_annotation_stripped_from_biolink_column(self):
        """Type annotations on Biolink columns should be stripped but not prefixed."""
        columns = ["publications:string[]", "all_categories:string[]"]
        rename_mapping, changelog = normalize_column_names(columns, "nodes")
        self.assertEqual(rename_mapping["publications:string[]"], "publications")
        self.assertEqual(rename_mapping["all_categories:string[]"], "all_categories")
        self.assertEqual(len(changelog), 2)

    def test_internal_node_columns_prefixed(self):
        """Non-Biolink node columns should be prefixed with _."""
        columns = ["all_names", "equivalent_curies", ":LABEL"]
        rename_mapping, changelog = normalize_column_names(columns, "nodes")
        self.assertEqual(rename_mapping["all_names"], "_all_names")
        self.assertEqual(rename_mapping["equivalent_curies"], "_equivalent_curies")
        self.assertEqual(rename_mapping[":LABEL"], "_:LABEL")

    def test_internal_edge_columns_prefixed(self):
        """Non-Biolink edge columns should be prefixed with _."""
        columns = ["publications_info", "kg2_ids", ":TYPE", ":START_ID", ":END_ID"]
        rename_mapping, changelog = normalize_column_names(columns, "edges")
        self.assertEqual(rename_mapping["publications_info"], "_publications_info")
        self.assertEqual(rename_mapping["kg2_ids"], "_kg2_ids")
        self.assertEqual(rename_mapping[":TYPE"], "_:TYPE")
        self.assertEqual(rename_mapping[":START_ID"], "_:START_ID")
        self.assertEqual(rename_mapping[":END_ID"], "_:END_ID")

    def test_type_annotation_on_internal_column(self):
        """Type annotations should be stripped AND column prefixed."""
        columns = ["kg2_ids:string[]", "all_names:string[]"]
        rename_mapping, changelog = normalize_column_names(columns, "edges")
        self.assertEqual(rename_mapping["kg2_ids:string[]"], "_kg2_ids")

        rename_mapping, changelog = normalize_column_names(columns, "nodes")
        self.assertEqual(rename_mapping["all_names:string[]"], "_all_names")

    def test_already_prefixed_columns_unchanged(self):
        """Columns already prefixed with _ should not be changed."""
        columns = ["_kg2_ids", "_all_names", "_:LABEL"]
        rename_mapping, changelog = normalize_column_names(columns, "nodes")
        self.assertEqual(rename_mapping, {})
        self.assertEqual(changelog, [])

    def test_rtxkg2_nodes_header(self):
        """Test with actual RTX-KG2 node header columns."""
        columns = [
            "id",
            "name",
            "category",
            "all_names:string[]",
            "all_categories:string[]",
            "iri",
            "description",
            "equivalent_curies:string[]",
            "publications:string[]",
            ":LABEL",
        ]
        rename_mapping, changelog = normalize_column_names(columns, "nodes")

        # Biolink columns with annotations should be stripped
        self.assertEqual(rename_mapping.get("all_categories:string[]"), "all_categories")
        self.assertEqual(rename_mapping.get("publications:string[]"), "publications")

        # Internal columns should be prefixed
        self.assertEqual(rename_mapping.get("all_names:string[]"), "_all_names")
        self.assertEqual(rename_mapping.get("equivalent_curies:string[]"), "_equivalent_curies")
        self.assertEqual(rename_mapping.get(":LABEL"), "_:LABEL")

        # Biolink columns without annotations should not be changed
        self.assertNotIn("id", rename_mapping)
        self.assertNotIn("name", rename_mapping)
        self.assertNotIn("category", rename_mapping)
        self.assertNotIn("iri", rename_mapping)
        self.assertNotIn("description", rename_mapping)

    def test_rtxkg2_edges_header(self):
        """Test with actual RTX-KG2 edge header columns."""
        columns = [
            "subject",
            "object",
            "predicate",
            "primary_knowledge_source",
            "publications:string[]",
            "publications_info",
            "kg2_ids:string[]",
            "qualified_predicate",
            "qualified_object_aspect",
            "qualified_object_direction",
            "domain_range_exclusion",
            "knowledge_level",
            "agent_type",
            "id",
            ":TYPE",
            ":START_ID",
            ":END_ID",
        ]
        rename_mapping, changelog = normalize_column_names(columns, "edges")

        # Biolink column with annotation stripped
        self.assertEqual(rename_mapping.get("publications:string[]"), "publications")

        # Internal columns prefixed
        self.assertEqual(rename_mapping.get("publications_info"), "_publications_info")
        self.assertEqual(rename_mapping.get("kg2_ids:string[]"), "_kg2_ids")
        self.assertEqual(rename_mapping.get(":TYPE"), "_:TYPE")
        self.assertEqual(rename_mapping.get(":START_ID"), "_:START_ID")
        self.assertEqual(rename_mapping.get(":END_ID"), "_:END_ID")

        # Biolink columns unchanged
        for col in ["subject", "object", "predicate", "primary_knowledge_source", "knowledge_level", "agent_type"]:
            self.assertNotIn(col, rename_mapping)


class TestNormalizeDataFrame(unittest.TestCase):
    """Test DataFrame normalization."""

    def test_normalize_node_dataframe(self):
        """Test normalizing a polars DataFrame with node columns."""
        df = pl.DataFrame(
            {
                "id": ["CHEBI:123"],
                "category": ["biolink:ChemicalEntity"],
                "all_names:string[]": ["aspirin"],
                "publications:string[]": ["PMID:123"],
                ":LABEL": ["ChemicalEntity"],
            }
        )
        normalized_df, changelog = normalize_dataframe(df, "nodes")

        self.assertIn("id", normalized_df.columns)
        self.assertIn("category", normalized_df.columns)
        self.assertIn("_all_names", normalized_df.columns)
        self.assertIn("publications", normalized_df.columns)
        self.assertIn("_:LABEL", normalized_df.columns)

        # Original columns should be gone
        self.assertNotIn("all_names:string[]", normalized_df.columns)
        self.assertNotIn("publications:string[]", normalized_df.columns)
        self.assertNotIn(":LABEL", normalized_df.columns)

        # Data should be preserved
        self.assertEqual(normalized_df["_all_names"][0], "aspirin")
        self.assertEqual(normalized_df["publications"][0], "PMID:123")

    def test_normalize_edge_dataframe(self):
        """Test normalizing a polars DataFrame with edge columns."""
        df = pl.DataFrame(
            {
                "subject": ["node1"],
                "predicate": ["biolink:related_to"],
                "object": ["node2"],
                "kg2_ids:string[]": ["id1"],
                ":TYPE": ["biolink:related_to"],
            }
        )
        normalized_df, changelog = normalize_dataframe(df, "edges")

        self.assertIn("subject", normalized_df.columns)
        self.assertIn("_kg2_ids", normalized_df.columns)
        self.assertIn("_:TYPE", normalized_df.columns)

    def test_no_changes_needed(self):
        """Test DataFrame that needs no normalization."""
        df = pl.DataFrame({"id": ["node1"], "category": ["biolink:Gene"]})
        normalized_df, changelog = normalize_dataframe(df, "nodes")
        self.assertEqual(changelog, [])
        self.assertEqual(df.columns, normalized_df.columns)


class TestNormalizeHeader(unittest.TestCase):
    """Test header normalization for pure Python validator."""

    def test_normalize_node_header(self):
        """Test normalizing a node header list."""
        header = ["id", "name", "category", "all_names:string[]", ":LABEL"]
        normalized, rename_mapping, changelog = normalize_header(header, "nodes")
        self.assertEqual(normalized, ["id", "name", "category", "_all_names", "_:LABEL"])

    def test_normalize_edge_header(self):
        """Test normalizing an edge header list."""
        header = ["subject", "object", "predicate", "kg2_ids:string[]", ":TYPE"]
        normalized, rename_mapping, changelog = normalize_header(header, "edges")
        self.assertEqual(normalized, ["subject", "object", "predicate", "_kg2_ids", "_:TYPE"])

    def test_backwards_compat_already_prefixed(self):
        """Headers with already _-prefixed columns should pass through unchanged."""
        header = ["id", "category", "_all_names", "_:LABEL"]
        normalized, rename_mapping, changelog = normalize_header(header, "nodes")
        self.assertEqual(normalized, header)
        self.assertEqual(changelog, [])


class TestBiolinkColumnSets(unittest.TestCase):
    """Test that the Biolink column sets contain expected entries."""

    def test_required_node_columns_in_set(self):
        self.assertIn("id", BIOLINK_NODE_COLUMNS)
        self.assertIn("category", BIOLINK_NODE_COLUMNS)

    def test_required_edge_columns_in_set(self):
        self.assertIn("subject", BIOLINK_EDGE_COLUMNS)
        self.assertIn("predicate", BIOLINK_EDGE_COLUMNS)
        self.assertIn("object", BIOLINK_EDGE_COLUMNS)
        self.assertIn("primary_knowledge_source", BIOLINK_EDGE_COLUMNS)
        self.assertIn("knowledge_level", BIOLINK_EDGE_COLUMNS)
        self.assertIn("agent_type", BIOLINK_EDGE_COLUMNS)

    def test_internal_columns_not_in_biolink_sets(self):
        """Internal columns should NOT be in the Biolink column sets."""
        self.assertNotIn("all_names", BIOLINK_NODE_COLUMNS)
        self.assertNotIn("equivalent_curies", BIOLINK_NODE_COLUMNS)
        self.assertNotIn(":LABEL", BIOLINK_NODE_COLUMNS)
        self.assertNotIn("kg2_ids", BIOLINK_EDGE_COLUMNS)
        self.assertNotIn("publications_info", BIOLINK_EDGE_COLUMNS)
        self.assertNotIn(":TYPE", BIOLINK_EDGE_COLUMNS)
        self.assertNotIn(":START_ID", BIOLINK_EDGE_COLUMNS)
        self.assertNotIn(":END_ID", BIOLINK_EDGE_COLUMNS)


if __name__ == "__main__":
    unittest.main()
