import unittest
from unittest.mock import MagicMock
import sys
import os

# Add scripts to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))

from kag_poc import Graph, ExtractionContext, extract_gliner_entities, extract_rebel_relations, normalize_term

class TestSotaIntegration(unittest.TestCase):
    def setUp(self):
        self.graph = Graph()
        self.source_id = self.graph.add_node("File", "dummy.txt")
        self.context = ExtractionContext(
            nlp=None, ner_enabled=False, llm_enabled=False, csv_enabled=False, llm_calls_left=0
        )

    def test_gliner_extraction(self):
        # Mock GLiNER model
        mock_gliner = MagicMock()
        mock_gliner.predict_entities.return_value = [
            {"text": "Apple", "label": "Organization"},
            {"text": "California", "label": "Location"}
        ]
        self.context.gliner_model = mock_gliner

        entities = extract_gliner_entities(self.graph, "Apple is in California.", self.source_id, self.context)

        # Verify entities returned
        self.assertEqual(len(entities), 2)
        self.assertIn("Apple", entities)
        self.assertIn("California", entities)

        # Verify nodes in graph
        nodes = {n.label: n for n in self.graph.nodes}
        self.assertIn("Apple", nodes)
        self.assertIn("California", nodes)
        self.assertEqual(nodes["Apple"].attrs.get("ner_label"), "Organization")
        self.assertEqual(nodes["California"].attrs.get("ner_label"), "Location")
        self.assertEqual(nodes["Apple"].attrs.get("source"), "gliner")

    def test_rebel_extraction(self):
        # Mock REBEL model and tokenizer
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        # tokenizer(text, ...) returns a dict-like object
        mock_tokenizer.return_value = {"input_ids": "ids", "attention_mask": "mask"}

        # model.generate(...) returns generated tokens
        mock_model.generate.return_value = "generated_tokens"

        # tokenizer.batch_decode(...) returns list of strings
        # We simulate that special tokens are preserved
        mock_tokenizer.batch_decode.return_value = ["<triplet> Apple <subj> Cupertino <obj> located in"]

        self.context.rebel_model = mock_model
        self.context.rebel_tokenizer = mock_tokenizer

        extract_rebel_relations(self.graph, "Apple is located in Cupertino.", self.source_id, self.context)

        # Verify edges
        # We expect Apple -> located_in -> Cupertino
        # And mentions edges from file to Apple and Cupertino

        nodes = {n.label: n for n in self.graph.nodes}
        self.assertIn("Apple", nodes)
        self.assertIn("Cupertino", nodes)

        apple_id = nodes["Apple"].id
        cupertino_id = nodes["Cupertino"].id

        # Find edge
        edge_found = False
        for edge in self.graph.edges:
            if edge.source == apple_id and edge.target == cupertino_id and edge.type == "located_in":
                # Check provenance
                if edge.attrs.get("provenance") == "rebel":
                    edge_found = True
                    break
        self.assertTrue(edge_found, "Relation Apple -> located_in -> Cupertino not found with correct provenance")

if __name__ == "__main__":
    unittest.main()
