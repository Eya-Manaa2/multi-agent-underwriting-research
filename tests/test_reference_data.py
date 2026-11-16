"""Tests for the reference-data connector and the agents wired to it."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.claims_agent import ClaimsAgent
from agents.external_agent import ExternalAgent
from connectors.agira import agira_mode, lookup_vehicle, normalize_registration
from connectors.reference_data import load_reference_dataset, reference_dir

SOURCE_IDS = ("geo_risk", "claims_history")


class ReferenceDataSetTests(unittest.TestCase):
    def test_bundled_datasets_load_with_provenance(self):
        for source_id in SOURCE_IDS:
            dataset = load_reference_dataset(source_id)
            self.assertTrue(dataset.available, dataset.reason)
            self.assertTrue(dataset.version)
            self.assertTrue(dataset.checksum)
            self.assertGreater(dataset.row_count, 0)
            descriptor = dataset.descriptor()
            self.assertEqual(descriptor["source_id"], source_id)
            self.assertTrue(descriptor["license"])

    def test_unknown_source_is_not_available(self):
        dataset = load_reference_dataset("does_not_exist")
        self.assertFalse(dataset.available)
        self.assertIn("inconnue", dataset.reason)

    def test_missing_dataset_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"AURELIA_REFERENCE_DATA_DIR": tmp}):
                self.assertEqual(reference_dir(), Path(tmp))
                dataset = load_reference_dataset("geo_risk")
        self.assertFalse(dataset.available)
        self.assertIn("absent", dataset.reason)

    def test_invalid_dataset_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "geographic_risk.json").write_text("{not json", encoding="utf-8")
            with patch.dict(os.environ, {"AURELIA_REFERENCE_DATA_DIR": tmp}):
                dataset = load_reference_dataset("geo_risk")
        self.assertFalse(dataset.available)
        self.assertIn("illisible", dataset.reason)


class ExternalAgentTests(unittest.TestCase):
    def test_uses_reference_record_for_area_risk(self):
        result = ExternalAgent().process({"area": "F"})
        self.assertEqual(result["source_status"], "ok")
        self.assertTrue(any("F" in factor for factor in result["external_factors"]))
        self.assertTrue(result["risk_adjustments"])
        self.assertTrue(result["data_sources"])
        self.assertEqual(result["source"]["source_id"], "geo_risk")

    def test_unknown_area_is_reported_without_inventing_a_factor(self):
        result = ExternalAgent().process({"area": "Z"})
        self.assertEqual(result["source_status"], "ok")
        self.assertEqual(result["risk_adjustments"], [])
        self.assertTrue(any("non référencée" in f for f in result["external_factors"]))

    def test_fails_closed_when_source_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"AURELIA_REFERENCE_DATA_DIR": tmp}):
                result = ExternalAgent().process({"area": "F"})
        self.assertEqual(result["source_status"], "unavailable")
        self.assertEqual(result["risk_adjustments"], [])
        self.assertEqual(result["data_sources"], [])


class AgiraConnectorTests(unittest.TestCase):
    def test_replay_mode_returns_the_bundled_fixture_record(self):
        with patch.dict(os.environ, {"AGIRA_MODE": "replay"}):
            lookup = lookup_vehicle("ab-123-cd")
        self.assertTrue(lookup.available)
        self.assertEqual(lookup.status, "insured")
        self.assertEqual(lookup.prior_claims_count, 2)
        self.assertFalse(lookup.real_data)
        descriptor = lookup.descriptor()
        self.assertEqual(descriptor["source_id"], "agira")
        self.assertFalse(descriptor["real_data"])
        self.assertTrue(descriptor["checksum"])
        self.assertTrue(descriptor["license"])

    def test_replay_reports_a_non_insured_vehicle_with_its_resiliation(self):
        with patch.dict(os.environ, {"AGIRA_MODE": "replay"}):
            lookup = lookup_vehicle("EF-456-GH")
        self.assertTrue(lookup.available)
        self.assertEqual(lookup.status, "not_insured")
        self.assertEqual(lookup.resiliation["reason"], "non_paiement")
        self.assertEqual(lookup.prior_claims_count, 0)

    def test_unknown_registration_is_unknown_without_inventing_history(self):
        with patch.dict(os.environ, {"AGIRA_MODE": "replay"}):
            lookup = lookup_vehicle("ZZ-999-ZZ")
        self.assertTrue(lookup.available)
        self.assertEqual(lookup.status, "unknown")
        self.assertEqual(lookup.claims_history, [])

    def test_missing_registration_fails_closed(self):
        with patch.dict(os.environ, {"AGIRA_MODE": "replay"}):
            lookup = lookup_vehicle("")
        self.assertFalse(lookup.available)
        self.assertIn("Immatriculation absente", lookup.reason)

    def test_disabled_mode_fails_closed(self):
        with patch.dict(os.environ, {"AGIRA_MODE": "disabled"}):
            self.assertEqual(agira_mode(), "disabled")
            lookup = lookup_vehicle("AB-123-CD")
        self.assertFalse(lookup.available)
        self.assertIn("désactivé", lookup.reason)

    def test_live_mode_without_credentials_fails_closed(self):
        with patch.dict(os.environ, {"AGIRA_MODE": "live"}, clear=False):
            os.environ.pop("AGIRA_API_URL", None)
            os.environ.pop("AGIRA_API_TOKEN", None)
            lookup = lookup_vehicle("AB-123-CD")
        self.assertFalse(lookup.available)
        self.assertEqual(lookup.mode, "live")
        self.assertIn("non configuré", lookup.reason)

    def test_unknown_mode_fails_closed(self):
        with patch.dict(os.environ, {"AGIRA_MODE": "guessing"}):
            lookup = lookup_vehicle("AB-123-CD")
        self.assertFalse(lookup.available)
        self.assertIn("inconnu", lookup.reason)

    def test_normalize_registration_ignores_separators_and_case(self):
        self.assertEqual(normalize_registration(" ab 123 cd "), "AB123CD")
        self.assertEqual(normalize_registration("AB-123-CD"), normalize_registration("ab123cd"))


class ExternalAgentAgiraTests(unittest.TestCase):
    def test_uses_the_registration_declared_in_the_application(self):
        with patch.dict(os.environ, {"AGIRA_MODE": "replay"}):
            result = ExternalAgent().process({"area": "F", "registration": "AB-123-CD"})
        self.assertTrue(result["agira"]["available"])
        self.assertFalse(result["agira"]["real_data"])
        self.assertTrue(any("AGIRA" in source for source in result["data_sources"]))
        self.assertTrue(any("assuré au fichier AGIRA" in f for f in result["external_factors"]))

    def test_reads_the_registration_from_the_uploaded_documents(self):
        with patch.dict(os.environ, {"AGIRA_MODE": "replay"}):
            result = ExternalAgent().process(
                {"area": "F"},
                {"combined_text": "Certificat d'immatriculation AB-123-CD véhicule"},
            )
        self.assertTrue(result["agira"]["available"])
        self.assertTrue(any("2 sinistre" in f for f in result["external_factors"]))

    def test_no_registration_leaves_agira_unavailable_without_a_factor(self):
        with patch.dict(os.environ, {"AGIRA_MODE": "replay"}):
            result = ExternalAgent().process({"area": "F"})
        self.assertFalse(result["agira"]["available"])
        self.assertFalse(any("AGIRA" in f for f in result["external_factors"]))


class ClaimsAgentTests(unittest.TestCase):
    def test_reads_prior_claims_from_reference_history(self):
        # 2241683 is a real freMTPL2 policy with 16 prior claims in the reference history.
        result = ClaimsAgent().process({"application_id": "2241683"})
        self.assertEqual(result["history_source_status"], "ok")
        self.assertEqual(result["claims_analysis"]["prior_claims_count"], 16)
        self.assertGreater(
            result["claims_analysis"]["prior_claims_total_amount"], 30000
        )
        self.assertGreaterEqual(len(result["fraud_indicators"]), 2)
        self.assertGreater(result["similarity_score"], 0.8)

    def test_unknown_application_reports_no_history_without_fabricating(self):
        result = ClaimsAgent().process({"application_id": "INCONNU-XYZ"})
        self.assertEqual(result["history_source_status"], "ok")
        self.assertEqual(result["claims_analysis"]["prior_claims_count"], 0)
        self.assertEqual(result["similarity_score"], 0.0)
        self.assertEqual(result["fraud_indicators"], [])

    def test_fails_closed_when_source_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"AURELIA_REFERENCE_DATA_DIR": tmp}):
                result = ClaimsAgent().process({"application_id": "2241683"})
        self.assertEqual(result["history_source_status"], "unavailable")
        self.assertIsNone(result["claims_analysis"]["prior_claims_count"])
        self.assertEqual(result["fraud_indicators"], [])

    def test_severity_is_derived_from_aggregate_prior_claims(self):
        result = ClaimsAgent().process({"application_id": "2241683"})
        self.assertEqual(result["severity_assessment"], "High severity")


if __name__ == "__main__":
    unittest.main()
