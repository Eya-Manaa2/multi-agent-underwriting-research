"""Tests for the consolidated underwriting report (bilan de souscription)."""

import os
import tempfile
import unittest
from unittest.mock import patch

from orchestrator.langgraph_orchestrator import LangGraphOrchestrator
from utils.document_bundle import build_document_bundle

VALID_APPLICATION = {
    'application_id': 'TEST-001',
    'driver_age': 35,
    'bonus_malus': 60.0,
    'vehicle_age': 5,
    'vehicle_power': 6,
    'vehicle_gas': 'Regular',
    'density': 1200.0,
    'area': 'C',
}


def _bundle(*declared_types):
    files = [
        {
            'filename': f'{code}.txt',
            'content': "Contrat d'assurance automobile. Vehicule assure.".encode('utf-8'),
            'declared_type': code,
        }
        for code in declared_types
    ]
    return build_document_bundle(files, product_line='auto')


class UnderwritingReportTests(unittest.TestCase):
    def test_report_aggregates_agents_and_reference_sources(self):
        result = LangGraphOrchestrator().run_workflow(
            VALID_APPLICATION,
            document_bundle=_bundle('releve_information', 'carte_grise', 'permis_conduire'),
        )
        report = result['report_result']

        self.assertEqual(report['recommendation']['human_review_required'], True)
        self.assertEqual(report['recommendation']['code'], result['decision_result']['final_decision'])
        self.assertTrue(report['summary_lines'])
        self.assertEqual(report['provenance']['agents'], [
            'intake', 'customer', 'risk', 'uncertainty', 'document',
            'external', 'claims', 'decision', 'fairness', 'compliance', 'explanation',
        ])
        self.assertEqual(report['provenance']['collaboration_mode'], 'sequential')
        self.assertEqual(report['uncertainty']['status'], 'evaluated')
        self.assertIn(report['uncertainty']['level'], {'low', 'moderate', 'high'})
        source_ids = {source['source_id'] for source in report['provenance']['sources']}
        self.assertEqual(source_ids, {'geo_risk', 'claims_history'})
        self.assertEqual(report['claims_history']['status'], 'ok')
        self.assertEqual(report['environment']['status'], 'ok')

    def test_report_surfaces_missing_mandatory_documents(self):
        result = LangGraphOrchestrator().run_workflow(
            VALID_APPLICATION,
            document_bundle=_bundle('releve_information'),
        )
        report = result['report_result']
        self.assertEqual(
            report['documents']['missing_mandatory'], ['carte_grise', 'permis_conduire']
        )
        self.assertFalse(report['documents']['complete'])
        self.assertTrue(
            any('obligatoires manquantes' in step for step in report['next_steps'])
        )

    def test_report_surfaces_history_indicators_for_flagged_case(self):
        # 2241683 is a real freMTPL2 policy with 16 prior claims in the reference history.
        application = {**VALID_APPLICATION, 'application_id': '2241683'}
        result = LangGraphOrchestrator().run_workflow(
            application, document_bundle=_bundle('releve_information', 'carte_grise', 'permis_conduire')
        )
        report = result['report_result']
        self.assertEqual(report['claims_history']['prior_claims_count'], 16)
        self.assertTrue(report['claims_history']['fraud_indicators'])
        self.assertEqual(report['claims_history']['source']['source_id'], 'claims_history')

    def test_report_fails_closed_when_reference_sources_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'AURELIA_REFERENCE_DATA_DIR': tmp}):
                result = LangGraphOrchestrator().run_workflow(
                    VALID_APPLICATION,
                    document_bundle=_bundle('releve_information', 'carte_grise', 'permis_conduire'),
                )
        report = result['report_result']
        self.assertEqual(report['environment']['status'], 'unavailable')
        self.assertEqual(report['claims_history']['status'], 'unavailable')
        self.assertIsNone(report['claims_history']['prior_claims_count'])
        self.assertTrue(
            any('indisponible' in line for line in report['summary_lines'])
        )


if __name__ == "__main__":
    unittest.main()
