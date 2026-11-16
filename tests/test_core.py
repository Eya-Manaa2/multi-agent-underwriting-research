import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from agents.decision_agent_enhanced import EnhancedDecisionAgent
from agents.document_agent import DocumentAgent
from agents.fairness_agent import FairnessAgent
from agents.intake import IntakeAgent as LegacyIntakeAgent
from agents.intake_agent import IntakeAgent
from agents.llm_document_agent import LLMDocumentAgent
from agents.risk_agent import RiskAgent
from agents.uncertainty_agent import UncertaintyAgent, binary_entropy
from experiments.baselines.evaluate_holdout import evaluate_holdout
from experiments.fairness.fairness_metrics import evaluate_fairness
from experiments.uncertainty.uncertainty_metrics import evaluate_uncertainty
from orchestrator.langgraph_orchestrator import LangGraphOrchestrator
from utils.document_bundle import build_document_bundle
from utils.document_upload import PdfExtractionUnavailable, extract_uploaded_document


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

RAW_FREMTPL2_FREQ = (
    Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'freMTPL2freq.arff'
)


class IntakeAgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = IntakeAgent()

    def test_accepts_complete_valid_application(self):
        result = self.agent.process(VALID_APPLICATION)
        self.assertEqual(result['status'], 'validated')
        self.assertEqual(result['errors'], [])

    def test_rejects_missing_values_without_imputation(self):
        result = self.agent.process({'driver_age': 35})
        self.assertEqual(result['status'], 'needs_review')
        self.assertIn('bonus_malus', result['missing_fields'])
        self.assertNotIn('bonus_malus', result['prepared_data'])

    def test_rejects_non_finite_and_wrong_type_values(self):
        application = {**VALID_APPLICATION, 'density': float('nan'), 'vehicle_gas': 0}
        result = self.agent.process(application)
        self.assertEqual(result['status'], 'needs_review')
        self.assertTrue(any('density' in error for error in result['errors']))
        self.assertTrue(any('vehicle_gas' in error for error in result['errors']))

    def test_rejects_out_of_range_values(self):
        result = self.agent.process({**VALID_APPLICATION, 'bonus_malus': 10.0})
        self.assertEqual(result['status'], 'needs_review')
        self.assertTrue(any('bonus_malus' in error for error in result['errors']))

    def test_legacy_validator_does_not_replace_invalid_values(self):
        result = LegacyIntakeAgent().validate({**VALID_APPLICATION, 'driver_age': 150})
        self.assertFalse(result['is_valid'])
        self.assertEqual(result['normalized_data']['driver_age'], 150)


class DocumentUploadTests(unittest.TestCase):
    def test_extracts_text_from_fictional_pdf_test_documents(self):
        test_documents = Path(__file__).resolve().parents[1] / 'examples' / 'test_documents'
        text = extract_uploaded_document(
            'demo_auto_policy.pdf',
            (test_documents / 'demo_auto_policy.pdf').read_bytes(),
        )
        self.assertIn('DOCUMENT FICTIF', text)
        self.assertIn('Reference fictive', text)
        document_result = DocumentAgent().process(
            VALID_APPLICATION,
            {},
            document_text=text,
            document_name='demo_auto_policy.pdf',
            document_provided=True,
        )
        self.assertEqual(
            document_result['uploaded_document']['scope_status'],
            'supported',
        )

    def test_extracts_utf8_text_for_document_agent_review(self):
        text = extract_uploaded_document(
            'conditions.txt',
            'Conditions d’assurance automobile. Exclusion de garantie après fraude. '
            'Déclaration sous 30 jours.'.encode('utf-8'),
        )
        result = DocumentAgent().process(
            VALID_APPLICATION,
            {},
            document_text=text,
            document_name='conditions.txt',
        )
        self.assertTrue(result['uploaded_document']['provided'])
        self.assertFalse(result['uploaded_document']['affects_decision'])
        self.assertIn(
            'Exclusion de garantie',
            result['uploaded_document']['possible_terms_for_human_review'],
        )
        self.assertIn(
            'Fraude ou déclaration frauduleuse',
            result['uploaded_document']['possible_terms_for_human_review'],
        )

    def test_health_insurance_document_is_out_of_scope(self):
        result = DocumentAgent().process(
            VALID_APPLICATION,
            {},
            document_text=(
                'Demande de remboursement de soins de santé et frais '
                'd’hospitalisation auprès de la mutuelle.'
            ),
            document_name='demande_assurance_sante_exemple.pdf',
            document_provided=True,
            document_status='text_extracted',
        )
        document = result['uploaded_document']
        self.assertEqual(document['scope_status'], 'unsupported')
        self.assertTrue(document['affects_decision'])
        self.assertFalse(document['affects_risk_score'])
        self.assertEqual(document['possible_terms_for_human_review'], [])
        self.assertIn('assurance santé', document['scope_reason'])

    def test_attachment_without_pdf_reader_is_not_reported_as_read(self):
        with patch.dict('sys.modules', {'pypdf': None}):
            with self.assertRaises(PdfExtractionUnavailable):
                extract_uploaded_document('policy.pdf', b'%PDF-1.4\nnot parsed')

        result = DocumentAgent().process(
            VALID_APPLICATION,
            {},
            document_name='policy.pdf',
            document_provided=True,
            document_size_bytes=19,
            document_status='attached_reader_unavailable',
        )
        self.assertTrue(result['uploaded_document']['provided'])
        self.assertEqual(
            result['uploaded_document']['extraction_status'],
            'attached_reader_unavailable',
        )
        self.assertEqual(result['uploaded_document']['scope_status'], 'unverified')
        self.assertTrue(result['uploaded_document']['affects_decision'])
        self.assertFalse(result['uploaded_document']['affects_risk_score'])

    def test_rejects_unsupported_document_format(self):
        with self.assertRaisesRegex(ValueError, 'PDF ou fichier texte'):
            extract_uploaded_document('policy.docx', b'document')

    def test_rejects_invalid_pdf_header(self):
        with self.assertRaisesRegex(ValueError, 'en-tête PDF'):
            extract_uploaded_document('policy.pdf', b'not a PDF')


class DocumentBundleTests(unittest.TestCase):
    def test_multi_file_bundle_combines_text_and_flags_missing_mandatory(self):
        bundle = build_document_bundle(
            [
                {
                    'filename': 'releve.txt',
                    'content': 'Relevé d’information assurance automobile du véhicule assuré.'.encode('utf-8'),
                    'declared_type': 'releve_information',
                },
                {
                    'filename': 'carte_grise.txt',
                    'content': 'Certificat d’immatriculation du véhicule assuré.'.encode('utf-8'),
                    'declared_type': 'carte_grise',
                },
            ],
            product_line='auto',
        )
        self.assertEqual(len(bundle['documents']), 2)
        self.assertIn('releve.txt', bundle['combined_text'])
        self.assertIn('carte_grise.txt', bundle['combined_text'])
        self.assertEqual(bundle['aggregate_status'], 'text_extracted')
        self.assertEqual(bundle['checklist']['missing_mandatory'], ['permis_conduire'])
        self.assertFalse(bundle['checklist']['complete'])

    def test_bundle_is_complete_when_all_mandatory_pieces_declared(self):
        bundle = build_document_bundle(
            [
                {'filename': 'a.txt', 'content': b'assurance automobile', 'declared_type': 'releve_information'},
                {'filename': 'b.txt', 'content': b'assurance automobile', 'declared_type': 'carte_grise'},
                {'filename': 'c.txt', 'content': b'assurance automobile', 'declared_type': 'permis_conduire'},
            ],
            product_line='auto',
        )
        self.assertTrue(bundle['checklist']['complete'])
        self.assertEqual(bundle['checklist']['missing_mandatory'], [])

    def test_image_without_ocr_is_attached_but_unreadable(self):
        with patch('utils.document_bundle.ocr_available', return_value=False):
            bundle = build_document_bundle(
                [{'filename': 'scan.png', 'content': b'\x89PNG\r\n', 'declared_type': 'carte_grise'}],
                product_line='auto',
            )
        document = bundle['documents'][0]
        self.assertEqual(document['extraction_status'], 'attached_ocr_unavailable')
        self.assertEqual(bundle['aggregate_status'], 'attached_unreadable')

    def test_unsupported_format_is_reported_without_raising(self):
        bundle = build_document_bundle(
            [{'filename': 'policy.docx', 'content': b'document', 'declared_type': None}],
            product_line='auto',
        )
        self.assertEqual(bundle['documents'][0]['extraction_status'], 'unsupported_format')

    def test_document_agent_consumes_bundle_and_fails_closed(self):
        bundle = build_document_bundle(
            [
                {
                    'filename': 'conditions.txt',
                    'content': (
                        'Conditions d’assurance automobile. Véhicule assuré. '
                        'Exclusion de garantie après fraude.'
                    ).encode('utf-8'),
                    'declared_type': 'releve_information',
                },
            ],
            product_line='auto',
        )
        result = DocumentAgent().process(VALID_APPLICATION, {}, document_bundle=bundle)
        document = result['uploaded_document']
        self.assertTrue(document['provided'])
        self.assertEqual(document['scope_status'], 'supported')
        self.assertTrue(document['affects_decision'])
        self.assertEqual(
            result['document_checklist']['missing_mandatory'],
            ['carte_grise', 'permis_conduire'],
        )
        self.assertTrue(
            any('obligatoires manquantes' in warning for warning in result['warnings'])
        )


class LLMDocumentAgentTests(unittest.TestCase):
    def setUp(self):
        self.document_text = (
            'Conditions d’assurance automobile.\n'
            'La garantie couvre les dommages au véhicule assuré.\n'
            'Les dommages intentionnels sont exclus de la garantie.\n'
            'Coordonnées de démonstration : telephone fictif 555-0100.'
        )
        self.client = Mock()
        self.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"document_type":"auto_policy",'
                            '"summary":"La police décrit une garantie automobile.",'
                            '"findings":[{"category":"coverage",'
                            '"finding":"Les dommages au véhicule sont couverts.",'
                            '"evidence_quote":"La garantie couvre les dommages au véhicule assuré."}],'
                            '"uncertainties":[]}'
                        )
                    )
                )
            ]
        )

    def test_returns_only_findings_cited_in_retrieved_document_passages(self):
        result = LLMDocumentAgent(client=self.client).analyze(self.document_text)

        self.assertEqual(result['status'], 'analyzed')
        self.assertEqual(result['document_type'], 'auto_policy')
        self.assertEqual(len(result['findings']), 1)
        self.assertEqual(
            result['findings'][0]['evidence_quote'],
            'La garantie couvre les dommages au véhicule assuré.',
        )
        self.assertFalse(result['affects_underwriting_decision'])
        prompt = self.client.chat.completions.create.call_args.kwargs['messages'][1]['content']
        self.assertNotIn('555-0100', prompt)
        self.assertIn('[EXTRAIT-', prompt)

    def test_rejects_a_llm_citation_not_present_in_retrieved_passages(self):
        self.client.chat.completions.create.return_value.choices[0].message.content = (
            '{"document_type":"auto_policy","summary":"Résumé.",'
            '"findings":[{"category":"coverage","finding":"Garantie santé.",'
            '"evidence_quote":"La santé est toujours couverte."}],"uncertainties":[]}'
        )
        with self.assertRaisesRegex(ValueError, 'ne correspond pas au texte fourni'):
            LLMDocumentAgent(client=self.client).analyze(self.document_text)

    def test_requires_api_key_only_when_llm_review_is_enabled(self):
        with patch.dict('os.environ', {'GROQ_API_KEY': ''}):
            with self.assertRaisesRegex(RuntimeError, 'GROQ_API_KEY'):
                LLMDocumentAgent().analyze(self.document_text)


class UncertaintyAgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = UncertaintyAgent()

    def test_entropy_helper_is_zero_when_certain_and_one_when_ambiguous(self):
        self.assertEqual(binary_entropy(0.0), 0.0)
        self.assertEqual(binary_entropy(1.0), 0.0)
        self.assertAlmostEqual(binary_entropy(0.5), 1.0)

    def test_confident_estimate_is_reported_as_low_uncertainty(self):
        result = self.agent.process(VALID_APPLICATION, {'claim_probability': 0.02})
        self.assertEqual(result['status'], 'evaluated')
        self.assertEqual(result['level'], 'low')
        self.assertFalse(result['escalation_recommended'])
        self.assertLess(result['predictive_uncertainty'], 0.30)

    def test_ambiguous_estimate_recommends_human_review(self):
        result = self.agent.process(VALID_APPLICATION, {'claim_probability': 0.5})
        self.assertEqual(result['level'], 'high')
        self.assertTrue(result['escalation_recommended'])
        self.assertAlmostEqual(result['predictive_uncertainty'], 1.0)

    def test_fails_closed_when_probability_is_missing(self):
        result = self.agent.process(VALID_APPLICATION, {})
        self.assertEqual(result['status'], 'unavailable')
        self.assertTrue(result['escalation_recommended'])
        self.assertIsNone(result['predictive_uncertainty'])


class DecisionAndPipelineTests(unittest.TestCase):
    def test_reject_recommendation_is_escalated_for_human_review(self):
        result = EnhancedDecisionAgent(mode='human_in_the_loop').process(
            VALID_APPLICATION,
            {
                'risk': {'decision': 'Reject', 'confidence': 0.99, 'risk_score': 75},
                'document': {},
                'claims': {},
                'customer': {},
            },
        )
        self.assertEqual(result['final_decision'], 'Escalate_to_Human')
        self.assertTrue(result['escalation_triggered'])

    def test_high_uncertainty_escalates_in_human_in_the_loop_mode(self):
        result = EnhancedDecisionAgent(mode='human_in_the_loop').process(
            VALID_APPLICATION,
            {
                'risk': {'decision': 'Accept_Standard', 'confidence': 0.9, 'risk_score': 45},
                'document': {},
                'claims': {},
                'customer': {},
                'uncertainty': {
                    'escalation_recommended': True,
                    'predictive_uncertainty': 0.9,
                    'level': 'high',
                },
            },
        )
        self.assertEqual(result['final_decision'], 'Escalate_to_Human')
        self.assertTrue(result['escalation_triggered'])
        self.assertEqual(result['uncertainty_level'], 'high')

    def test_single_case_fairness_does_not_claim_bias(self):
        result = FairnessAgent().process(
            {**VALID_APPLICATION, 'driver_age': 22},
            {'final_decision': 'Escalate_to_Human'},
        )
        self.assertFalse(result['bias_detected'])
        self.assertTrue(result['review_required'])

    def test_workflow_runs_all_governance_agents_once(self):
        result = LangGraphOrchestrator().run_workflow(VALID_APPLICATION)
        self.assertEqual(len(result['messages']), 11)
        self.assertEqual(len({str(message) for message in result['messages']}), 11)
        self.assertIn('uncertainty_result', result)
        self.assertIn('fairness_result', result)
        self.assertIn('compliance_result', result)
        self.assertIn('explanation_result', result)
        self.assertIn('not certified', result['compliance_result']['compliance_status'])
        self.assertAlmostEqual(sum(result['risk_result']['ml_probabilities'].values()), 1.0, places=5)

    def test_workflow_fails_closed_for_invalid_application(self):
        with self.assertRaisesRegex(ValueError, 'Missing required field'):
            LangGraphOrchestrator().run_workflow({'age': 35})

    def test_direct_risk_call_does_not_impute_missing_features(self):
        with self.assertRaisesRegex(ValueError, 'requires a valid application'):
            RiskAgent().process({}, {}, {})

    def test_parallel_workflow_returns_governance_results(self):
        result = LangGraphOrchestrator().run_workflow(
            VALID_APPLICATION, mode='parallel'
        )
        self.assertEqual(len(result['messages']), 11)
        self.assertIn('uncertainty_result', result)
        self.assertIn('explanation_result', result)

    def test_workflow_reports_document_attachment_metadata(self):
        result = LangGraphOrchestrator().run_workflow(
            VALID_APPLICATION,
            document_name='policy.pdf',
            document_provided=True,
            document_size_bytes=512,
            document_status='attached_reader_unavailable',
        )
        document = result['document_result']['uploaded_document']
        self.assertTrue(document['provided'])
        self.assertEqual(document['filename'], 'policy.pdf')
        self.assertEqual(document['size_bytes'], 512)
        self.assertEqual(document['scope_status'], 'unverified')
        self.assertTrue(document['affects_decision'])
        self.assertFalse(document['affects_risk_score'])

    def test_workflow_escalates_instead_of_recommending_on_health_document(self):
        orchestrator = LangGraphOrchestrator()
        orchestrator.llm_document_agent.analyze = Mock(
            side_effect=AssertionError('out-of-scope text must not be sent to the LLM')
        )
        result = orchestrator.run_workflow(
            VALID_APPLICATION,
            document_name='demande_assurance_sante_exemple.pdf',
            document_text='Demande de remboursement de soins de santé et hospitalisation.',
            document_provided=True,
            document_status='text_extracted',
            enable_llm_document_analysis=True,
        )
        self.assertEqual(
            result['decision_result']['final_decision'],
            'Escalate_to_Human',
        )
        self.assertTrue(result['decision_result']['escalation_triggered'])
        self.assertIn(
            'assurance santé',
            result['explanation_result']['explanations']['customer']['summary'],
        )
        self.assertEqual(
            result['llm_document_result']['status'],
            'skipped_out_of_scope',
        )
        orchestrator.llm_document_agent.analyze.assert_not_called()

    def test_llm_document_findings_do_not_change_underwriting_decision(self):
        orchestrator = LangGraphOrchestrator()
        orchestrator.llm_document_agent.analyze = Mock(return_value={
            'agent': 'LLM_Document',
            'status': 'analyzed',
            'summary': 'La police décrit une garantie automobile.',
            'document_type': 'auto_policy',
            'findings': [],
            'uncertainties': [],
            'provider': 'Groq',
            'model': 'test-model',
            'retrieved_passages': 1,
            'affects_underwriting_decision': False,
        })
        result = orchestrator.run_workflow(
            VALID_APPLICATION,
            document_name='conditions.txt',
            document_text='Conditions d’assurance automobile. La garantie couvre le véhicule assuré.',
            document_provided=True,
            document_status='text_extracted',
            enable_llm_document_analysis=True,
        )
        self.assertEqual(result['llm_document_result']['status'], 'analyzed')
        self.assertEqual(
            result['decision_result']['ml_decision'],
            result['risk_result']['decision'],
        )
        self.assertFalse(
            result['llm_document_result']['affects_underwriting_decision']
        )
        orchestrator.llm_document_agent.analyze.assert_called_once()


class ResearchMetricsTests(unittest.TestCase):
    @unittest.skipUnless(
        RAW_FREMTPL2_FREQ.is_file(),
        'real freMTPL2 raw data not downloaded (run experiments/data/fetch_fremtpl2.py)',
    )
    def test_holdout_evaluation_uses_the_real_model_and_partition(self):
        result = evaluate_holdout()
        self.assertEqual(result['dataset_size'], 678013)
        self.assertEqual(result['test_size'], 135603)
        self.assertLess(result['base_rate_test'], 0.1)
        self.assertGreater(result['roc_auc'], 0.55)
        self.assertLess(result['roc_auc'], 0.9)
        self.assertLess(result['accuracy'], 0.99)
        self.assertEqual(result['confusion_matrix'].shape, (2, 2))
        self.assertEqual(
            sum(result['mapped_decisions'].values()), result['test_size']
        )

    def test_fairness_counts_approval_as_positive_outcome(self):
        records = [
            {'driver_age': 25, 'bonus_malus': 120.0},
            {'driver_age': 30, 'bonus_malus': 120.0},
            {'driver_age': 45, 'bonus_malus': 60.0},
            {'driver_age': 50, 'bonus_malus': 60.0},
        ]
        result = evaluate_fairness(
            records,
            ['Accept_Standard', 'Accept_Standard', 'Accept_Standard', 'Reject'],
        )
        self.assertAlmostEqual(result['demographic_parity_age']['ratio'], 0.5)
        self.assertAlmostEqual(result['demographic_parity_bonus_malus']['ratio'], 0.5)
        self.assertEqual(result['positive_outcome'], 'approval (decision other than Reject)')

    def test_uncertainty_metrics_use_predictions_and_probabilities(self):
        result = evaluate_uncertainty(
            predictions=np.array([0, 1, 1]),
            true_labels=np.array([0, 0, 1]),
            confidences=np.array([0.9, 0.6, 0.8]),
            probabilities=np.array([
                [0.9, 0.1],
                [0.4, 0.6],
                [0.2, 0.8],
            ]),
        )
        self.assertAlmostEqual(result['accuracy'], 2 / 3)
        self.assertIn('multiclass_brier_score', result)
        self.assertEqual(result['risk_coverage_curve'][-1][0], 1.0)
        self.assertAlmostEqual(result['risk_coverage_curve'][-1][1], 2 / 3)


if __name__ == '__main__':
    unittest.main()
