import json
import tempfile
import unittest
from pathlib import Path

from agents.intake_agent import IntakeAgent
from products.registry import get_product_pack, list_product_lines, validate_product_line
from utils.case_store import CaseStore, role_can


VALID_APPLICATION = {
    "application_id": "PILOT-001",
    "product_line": "auto",
    "driver_age": 35,
    "bonus_malus": 60.0,
    "vehicle_age": 5,
    "vehicle_power": 6,
    "vehicle_gas": "Regular",
    "density": 1200.0,
    "area": "C",
}


def _sample_result(decision="Escalate_to_Human", escalation=True):
    return {
        "decision_result": {
            "final_decision": decision,
            "escalation_triggered": escalation,
            "reasons": ["pilot test"],
        },
        "document_result": {"uploaded_document": {"provided": True}},
        "explanation_result": {
            "explanations": {
                "customer": {
                    "summary": "test",
                    "details": [],
                    "next_steps": [],
                }
            }
        },
    }


class ProductLineTests(unittest.TestCase):
    def test_auto_pack_is_active(self):
        pack = get_product_pack("auto")
        self.assertEqual(pack["status"], "active")
        self.assertTrue(pack["required_documents"])

    def test_home_is_reserved(self):
        with self.assertRaises(ValueError):
            validate_product_line("home", require_active=True)
        lines = list_product_lines(enabled_only=False)
        self.assertIn("home", {item["code"] for item in lines})

    def test_intake_defaults_to_auto_and_rejects_reserved_line(self):
        agent = IntakeAgent()
        ok = agent.process({k: v for k, v in VALID_APPLICATION.items() if k != "product_line"})
        self.assertEqual(ok["status"], "validated")
        self.assertEqual(ok["prepared_data"]["product_line"], "auto")

        bad = agent.process({**VALID_APPLICATION, "product_line": "home"})
        self.assertEqual(bad["status"], "needs_review")
        self.assertTrue(any("reserved" in error for error in bad["errors"]))


class CaseStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "pilot.db"
        self.store = CaseStore(self.db_path)

    def tearDown(self):
        # Drop the store reference so Windows can delete the SQLite file.
        self.store = None
        self.tmp.cleanup()

    def test_authenticate_default_users(self):
        user = self.store.authenticate("uw", "uw-demo-change-me")
        self.assertEqual(user["role"], "uw")
        self.assertIsNone(self.store.authenticate("uw", "wrong"))

    def test_role_capabilities(self):
        self.assertTrue(role_can("uw", "decide"))
        self.assertTrue(role_can("compliance", "view_queue"))
        self.assertFalse(role_can("compliance", "decide"))
        self.assertTrue(role_can("admin", "manage_users"))
        self.assertFalse(role_can("uw", "manage_users"))
        self.assertFalse(role_can("unknown", "view_queue"))

    def test_default_users_cover_all_pilot_roles(self):
        roles = {user["role"] for user in self.store.list_users()}
        self.assertEqual(roles, {"uw", "manager", "compliance", "admin"})
        self.assertTrue(
            all("password" not in key for key in self.store.list_users()[0])
        )

    def test_create_case_enters_hitl_queue_and_export_audit(self):
        case = self.store.create_case(
            application=VALID_APPLICATION,
            result=_sample_result(),
            product_line="auto",
            actor="uw",
            llm_enabled=False,
        )
        self.assertEqual(case["status"], "pending_review")
        pending = self.store.list_pending_cases()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["id"], case["id"])

        updated = self.store.record_human_decision(
            case_id=case["id"],
            human_decision="Accept_with_Surcharge",
            human_reason="Revue manuelle du dossier pilote.",
            actor="uw",
        )
        self.assertEqual(updated["status"], "decided")
        self.assertEqual(updated["human_decision"], "Accept_with_Surcharge")
        self.assertEqual(len(self.store.list_pending_cases()), 0)

        audit = self.store.export_audit(case["id"])
        event_types = [event["event_type"] for event in audit["audit_events"]]
        self.assertEqual(event_types, ["case_created", "human_decision"])
        payload = json.dumps(audit)
        self.assertIn("Accept_with_Surcharge", payload)
        self.assertIn("Revue manuelle", payload)

    def test_human_decision_requires_reason(self):
        case = self.store.create_case(
            application=VALID_APPLICATION,
            result=_sample_result(),
            product_line="auto",
            actor="admin",
        )
        with self.assertRaises(ValueError):
            self.store.record_human_decision(
                case_id=case["id"],
                human_decision="Reject",
                human_reason="non",
                actor="admin",
            )

    def test_audit_chain_links_events_and_flags_override(self):
        case = self.store.create_case(
            application=VALID_APPLICATION,
            result=_sample_result(decision="Reject"),
            product_line="auto",
            actor="uw",
        )
        self.store.record_human_decision(
            case_id=case["id"],
            human_decision="Accept_Standard",
            human_reason="Override documenté après revue du relevé d'information.",
            actor="manager",
        )
        audit = self.store.export_audit(case["id"])
        self.assertTrue(audit["audit_chain"]["valid"])
        self.assertEqual(audit["audit_chain"]["events"], 2)
        created, decision = audit["audit_events"]
        self.assertEqual(decision["prev_hash"], created["event_hash"])
        self.assertTrue(decision["payload"]["is_override"])
        self.assertEqual(decision["payload"]["ai_decision"], "Reject")

    def test_audit_chain_detects_tampering(self):
        case = self.store.create_case(
            application=VALID_APPLICATION,
            result=_sample_result(),
            product_line="auto",
            actor="uw",
        )
        connection = self.store._connect()
        try:
            connection.execute(
                "UPDATE audit_events SET payload_json = ? WHERE case_id = ?",
                ('{"tampered": true}', case["id"]),
            )
            connection.commit()
        finally:
            connection.close()

        audit = self.store.export_audit(case["id"])
        self.assertFalse(audit["audit_chain"]["valid"])
        self.assertEqual(audit["audit_chain"]["reason"], "hash_mismatch")


if __name__ == "__main__":
    unittest.main()
