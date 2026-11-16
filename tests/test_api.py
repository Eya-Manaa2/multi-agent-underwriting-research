import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app as api_module
from utils.case_store import CaseStore


def _sample_result():
    return {
        "decision_result": {
            "final_decision": "Escalate_to_Human",
            "escalation_triggered": True,
            "reasons": ["api test"],
            "final_risk_score": 55.0,
        },
        "document_result": {"uploaded_document": {"provided": True, "scope_status": "supported"}},
        "explanation_result": {
            "explanations": {
                "customer": {"summary": "test", "details": [], "next_steps": []}
            }
        },
        "llm_document_result": {"status": "disabled"},
    }


class ApiPilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CaseStore(Path(self.tmp.name) / "api_pilot.db")
        api_module._store = self.store
        api_module._sessions.clear()
        api_module._orchestrator = None
        self.client = TestClient(api_module.app)

    def tearDown(self):
        self.client.close()
        api_module._sessions.clear()
        api_module._store = None
        api_module._orchestrator = None
        self.store = None
        self.tmp.cleanup()

    def _login(self, username="uw", password="uw-demo-change-me"):
        response = self.client.post(
            "/v1/auth/login",
            json={"username": username, "password": password},
        )
        self.assertEqual(response.status_code, 200, response.text)
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def test_health_and_auth_required(self):
        self.assertEqual(self.client.get("/health").json()["status"], "ok")
        denied = self.client.get("/v1/cases/pending")
        self.assertEqual(denied.status_code, 401)

    def test_login_rejects_bad_password(self):
        response = self.client.post(
            "/v1/auth/login",
            json={"username": "uw", "password": "nope"},
        )
        self.assertEqual(response.status_code, 401)

    def test_analyze_queues_hitl_decision_and_audit(self):
        headers = self._login()
        with patch.object(
            api_module,
            "get_orchestrator",
        ) as get_orch:
            orchestrator = get_orch.return_value
            orchestrator.run_workflow.return_value = _sample_result()
            response = self.client.post(
                "/v1/cases/analyze",
                headers=headers,
                data={
                    "application_id": "API-001",
                    "product_line": "auto",
                    "driver_age": "35",
                    "bonus_malus": "60",
                    "vehicle_age": "5",
                    "vehicle_power": "6",
                    "vehicle_gas": "Regular",
                    "density": "1200",
                    "area": "C",
                    "enable_llm_document_analysis": "false",
                },
                files={
                    "document": (
                        "policy.txt",
                        b"DOCUMENT FICTIF assurance automobile conditions generales",
                        "text/plain",
                    )
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        case_id = body["case"]["id"]
        self.assertEqual(body["case"]["status"], "pending_review")
        self.assertEqual(body["case"]["application_id"], "API-001")

        pending = self.client.get("/v1/cases/pending", headers=headers)
        self.assertEqual(pending.status_code, 200)
        self.assertEqual(len(pending.json()), 1)

        decision = self.client.post(
            f"/v1/cases/{case_id}/decision",
            headers=headers,
            json={
                "human_decision": "Accept_Standard",
                "human_reason": "Revue API pilote apres analyse documentaire.",
            },
        )
        self.assertEqual(decision.status_code, 200, decision.text)
        self.assertEqual(decision.json()["status"], "decided")

        audit = self.client.get(f"/v1/cases/{case_id}/audit", headers=headers)
        self.assertEqual(audit.status_code, 200)
        events = [event["event_type"] for event in audit.json()["audit_events"]]
        self.assertEqual(events, ["case_created", "human_decision"])

    def test_analyze_accepts_multiple_documents_and_reports_checklist(self):
        headers = self._login()
        with patch.object(api_module, "get_orchestrator") as get_orch:
            orchestrator = get_orch.return_value
            orchestrator.run_workflow.return_value = _sample_result()
            response = self.client.post(
                "/v1/cases/analyze",
                headers=headers,
                data={
                    "application_id": "API-MULTI",
                    "product_line": "auto",
                    "driver_age": "35",
                    "bonus_malus": "60",
                    "vehicle_age": "5",
                    "vehicle_power": "6",
                    "vehicle_gas": "Regular",
                    "density": "1200",
                    "area": "C",
                    "enable_llm_document_analysis": "false",
                    "document_types": [
                        "releve_information",
                        "carte_grise",
                        "permis_conduire",
                    ],
                },
                files=[
                    ("documents", ("releve.txt", b"assurance automobile", "text/plain")),
                    ("documents", ("carte.txt", b"assurance automobile", "text/plain")),
                    ("documents", ("permis.txt", b"assurance automobile", "text/plain")),
                ],
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["document_checklist"]["complete"])
        bundle = orchestrator.run_workflow.call_args.kwargs["document_bundle"]
        self.assertEqual(len(bundle["documents"]), 3)
        self.assertEqual(
            [document["declared_type"] for document in bundle["documents"]],
            ["releve_information", "carte_grise", "permis_conduire"],
        )

    def test_analyze_forwards_collaboration_mode_and_rejects_unknown_mode(self):
        headers = self._login()
        with patch.object(api_module, "get_orchestrator") as get_orch:
            get_orch.return_value.run_workflow.return_value = _sample_result()
            response = self.client.post(
                "/v1/cases/analyze",
                headers=headers,
                data={
                    "application_id": "API-PARALLEL",
                    "product_line": "auto",
                    "driver_age": "35",
                    "bonus_malus": "60",
                    "vehicle_age": "5",
                    "vehicle_power": "6",
                    "vehicle_gas": "Regular",
                    "density": "1200",
                    "area": "C",
                    "collaboration_mode": "parallel",
                },
                files={"document": ("policy.txt", b"assurance automobile", "text/plain")},
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(
                get_orch.return_value.run_workflow.call_args.kwargs["mode"], "parallel"
            )

        invalid = self.client.post(
            "/v1/cases/analyze",
            headers=headers,
            data={
                "application_id": "API-BADMODE",
                "product_line": "auto",
                "driver_age": "35",
                "bonus_malus": "60",
                "vehicle_age": "5",
                "vehicle_power": "6",
                "vehicle_gas": "Regular",
                "density": "1200",
                "area": "C",
                "collaboration_mode": "quantum",
            },
            files={"document": ("policy.txt", b"assurance automobile", "text/plain")},
        )
        self.assertEqual(invalid.status_code, 400, invalid.text)

    def test_products_endpoint_lists_auto_and_reserved_home(self):
        headers = self._login("admin", "admin-demo-change-me")
        response = self.client.get("/v1/products", headers=headers)
        self.assertEqual(response.status_code, 200)
        codes = {item["code"] for item in response.json()["lines"]}
        self.assertIn("auto", codes)
        self.assertIn("home", codes)

    def test_compliance_role_is_read_only(self):
        headers = self._login("compliance", "compliance-demo-change-me")
        pending = self.client.get("/v1/cases/pending", headers=headers)
        self.assertEqual(pending.status_code, 200)

        analyze = self.client.post(
            "/v1/cases/analyze",
            headers=headers,
            data={
                "application_id": "API-RBAC",
                "product_line": "auto",
                "driver_age": "35",
                "bonus_malus": "60",
                "vehicle_age": "5",
                "vehicle_power": "6",
                "vehicle_gas": "Regular",
                "density": "1200",
                "area": "C",
                "enable_llm_document_analysis": "false",
            },
            files={"document": ("policy.txt", b"DOCUMENT FICTIF", "text/plain")},
        )
        self.assertEqual(analyze.status_code, 403, analyze.text)

        decision = self.client.post(
            "/v1/cases/unknown/decision",
            headers=headers,
            json={"human_decision": "Accept_Standard", "human_reason": "tentative"},
        )
        self.assertEqual(decision.status_code, 403, decision.text)

    def test_admin_users_endpoint_requires_admin(self):
        headers = self._login()
        denied = self.client.get("/v1/admin/users", headers=headers)
        self.assertEqual(denied.status_code, 403)

        admin = self._login("admin", "admin-demo-change-me")
        allowed = self.client.get("/v1/admin/users", headers=admin)
        self.assertEqual(allowed.status_code, 200)
        roles = {user["role"] for user in allowed.json()["users"]}
        self.assertEqual(roles, {"uw", "manager", "compliance", "admin"})

    def test_logout_revokes_token(self):
        headers = self._login()
        self.assertEqual(
            self.client.get("/v1/cases/pending", headers=headers).status_code, 200
        )
        logged_out = self.client.post("/v1/auth/logout", headers=headers)
        self.assertEqual(logged_out.status_code, 200)
        self.assertEqual(
            self.client.get("/v1/cases/pending", headers=headers).status_code, 401
        )


if __name__ == "__main__":
    unittest.main()
