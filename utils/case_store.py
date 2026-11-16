"""SQLite persistence for underwriting cases, HITL decisions, and audit events."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "runtime" / "aurelia_cases.db"

# Pilot role model. Capabilities are explicit so authorization stays declarative.
ROLE_CAPABILITIES: dict[str, frozenset[str]] = {
    "uw": frozenset({"analyze", "view_queue", "decide", "view_audit"}),
    "manager": frozenset(
        {"analyze", "view_queue", "decide", "override", "view_audit"}
    ),
    "compliance": frozenset({"view_queue", "view_audit"}),
    "admin": frozenset(
        {"analyze", "view_queue", "decide", "override", "view_audit", "manage_users"}
    ),
}
VALID_ROLES = frozenset(ROLE_CAPABILITIES)
CASE_STATUSES = frozenset({"pending_review", "decided", "archived"})
HUMAN_DECISIONS = frozenset(
    {
        "Accept_Standard",
        "Accept_with_Surcharge",
        "Accept_with_Exclusion",
        "Reject",
        "Request_More_Info",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# Sentinel used as the previous hash of the first audit event of a case.
GENESIS_HASH = "0" * 64


def _hash_audit_event(
    prev_hash: str,
    case_id: str,
    event_type: str,
    actor: str,
    created_at: str,
    payload_json: str,
) -> str:
    """Deterministic hash binding an audit event to its predecessor."""
    material = "\x1f".join(
        [prev_hash, case_id, event_type, actor, created_at, payload_json]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _hash_password(password: str, salt: str | None = None) -> str:
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        120_000,
    )
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, _ = stored.split("$", 1)
    except ValueError:
        return False
    return secrets.compare_digest(_hash_password(password, salt), stored)


def role_can(role: str, capability: str) -> bool:
    """Return whether a pilot role is allowed to perform a capability."""
    return capability in ROLE_CAPABILITIES.get(role, frozenset())


class CaseStore:
    """Append-oriented case and audit store for the pilot workbench."""

    def __init__(self, db_path: str | Path | None = None):
        env_path = os.environ.get("AURELIA_DB_PATH")
        self.db_path = Path(db_path or env_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self._ensure_default_users()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _init_schema(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS cases (
                    id TEXT PRIMARY KEY,
                    application_id TEXT NOT NULL,
                    product_line TEXT NOT NULL,
                    status TEXT NOT NULL,
                    ai_decision TEXT,
                    human_decision TEXT,
                    human_reason TEXT,
                    decided_by TEXT,
                    decided_at TEXT,
                    created_at TEXT NOT NULL,
                    application_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    escalation_triggered INTEGER NOT NULL DEFAULT 0,
                    llm_enabled INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    prev_hash TEXT,
                    event_hash TEXT,
                    FOREIGN KEY (case_id) REFERENCES cases(id)
                );

                CREATE INDEX IF NOT EXISTS idx_cases_status
                    ON cases(status, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_audit_case
                    ON audit_events(case_id, created_at);
                """
            )
            self._ensure_columns(
                connection,
                "audit_events",
                {"prev_hash": "TEXT", "event_hash": "TEXT"},
            )
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _ensure_columns(
        connection: sqlite3.Connection,
        table: str,
        columns: dict[str, str],
    ) -> None:
        """Add missing columns to a pre-existing table (pilot migration)."""
        existing = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for name, ddl_type in columns.items():
            if name not in existing:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl_type}")

    def _ensure_default_users(self) -> None:
        defaults = [
            (
                "uw",
                os.environ.get("AURELIA_UW_PASSWORD", "uw-demo-change-me"),
                "uw",
            ),
            (
                "manager",
                os.environ.get("AURELIA_MANAGER_PASSWORD", "manager-demo-change-me"),
                "manager",
            ),
            (
                "compliance",
                os.environ.get("AURELIA_COMPLIANCE_PASSWORD", "compliance-demo-change-me"),
                "compliance",
            ),
            (
                "admin",
                os.environ.get("AURELIA_ADMIN_PASSWORD", "admin-demo-change-me"),
                "admin",
            ),
        ]
        connection = self._connect()
        try:
            for username, password, role in defaults:
                existing = connection.execute(
                    "SELECT id FROM users WHERE username = ?",
                    (username,),
                ).fetchone()
                if existing:
                    continue
                connection.execute(
                    """
                    INSERT INTO users (username, password_hash, role, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (username, _hash_password(password), role, utc_now()),
                )
            connection.commit()
        finally:
            connection.close()

    def authenticate(self, username: str, password: str) -> dict[str, Any] | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT username, password_hash, role FROM users WHERE username = ?",
                (username.strip(),),
            ).fetchone()
        finally:
            connection.close()
        if row is None or not verify_password(password, row["password_hash"]):
            return None
        return {"username": row["username"], "role": row["role"]}

    def append_audit(
        self,
        case_id: str,
        event_type: str,
        actor: str,
        payload: dict[str, Any],
        connection: sqlite3.Connection | None = None,
    ) -> dict[str, Any]:
        owns_connection = connection is None
        if owns_connection:
            connection = self._connect()
        assert connection is not None
        payload_json = json.dumps(payload, ensure_ascii=False, default=str)
        created_at = utc_now()
        try:
            previous = connection.execute(
                """
                SELECT event_hash FROM audit_events
                WHERE case_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (case_id,),
            ).fetchone()
            prev_hash = (previous["event_hash"] if previous else None) or GENESIS_HASH
            event_hash = _hash_audit_event(
                prev_hash, case_id, event_type, actor, created_at, payload_json
            )
            connection.execute(
                """
                INSERT INTO audit_events (
                    case_id, event_type, actor, payload_json, created_at,
                    prev_hash, event_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    event_type,
                    actor,
                    payload_json,
                    created_at,
                    prev_hash,
                    event_hash,
                ),
            )
            if owns_connection:
                connection.commit()
        finally:
            if owns_connection:
                connection.close()
        return {
            "event_type": event_type,
            "created_at": created_at,
            "prev_hash": prev_hash,
            "event_hash": event_hash,
        }

    def create_case(
        self,
        *,
        application: dict[str, Any],
        result: dict[str, Any],
        product_line: str,
        actor: str,
        llm_enabled: bool = False,
    ) -> dict[str, Any]:
        case_id = str(uuid.uuid4())
        decision = result.get("decision_result") or {}
        ai_decision = decision.get("final_decision")
        escalation = bool(decision.get("escalation_triggered"))
        # Pilot policy: every recommendation enters the HITL queue.
        status = "pending_review"
        created_at = utc_now()
        application_id = str(application.get("application_id") or case_id)
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO cases (
                    id, application_id, product_line, status, ai_decision,
                    created_at, application_json, result_json,
                    escalation_triggered, llm_enabled
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    application_id,
                    product_line,
                    status,
                    ai_decision,
                    created_at,
                    json.dumps(application, ensure_ascii=False, default=str),
                    json.dumps(result, ensure_ascii=False, default=str),
                    int(escalation),
                    int(llm_enabled),
                ),
            )
            self.append_audit(
                case_id,
                "case_created",
                actor,
                {
                    "application_id": application_id,
                    "product_line": product_line,
                    "ai_decision": ai_decision,
                    "escalation_triggered": escalation,
                    "llm_enabled": llm_enabled,
                },
                connection=connection,
            )
            connection.commit()
        finally:
            connection.close()
        return self.get_case(case_id)

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM cases WHERE id = ?",
                (case_id,),
            ).fetchone()
        finally:
            connection.close()
        return self._row_to_case(row) if row else None

    def list_pending_cases(self, product_line: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM cases WHERE status = 'pending_review'"
        params: list[Any] = []
        if product_line:
            query += " AND product_line = ?"
            params.append(product_line)
        query += " ORDER BY created_at ASC"
        connection = self._connect()
        try:
            rows = connection.execute(query, params).fetchall()
        finally:
            connection.close()
        return [self._row_to_case(row) for row in rows]

    def list_recent_cases(self, limit: int = 50) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM cases ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            connection.close()
        return [self._row_to_case(row) for row in rows]

    def record_human_decision(
        self,
        *,
        case_id: str,
        human_decision: str,
        human_reason: str,
        actor: str,
    ) -> dict[str, Any]:
        if human_decision not in HUMAN_DECISIONS:
            raise ValueError(f"Unsupported human decision: {human_decision}")
        reason = (human_reason or "").strip()
        if len(reason) < 5:
            raise ValueError("A human decision reason of at least 5 characters is required")

        decided_at = utc_now()
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT status, ai_decision FROM cases WHERE id = ?",
                (case_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown case: {case_id}")
            if row["status"] != "pending_review":
                raise ValueError("Only pending cases can receive a human decision")

            connection.execute(
                """
                UPDATE cases
                SET status = 'decided',
                    human_decision = ?,
                    human_reason = ?,
                    decided_by = ?,
                    decided_at = ?
                WHERE id = ?
                """,
                (human_decision, reason, actor, decided_at, case_id),
            )
            self.append_audit(
                case_id,
                "human_decision",
                actor,
                {
                    "human_decision": human_decision,
                    "human_reason": reason,
                    "ai_decision": row["ai_decision"],
                    "is_override": human_decision != row["ai_decision"],
                    "decided_at": decided_at,
                },
                connection=connection,
            )
            connection.commit()
        finally:
            connection.close()
        case = self.get_case(case_id)
        assert case is not None
        return case

    def export_audit(self, case_id: str) -> dict[str, Any]:
        case = self.get_case(case_id)
        if case is None:
            raise KeyError(f"Unknown case: {case_id}")
        connection = self._connect()
        try:
            events = connection.execute(
                """
                SELECT id, event_type, actor, payload_json, created_at,
                       prev_hash, event_hash
                FROM audit_events
                WHERE case_id = ?
                ORDER BY id ASC
                """,
                (case_id,),
            ).fetchall()
        finally:
            connection.close()

        audit_events = [
            {
                "id": event["id"],
                "event_type": event["event_type"],
                "actor": event["actor"],
                "created_at": event["created_at"],
                "payload": json.loads(event["payload_json"]),
                "prev_hash": event["prev_hash"],
                "event_hash": event["event_hash"],
            }
            for event in events
        ]
        return {
            "case_id": case_id,
            "exported_at": utc_now(),
            "case": {
                "id": case["id"],
                "application_id": case["application_id"],
                "product_line": case["product_line"],
                "status": case["status"],
                "ai_decision": case["ai_decision"],
                "human_decision": case["human_decision"],
                "human_reason": case["human_reason"],
                "decided_by": case["decided_by"],
                "decided_at": case["decided_at"],
                "created_at": case["created_at"],
                "escalation_triggered": case["escalation_triggered"],
                "llm_enabled": case["llm_enabled"],
                "application": case["application"],
                "result": case["result"],
            },
            "audit_events": audit_events,
            "audit_chain": self._verify_chain(case_id, audit_events),
        }

    @staticmethod
    def _verify_chain(case_id: str, audit_events: list[dict[str, Any]]) -> dict[str, Any]:
        """Recompute the append-only hash chain to detect tampering."""
        previous = GENESIS_HASH
        for index, event in enumerate(audit_events):
            if event.get("prev_hash") != previous:
                return {"valid": False, "reason": "broken_link", "at_index": index}
            expected = _hash_audit_event(
                previous,
                case_id,
                event["event_type"],
                event["actor"],
                event["created_at"],
                json.dumps(event["payload"], ensure_ascii=False, default=str),
            )
            if event.get("event_hash") != expected:
                return {"valid": False, "reason": "hash_mismatch", "at_index": index}
            previous = event["event_hash"]
        return {"valid": True, "events": len(audit_events)}

    def list_users(self) -> list[dict[str, Any]]:
        """List pilot users without exposing password material."""
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT username, role, created_at FROM users ORDER BY username"
            ).fetchall()
        finally:
            connection.close()
        return [
            {
                "username": row["username"],
                "role": row["role"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    @staticmethod
    def _row_to_case(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "application_id": row["application_id"],
            "product_line": row["product_line"],
            "status": row["status"],
            "ai_decision": row["ai_decision"],
            "human_decision": row["human_decision"],
            "human_reason": row["human_reason"],
            "decided_by": row["decided_by"],
            "decided_at": row["decided_at"],
            "created_at": row["created_at"],
            "application": json.loads(row["application_json"]),
            "result": json.loads(row["result_json"]),
            "escalation_triggered": bool(row["escalation_triggered"]),
            "llm_enabled": bool(row["llm_enabled"]),
        }
