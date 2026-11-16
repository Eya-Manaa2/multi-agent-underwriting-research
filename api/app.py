"""FastAPI surface for analysis, HITL queue, and audit export."""

from __future__ import annotations

import secrets
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestrator.langgraph_orchestrator import LangGraphOrchestrator
from products.registry import DEFAULT_PRODUCT_LINE, get_product_pack, list_product_lines
from utils.case_store import HUMAN_DECISIONS, CaseStore, role_can
from utils.document_bundle import build_document_bundle


TOKEN_TTL_HOURS = 12
_security = HTTPBearer(auto_error=False)
_sessions: dict[str, dict[str, Any]] = {}
_store: CaseStore | None = None
_orchestrator: LangGraphOrchestrator | None = None


app = FastAPI(
    title="AURELIA Underwriting Pilot API",
    description=(
        "Aide à la décision de souscription. Toute recommandation entre en file HITL. "
        "Ce n'est pas un système de production ni une décision automatisée certifiée."
    ),
    version="0.1.0-pilot",
)


def get_store() -> CaseStore:
    global _store
    if _store is None:
        _store = CaseStore()
    return _store


def get_orchestrator() -> LangGraphOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = LangGraphOrchestrator()
    return _orchestrator


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str
    expires_at: str


class HumanDecisionRequest(BaseModel):
    human_decision: str
    human_reason: str = Field(min_length=5)


class CaseSummary(BaseModel):
    id: str
    application_id: str
    product_line: str
    status: str
    ai_decision: str | None
    human_decision: str | None
    human_reason: str | None
    decided_by: str | None
    decided_at: str | None
    created_at: str
    escalation_triggered: bool
    llm_enabled: bool


def _purge_expired_sessions() -> None:
    now = datetime.now(timezone.utc)
    expired = [
        token
        for token, session in _sessions.items()
        if session["expires_at"] <= now
    ]
    for token in expired:
        _sessions.pop(token, None)


def _issue_token(user: dict[str, str]) -> LoginResponse:
    _purge_expired_sessions()
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=TOKEN_TTL_HOURS)
    _sessions[token] = {
        "username": user["username"],
        "role": user["role"],
        "expires_at": expires_at,
    }
    return LoginResponse(
        access_token=token,
        role=user["role"],
        username=user["username"],
        expires_at=expires_at.replace(microsecond=0).isoformat(),
    )


def current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> dict[str, str]:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required",
        )
    _purge_expired_sessions()
    session = _sessions.get(credentials.credentials)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    return {"username": session["username"], "role": session["role"]}


def require_capability(capability: str):
    """Dependency factory enforcing a pilot role capability."""

    def dependency(user: dict[str, str] = Depends(current_user)) -> dict[str, str]:
        if not role_can(user["role"], capability):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user['role']}' is not allowed to {capability}",
            )
        return user

    return dependency


def _case_summary(case: dict[str, Any]) -> CaseSummary:
    return CaseSummary(
        id=case["id"],
        application_id=case["application_id"],
        product_line=case["product_line"],
        status=case["status"],
        ai_decision=case["ai_decision"],
        human_decision=case["human_decision"],
        human_reason=case["human_reason"],
        decided_by=case["decided_by"],
        decided_at=case["decided_at"],
        created_at=case["created_at"],
        escalation_triggered=case["escalation_triggered"],
        llm_enabled=case["llm_enabled"],
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "aurelia-pilot", "mode": "human_in_the_loop"}


@app.post("/v1/auth/login", response_model=LoginResponse)
def login(payload: LoginRequest) -> LoginResponse:
    user = get_store().authenticate(payload.username, payload.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    return _issue_token(user)


@app.get("/v1/products")
def products(_: dict[str, str] = Depends(current_user)) -> dict[str, Any]:
    return {
        "default": DEFAULT_PRODUCT_LINE,
        "lines": list_product_lines(enabled_only=False),
        "active_pack": get_product_pack(DEFAULT_PRODUCT_LINE),
    }


@app.post("/v1/auth/logout")
def logout(
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> dict[str, str]:
    if credentials is not None:
        _sessions.pop(credentials.credentials, None)
    return {"status": "logged_out"}


@app.get("/v1/admin/users")
def admin_list_users(
    _: dict[str, str] = Depends(require_capability("manage_users")),
) -> dict[str, Any]:
    return {"users": get_store().list_users()}


@app.post("/v1/cases/analyze")
async def analyze_case(
    application_id: str = Form("DEMO-001"),
    product_line: str = Form(DEFAULT_PRODUCT_LINE),
    driver_age: int = Form(...),
    bonus_malus: float = Form(...),
    vehicle_age: int = Form(...),
    vehicle_power: int = Form(...),
    vehicle_gas: str = Form("Regular"),
    density: float = Form(...),
    area: str = Form("C"),
    enable_llm_document_analysis: bool = Form(False),
    collaboration_mode: str = Form("sequential"),
    document: UploadFile | None = File(default=None),
    documents: list[UploadFile] = File(default=[]),
    document_types: list[str] = Form(default=[]),
    user: dict[str, str] = Depends(require_capability("analyze")),
) -> dict[str, Any]:
    uploaded_files = ([document] if document is not None else []) + list(documents)
    if not uploaded_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one document is required.",
        )
    if collaboration_mode not in {"sequential", "parallel"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown collaboration mode: {collaboration_mode}",
        )

    payload = []
    for index, upload in enumerate(uploaded_files):
        name = Path(upload.filename or "upload.bin").name
        declared = document_types[index] if index < len(document_types) else None
        payload.append(
            {
                "filename": name,
                "content": await upload.read(),
                "declared_type": declared or None,
            }
        )

    bundle = build_document_bundle(payload, product_line=product_line)
    invalid = [
        doc for doc in bundle["documents"]
        if doc["extraction_status"] in {"invalid", "unsupported_format"}
    ]
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{invalid[0]['filename']}: {invalid[0]['reason']}",
        )

    application = {
        "application_id": application_id.strip() or "DEMO-001",
        "product_line": product_line,
        "driver_age": int(driver_age),
        "bonus_malus": float(bonus_malus),
        "vehicle_age": int(vehicle_age),
        "vehicle_power": int(vehicle_power),
        "vehicle_gas": vehicle_gas,
        "density": float(density),
        "area": area,
    }

    try:
        result = get_orchestrator().run_workflow(
            application,
            mode=collaboration_mode,
            decision_mode="human_in_the_loop",
            document_bundle=bundle,
            enable_llm_document_analysis=enable_llm_document_analysis,
        )
        case = get_store().create_case(
            application=application,
            result=result,
            product_line=application["product_line"],
            actor=user["username"],
            llm_enabled=enable_llm_document_analysis,
        )
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    except (FileNotFoundError, ImportError, RuntimeError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error

    return {
        "case": _case_summary(case),
        "decision": result.get("decision_result", {}),
        "document_checklist": bundle.get("checklist"),
        "report": result.get("report_result", {}),
        "message": "Recommendation stored and queued for human review",
    }


@app.get("/v1/cases/pending", response_model=list[CaseSummary])
def list_pending(
    product_line: str | None = None,
    _: dict[str, str] = Depends(require_capability("view_queue")),
) -> list[CaseSummary]:
    return [_case_summary(case) for case in get_store().list_pending_cases(product_line)]


@app.get("/v1/cases/{case_id}")
def get_case(
    case_id: str,
    _: dict[str, str] = Depends(require_capability("view_queue")),
) -> dict[str, Any]:
    case = get_store().get_case(case_id)
    if case is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")
    return {
        "case": _case_summary(case),
        "application": case["application"],
        "result": case["result"],
    }


@app.post("/v1/cases/{case_id}/decision", response_model=CaseSummary)
def record_decision(
    case_id: str,
    payload: HumanDecisionRequest,
    user: dict[str, str] = Depends(require_capability("decide")),
) -> CaseSummary:
    if payload.human_decision not in HUMAN_DECISIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported human_decision. Allowed: {sorted(HUMAN_DECISIONS)}",
        )
    try:
        case = get_store().record_human_decision(
            case_id=case_id,
            human_decision=payload.human_decision,
            human_reason=payload.human_reason,
            actor=user["username"],
        )
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    return _case_summary(case)


@app.get("/v1/cases/{case_id}/audit")
def export_audit(
    case_id: str,
    _: dict[str, str] = Depends(require_capability("view_audit")),
) -> dict[str, Any]:
    try:
        return get_store().export_audit(case_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
