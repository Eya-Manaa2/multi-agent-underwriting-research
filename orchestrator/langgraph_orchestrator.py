"""
LangGraph Orchestrator: Flexible coordination of multi-agent workflows
Demonstrates different collaboration patterns: sequential, parallel, conditional
"""

from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
import operator
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agents.claims_agent import ClaimsAgent
from agents.compliance_agent import ComplianceAgent
from agents.customer_agent import CustomerAgent
from agents.decision_agent_enhanced import EnhancedDecisionAgent
from agents.document_agent import DocumentAgent
from agents.explanation_agent import ExplanationAgent
from agents.fairness_agent import FairnessAgent
from agents.external_agent import ExternalAgent
from agents.intake_agent import IntakeAgent
from agents.llm_document_agent import LLMDocumentAgent
from agents.risk_agent import RiskAgent
from agents.uncertainty_agent import UncertaintyAgent
from orchestrator.underwriting_report import build_underwriting_report

# Define the state structure
class AgentState(TypedDict):
    """State passed between agents"""
    application: dict
    intake_result: dict
    customer_result: dict
    risk_result: dict
    uncertainty_result: dict
    document_result: dict
    llm_document_result: dict
    external_result: dict
    claims_result: dict
    decision_result: dict
    fairness_result: dict
    compliance_result: dict
    explanation_result: dict
    report_result: dict
    decision_mode: str
    document_text: str
    document_name: str
    document_provided: bool
    document_size_bytes: int
    document_status: str
    document_bundle: dict
    enable_llm_document_analysis: bool
    messages: Annotated[Sequence[str], operator.add]
    collaboration_mode: str  # sequential, parallel, conditional
    disabled_agents: Sequence[str]  # research ablation axis: nodes turned into no-ops


# Node names, in the order of the consolidated report. Used to validate the
# ``disabled_agents`` ablation list.
AGENT_NODES = (
    'intake',
    'customer',
    'risk',
    'uncertainty',
    'document',
    'external',
    'claims',
    'decision',
    'fairness',
    'compliance',
    'explanation',
)


class LangGraphOrchestrator:
    """Orchestrates multi-agent workflows using LangGraph"""

    def __init__(self):
        self.intake_agent = IntakeAgent()
        self.customer_agent = CustomerAgent()
        self.risk_agent = RiskAgent()
        self.uncertainty_agent = UncertaintyAgent()
        self.document_agent = DocumentAgent()
        self.llm_document_agent = LLMDocumentAgent()
        self.external_agent = ExternalAgent()
        self.claims_agent = ClaimsAgent()
        self.fairness_agent = FairnessAgent()
        self.compliance_agent = ComplianceAgent()
        self.explanation_agent = ExplanationAgent()
        print("[OK] LangGraph Orchestrator initialized")

    def _guard(self, name, node):
        """Turn a node into a no-op when its name is in ``disabled_agents``.

        Used only by the research ablation study: the graph keeps its structure, but
        an ablated agent contributes nothing to the state (its result field stays
        empty), so the effect of removing that agent can be measured on real cases.
        """
        def guarded(state):
            if name in (state.get("disabled_agents") or ()):
                return {}
            return node(state)

        return guarded

    def build_sequential_graph(self):
        """Build a sequential workflow graph"""
        workflow = StateGraph(AgentState)

        # Define nodes (agent functions)
        workflow.add_node("intake", self._guard("intake", self.intake_node))
        workflow.add_node("customer", self._guard("customer", self.customer_node))
        workflow.add_node("risk", self._guard("risk", self.risk_node))
        workflow.add_node("uncertainty", self._guard("uncertainty", self.uncertainty_node))
        workflow.add_node("document", self._guard("document", self.document_node))
        workflow.add_node("external", self._guard("external", self.external_node))
        workflow.add_node("claims", self._guard("claims", self.claims_node))
        workflow.add_node("decision", self._guard("decision", self.decision_node))
        workflow.add_node("fairness", self._guard("fairness", self.fairness_node))
        workflow.add_node("compliance", self._guard("compliance", self.compliance_node))
        workflow.add_node("explanation", self._guard("explanation", self.explanation_node))

        # Define edges (sequential flow)
        workflow.set_entry_point("intake")
        workflow.add_edge("intake", "customer")
        workflow.add_edge("customer", "risk")
        workflow.add_edge("risk", "uncertainty")
        workflow.add_edge("uncertainty", "document")
        workflow.add_edge("document", "external")
        workflow.add_edge("external", "claims")
        workflow.add_edge("claims", "decision")
        workflow.add_edge("decision", "fairness")
        workflow.add_edge("fairness", "compliance")
        workflow.add_edge("compliance", "explanation")
        workflow.add_edge("explanation", END)

        return workflow.compile()

    def build_parallel_graph(self):
        """Build a parallel workflow graph (some agents run in parallel)"""
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("intake", self._guard("intake", self.intake_node))
        workflow.add_node("customer", self._guard("customer", self.customer_node))
        workflow.add_node("risk", self._guard("risk", self.risk_node))
        workflow.add_node("uncertainty", self._guard("uncertainty", self.uncertainty_node))
        workflow.add_node("document", self._guard("document", self.document_node))
        workflow.add_node("external", self._guard("external", self.external_node))
        workflow.add_node("claims", self._guard("claims", self.claims_node))
        workflow.add_node("decision", self._guard("decision", self.decision_node))
        workflow.add_node("fairness", self._guard("fairness", self.fairness_node))
        workflow.add_node("compliance", self._guard("compliance", self.compliance_node))
        workflow.add_node("explanation", self._guard("explanation", self.explanation_node))

        # Entry point
        workflow.set_entry_point("intake")

        # After intake, customer runs in parallel with other agents
        workflow.add_edge("intake", "customer")
        workflow.add_edge("intake", "document")
        workflow.add_edge("intake", "external")
        workflow.add_edge("intake", "claims")

        # All must complete before risk (which needs customer data)
        workflow.add_edge("customer", "risk")
        workflow.add_edge("document", "risk")
        workflow.add_edge("external", "risk")
        workflow.add_edge("claims", "risk")

        # Final decision (uncertainty qualifies the model estimate first)
        workflow.add_edge("risk", "uncertainty")
        workflow.add_edge("uncertainty", "decision")
        workflow.add_edge("decision", "fairness")
        workflow.add_edge("fairness", "compliance")
        workflow.add_edge("compliance", "explanation")
        workflow.add_edge("explanation", END)

        return workflow.compile()

    # Node functions
    def intake_node(self, state: AgentState):
        """Intake agent node"""
        result = self.intake_agent.process(state["application"])
        return {
            "intake_result": result,
            "messages": [f"Intake: {result['status']}"],
        }

    def customer_node(self, state: AgentState):
        """Customer agent node"""
        result = self.customer_agent.process(state["application"], state.get("intake_result", {}))
        return {
            "customer_result": result,
            "messages": [f"Customer: Bonus-malus band {result['bonus_malus_band']}"],
        }

    def risk_node(self, state: AgentState):
        """Risk agent node"""
        result = self.risk_agent.process(
            state["application"],
            state.get("intake_result", {}),
            state.get("customer_result", {})
        )
        return {
            "risk_result": result,
            "messages": [f"Risk: Decision {result['decision']} (confidence {result['confidence']:.2f})"],
        }

    def uncertainty_node(self, state: AgentState):
        """Uncertainty agent node"""
        result = self.uncertainty_agent.process(
            state["application"], state.get("risk_result", {})
        )
        return {
            "uncertainty_result": result,
            "messages": [
                f"Uncertainty: {result['level']} "
                f"(index {result['predictive_uncertainty']})"
            ],
        }

    def document_node(self, state: AgentState):
        """Run local document checks and the optional grounded LLM review."""
        result = self.document_agent.process(
            state["application"],
            state.get("risk_result", {}),
            document_text=state.get("document_text"),
            document_name=state.get("document_name"),
            document_provided=state.get("document_provided"),
            document_size_bytes=state.get("document_size_bytes"),
            document_status=state.get("document_status"),
            document_bundle=state.get("document_bundle"),
        )
        llm_result = {'status': 'disabled'}
        messages = [f"Document: {len(result['exclusions'])} exclusions"]
        if state.get("enable_llm_document_analysis", False):
            bundle = state.get("document_bundle") or {}
            llm_document_text = (
                bundle.get("combined_text") or state.get("document_text") or ""
            ).strip()
            uploaded_document = result.get("uploaded_document", {})
            if uploaded_document.get("scope_status") != "supported":
                llm_result = {
                    "status": "skipped_out_of_scope",
                    "reason": uploaded_document.get("scope_reason"),
                }
                messages.append(
                    "LLM document review skipped: document scope is not supported"
                )
            elif not llm_document_text:
                llm_result = {
                    "status": "skipped_no_text",
                    "reason": "Aucun texte extractible dans les pièces jointes.",
                }
                messages.append("LLM document review skipped: no extractable text")
            else:
                try:
                    llm_result = self.llm_document_agent.analyze(llm_document_text)
                    messages.append(
                        f"LLM document review: {len(llm_result['findings'])} cited findings"
                    )
                except (ValueError, RuntimeError) as error:
                    # The LLM review is secondary: a missing key or provider failure
                    # must not abort the underwriting analysis.
                    llm_result = {
                        "status": "unavailable",
                        "reason": str(error),
                    }
                    messages.append(f"LLM document review unavailable: {error}")
        return {
            "document_result": result,
            "llm_document_result": llm_result,
            "messages": messages,
        }

    def external_node(self, state: AgentState):
        """External agent node"""
        result = self.external_agent.process(
            state["application"], state.get("document_bundle")
        )
        return {
            "external_result": result,
            "messages": [f"External: {len(result['external_factors'])} factors"],
        }

    def claims_node(self, state: AgentState):
        """Claims agent node"""
        result = self.claims_agent.process(state["application"])
        return {
            "claims_result": result,
            "messages": [f"Claims: Severity {result['severity_assessment']}"],
        }

    def decision_node(self, state: AgentState):
        """Decision agent node"""
        all_results = {
            'intake': state.get("intake_result", {}),
            'customer': state.get("customer_result", {}),
            'risk': state.get("risk_result", {}),
            'uncertainty': state.get("uncertainty_result", {}),
            'document': state.get("document_result", {}),
            'external': state.get("external_result", {}),
            'claims': state.get("claims_result", {})
        }
        result = EnhancedDecisionAgent(mode=state["decision_mode"]).process(
            state["application"], all_results
        )
        return {
            "decision_result": result,
            "messages": [
                f"Decision: {result['final_decision']} "
                f"(risk score {result['final_risk_score']:.1f})"
            ],
        }

    def fairness_node(self, state: AgentState):
        result = self.fairness_agent.process(
            state["application"], state["decision_result"]
        )
        return {
            "fairness_result": result,
            "messages": [f"Fairness review required: {result['review_required']}"],
        }

    def compliance_node(self, state: AgentState):
        all_results = {
            'intake': state.get("intake_result", {}),
            'customer': state.get("customer_result", {}),
            'risk': state.get("risk_result", {}),
            'document': state.get("document_result", {}),
            'external': state.get("external_result", {}),
            'claims': state.get("claims_result", {}),
            'fairness': state.get("fairness_result", {}),
        }
        result = self.compliance_agent.process(
            state["application"], state["decision_result"], all_results
        )
        return {
            "compliance_result": result,
            "messages": [f"Compliance: {result['compliance_status']}"],
        }

    def explanation_node(self, state: AgentState):
        all_results = {
            'intake': state.get("intake_result", {}),
            'customer': state.get("customer_result", {}),
            'risk': state.get("risk_result", {}),
            'document': state.get("document_result", {}),
            'external': state.get("external_result", {}),
            'claims': state.get("claims_result", {}),
            'fairness': state.get("fairness_result", {}),
            'compliance': state.get("compliance_result", {}),
        }
        result = self.explanation_agent.process(
            state["application"], state["decision_result"], all_results
        )
        return {
            "explanation_result": result,
            "messages": ["Explanation: generated for customer, regulator and auditor"],
        }

    def run_workflow(
        self,
        application,
        mode="sequential",
        decision_mode="human_in_the_loop",
        document_text=None,
        document_name=None,
        document_provided=None,
        document_size_bytes=None,
        document_status=None,
        document_bundle=None,
        enable_llm_document_analysis=False,
        disabled_agents=None,
    ):
        """Run the workflow with specified collaboration mode"""

        if decision_mode not in {"automated", "human_in_the_loop"}:
            raise ValueError(f"Unknown decision mode: {decision_mode}")
        if document_text is not None and not isinstance(document_text, str):
            raise TypeError('document_text must be text')
        if document_name is not None and not isinstance(document_name, str):
            raise TypeError('document_name must be text')
        if document_provided is not None and not isinstance(document_provided, bool):
            raise TypeError('document_provided must be a boolean')
        if document_size_bytes is not None and (
            isinstance(document_size_bytes, bool)
            or not isinstance(document_size_bytes, int)
            or document_size_bytes < 0
        ):
            raise TypeError('document_size_bytes must be a non-negative integer')
        if document_status is not None and not isinstance(document_status, str):
            raise TypeError('document_status must be text')
        if document_bundle is not None and not isinstance(document_bundle, dict):
            raise TypeError('document_bundle must be a dictionary')
        if not isinstance(enable_llm_document_analysis, bool):
            raise TypeError('enable_llm_document_analysis must be a boolean')
        disabled = tuple(disabled_agents or ())
        unknown = [name for name in disabled if name not in AGENT_NODES]
        if unknown:
            raise ValueError(f"Unknown agent(s) to ablate: {unknown}")
        validation = self.intake_agent.process(application)
        if validation["status"] != "validated":
            details = "; ".join(validation["errors"])
            raise ValueError(f"Application cannot be evaluated: {details}")

        if mode == "sequential":
            graph = self.build_sequential_graph()
        elif mode == "parallel":
            graph = self.build_parallel_graph()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Initialize state
        initial_state = {
            "application": application,
            "intake_result": {},
            "customer_result": {},
            "risk_result": {},
            "uncertainty_result": {},
            "document_result": {},
            "llm_document_result": {},
            "external_result": {},
            "claims_result": {},
            "decision_result": {},
            "fairness_result": {},
            "compliance_result": {},
            "explanation_result": {},
            "report_result": {},
            "decision_mode": decision_mode,
            "document_text": document_text,
            "document_name": document_name,
            "document_provided": document_provided,
            "document_size_bytes": document_size_bytes,
            "document_status": document_status,
            "document_bundle": document_bundle,
            "enable_llm_document_analysis": enable_llm_document_analysis,
            "messages": [],
            "collaboration_mode": mode,
            "disabled_agents": disabled,
        }

        # Run the graph
        final_state = graph.invoke(initial_state)
        # Extracted document text is stripped from the result: only metadata,
        # per-file statuses and the product checklist are kept for audit.
        final_state.pop("document_text", None)
        final_state.pop("document_bundle", None)
        # Consolidate every agent output into a single auditable report.
        final_state["report_result"] = build_underwriting_report(final_state)

        return final_state


if __name__ == "__main__":
    orchestrator = LangGraphOrchestrator()

    # Test with a sample application (real freMTPL2 variable schema)
    sample_application = {
        'application_id': 'TEST_001',
        'product_line': 'auto',
        'driver_age': 35,
        'bonus_malus': 60.0,
        'vehicle_age': 5,
        'vehicle_power': 6,
        'vehicle_gas': 'Regular',
        'density': 1200.0,
        'area': 'C',
    }

    print("=" * 60)
    print("Testing Sequential Workflow (LangGraph)")
    print("=" * 60)
    result_seq = orchestrator.run_workflow(sample_application, mode="sequential")
    print(f"\nFinal Decision: {result_seq['decision_result']['final_decision']}")
    print(f"Risk Score: {result_seq['decision_result']['final_risk_score']:.1f}")
    print(f"Reasons: {result_seq['decision_result']['reasons']}")
    print(f"\nAgent Execution Sequence:")
    for msg in result_seq['messages']:
        print(f"  - {msg}")

    print("\n[INFO] Parallel workflow requires concurrent state updates - documented for future implementation")
