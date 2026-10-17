from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from threading import RLock
from typing import Annotated, Any, Literal, Protocol

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from engine import CostRuleEngine

logger = logging.getLogger(__name__)

PositiveDecimal = Annotated[Decimal, Field(gt=0, allow_inf_nan=False)]
DEFAULT_BUDGETS = {
    "MAT-03": Decimal("50000"),
    "EQ-31": Decimal("40000"),
    "LAB-09": Decimal("30000"),
}
MAX_AGENT_TURNS = 6


class TransactionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    invoice_id: str = Field(min_length=1)
    vendor_name: str = Field(min_length=1)
    raw_description: str = Field(min_length=1)
    amount: PositiveDecimal


class LedgerEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    cost_code: str
    amount: PositiveDecimal


class RuleMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    cost_code: str
    category: str
    section: str | None = None
    source: str | None = None


class BudgetLedgerState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    budgets: dict[str, PositiveDecimal]
    approved_transactions: dict[str, LedgerEntry] = Field(default_factory=dict)


class AuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    invoice_id: str | None
    vendor_name: str | None
    description: str | None
    amount: Decimal | None
    cost_code: str | None
    status: Literal[
        "APPROVED & CODED",
        "REJECTED - OVER BUDGET",
        "FLAGGED - HUMAN AUDIT REQUIRED (No Matching Spec Class)",
        "INVALID INPUT",
        "ALREADY POSTED",
    ]
    is_valid: bool
    validation_errors: list[str] = Field(default_factory=list)
    matched_rule: str | None = None
    rule_metadata: RuleMetadata | None = None
    nearest_distance: float | None = None
    distance_cutoff: float | None = None
    budget: Decimal | None = None
    spent_before: Decimal | None = None
    remaining_before: Decimal | None = None
    remaining_after: Decimal | None = None
    tool_calls: list[str] = Field(default_factory=list)
    reasoning_mode: Literal["local_tool_planner", "ollama_tool_calling"]
    audit_reason: str


class AuditBatchReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["1.0"] = "1.0"
    generated_at: datetime
    total_input: int = Field(ge=0)
    valid_transactions: int = Field(ge=0)
    approved: int = Field(ge=0)
    rejected: int = Field(ge=0)
    flagged: int = Field(ge=0)
    invalid: int = Field(ge=0)
    already_posted: int = Field(ge=0)
    reports: list[AuditReport]


class RuleLookup(Protocol):
    def lookup(self, description: str) -> dict[str, Any]: ...


class BudgetLedger:
    """A small persistent, idempotent ledger for approved invoice commitments."""

    def __init__(self, ledger_path: Path) -> None:
        self.ledger_path = ledger_path
        self._lock = RLock()
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        if self.ledger_path.exists():
            self.state = BudgetLedgerState.model_validate_json(
                self.ledger_path.read_text(encoding="utf-8")
            )
        else:
            self.state = BudgetLedgerState(budgets=DEFAULT_BUDGETS.copy())
            self._save()

    def check_and_post(self, transaction: TransactionInput, cost_code: str) -> dict[str, Any]:
        with self._lock:
            if cost_code not in self.state.budgets:
                raise ValueError(f"No budget is configured for cost code {cost_code}")

            prior_entry = self.state.approved_transactions.get(transaction.invoice_id)
            if prior_entry is not None:
                if prior_entry.cost_code != cost_code or prior_entry.amount != transaction.amount:
                    raise ValueError(
                        f"Invoice {transaction.invoice_id} was already posted with different values"
                    )
                spent = self._spent(cost_code)
                budget = self.state.budgets[cost_code]
                return {
                    "status": "ALREADY POSTED",
                    "approved": True,
                    "budget": str(budget),
                    "spent_before": str(spent),
                    "remaining_before": str(budget - spent),
                    "remaining_after": str(budget - spent),
                    "amount": str(transaction.amount),
                }

            budget = self.state.budgets[cost_code]
            spent_before = self._spent(cost_code)
            remaining_before = budget - spent_before
            approved = transaction.amount <= remaining_before
            if approved:
                self.state.approved_transactions[transaction.invoice_id] = LedgerEntry(
                    cost_code=cost_code,
                    amount=transaction.amount,
                )
                self._save()

            remaining_after = (
                remaining_before - transaction.amount if approved else remaining_before
            )
            return {
                "status": "APPROVED & CODED" if approved else "REJECTED - OVER BUDGET",
                "approved": approved,
                "budget": str(budget),
                "spent_before": str(spent_before),
                "remaining_before": str(remaining_before),
                "remaining_after": str(remaining_after),
                "amount": str(transaction.amount),
            }

    def _spent(self, cost_code: str) -> Decimal:
        return sum(
            (
                entry.amount
                for entry in self.state.approved_transactions.values()
                if entry.cost_code == cost_code
            ),
            start=Decimal("0"),
        )

    def _save(self) -> None:
        temporary_path = self.ledger_path.with_name(f".{self.ledger_path.name}.tmp")
        temporary_path.write_text(
            json.dumps(self.state.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.ledger_path)


class CostAuditOrchestrator:
    """Coordinate RAG retrieval, budget validation, and auditable report output."""

    _SYSTEM_PROMPT = """You audit construction invoices using the available tools.
For each invoice, call lookup_spec_rule first and use its returned classification and
distance as the only source of cost-code truth. If in_distribution is true, call
check_budget_ledger exactly once with the invoice_id and returned cost_code. If it is
false, do not call the ledger tool and flag the invoice for human review. Do not invent
invoice values or codes. After the required tool calls, provide a concise audit summary."""

    def __init__(
        self,
        rule_engine: RuleLookup | None = None,
        ledger: BudgetLedger | None = None,
        llm: Any | None = None,
        data_dir: Path | None = None,
    ) -> None:
        self.project_root = Path(__file__).resolve().parent
        self.data_dir = data_dir or self.project_root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.rule_engine = rule_engine or CostRuleEngine()
        self.ledger = ledger or BudgetLedger(self.data_dir / "budget_ledger.json")
        self.llm = llm if llm is not None else self._configured_llm()
        self.reasoning_mode: Literal["local_tool_planner", "ollama_tool_calling"] = (
            "ollama_tool_calling" if self.llm is not None else "local_tool_planner"
        )

        self.tools = [
            StructuredTool.from_function(
                func=self.lookup_spec_rule,
                name="lookup_spec_rule",
                description=(
                    "Search local construction specification rules for an invoice "
                    "description and report its cost code and geometric distance."
                ),
            ),
            StructuredTool.from_function(
                func=self.check_budget_ledger,
                name="check_budget_ledger",
                description=(
                    "Check an invoice against the current remaining balance for the "
                    "RAG-verified cost code and record it if it fits."
                ),
            ),
        ]
        self._tools_by_name = {tool.name: tool for tool in self.tools}
        self._active_transaction: TransactionInput | None = None
        self._rag_result: dict[str, Any] | None = None

    @staticmethod
    def _configured_llm() -> Any | None:
        model_name = os.environ.get("OLLAMA_MODEL")
        if not model_name:
            return None
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model_name, temperature=0)

    def lookup_spec_rule(self, description: str) -> dict[str, Any]:
        """Tool A: perform a local persistent-vector lookup."""
        transaction = self._active_transaction
        if transaction is not None and description != transaction.raw_description:
            raise ValueError("The RAG tool must search the active invoice description")
        self._rag_result = self.rule_engine.lookup(description)
        return self._rag_result

    def check_budget_ledger(self, invoice_id: str, cost_code: str) -> dict[str, Any]:
        """Tool B: calculate remaining funds and post an approved invoice once."""
        transaction = self._active_transaction
        if transaction is None or invoice_id != transaction.invoice_id:
            raise ValueError("The ledger tool can only check the active invoice")
        rag_result = self._rag_result
        if (
            rag_result is None
            or not rag_result["in_distribution"]
            or cost_code != rag_result["cost_code"]
        ):
            raise ValueError("The ledger tool requires the cost code returned by the RAG tool")
        return self.ledger.check_and_post(transaction, cost_code)

    def audit(self, transaction: TransactionInput) -> AuditReport:
        self._active_transaction = transaction
        self._rag_result = None
        tool_results: dict[str, dict[str, Any]] = {}
        tool_calls: list[str] = []
        agent_summary = ""

        if self.llm is None:
            rag_result = self.tools[0].invoke({"description": transaction.raw_description})
            tool_results["lookup_spec_rule"] = rag_result
            tool_calls.append("lookup_spec_rule")
            if rag_result["in_distribution"]:
                ledger_result = self.tools[1].invoke(
                    {
                        "invoice_id": transaction.invoice_id,
                        "cost_code": rag_result["cost_code"],
                    }
                )
                tool_results["check_budget_ledger"] = ledger_result
                tool_calls.append("check_budget_ledger")
        else:
            tool_results, tool_calls, agent_summary = self._run_tool_calling_loop(transaction)

        rag_result = tool_results.get("lookup_spec_rule")
        if rag_result is None:
            raise RuntimeError("The agent completed without calling the required RAG tool")

        if rag_result["in_distribution"]:
            ledger_result = tool_results.get("check_budget_ledger")
            if ledger_result is None:
                raise RuntimeError("The agent completed without checking the required budget ledger")
            status = ledger_result["status"]
            reason = (
                f"RAG distance {rag_result['distance']:.4f} is within the "
                f"{rag_result['distance_cutoff']:.4f} cutoff; ledger reports {status.lower()}."
            )
            budget_values = {
                "budget": Decimal(ledger_result["budget"]),
                "spent_before": Decimal(ledger_result["spent_before"]),
                "remaining_before": Decimal(ledger_result["remaining_before"]),
                "remaining_after": Decimal(ledger_result["remaining_after"]),
            }
            cost_code = rag_result["cost_code"]
        else:
            status = "FLAGGED - HUMAN AUDIT REQUIRED (No Matching Spec Class)"
            reason = (
                f"Nearest rule distance {rag_result['distance']:.4f} exceeds the "
                f"{rag_result['distance_cutoff']:.4f} out-of-distribution cutoff."
            )
            budget_values = {}
            cost_code = None

        return AuditReport(
            invoice_id=transaction.invoice_id,
            vendor_name=transaction.vendor_name,
            description=transaction.raw_description,
            amount=transaction.amount,
            cost_code=cost_code,
            status=status,
            is_valid=True,
            matched_rule=rag_result["matched_rule"],
            rule_metadata=RuleMetadata.model_validate(rag_result["metadata"]),
            nearest_distance=rag_result["distance"],
            distance_cutoff=rag_result["distance_cutoff"],
            tool_calls=tool_calls,
            reasoning_mode=self.reasoning_mode,
            audit_reason=f"{reason} {agent_summary}".strip(),
            **budget_values,
        )

    def _run_tool_calling_loop(
        self,
        transaction: TransactionInput,
    ) -> tuple[dict[str, dict[str, Any]], list[str], str]:
        assert self.llm is not None
        model_with_tools = self.llm.bind_tools(self.tools)
        messages: list[Any] = [
            SystemMessage(content=self._SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Audit invoice {transaction.invoice_id} from {transaction.vendor_name}. "
                    f"Description: {transaction.raw_description}. "
                    f"Amount: {transaction.amount}."
                )
            ),
        ]
        tool_results: dict[str, dict[str, Any]] = {}
        tool_calls: list[str] = []
        agent_summary = ""

        for _ in range(MAX_AGENT_TURNS):
            response = model_with_tools.invoke(messages)
            messages.append(response)
            requested_calls = response.tool_calls
            if not requested_calls:
                agent_summary = str(response.content).strip()
                break

            for requested_call in requested_calls:
                tool_name = requested_call["name"]
                tool = self._tools_by_name.get(tool_name)
                if tool is None:
                    raise ValueError(f"Agent requested an unknown tool: {tool_name}")
                result = tool.invoke(requested_call["args"])
                if not isinstance(result, dict):
                    raise TypeError(f"Tool {tool_name} returned an unexpected result type")
                tool_results[tool_name] = result
                tool_calls.append(tool_name)
                logger.info("Agent invoked tool %s for invoice %s", tool_name, transaction.invoice_id)
                messages.append(
                    ToolMessage(
                        content=json.dumps(result, default=str),
                        tool_call_id=requested_call["id"],
                    )
                )
        else:
            raise RuntimeError(f"Agent exceeded the {MAX_AGENT_TURNS}-turn tool-call limit")

        return tool_results, tool_calls, agent_summary

    def run(self, invoices_path: Path | None = None) -> AuditBatchReport:
        input_path = invoices_path or self.data_dir / "invoices.csv"
        if not input_path.is_file():
            raise FileNotFoundError(f"Invoice input not found: {input_path}")

        rows = pd.read_csv(input_path, dtype=str, keep_default_na=False).to_dict(
            orient="records"
        )
        reports: list[AuditReport] = []
        for row in rows:
            try:
                transaction = self._validate_transaction(row)
            except ValidationError as error:
                errors = [
                    f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
                    for item in error.errors()
                ]
                reports.append(self._invalid_report(row, errors))
                logger.warning("Invoice input failed validation: %s", "; ".join(errors))
                continue

            logger.info(
                "Auditing invoice %s from %s",
                transaction.invoice_id,
                transaction.vendor_name,
            )
            reports.append(self.audit(transaction))

        report = AuditBatchReport(
            generated_at=datetime.now(UTC),
            total_input=len(rows),
            valid_transactions=sum(result.is_valid for result in reports),
            approved=sum(result.status == "APPROVED & CODED" for result in reports),
            rejected=sum(result.status == "REJECTED - OVER BUDGET" for result in reports),
            flagged=sum(
                result.status
                == "FLAGGED - HUMAN AUDIT REQUIRED (No Matching Spec Class)"
                for result in reports
            ),
            invalid=sum(result.status == "INVALID INPUT" for result in reports),
            already_posted=sum(result.status == "ALREADY POSTED" for result in reports),
            reports=reports,
        )
        self._write_report(report, self.data_dir / "audit_report.json")
        logger.info(
            "Audit completed: %d invoices, %d approved, %d rejected, %d flagged, %d invalid",
            report.total_input,
            report.approved,
            report.rejected,
            report.flagged,
            report.invalid,
        )
        return report

    @staticmethod
    def _validate_transaction(row: dict[str, str]) -> TransactionInput:
        raw_amount: Any = row.get("amount")
        try:
            amount: Any = Decimal(raw_amount)
        except (InvalidOperation, TypeError, ValueError):
            amount = raw_amount

        normalized_row: dict[str, Any] = dict(row)
        normalized_row["amount"] = amount
        return TransactionInput.model_validate(normalized_row)

    @staticmethod
    def _invalid_report(row: dict[str, str], errors: list[str]) -> AuditReport:
        return AuditReport(
            invoice_id=row.get("invoice_id") or None,
            vendor_name=row.get("vendor_name") or None,
            description=row.get("raw_description") or None,
            amount=None,
            cost_code=None,
            status="INVALID INPUT",
            is_valid=False,
            validation_errors=errors,
            reasoning_mode="local_tool_planner",
            audit_reason="Invoice failed the strict transaction input contract.",
        )

    @staticmethod
    def _write_report(report: AuditBatchReport, output_path: Path) -> None:
        temporary_path = output_path.with_name(f".{output_path.name}.tmp")
        temporary_path.write_text(
            report.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(output_path)


def run_agentic_pipeline() -> AuditBatchReport:
    """Run the configured invoice audit and persist its validated report."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return CostAuditOrchestrator().run()


if __name__ == "__main__":
    run_agentic_pipeline()
