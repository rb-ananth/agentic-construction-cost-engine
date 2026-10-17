from __future__ import annotations

import json
import os
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import patch

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from agent import BudgetLedger, CostAuditOrchestrator, TransactionInput


class StubRuleEngine:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    def lookup(self, description: str) -> dict[str, Any]:
        return self.result


class StubToolCallingModel:
    def __init__(self) -> None:
        self.responses = iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "lookup_spec_rule",
                            "args": {"description": "rebar supply"},
                            "id": "lookup-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "check_budget_ledger",
                            "args": {"invoice_id": "INV-LLM", "cost_code": "MAT-03"},
                            "id": "budget-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Rule matched and the remaining balance is sufficient."),
            ]
        )

    def bind_tools(self, tools: list[object]) -> StubToolCallingModel:
        return self

    def invoke(self, messages: list[object]) -> AIMessage:
        return next(self.responses)


class CostAuditTests(unittest.TestCase):
    def test_ledger_checks_remaining_balance_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            ledger = BudgetLedger(Path(temporary_directory) / "budget.json")
            first = TransactionInput(
                invoice_id="INV-1",
                vendor_name="Steel",
                raw_description="rebar supply",
                amount=Decimal("30000"),
            )
            second = TransactionInput(
                invoice_id="INV-2",
                vendor_name="Steel",
                raw_description="rebar supply",
                amount=Decimal("21000"),
            )

            approved = ledger.check_and_post(first, "MAT-03")
            rejected = ledger.check_and_post(second, "MAT-03")
            repeated = BudgetLedger(Path(temporary_directory) / "budget.json").check_and_post(
                first,
                "MAT-03",
            )

            self.assertEqual(approved["remaining_after"], "20000")
            self.assertEqual(rejected["status"], "REJECTED - OVER BUDGET")
            self.assertEqual(repeated["status"], "ALREADY POSTED")
            stored_state = json.loads((Path(temporary_directory) / "budget.json").read_text())
            self.assertEqual(list(stored_state["approved_transactions"]), ["INV-1"])

    def test_out_of_distribution_invoice_is_flagged_without_ledger_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            invoice_path = data_dir / "invoices.csv"
            invoice_path.write_text(
                "invoice_id,vendor_name,raw_description,amount\n"
                "INV-OUT,Global Supply,site security services,8000\n",
                encoding="utf-8",
            )
            rule_engine = StubRuleEngine(
                {
                    "in_distribution": False,
                    "cost_code": None,
                    "matched_rule": "nearest rule",
                    "metadata": {
                        "cost_code": "LAB-09",
                        "category": "labor",
                    },
                    "distance": 1.4,
                    "distance_cutoff": 1.2,
                }
            )
            with patch.dict(os.environ, {"OLLAMA_MODEL": ""}):
                orchestrator = CostAuditOrchestrator(
                    rule_engine=rule_engine,
                    ledger=BudgetLedger(data_dir / "budget_ledger.json"),
                    data_dir=data_dir,
                )
                report = orchestrator.run(invoice_path)

            item = report.reports[0]
            self.assertEqual(
                item.status,
                "FLAGGED - HUMAN AUDIT REQUIRED (No Matching Spec Class)",
            )
            self.assertEqual(item.tool_calls, ["lookup_spec_rule"])
            self.assertEqual(report.flagged, 1)
            self.assertEqual(item.cost_code, None)
            self.assertTrue((data_dir / "audit_report.json").is_file())

    def test_model_loop_calls_both_tools_before_finalizing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            rule_engine = StubRuleEngine(
                {
                    "in_distribution": True,
                    "cost_code": "MAT-03",
                    "matched_rule": "rebar rule",
                    "metadata": {"cost_code": "MAT-03", "category": "materials"},
                    "distance": 0.2,
                    "distance_cutoff": 0.76,
                }
            )
            orchestrator = CostAuditOrchestrator(
                rule_engine=rule_engine,
                ledger=BudgetLedger(data_dir / "budget_ledger.json"),
                llm=StubToolCallingModel(),
                data_dir=data_dir,
            )
            transaction = TransactionInput(
                invoice_id="INV-LLM",
                vendor_name="Steel",
                raw_description="rebar supply",
                amount=Decimal("100"),
            )

            result = orchestrator.audit(transaction)

            self.assertEqual(result.status, "APPROVED & CODED")
            self.assertEqual(result.reasoning_mode, "ollama_tool_calling")
            self.assertEqual(
                result.tool_calls,
                ["lookup_spec_rule", "check_budget_ledger"],
            )
            self.assertIn("remaining balance is sufficient", result.audit_reason)

    def test_transaction_contract_rejects_invalid_amount_and_extra_fields(self) -> None:
        with self.assertRaises(ValidationError):
            CostAuditOrchestrator._validate_transaction(
                {
                    "invoice_id": "INV-BAD",
                    "vendor_name": "Vendor",
                    "raw_description": "test",
                    "amount": "0",
                }
            )

    def test_invalid_input_is_included_in_the_exported_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            invoice_path = data_dir / "invoices.csv"
            invoice_path.write_text(
                "invoice_id,vendor_name,raw_description,amount\n"
                "INV-BAD,Vendor,invalid amount,0\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"OLLAMA_MODEL": ""}):
                orchestrator = CostAuditOrchestrator(
                    rule_engine=StubRuleEngine({}),
                    ledger=BudgetLedger(data_dir / "budget_ledger.json"),
                    data_dir=data_dir,
                )
                report = orchestrator.run(invoice_path)

            self.assertEqual(report.invalid, 1)
            self.assertFalse(report.reports[0].is_valid)
            self.assertTrue(report.reports[0].validation_errors)
            self.assertTrue((data_dir / "audit_report.json").is_file())

        with self.assertRaises(ValidationError):
            CostAuditOrchestrator._validate_transaction(
                {
                    "invoice_id": "INV-EXTRA",
                    "vendor_name": "Vendor",
                    "raw_description": "test",
                    "amount": "10",
                    "unrecognized": "value",
                }
            )


if __name__ == "__main__":
    unittest.main()
