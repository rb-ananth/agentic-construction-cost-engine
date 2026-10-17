# Agentic Construction Cost Engine

A local-first, agentic RAG pipeline for classifying construction invoices,
checking specification compliance, and validating budget availability. The
system combines persistent vector search, strict Pydantic contracts, structured
tool invocation, and a transaction-aware budget ledger to produce a
machine-readable audit report.

> **Public-data notice:** The checked-in invoice sample, audit report, and
> budget ledger contain vendor names and financial amounts. They are included
> intentionally as project demo data. Review and replace them before publishing
> this project with any different or real business data.

## Capabilities

- **Persistent RAG:** `SentenceTransformer` embeds cost specification rules and
  Chroma stores vectors locally in `data/vector_store/`.
- **Geometric OOD detection:** The nearest cosine distance is compared against a
  configurable cutoff (default `0.76`). Results beyond the cutoff are flagged
  for human review and are not sent to the budget tool.
- **Structured tools:** The orchestrator exposes a specification lookup and a
  budget-ledger check as LangChain tools. Its deterministic local planner runs
  without a chat service; an optional Ollama model can drive the tool-calling
  loop.
- **Validated audit contracts:** Pydantic validates input transactions, rule
  metadata, ledger state, and the final batch report. Invalid rows are included
  in the report with field-level validation errors.
- **Persistent, idempotent budget checks:** Approved invoices are recorded in
  `data/budget_ledger.json`; rerunning an already-posted invoice does not charge
  the budget a second time. Budget checks use the remaining category balance.
- **Operational logging:** Python's standard `logging` module records pipeline
  activity without printing invoice contents to the console.

## Architecture

```text
data/invoices.csv
       |
       v
Pydantic transaction validation
       |
       v
CostAuditOrchestrator
   |               |
   | Tool A        | Tool B (only for in-distribution matches)
   v               v
Chroma RAG      Persistent budget ledger
   |               |
   +-------+-------+
           v
   Pydantic audit report
           |
           v
data/audit_report.json
```

The RAG engine reads `data/spec_rules.txt`, extracts each rule's cost code and
section, and stores this metadata with its vector. Every invoice's nearest
distance, cutoff, matched rule, and tool calls are recorded for traceability.

## Requirements

- Python 3.14.8 (the version used to verify this workspace)
- The pinned packages in [`requirements.txt`](./requirements.txt)
- Network access on first run to download the `all-MiniLM-L6-v2` embedding
  model, unless it is already cached
- Optional: [Ollama](https://ollama.com/) and a tool-capable model for
  model-directed tool selection

## Quick start

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python agent.py
```

The first run downloads the sentence-transformer model and creates the
persistent Chroma index. Subsequent runs reuse the local vector store and
ledger.

### Optional Ollama tool-calling mode

Start Ollama, pull a model that supports tool calling, then set `OLLAMA_MODEL`
to the model name:

```bash
ollama pull qwen3
export OLLAMA_MODEL=qwen3
python agent.py
```

Without `OLLAMA_MODEL`, the local planner calls the same registered tools in a
deterministic order: retrieve and evaluate the matching rule, then check the
ledger only if the match is within the distance cutoff. Model-directed mode
uses a bounded tool-call loop and rejects unknown tools or invalid tool
sequences.

## Inputs and outputs

| Path | Purpose |
| --- | --- |
| `data/invoices.csv` | Invoice rows: `invoice_id`, `vendor_name`, `raw_description`, and positive `amount` |
| `data/spec_rules.txt` | Specification rules with cost codes |
| `data/budget_ledger.json` | Category budgets and approved invoice commitments |
| `data/audit_report.json` | Complete Pydantic-validated audit output |
| `data/vector_store/` | Generated local Chroma database; excluded from Git |

The cost-code budgets are initialized in `agent.py` as sample values for
`MAT-03`, `EQ-31`, and `LAB-09`. Change these values to match an authorized
project ledger before production use. The ledger uses invoice IDs for
idempotency; reusing an invoice ID with different cost code or amount is an
error.

## Distance cutoff calibration

The default cutoff is `0.76` for cosine distance. It separates the three
in-scope sample invoices from the unmatched security/cleanup invoice in the
checked-in dataset. This is a demonstration value, not a universal confidence
threshold. Calibrate the cutoff against representative, labeled invoices and
specification rules before operational use.

## Tests

Run the focused test suite:

```bash
python -m unittest discover -s tests -v
```

Tests cover strict input contracts, budget arithmetic and idempotency,
out-of-distribution routing, exported invalid-input reports, and the model
tool-calling loop.

## Security and operational notes

- Do not commit `.env` files, credentials, local environments, or Chroma's
  generated vector database. These are excluded by `.gitignore`.
- The checked-in invoice, report, and ledger data are included as requested for
  this public demo; replace them before using the repository with confidential
  project data.
- Protect the ledger and report with appropriate access controls when running
  on real project data.
- A vector match is a classification aid, not a substitute for professional
  review. Route low-confidence or out-of-distribution items to an authorized
  reviewer.
