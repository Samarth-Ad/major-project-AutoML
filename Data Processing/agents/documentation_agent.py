"""
agents/documentation_agent.py
------------------------------
Documentation Agent — generates a comprehensive report.md after
the pipeline finishes.

The report includes:
  - Dataset analysis summary
  - Pipeline decisions with justifications
  - Each step's reasoning (from agent results)
  - Model selection rationale
  - Validation metrics
  - Architecture diagram
"""

from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from utils.logger import PipelineLogger

OUTPUTS_DIR = Path("outputs")


class DocumentationAgent:
    """
    Generates ``outputs/report.md`` — a human-readable report
    explaining every decision the pipeline made.
    """

    def __init__(self) -> None:
        self._logger = PipelineLogger("agents.DocumentationAgent")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_report(
        self,
        result: Any,
        saved_paths: Dict[str, str],
    ) -> str:
        """
        Build and save report.md from pipeline result data.

        Parameters
        ----------
        result
            The PipelineResult returned by MasterAgent.run().
        saved_paths
            Dict of output paths (processed_csv, trained_model, etc.)

        Returns
        -------
        str  — absolute path to the saved report.md
        """
        self._logger.info("Generating documentation report ...")

        sections = []
        sections.append(self._header(result))
        sections.append(self._dataset_analysis(result))
        sections.append(self._pipeline_decisions(result))
        sections.append(self._step_details(result))
        sections.append(self._model_info(result))
        sections.append(self._output_files(saved_paths))
        sections.append(self._architecture_diagram())
        sections.append(self._footer(result))

        report = "\n\n".join(s for s in sections if s)

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = OUTPUTS_DIR / "report.md"
        report_path.write_text(report, encoding="utf-8")

        self._logger.info(f"Report saved -> {report_path}")
        return str(report_path)

    # ------------------------------------------------------------------
    # Private: build each section
    # ------------------------------------------------------------------

    def _header(self, result: Any) -> str:
        status = "✅ SUCCESS" if result.success else "❌ FAILED"
        return (
            f"# Agentic AutoML Pipeline — Report\n\n"
            f"| Property | Value |\n"
            f"|----------|-------|\n"
            f"| **Status** | {status} |\n"
            f"| **Pipeline ID** | `{result.pipeline_id}` |\n"
            f"| **Total Time** | {result.total_elapsed_s:.2f}s |\n"
            f"| **Steps Completed** | {result.report.successful_steps}/{result.step_count} |\n"
            f"| **Generated At** | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} |"
        )

    def _dataset_analysis(self, result: Any) -> str:
        """Build the Dataset Analysis section from the pipeline decision."""
        lines = ["## 1. Dataset Analysis"]

        decision = getattr(result, "decision", None)
        if decision is None:
            lines.append("_No dataset analysis available (decision not stored)._")
            return "\n\n".join(lines)

        profile = getattr(decision, "profile", None) or {}
        lines.append(
            f"| Property | Value |\n"
            f"|----------|-------|\n"
            f"| **Problem Type** | {getattr(decision, 'problem_type', 'unknown')} |\n"
            f"| **Target Column** | `{getattr(decision, 'target_column', 'auto-inferred')}` |\n"
            f"| **Rows** | {profile.get('n_rows', '?')} |\n"
            f"| **Columns** | {profile.get('n_cols', '?')} |\n"
            f"| **Numeric Features** | {profile.get('n_numeric', '?')} |\n"
            f"| **Null Values** | {profile.get('total_nulls', '?')} |\n"
            f"| **Class Imbalanced** | {profile.get('is_imbalanced', '?')} |"
        )

        # Skewed columns
        skewed = profile.get("skewed_columns", [])
        if skewed:
            lines.append(f"**Skewed columns:** {', '.join(f'`{c}`' for c in skewed)}")

        return "\n\n".join(lines)

    def _pipeline_decisions(self, result: Any) -> str:
        """Build the Pipeline Plan section showing step/method/justification."""
        lines = ["## 2. Pipeline Decisions"]

        decision = getattr(result, "decision", None)
        if decision is None:
            lines.append("_No pipeline decision available._")
            return "\n\n".join(lines)

        # Steps included
        steps = getattr(decision, "steps", [])
        reasoning = getattr(decision, "reasoning", {})
        skipped = getattr(decision, "skipped", {})

        if steps:
            lines.append("### Steps Included\n")
            lines.append("| # | Step | Justification |")
            lines.append("|---|------|---------------|")
            for i, step in enumerate(steps, 1):
                reason = reasoning.get(step, "Data-driven decision")
                lines.append(f"| {i} | `{step}` | {reason} |")

        # Steps skipped
        if skipped:
            lines.append("\n### Steps Skipped\n")
            lines.append("| Step | Reason |")
            lines.append("|------|--------|")
            for step, reason in skipped.items():
                lines.append(f"| `{step}` | {reason} |")

        # Models selected
        models = getattr(decision, "models", [])
        if models:
            lines.append(f"\n**Models selected:** {', '.join(f'`{m}`' for m in models)}")

        return "\n".join(lines)

    def _step_details(self, result: Any) -> str:
        """Build per-step execution details."""
        lines = ["## 3. Step Execution Details"]

        if not hasattr(result, "report") or not result.report.outcomes:
            lines.append("_No step execution data available._")
            return "\n\n".join(lines)

        lines.append("| # | Step | Status | Time | Attempts |")
        lines.append("|---|------|--------|------|----------|")

        for outcome in result.report.outcomes:
            icon = "✅" if outcome.succeeded else ("⏭️" if outcome.status == "skipped" else "❌")
            lines.append(
                f"| {outcome.step_index} "
                f"| `{outcome.step_name}` "
                f"| {icon} {outcome.status} "
                f"| {outcome.elapsed_s:.2f}s "
                f"| {outcome.attempts} |"
            )

        # Step reasoning from stored results
        step_results = getattr(result, "step_results", {})
        if step_results:
            lines.append("\n### Step Reasoning\n")
            for step_name, step_result in step_results.items():
                if isinstance(step_result, dict):
                    reasoning = step_result.get("reasoning", "")
                    if reasoning:
                        lines.append(f"**{step_name}:** {reasoning}\n")

        return "\n".join(lines)

    def _model_info(self, result: Any) -> str:
        """Build model selection and validation section."""
        lines = ["## 4. Model Selection & Validation"]

        decision = getattr(result, "decision", None)
        if decision:
            models = getattr(decision, "models", [])
            problem = getattr(decision, "problem_type", "unknown")
            lines.append(f"**Problem type:** {problem}\n")
            if models:
                lines.append("**Models considered:**\n")
                for m in models:
                    lines.append(f"- `{m}`")

            lines.append(
                f"\n**Selection rationale:** Models were selected based on the "
                f"dataset characteristics ({problem} task). "
                f"Tree-based models (Random Forest, XGBoost, LightGBM) were "
                f"preferred for their robustness. Linear models were included "
                f"as baselines."
            )

        return "\n".join(lines)

    def _output_files(self, saved_paths: Dict[str, str]) -> str:
        """List generated output files."""
        lines = ["## 5. Output Files\n"]
        lines.append("| File | Path |")
        lines.append("|------|------|")

        file_map = {
            "processed_csv": ("Cleaned Dataset", "cleaned_data.csv"),
            "trained_model": ("Trained Model", "model.pkl"),
            "report":        ("Documentation", "report.md"),
            "metrics":       ("Metrics", "metrics.json"),
            "comparison":    ("Model Comparison", "model_comparison.csv"),
        }

        for key, (label, default) in file_map.items():
            if key in saved_paths:
                lines.append(f"| **{label}** | `{saved_paths[key]}` |")

        # Always add notebook
        lines.append(f"| **Notebook** | `outputs/pipeline.ipynb` |")
        lines.append(f"| **Pipeline Script** | `generated_code/pipeline_script.py` |")

        return "\n".join(lines)

    def _architecture_diagram(self) -> str:
        return (
            "## 6. System Architecture\n\n"
            "```\n"
            "CSV Input\n"
            "    │\n"
            "    ▼\n"
            "┌──────────────────────┐\n"
            "│   Analyzer Agent     │  ← DataUnderstandingAgent (Pass 1)\n"
            "│   (Data Profiling)   │     Detects: nulls, types, skew, imbalance\n"
            "└──────────┬───────────┘\n"
            "           │ data profile\n"
            "           ▼\n"
            "┌──────────────────────┐\n"
            "│   Planner Agent      │  ← DataUnderstandingAgent (Pass 2)\n"
            "│   (Pipeline Plan)    │     Outputs: structured JSON plan\n"
            "└──────────┬───────────┘\n"
            "           │ step list + reasoning\n"
            "           ▼\n"
            "┌──────────────────────┐\n"
            "│   Meta Agent         │  ← MasterAgent + AgentBuilder\n"
            "│   (Agent Factory)    │     Dynamically creates agents per step\n"
            "└──────────┬───────────┘\n"
            "           │ built agents\n"
            "           ▼\n"
            "┌──────────────────────────────────────────┐\n"
            "│   Dynamic Pipeline (Scheduler)           │\n"
            "│                                          │\n"
            "│   ┌─────────┐  ┌─────────┐  ┌────────┐  │\n"
            "│   │ Impute  │→ │ Encode  │→ │ Scale  │  │\n"
            "│   └─────────┘  └─────────┘  └────────┘  │\n"
            "│        │            │            │       │\n"
            "│   ┌─────────┐  ┌─────────┐  ┌────────┐  │\n"
            "│   │Feature  │→ │ Train   │→ │Evaluate│  │\n"
            "│   │Engineer │  │ Models  │  │ Models │  │\n"
            "│   └─────────┘  └─────────┘  └────────┘  │\n"
            "│                                          │\n"
            "└──────────────────┬───────────────────────┘\n"
            "                   │ results\n"
            "                   ▼\n"
            "    ┌──────────────────────────────────┐\n"
            "    │         Output Generation        │\n"
            "    │  ┌────────────┐ ┌─────────────┐  │\n"
            "    │  │ Doc Agent  │ │ Notebook    │  │\n"
            "    │  │ report.md  │ │ Generator   │  │\n"
            "    │  └────────────┘ └─────────────┘  │\n"
            "    │  ┌────────────┐ ┌─────────────┐  │\n"
            "    │  │cleaned_data│ │  model.pkl  │  │\n"
            "    │  │   .csv     │ │             │  │\n"
            "    │  └────────────┘ └─────────────┘  │\n"
            "    └──────────────────────────────────┘\n"
            "```"
        )

    def _footer(self, result: Any) -> str:
        return (
            "---\n\n"
            f"*Report generated automatically by the Agentic AutoML Pipeline "
            f"(Pipeline ID: `{result.pipeline_id}`)*"
        )
