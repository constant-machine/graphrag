# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License

from graphrag.callbacks.console_workflow_callbacks import ConsoleWorkflowCallbacks
from graphrag.index.typing.pipeline_run_result import PipelineRunResult


def test_pipeline_end_prints_failed_for_error_results(capsys) -> None:
    callbacks = ConsoleWorkflowCallbacks()

    callbacks.pipeline_end(
        [
            PipelineRunResult(
                workflow="extract_graph",
                result=None,
                state={},
                error=RuntimeError("boom"),
            )
        ]
    )

    assert capsys.readouterr().out.strip() == "Pipeline failed"


def test_pipeline_end_prints_complete_for_success_results(capsys) -> None:
    callbacks = ConsoleWorkflowCallbacks()

    callbacks.pipeline_end(
        [
            PipelineRunResult(
                workflow="extract_graph",
                result={"ok": True},
                state={},
                error=None,
            )
        ]
    )

    assert capsys.readouterr().out.strip() == "Pipeline complete"
