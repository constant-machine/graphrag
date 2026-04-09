# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing validate_config_names definition."""

import asyncio
import logging
import sys
from typing import TYPE_CHECKING

from graphrag_llm.completion import create_completion
from graphrag_llm.embedding import create_embedding

import graphrag.config.defaults as defs
from graphrag.config.enums import IndexingMethod
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.workflows.factory import PipelineFactory

if TYPE_CHECKING:
    from graphrag_llm.types import LLMEmbeddingResponse

logger = logging.getLogger(__name__)


def validate_config_names(
    parameters: GraphRagConfig,
    *,
    method: IndexingMethod | str = IndexingMethod.Standard,
    is_update_run: bool = False,
) -> None:
    """Validate config file for model deployment name typos, by running a quick test message for each."""
    _warn_on_aggressive_concurrency(
        parameters,
        workflows=_resolve_workflows(parameters, method=method, is_update_run=is_update_run),
    )
    for id, config in parameters.completion_models.items():
        llm = create_completion(config)
        try:
            llm.completion(messages="This is an LLM connectivity test. Say Hello World")
            logger.info("LLM Config Params Validated")
        except Exception as e:  # noqa: BLE001
            logger.error(f"LLM configuration error detected.\n{e}")  # noqa
            print(f"Failed to validate language model ({id}) params", e)  # noqa: T201
            sys.exit(1)
    for id, config in parameters.embedding_models.items():
        embed_llm = create_embedding(config)
        try:
            response = asyncio.run(
                embed_llm.embedding_async(
                    input=["This is an LLM Embedding Test String"]
                )
            )
            logger.info("Embedding LLM Config Params Validated")

            if id == parameters.embed_text.embedding_model_id:
                _sync_vector_store_dimensions(parameters, response)

        except Exception as e:  # noqa: BLE001
            logger.error(f"Embedding configuration error detected.\n{e}")  # noqa
            print(f"Failed to validate embedding model ({id}) params", e)  # noqa: T201
            sys.exit(1)


def _resolve_workflows(
    parameters: GraphRagConfig,
    *,
    method: IndexingMethod | str,
    is_update_run: bool,
) -> list[str]:
    if parameters.workflows is not None:
        return parameters.workflows

    resolved_method = method.value if isinstance(method, IndexingMethod) else method
    if is_update_run:
        resolved_method = f"{resolved_method}-update"

    return PipelineFactory.pipelines.get(resolved_method, [])


def _get_active_model_ids(
    parameters: GraphRagConfig,
    *,
    workflows: list[str],
) -> set[str]:
    model_ids: set[str] = set()

    for workflow in workflows:
        if workflow == "extract_graph":
            model_ids.add(parameters.extract_graph.completion_model_id)
            model_ids.add(parameters.summarize_descriptions.completion_model_id)
        elif workflow == "extract_covariates" and parameters.extract_claims.enabled:
            model_ids.add(parameters.extract_claims.completion_model_id)
        elif workflow == "create_community_reports":
            model_ids.add(parameters.community_reports.completion_model_id)
        elif workflow == "create_community_reports_text":
            model_ids.add(parameters.community_reports.completion_model_id)
        elif workflow == "generate_text_embeddings":
            model_ids.add(parameters.embed_text.embedding_model_id)
        elif workflow == "update_entities_relationships":
            model_ids.add(parameters.summarize_descriptions.completion_model_id)
        elif workflow == "update_text_embeddings":
            model_ids.add(parameters.embed_text.embedding_model_id)

    return model_ids


def _warn_on_aggressive_concurrency(
    parameters: GraphRagConfig,
    *,
    workflows: list[str],
) -> None:
    if (
        parameters.concurrent_requests
        <= defs.SAFE_CONCURRENT_REQUESTS_WITHOUT_RATE_LIMIT
    ):
        return

    active_model_ids = _get_active_model_ids(parameters, workflows=workflows)
    model_ids_without_limits = [
        model_id
        for model_id, config in (
            list(parameters.completion_models.items())
            + list(parameters.embedding_models.items())
        )
        if model_id in active_model_ids and config.rate_limit is None
    ]
    if model_ids_without_limits:
        logger.warning(
            "concurrent_requests=%d with no rate_limit configured for models %s. "
            "Provider-side throttling or 5xx errors are more likely; consider lowering "
            "concurrent_requests or configuring rate_limit.",
            parameters.concurrent_requests,
            ", ".join(sorted(model_ids_without_limits)),
        )


def _sync_vector_store_dimensions(
    parameters: GraphRagConfig,
    response: "LLMEmbeddingResponse",
) -> None:
    """Sync vector store dimensions to match the actual embedding model output."""
    detected = len(response.first_embedding)
    if detected == 0:
        return

    configured = parameters.vector_store.vector_size
    if detected == configured:
        return

    logger.warning(
        "Embedding model produces %d-dimensional vectors but vector_size is "
        "configured as %d. Overriding vector_size to match the model.",
        detected,
        configured,
    )
    parameters.vector_store.vector_size = detected
    for schema in parameters.vector_store.index_schema.values():
        schema.vector_size = detected
