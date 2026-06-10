# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from dynamo.llm import AicPerfConfig, KvRouter, KvRouterConfig
from dynamo.runtime import DistributedRuntime

from .preprocessor import Preprocessor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouterDecision:
    worker_id: int
    dp_rank: int
    overlap_blocks: int


@dataclass(frozen=True)
class RouterServer:
    worker_id: int
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Router:
    """Reusable Dynamo KV router facade for external services.

    ``KvRouter`` discovers backend workers through Dynamo runtime state.  This
    facade adds an externally managed allow-list so services such as TRT-LLM can
    add/delete live servers without mutating Dynamo discovery.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        block_size: int,
        runtime: DistributedRuntime | None = None,
        kv_router_config: KvRouterConfig | None = None,
        aic_perf_config: AicPerfConfig | None = None,
        discovery_backend: str | None = None,
        request_plane: str | None = None,
        event_plane: str | None = None,
        preprocessor: Preprocessor | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.block_size = block_size
        self.runtime = runtime
        self.discovery_backend = discovery_backend
        self.request_plane = request_plane
        self.event_plane = event_plane
        self.kv_router_config = kv_router_config or KvRouterConfig()
        self.aic_perf_config = aic_perf_config
        self.preprocessor = preprocessor

        self._kv_router: KvRouter | None = None
        self._servers: dict[int, RouterServer] = {}
        self._lock = asyncio.Lock()

    @property
    def kv_router(self) -> KvRouter:
        if self._kv_router is None:
            raise RuntimeError("Router is not initialized; call initialize() first")
        return self._kv_router

    async def initialize(self) -> None:
        if self._kv_router is not None:
            return
        if self.runtime is None:
            loop = asyncio.get_running_loop()
            self.runtime = DistributedRuntime(
                loop,
                self.discovery_backend or os.environ.get("DYN_DISCOVERY_BACKEND", "etcd"),
                self.request_plane or os.environ.get("DYN_REQUEST_PLANE", "tcp"),
                event_plane=self.event_plane,
            )
        worker_endpoint = self.runtime.endpoint(self.endpoint)
        self._kv_router = KvRouter(
            endpoint=worker_endpoint,
            block_size=self.block_size,
            kv_router_config=self.kv_router_config,
            aic_perf_config=self.aic_perf_config,
        )

    async def add_server(
        self,
        worker_id: int,
        url: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        async with self._lock:
            self._servers[int(worker_id)] = RouterServer(
                worker_id=int(worker_id),
                url=url,
                metadata=dict(metadata or {}),
            )

    async def delete_server(self, worker_id: int, purge_index: bool = True) -> None:
        async with self._lock:
            self._servers.pop(int(worker_id), None)
        if purge_index:
            logger.debug(
                "Dynamo KvRouter does not yet expose worker index purge; "
                "worker_id=%s was removed from the external allow-list",
                worker_id,
            )

    async def remove_server(self, worker_id: int, purge_index: bool = True) -> None:
        await self.delete_server(worker_id, purge_index=purge_index)

    def list_servers(self) -> dict[int, RouterServer]:
        return dict(self._servers)

    async def best_worker(
        self,
        token_ids: list[int],
        router_config_override: Any = None,
        request_id: str | None = None,
        update_indexer: bool = False,
        block_mm_infos: Any = None,
        lora_name: str | None = None,
        routing_constraints: Any = None,
        allowed_worker_ids: list[int] | None = None,
    ) -> RouterDecision:
        await self.initialize()
        async with self._lock:
            active_worker_ids = set(self._servers)
        if allowed_worker_ids is None:
            allowed_worker_ids = list(active_worker_ids)
        else:
            requested = {int(worker_id) for worker_id in allowed_worker_ids}
            missing = requested - active_worker_ids
            if missing:
                raise RuntimeError(
                    f"allowed_worker_ids contains inactive workers: {sorted(missing)}"
                )
            allowed_worker_ids = list(requested)
        if not allowed_worker_ids:
            raise RuntimeError("No externally added servers are available for routing")

        worker_id, dp_rank, overlap_blocks = await self.kv_router.best_worker(
            [int(t) for t in token_ids],
            router_config_override=router_config_override,
            request_id=request_id,
            update_indexer=update_indexer,
            block_mm_infos=block_mm_infos,
            lora_name=lora_name,
            routing_constraints=routing_constraints,
            allowed_worker_ids=allowed_worker_ids,
        )
        worker_id = int(worker_id)
        async with self._lock:
            if worker_id not in self._servers:
                raise RuntimeError(
                    f"Dynamo selected worker_id={worker_id}, but it is not in the active server set"
                )
        return RouterDecision(int(worker_id), int(dp_rank), int(overlap_blocks))

    async def best_worker_for_request(
        self,
        request: Any,
        *,
        request_id: str | None = None,
        update_indexer: bool = False,
    ) -> tuple[RouterDecision, dict[str, Any]]:
        if self.preprocessor is None:
            raise RuntimeError("best_worker_for_request requires a Preprocessor")
        preprocessed = await self.preprocessor.preprocess_openai_request(request)
        decision = await self.best_worker(
            preprocessed.token_ids,
            request_id=request_id,
            update_indexer=update_indexer,
        )
        return decision, preprocessed.preprocessed_request

    async def generate(self, request: dict[str, Any]):
        await self.initialize()
        async with self._lock:
            allowed_worker_ids = list(self._servers)
        if not allowed_worker_ids:
            raise RuntimeError("No externally added servers are available for routing")

        request = _wrap_preprocessed_request(request)
        routing = dict(request.get("routing") or {})
        routing["allowed_worker_ids"] = allowed_worker_ids
        request["routing"] = routing
        async for output in await self.kv_router.generate_from_request(request):
            yield output

    async def get_overlap_scores(self, request: dict[str, Any]):
        await self.initialize()
        return await self.kv_router.get_overlap_scores(
            request["token_ids"],
            request.get("router_config_override"),
            request.get("block_mm_infos"),
            request.get("lora_name"),
            request.get("include_shared", True),
        )

    async def free(self, request_id: str) -> None:
        await self.initialize()
        await self.kv_router.free(request_id)

    async def mark_prefill_complete(self, request_id: str) -> None:
        await self.initialize()
        await self.kv_router.mark_prefill_complete(request_id)


def _wrap_preprocessed_request(request: dict[str, Any]) -> dict[str, Any]:
    routing = request.get("routing")
    dp_rank = request.get("dp_rank")
    if routing is None and dp_rank is not None:
        routing = {"dp_rank": dp_rank}

    return {
        "model": request.get("model", "unknown"),
        "token_ids": request["token_ids"],
        "stop_conditions": request.get("stop_conditions", {}),
        "sampling_options": request.get("sampling_options", {}),
        "output_options": request.get("output_options", {}),
        "eos_token_ids": request.get("eos_token_ids", []),
        "annotations": request.get("annotations", []),
        "routing": routing,
        "router_config_override": request.get("router_config_override"),
        "prefill_result": request.get("prefill_result"),
        "bootstrap_info": request.get("bootstrap_info"),
        "extra_args": request.get("extra_args"),
        "mm_processor_kwargs": request.get("mm_processor_kwargs"),
    }
