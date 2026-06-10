# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.router.router import Router, RouterServer


@pytest.mark.asyncio
async def test_router_external_server_membership():
    router = Router(endpoint="dynamo.trtllm.generate", block_size=128)

    await router.add_server(7, url="http://server:8000", metadata={"role": "gen"})
    assert router.list_servers() == {
        7: RouterServer(7, "http://server:8000", {"role": "gen"})
    }

    await router.delete_server(7)
    assert router.list_servers() == {}


@pytest.mark.asyncio
async def test_router_best_worker_requires_active_servers(monkeypatch):
    router = Router(endpoint="dynamo.trtllm.generate", block_size=128)

    async def fake_initialize():
        router._kv_router = object()

    monkeypatch.setattr(router, "initialize", fake_initialize)

    with pytest.raises(RuntimeError, match="No externally added servers"):
        await router.best_worker([1, 2, 3])
