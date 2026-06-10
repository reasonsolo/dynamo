# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dynamo.llm import AsyncTokenizer, fetch_model


@dataclass(frozen=True)
class PreprocessedRequest:
    token_ids: list[int]
    preprocessed_request: dict[str, Any]
    annotations: dict[str, Any]
    formatted_prompt: str | None = None


class Preprocessor:
    """Rust-backed tokenizer/preprocessor helper for external routers.

    This class intentionally stays separate from ``Router``. Dynamo's
    ``KvRouter`` routes already-tokenized requests; this helper provides the
    tokenization surface external services need before calling the router.
    """

    def __init__(self, tokenizer_path: str):
        self._tokenizer = AsyncTokenizer(tokenizer_path)

    @classmethod
    async def create(cls, model_path: str, *, ignore_weights: bool = True) -> "Preprocessor":
        tokenizer_path = model_path
        if not os.path.exists(tokenizer_path):
            tokenizer_path = await fetch_model(model_path, ignore_weights)
        return cls(tokenizer_path)

    @property
    def tokenizer_path(self) -> str:
        return self._tokenizer.tokenizer_path

    async def tokenize_text(self, text: str) -> list[int]:
        return await self._tokenizer.tokenize_text(text)

    async def tokenize_batch(self, texts: list[str]) -> list[list[int]]:
        return await self._tokenizer.tokenize_batch(texts)

    async def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        return await self._tokenizer.decode(token_ids, skip_special_tokens)

    async def preprocess_openai_request(self, request: Any) -> PreprocessedRequest:
        request_dict = _to_request_dict(request)
        token_ids = await self._extract_token_ids(request_dict)
        preprocessed = {
            "model": request_dict.get("model", "unknown"),
            "token_ids": token_ids,
            "stop_conditions": request_dict.get("stop_conditions", {}),
            "sampling_options": request_dict.get("sampling_options", {}),
            "output_options": request_dict.get("output_options", {}),
            "eos_token_ids": request_dict.get("eos_token_ids", []),
            "annotations": request_dict.get("annotations", []),
            "routing": request_dict.get("routing"),
            "router_config_override": request_dict.get("router_config_override"),
            "prefill_result": request_dict.get("prefill_result"),
            "bootstrap_info": request_dict.get("bootstrap_info"),
            "extra_args": request_dict.get("extra_args"),
            "mm_processor_kwargs": request_dict.get("mm_processor_kwargs"),
        }
        return PreprocessedRequest(
            token_ids=token_ids,
            preprocessed_request=preprocessed,
            annotations={},
            formatted_prompt=None,
        )

    async def _extract_token_ids(self, request: dict[str, Any]) -> list[int]:
        prompt_token_ids = request.get("prompt_token_ids")
        if prompt_token_ids is not None:
            if prompt_token_ids and isinstance(prompt_token_ids[0], list):
                if len(prompt_token_ids) != 1:
                    raise ValueError("batched prompt_token_ids are not supported for routing")
                return [int(t) for t in prompt_token_ids[0]]
            return [int(t) for t in prompt_token_ids]

        prompt = request.get("prompt")
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            return [int(t) for t in prompt]
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], list):
            if len(prompt) != 1:
                raise ValueError("batched token prompts are not supported for routing")
            return [int(t) for t in prompt[0]]
        if isinstance(prompt, str):
            return await self.tokenize_text(prompt)

        raise ValueError(
            "Dynamo router preprocessor needs prompt_token_ids, token prompt, "
            "or a string completion prompt. Chat-template preprocessing is not "
            "available through this lightweight Python facade yet."
        )


def _to_request_dict(request: Any) -> dict[str, Any]:
    if isinstance(request, dict):
        return dict(request)
    if hasattr(request, "model_dump"):
        return request.model_dump(exclude_none=True)
    if hasattr(request, "dict"):
        return request.dict(exclude_none=True)
    raise TypeError(f"unsupported request type: {type(request)!r}")
