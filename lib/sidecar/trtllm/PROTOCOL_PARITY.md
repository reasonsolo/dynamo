<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent protocol compatibility over TensorRT-LLM OpenEngine

The TensorRT-LLM serving profile supports Kimi K3 and GLM agent protocols through
the Dynamo HTTP frontend, prompt/token/parser pipeline, TensorRT-LLM sidecar, and
existing `openengine.v1.Inference/Generate` RPC. HTTP compatibility belongs to the
frontend. The sidecar transports sampling controls and results; it does not proxy
an HTTP body into the engine. No protobuf wire definitions change.

## Public Contracts

The user-facing contracts live in the public documentation:

- [Reasoning output, replay format, and GLM thinking controls](../../../docs/fern/pages/use-cases/tool-calling-and-reasoning/reasoning-parsing.mdx#responses-reasoning-replay)
- [Responses and Chat Completions custom tools](../../../docs/fern/pages/use-cases/tool-calling-and-reasoning/tool-call-parsing.mdx)
  and [advisory grammar constraints](../../../docs/fern/pages/use-cases/tool-calling-and-reasoning/troubleshooting-tool-calls.md#custom-tool-grammar-constraints)
- [Responses session-hint precedence](../../../docs/fern/pages/use-cases/agents/session-ids.mdx#responses-session-hints)
- [Responses streaming context-overflow failures](../../../docs/fern/pages/reference/components/frontend-configuration.mdx#responses-streaming-errors)
- [Incomplete GLM tool-call failures](../../../docs/fern/pages/use-cases/tool-calling-and-reasoning/troubleshooting-tool-calls.md#incomplete-glm-tool-calls)
- [Image omission and media refusal](../../../docs/fern/pages/reference/components/frontend-configuration.mdx#media-refusal-for-text-only-models)

The sidecar advertises `responses_context_overflow_events=true`,
`incomplete_glm_tool_call_errors=true`, and `strict_glm_thinking_off=true` in
model runtime metadata. These capabilities gate the corresponding frontend
behavior. Existing workers without the metadata retain their prior behavior.
Other backends can advertise the same contracts. Image omission separately uses
the selected deployment's `supported_input_modalities` metadata.

## Request and Replay Adaptation

Responses `additional_tools` input items are moved into `tools` before typed
validation. Existing declarations are retained; malformed arrays fail before
dispatch. Namespace and custom-tool validation use Dynamo's tool registry.

Kimi Responses history merges split results for one call within a consecutive
result run. Unmatched results become user notes after that run, preserving their
content and the binding of valid results. Other families keep their existing
history policy.

Kimi namespaced calls retain the reserved `dynamo_kimi_bare_v1_` call-ID prefix
followed by URL-safe base64 of the original ID when the model wrote a bare name.
Native replay preserves that spelling; ordinary IDs ending in `_bare` retain
their supplied names. When an older client drops a qualified call's namespace,
a unique declared namespace is restored; ambiguous declarations are not guessed.

## Generation and Streaming

Dynamo retains incremental function/custom-tool payload events, usage, response
IDs, monotonic sequence numbers, early opening, and disconnect handling. Responses
backend failures close active output items and terminate with a full
`response.failed` snapshot. Clients should inspect its error, rather than relying
on a separate standalone `error` event.

With `parallel_tool_calls=false`, the frontend uses parsed call boundaries to
limit returned calls. It does not inject Kimi/GLM closing strings as native
stops: those strings can occur inside argument text, and early termination can
lose terminal usage. The backend may generate further calls that are discarded.
See [Single-Call Output](../../../docs/fern/pages/use-cases/tool-calling-and-reasoning/tool-call-parsing.mdx#single-call-output).

The legacy `glm45` reasoning adapter retains a one-character marker prefix so
`</think>` split after `<` is recognized. Literal trailing `<` bytes flush at
stream end. Default GLM routing remains unified/V2 unless V1 is selected explicitly.

## Deployment Boundaries

The sidecar does not add stored Responses history: `previous_response_id` remains
unsupported on the stateless serving path. It does not introduce Message Batches,
new KV-write verification guarantees, or a deployment gateway.

## How to Test

Run in a development environment with the repository-pinned Rust toolchain:

```bash
cargo test --locked -p dynamo-trtllm-sidecar --lib
cargo test --locked -p dynamo-llm --no-default-features --features mm-routing --lib -- protocols::openai::responses::
cargo test --locked -p dynamo-llm --no-default-features --features mm-routing --lib -- http::service::openai::tests::
cargo fmt --all -- --check
```

The sidecar suite uses a local fake gRPC engine; frontend tests cover conversion,
replay, parsing, and SSE serialization. These checks do not establish GPU
inference, native engine stop matching, physical KV-transfer safety, or serving
performance. Live-engine cases in `src/tests/e2e.rs` require running OpenEngine
endpoints.
