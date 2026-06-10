#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

try:
    from ._version import __version__
except Exception:
    try:
        from importlib.metadata import version as _pkg_version

        __version__ = _pkg_version("ai-dynamo")
    except Exception:
        __version__ = "0.0.0+unknown"

from .preprocessor import PreprocessedRequest, Preprocessor
from .router import Router, RouterDecision, RouterServer

__all__ = [
    "PreprocessedRequest",
    "Preprocessor",
    "Router",
    "RouterDecision",
    "RouterServer",
    "__version__",
]
