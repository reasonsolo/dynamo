// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use llm_rs::preprocessor::media::{MediaDecoder as RsMediaDecoder, MediaFetcher as RsMediaFetcher};
use llm_rs::tokenizers::traits::{Decoder, Encoder};
use llm_rs::tokenizers::Tokenizer;

#[pyclass]
#[derive(Clone)]
pub struct MediaDecoder {
    pub(crate) inner: RsMediaDecoder,
}

#[pymethods]
impl MediaDecoder {
    #[new]
    fn new() -> Self {
        Self {
            inner: RsMediaDecoder::default(),
        }
    }

    fn enable_image(&mut self, decoder_options: &Bound<'_, PyDict>) -> PyResult<()> {
        let decoder_options = pythonize::depythonize(decoder_options).map_err(|err| {
            PyErr::new::<PyException, _>(format!("Failed to parse image decoder config: {}", err))
        })?;
        self.inner.image = Some(decoder_options);
        Ok(())
    }

    #[cfg(feature = "media-ffmpeg")]
    fn enable_video(&mut self, decoder_options: &Bound<'_, PyDict>) -> PyResult<()> {
        let decoder_options = pythonize::depythonize(decoder_options).map_err(|err| {
            PyErr::new::<PyException, _>(format!("Failed to parse video decoder config: {}", err))
        })?;
        self.inner.video = Some(decoder_options);
        Ok(())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct MediaFetcher {
    pub(crate) inner: RsMediaFetcher,
}

#[pymethods]
impl MediaFetcher {
    #[new]
    fn new() -> Self {
        // Use from_env so DYN_MM_ALLOW_INTERNAL is honored by the
        // Rust-side frontend-decode fetch path, matching the Python
        // UrlValidationPolicy.from_env() behavior on the backend.
        Self {
            inner: RsMediaFetcher::from_env(),
        }
    }
    fn user_agent(&mut self, user_agent: String) {
        self.inner.user_agent = user_agent;
    }

    fn allow_direct_ip(&mut self, allow: bool) {
        self.inner.allow_direct_ip = allow;
    }

    fn allow_direct_port(&mut self, allow: bool) {
        self.inner.allow_direct_port = allow;
    }

    fn allow_private_ips(&mut self, allow: bool) {
        self.inner.allow_private_ips = allow;
    }

    fn allowed_media_domains(&mut self, domains: Vec<String>) {
        self.inner.allowed_media_domains = Some(domains.into_iter().collect());
    }

    fn timeout_ms(&mut self, timeout_ms: u64) {
        self.inner.timeout = Some(Duration::from_millis(timeout_ms));
    }
}

fn resolve_tokenizer_file(path: &str) -> PyResult<PathBuf> {
    let path = Path::new(path);
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    if !path.is_dir() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            format!("tokenizer path does not exist: {}", path.display()),
        ));
    }

    for candidate in [
        "tokenizer.json",
        "tiktoken.model",
        "tokenizer.model",
        "tokenizer.tiktoken",
    ] {
        let candidate_path = path.join(candidate);
        if candidate_path.is_file() {
            return Ok(candidate_path);
        }
    }

    Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
        format!(
            "no supported tokenizer artifact found under {}; expected tokenizer.json, tiktoken.model, tokenizer.model, or tokenizer.tiktoken",
            path.display()
        ),
    ))
}

#[pyclass]
#[derive(Clone)]
pub struct AsyncTokenizer {
    inner: Arc<Tokenizer>,
    tokenizer_path: String,
}

#[pymethods]
impl AsyncTokenizer {
    #[new]
    #[pyo3(signature = (tokenizer_path))]
    fn new(tokenizer_path: String) -> PyResult<Self> {
        let resolved = resolve_tokenizer_file(&tokenizer_path)?;
        let path_str = resolved.to_string_lossy().to_string();
        let tokenizer = Tokenizer::from_file(&path_str).map_err(to_pyerr)?;
        Ok(Self {
            inner: Arc::new(tokenizer),
            tokenizer_path: path_str,
        })
    }

    #[getter]
    fn tokenizer_path(&self) -> &str {
        &self.tokenizer_path
    }

    fn tokenize_text<'p>(&self, py: Python<'p>, text: String) -> PyResult<Bound<'p, PyAny>> {
        let tokenizer = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let encoding = tokio::task::spawn_blocking(move || tokenizer.encode(&text))
                .await
                .map_err(to_pyerr)?
                .map_err(to_pyerr)?;
            Ok(encoding.token_ids().to_vec())
        })
    }

    fn tokenize_batch<'p>(
        &self,
        py: Python<'p>,
        texts: Vec<String>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let tokenizer = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let encodings = tokio::task::spawn_blocking(move || {
                let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
                tokenizer.encode_batch(&refs)
            })
            .await
            .map_err(to_pyerr)?
            .map_err(to_pyerr)?;
            Ok(encodings
                .into_iter()
                .map(|encoding| encoding.token_ids().to_vec())
                .collect::<Vec<_>>())
        })
    }

    fn decode<'p>(
        &self,
        py: Python<'p>,
        token_ids: Vec<u32>,
        skip_special_tokens: Option<bool>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let tokenizer = self.inner.clone();
        let skip_special_tokens = skip_special_tokens.unwrap_or(false);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let decoded = tokio::task::spawn_blocking(move || {
                tokenizer.decode(&token_ids, skip_special_tokens)
            })
            .await
            .map_err(to_pyerr)?
            .map_err(to_pyerr)?;
            Ok(String::from(decoded))
        })
    }
}
