# Multi-Agent RAG v4 — Evaluation Benchmark Report

## 1. Runtime Hardware & Capability Matrix
- **Local Hardware**: Lenovo ThinkBook 14 G8+ (`AMD Ryzen 7 H 255` + `AMD Radeon 780M` Vulkan iGPU, 32GB LPDDR5X)
- **Ollama Local Status**: `{}`
- **Gemini API Status**: `{'online': True, 'api_key_configured': True, 'used_today': 0, 'daily_limit': 1500}`
- **Kaggle Tunnel Status**: `{}`

## 2. Aggregate Metrics Across All 6 Execution Modes

| Mode | Questions | Groundedness | Citation F1 | Numeric/Date Exactness | Abstention Acc | Latency p50 (s) | Latency p95 (s) | Avg Outer Retries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cpu_min` | 4 | **100.0%** | **81.4%** | **100.0%** | **100.0%** | 0.39s | 0.39s | 0 |
| `cpu_max` | 4 | **100.0%** | **85.0%** | **100.0%** | **100.0%** | 4.08s | 4.14s | 0 |
| `gemini` | 4 | **91.7%** | **60.0%** | **100.0%** | **100.0%** | 9.11s | 20.56s | 0.5 |
| `kaggle_fn` | 4 | **100.0%** | **85.0%** | **100.0%** | **100.0%** | 0.53s | 0.54s | 0 |
| `kaggle_max` | 4 | **100.0%** | **85.0%** | **100.0%** | **100.0%** | 0.52s | 0.59s | 0 |
| `hybrid` | 4 | **100.0%** | **91.7%** | **100.0%** | **100.0%** | 20.35s | 34.08s | 2 |

## 3. Architectural Verification Highlights
- **Document-Scoped Retrieval Fix**: All 6 modes dynamically parse and cache `selected_doc` (`load_or_parse_selected_pdf`), eliminating cross-document contamination.
- **Two-Tier Self-Correction Loop**: Inner Generator-Critic loop (`route_after_summarizer`) + Outer Fact-Checker & Gap Re-Query loop (`route_after_fact_check` -> `retry_loop_node`).
- **Cross-Model Adversarial Verification (`Hybrid`)**: Guarantees Generator and Fact-Checker run on distinct model families to eliminate self-confirmation bias.
