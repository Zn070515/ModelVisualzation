# ModelViz

Interactive web app for visualizing and analyzing deep learning models (PyTorch, ONNX, TensorFlow Lite). Upload a model, inspect its graph, profile parameters, scan for health issues, compare models, run real inference for activation analysis, simulate quantization, and estimate per-layer performance on specific hardware.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + TypeScript, Vite, React Flow v12, ECharts, Zustand |
| Backend | FastAPI (Python 3.11), Pydantic v2, onnxruntime, NumPy |
| Parsers | onnx, torch, tensorflow |
| Theme | Catppuccin dark |

## Architecture

```
frontend (React 18 + Vite)            backend (FastAPI + Pydantic v2)
├── src/pages/      page components   ├── models.py            all request/response types & enums
├── src/components/ shared UI          ├── main.py              FastAPI app entry
├── src/api/        HTTP client        ├── routers/parse.py     upload & model info (7 ep)
├── src/store/      Zustand            ├── routers/compare.py   analysis endpoints (7 ep)
├── src/types.ts    shared types       ├── parser/              pt/onnx/tflite → IRModel
└── dist/           build output       ├── analyzer/            profile, weights, health, compare,
                                       │                          chain, perf, quant, activation,
                                       │                          prune, report
                                       ├── tests/               27 tests
                                       └── uploads/             saved model files
```

All parsers convert to a unified IR (`IRModel` / `IRLayer` / `TensorSpec` frozen dataclasses) so the analyzer and frontend work identically across formats. Every API endpoint uses Pydantic `response_model=` for runtime output validation and OpenAPI schema generation. Magic strings are replaced with typed enums (`Hardware`, `Bound`, `Severity`, `LayerDiffStatus`, `ActivationMethod`).

## Features

### Phase 1 — Model Upload & Graph Viewer

- Upload `.pt`, `.pth`, `.onnx`, `.tflite` files
- Interactive node-link graph (React Flow) with layout, zoom, drag
- Collapsible 3-column layout
- Basic model profile (params, FLOPs, memory)

### Phase 2 — Single-Model Deep Analysis

- **Weight Analysis** — per-layer histogram, sparsity, mean/std for every weight tensor (TFLite weights extracted from flatbuffer buffers; PyTorch weights from state_dict)
- **Dashboard** — pie chart (op type distribution), bar chart (params per layer), layer detail table
- **Health Scan** — dead neurons, weight outliers, BN anomalies, high sparsity, issue cards with severity

### Phase 3 — Multi-Model Comparison

- **Compare** — side-by-side diff of any two models (name-based layer matching, weight diffs, similarity score clamped 0–1)
- **Conversion Chain** — trace operator additions/removals/renames across a chain of converted models
- **Batch Reports** — generate JSON or HTML reports across multiple models

### Phase 4 — Advanced Analysis

- **Performance Estimation** — per-layer latency for 4 real hardware profiles (Core i9-13900K, RTX 4090, Apple M2, Raspberry Pi 4) with memory bandwidth modeling and compute/memory/bottleneck bound classification
- **Quantization Simulation** — INT4/INT8/FP16, per-tensor and per-channel modes, unsigned quantization, RMSE/SNR/per-channel error
- **Activation Analysis** — real inference via ONNX Runtime (modified-graph intermediate outputs), PyTorch (forward hooks), and TFLite (interpreter), with dead neuron and saturation detection; falls back to synthetic forward for unsupported formats
- **Pruning Assist** — multi-signal channel importance: L1 norm, L2 norm, Fisher information (E[w²]), optional activation sensitivity, weighted combined score, prune priority ranking

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- npm 9+

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The app runs at `http://localhost:5173` with the API proxied to `http://localhost:8000`.

### Tests

```bash
# All 27 tests (project root as PYTHONPATH)
PYTHONPATH=. python -m pytest backend/tests/ -v

# By phase
PYTHONPATH=. python -m pytest backend/tests/test_compare_phase3.py -v
PYTHONPATH=. python -m pytest backend/tests/test_phase4_analyzers.py -v

# Type check frontend
cd frontend && npx tsc --noEmit
```

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/model/upload` | Upload model file (multipart) |
| GET | `/api/model/{id}/info` | Model metadata (format, producer, layer count, I/O specs) |
| GET | `/api/model/{id}/graph` | React Flow graph data (nodes + edges) |
| GET | `/api/model/{id}/profile` | FLOPs / params / memory profile |
| GET | `/api/model/{id}/weights/overview` | All-layer weight summary |
| GET | `/api/model/{id}/weights?layer=` | Single-layer weight stats with histogram |
| GET | `/api/model/{id}/health` | Health scan (dead neurons, outliers, BN issues) |
| POST | `/api/compare` | Compare two models (layer diff, weight diff, similarity) |
| POST | `/api/chain` | Trace conversion chain across multiple models |
| POST | `/api/report/generate` | Batch report (JSON or HTML) |
| POST | `/api/model/{id}/perf` | Per-layer latency on selected hardware |
| POST | `/api/quant/simulate` | Quantization simulation (bits, mode, per-channel) |
| POST | `/api/model/{id}/activation` | Activation analysis with real inference (multipart) |
| POST | `/api/prune/analyze` | Channel importance + pruning recommendations |
| GET | `/api/health` | Backend health check |

All POST endpoints accept and return JSON validated against Pydantic models. See `backend/models.py` for the full type definitions.

## Project Structure

```
ModelVisualzation/
├── backend/
│   ├── main.py                   FastAPI app entry + global error handler
│   ├── models.py                 Pydantic models, enums, request/response types
│   ├── requirements.txt
│   ├── parser/
│   │   ├── ir.py                 IRModel / IRLayer / TensorSpec frozen dataclasses
│   │   ├── onnx_parser.py
│   │   ├── pytorch_parser.py     Includes shape inference via forward hooks
│   │   └── tflite_parser.py      Weight extraction from flatbuffer buffers
│   ├── analyzer/
│   │   ├── profile.py            FLOPs / memory estimation
│   │   ├── weights.py            Weight stats & histograms
│   │   ├── health.py             Model health scanner
│   │   ├── compare.py            Two-model diff with similarity scoring
│   │   ├── chain.py              Multi-model conversion chain tracer
│   │   ├── perf.py               HW latency estimation (4 real profiles)
│   │   ├── quant.py              INT4/INT8/FP16 quantization simulation
│   │   ├── activation.py         ONNX / PyTorch hooks / TFLite interpreter inference
│   │   ├── prune.py              Fisher + L1/L2 + activation channel importance
│   │   └── report.py             Batch report generator (JSON + HTML)
│   ├── routers/
│   │   ├── parse.py              Upload, model info, graph, profile (7 endpoints)
│   │   └── compare.py            Compare, chain, report, perf, quant, activation, prune (7 ep)
│   └── tests/
│       ├── test_ir.py
│       ├── test_onnx_parser.py
│       ├── test_pytorch_parser.py
│       ├── test_tflite_parser.py
│       ├── test_weights.py
│       ├── test_health.py
│       ├── test_compare_phase3.py
│       └── test_phase4_analyzers.py
├── frontend/
│   └── src/
│       ├── App.tsx               Routes
│       ├── main.tsx              Entry point
│       ├── types.ts              Shared TypeScript types
│       ├── api/client.ts         HTTP client functions
│       ├── utils.ts              Formatters
│       ├── store/                Zustand state
│       ├── pages/
│       │   ├── Home.tsx          Upload + model list
│       │   ├── ModelViewer.tsx   React Flow graph
│       │   ├── WeightsPage.tsx   Weight analysis
│       │   ├── DashboardPage.tsx
│       │   ├── HealthPage.tsx
│       │   ├── ComparePage.tsx
│       │   ├── ChainPage.tsx
│       │   ├── BatchPage.tsx
│       │   ├── PerformancePage.tsx
│       │   ├── QuantPage.tsx
│       │   ├── ActivationPage.tsx
│       │   └── PrunePage.tsx
│       └── components/
│           ├── ModelLayout.tsx   Tabbed model wrapper
│           ├── ModelTabs.tsx     Navigation tabs
│           ├── ChainTimeline.tsx
│           ├── DiffBadge.tsx
│           ├── ReportPreview.tsx
│           ├── LatencyBarChart.tsx
│           ├── QuantHeatmap.tsx
│           ├── ActivationHistogram.tsx
│           └── ImportanceChart.tsx
└── docs/
    └── superpowers/
        ├── plans/                Implementation plans (phase 1–4)
        └── specs/                Design documents
```
