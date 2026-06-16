# ModelViz

Interactive web app for visualizing and analyzing deep learning models (PyTorch, ONNX, TensorFlow Lite). Upload a model, inspect its graph, profile parameters, scan for health issues, compare models, and estimate deployment performance.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + TypeScript, Vite, React Flow v12, ECharts, Zustand |
| Backend | FastAPI (Python 3.11), onnxruntime, NumPy |
| Parsers | onnx, torch, tensorflow/keras |
| Theme | Catppuccin dark |

## Architecture

```
frontend (React 18 + Vite)          backend (FastAPI)
├── src/pages/      page components ├── routers/parse.py     upload & model info
├── src/components/ shared UI       ├── routers/compare.py   Phase 3/4 analysis
├── src/api/        HTTP client     ├── parser/              pt/onnx/tflite → IRModel
├── src/store/      Zustand         ├── analyzer/            profile, weights, health,
├── src/types.ts    shared types        compare, chain, perf, quant,
└── dist/           build output        activation, prune, report
                                    ├── tests/               27 tests
                                    └── uploads/             saved model files
```

All parsers convert to a unified intermediate representation (`IRModel` / `IRLayer` / `TensorSpec`) so the analyzer and frontend layers work identically across formats.

## Features

### Phase 1 — Model Upload & Graph Viewer

- Upload `.pt`, `.pth`, `.onnx`, `.tflite` files
- Interactive node-link graph (React Flow) with layout, zoom, drag
- Collapsible 3-column layout
- Basic model profile (params, FLOPs, memory)

### Phase 2 — Single-Model Deep Analysis

- **Weight Analysis** — per-layer histogram, sparsity, mean/std, overview dashboard
- **Dashboard** — pie chart (op type distribution), bar chart (params per layer), layer detail table
- **Health Scan** — dead neurons, weight outliers, BN anomalies, high sparsity, issue cards

### Phase 3 — Multi-Model Comparison

- **Compare** — side-by-side diff of any two models (name-based layer matching, weight diffs, similarity score)
- **Conversion Chain** — trace operator additions/removals/renames across a chain of converted models
- **Batch Reports** — generate JSON or HTML reports across multiple models

### Phase 4 — Advanced Analysis

- **Performance Estimation** — per-layer latency estimates for CPU/GPU/Edge TPU, bottleneck detection
- **Quantization Simulation** — INT8/FP16 error analysis with RMSE, SNR, per-channel heatmaps
- **Activation Analysis** — upload sample input, run ONNX Runtime inference (or synthetic fallback), detect dead/saturated neurons
- **Pruning Assist** — L1/L2 channel importance ranking, sparsity heatmaps, recommended prune ratios

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
# All 27 tests
python -m pytest backend/tests/ -v

# By phase
python -m pytest backend/tests/test_compare_phase3.py -v
python -m pytest backend/tests/test_phase4_analyzers.py -v
```

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/model/upload` | Upload model file (multipart) |
| GET | `/api/model/{id}/info` | Model metadata |
| GET | `/api/model/{id}/graph` | React Flow graph data |
| GET | `/api/model/{id}/profile` | FLOPs/params profile |
| GET | `/api/model/{id}/weights/overview` | All-layer weight summary |
| GET | `/api/model/{id}/weights?layer=` | Single-layer weight stats |
| GET | `/api/model/{id}/health` | Health scan results |
| POST | `/api/compare` | Compare two models |
| POST | `/api/chain` | Trace conversion chain |
| POST | `/api/report/generate` | Batch report (JSON/HTML) |
| POST | `/api/model/{id}/perf` | Performance estimation |
| POST | `/api/quant/simulate` | Quantization simulation |
| POST | `/api/model/{id}/activation` | Activation analysis (multipart) |
| POST | `/api/prune/analyze` | Pruning analysis |
| GET | `/api/health` | Backend health check |

## Project Structure

```
ModelVisualzation/
├── backend/
│   ├── main.py                  FastAPI app entry
│   ├── requirements.txt
│   ├── parser/
│   │   ├── ir.py                IRModel / IRLayer dataclasses
│   │   ├── onnx_parser.py
│   │   ├── pytorch_parser.py
│   │   └── tflite_parser.py
│   ├── analyzer/
│   │   ├── profile.py           FLOPs/memory estimation
│   │   ├── weights.py           Weight stats & histograms
│   │   ├── health.py            Model health scanner
│   │   ├── compare.py           Two-model diff
│   │   ├── chain.py             Multi-model conversion chain
│   │   ├── perf.py              Hardware latency estimation
│   │   ├── quant.py             Quantization simulation
│   │   ├── activation.py        ONNX Runtime / synthetic forward
│   │   ├── prune.py             Channel importance ranking
│   │   └── report.py            Batch report generator
│   ├── routers/
│   │   ├── parse.py             Upload, model info, graph, profile
│   │   └── compare.py           Phase 3/4 analysis endpoints
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
│       ├── App.tsx              Routes
│       ├── main.tsx             Entry point
│       ├── types.ts             Shared TypeScript types
│       ├── api/client.ts        HTTP client functions
│       ├── utils.ts             Formatters
│       ├── store/               Zustand state
│       ├── pages/
│       │   ├── Home.tsx         Upload + model list
│       │   ├── ModelViewer.tsx  React Flow graph
│       │   ├── WeightsPage.tsx  Weight analysis
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
        ├── plans/                Implementation plans
        └── specs/                Design documents
```
