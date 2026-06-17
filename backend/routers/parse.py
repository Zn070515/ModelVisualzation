from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from ..models import (
    GraphResponse,
    HealthResponse,
    ModelInfoResponse,
    ProfileResponse,
    UploadResponse,
    WeightLayerStatsResponse,
    WeightOverviewResponse,
)
from ..parser.ir import IRModel
from ..parser.onnx_parser import parse_onnx
from ..parser.pytorch_parser import parse_pytorch
from ..parser.tflite_parser import parse_tflite

router = APIRouter(prefix="/api/model", tags=["model"])

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

_model_store: dict[str, IRModel] = {}
_model_metas: dict[str, dict] = {}
_model_paths: dict[str, str] = {}
_active_progress: dict[str, int] = {}

_PARSER_MAP: dict[str, object] = {
    ".onnx": parse_onnx,
    ".tflite": parse_tflite,
    ".pt": parse_pytorch,
    ".pth": parse_pytorch,
}


def get_model(model_id: str) -> IRModel:
    if model_id not in _model_store:
        raise HTTPException(404, f"Model {model_id} not found")
    return _model_store[model_id]


def _build_meta(file_id: str, ir_model: IRModel, save_path: Path) -> dict:
    return {
        "id": file_id,
        "format": ir_model.format,
        "producer": ir_model.producer,
        "opset_version": ir_model.opset_version,
        "layer_count": len(ir_model.layers),
        "file_size_bytes": save_path.stat().st_size,
        "file_name": save_path.name,
        "inputs": [{"name": s.name, "shape": s.shape, "dtype": s.dtype} for s in ir_model.inputs],
        "outputs": [{"name": s.name, "shape": s.shape, "dtype": s.dtype} for s in ir_model.outputs],
    }


@router.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in _PARSER_MAP:
        raise HTTPException(400, f"Unsupported format: {ext}. Supported: {list(_PARSER_MAP.keys())}")

    file_id = uuid.uuid4().hex[:12]
    _active_progress[file_id] = 0
    save_path = UPLOAD_DIR / f"{file_id}{ext}"
    content = await file.read()

    _active_progress[file_id] = 20
    save_path.write_bytes(content)

    _active_progress[file_id] = 40
    loop = asyncio.get_event_loop()
    try:
        _active_progress[file_id] = 60
        ir_model = await loop.run_in_executor(None, _PARSER_MAP[ext], str(save_path))
        _active_progress[file_id] = 90
    except Exception:
        _active_progress.pop(file_id, None)
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(500, "Failed to parse model")

    _model_store[file_id] = ir_model
    _model_paths[file_id] = str(save_path)
    _model_metas[file_id] = _build_meta(file_id, ir_model, save_path)
    _active_progress[file_id] = 100

    return {"model_id": file_id}


@router.get("/{model_id}/info", response_model=ModelInfoResponse)
def model_info(model_id: str):
    if model_id not in _model_metas:
        raise HTTPException(404, f"Model {model_id} not found")
    return _model_metas[model_id]


@router.get("/{model_id}/graph", response_model=GraphResponse)
def model_graph(model_id: str):
    model = get_model(model_id)
    nodes = []
    edges = []

    for i, layer in enumerate(model.layers):
        nodes.append({
            "id": layer.name,
            "type": "modelNode",
            "position": {"x": 0, "y": i * 80},
            "data": {
                "label": _make_label(layer),
                "opType": layer.op_type,
                "inputShapes": layer.input_shapes,
                "outputShapes": layer.output_shapes,
                "params": layer.params,
            },
        })

    for i in range(len(model.layers) - 1):
        edges.append({
            "id": f"e_{model.layers[i].name}_{model.layers[i + 1].name}",
            "source": model.layers[i].name,
            "target": model.layers[i + 1].name,
        })

    return {"nodes": nodes, "edges": edges}


@router.get("/{model_id}/profile", response_model=ProfileResponse)
def model_profile(model_id: str):
    from ..analyzer.profile import compute_profile
    return compute_profile(get_model(model_id))


@router.get("/{model_id}/weights/overview", response_model=WeightOverviewResponse)
def model_weights_overview(model_id: str):
    from ..analyzer.weights import compute_weight_overview
    return compute_weight_overview(get_model(model_id))


@router.get("/{model_id}/weights", response_model=WeightLayerStatsResponse)
def model_weights_layer(model_id: str, layer: str):
    model = get_model(model_id)
    for lyr in model.layers:
        if lyr.name == layer:
            from ..analyzer.weights import compute_layer_weight_stats
            return compute_layer_weight_stats(lyr)
    raise HTTPException(404, f"Layer {layer} not found")


@router.get("/{model_id}/health", response_model=HealthResponse)
def model_health(model_id: str):
    from ..analyzer.health import health_check
    return health_check(get_model(model_id), model_id)


def _make_label(layer) -> str:
    name = layer.op_type
    if layer.params.get("kernel_size"):
        k = layer.params["kernel_size"]
        k_str = f"k{k[0]}" if isinstance(k, (list, tuple)) and len(k) == 1 else f"k{k}"
        name += f"\n{k_str}"
    if layer.params.get("stride"):
        s = layer.params["stride"]
        s_str = f"s{s[0]}" if isinstance(s, (list, tuple)) and len(s) == 1 else f"s{s}"
        if s_str != "s1":
            name += f" {s_str}"
    if layer.params.get("out_channels"):
        name += f"\n→{layer.params['out_channels']}"
    if layer.params.get("out_features"):
        name += f"\n→{layer.params['out_features']}"
    return name


@router.websocket("/ws/model/{model_id}/progress")
async def model_progress(websocket: WebSocket, model_id: str):
    await websocket.accept()
    try:
        while True:
            progress = _active_progress.get(model_id, 100)
            await websocket.send_json({"model_id": model_id, "progress": progress})
            if progress >= 100:
                break
            await asyncio.sleep(0.3)
    except WebSocketDisconnect:
        pass
