import uuid
import os
import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from ..parser.ir import IRModel
from ..parser.onnx_parser import parse_onnx
from ..parser.tflite_parser import parse_tflite
from ..parser.pytorch_parser import parse_pytorch

router = APIRouter(prefix="/api/model", tags=["model"])
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")

_model_store: dict[str, IRModel] = {}
_model_metas: dict[str, dict] = {}
_model_paths: dict[str, str] = {}

os.makedirs(UPLOAD_DIR, exist_ok=True)

_PARSER_MAP = {
    ".onnx": parse_onnx,
    ".tflite": parse_tflite,
    ".pt": parse_pytorch,
    ".pth": parse_pytorch,
}


def get_model(model_id: str) -> IRModel:
    if model_id not in _model_store:
        raise HTTPException(404, f"Model {model_id} not found")
    return _model_store[model_id]


# ---- Progress tracking (also used by Task 9 WS) ----
_active_progress: dict[str, int] = {}


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in _PARSER_MAP:
        raise HTTPException(400, f"Unsupported format: {ext}. Supported: {list(_PARSER_MAP.keys())}")

    file_id = uuid.uuid4().hex[:12]
    _active_progress[file_id] = 0
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    content = await file.read()

    _active_progress[file_id] = 20
    with open(save_path, "wb") as f:
        f.write(content)

    _active_progress[file_id] = 40
    loop = asyncio.get_event_loop()
    try:
        _active_progress[file_id] = 60
        ir_model = await loop.run_in_executor(None, _PARSER_MAP[ext], save_path)
        _active_progress[file_id] = 90
    except Exception as e:
        _active_progress.pop(file_id, None)
        raise HTTPException(500, f"Failed to parse model: {str(e)}")

    _model_store[file_id] = ir_model
    _model_paths[file_id] = save_path
    _model_metas[file_id] = {
        "id": file_id,
        "format": ir_model.format,
        "producer": ir_model.producer,
        "opset_version": ir_model.opset_version,
        "layer_count": len(ir_model.layers),
        "file_size_bytes": os.path.getsize(save_path),
        "file_name": file.filename,
    }
    _active_progress[file_id] = 100

    return {"model_id": file_id}


@router.get("/{model_id}/info")
def model_info(model_id: str):
    if model_id not in _model_metas:
        raise HTTPException(404, f"Model {model_id} not found")
    return _model_metas[model_id]


@router.get("/{model_id}/graph")
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
            "id": f"e_{model.layers[i].name}_{model.layers[i+1].name}",
            "source": model.layers[i].name,
            "target": model.layers[i+1].name,
        })

    return {"nodes": nodes, "edges": edges}


@router.get("/{model_id}/profile")
def model_profile(model_id: str):
    model = get_model(model_id)
    from ..analyzer.profile import compute_profile
    return compute_profile(model)


@router.get("/{model_id}/weights/overview")
def model_weights_overview(model_id: str):
    model = get_model(model_id)
    from ..analyzer.weights import compute_weight_overview
    return compute_weight_overview(model)


@router.get("/{model_id}/weights")
def model_weights_layer(model_id: str, layer: str):
    model = get_model(model_id)
    for l in model.layers:
        if l.name == layer:
            from ..analyzer.weights import compute_layer_weight_stats
            return compute_layer_weight_stats(l)
    raise HTTPException(404, f"Layer {layer} not found")


@router.get("/{model_id}/health")
def model_health(model_id: str):
    model = get_model(model_id)
    from ..analyzer.health import health_check
    return health_check(model, model_id)


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


# ---- WebSocket progress endpoint (Task 9) ----
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
    finally:
        _active_progress.pop(model_id, None)
