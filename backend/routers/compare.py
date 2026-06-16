from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ..analyzer.activation import collect_activations
from ..analyzer.chain import trace_conversion_chain
from ..analyzer.compare import compare_models
from ..analyzer.perf import estimate_perf
from ..analyzer.prune import analyze_pruning
from ..analyzer.quant import simulate_quantization
from ..analyzer.report import generate_batch_report
from .parse import _model_store, get_model

router = APIRouter(prefix="/api", tags=["analysis"])


class CompareRequest(BaseModel):
    model_a_id: str
    model_b_id: str


class ChainRequest(BaseModel):
    model_ids: list[str]
    labels: list[str] = []


class ReportRequest(BaseModel):
    model_ids: list[str]
    format: str = "json"


class PerfRequest(BaseModel):
    hardware: str = "cpu"


class QuantRequest(BaseModel):
    model_id: str
    bits: int = 8


class PruneRequest(BaseModel):
    model_id: str


@router.post("/compare")
def compare(req: CompareRequest):
    return compare_models(get_model(req.model_a_id), get_model(req.model_b_id), req.model_a_id, req.model_b_id)


@router.post("/chain")
def chain(req: ChainRequest):
    if len(req.model_ids) < 1:
        raise HTTPException(400, "At least 1 model_id required")
    models = [get_model(mid) for mid in req.model_ids]
    labels = req.labels if len(req.labels) == len(req.model_ids) else req.model_ids
    return trace_conversion_chain(models, labels, req.model_ids)


@router.post("/report/generate")
def report(req: ReportRequest):
    missing = [model_id for model_id in req.model_ids if model_id not in _model_store]
    if missing:
        raise HTTPException(404, f"Models not found: {missing}")
    models = {model_id: _model_store[model_id] for model_id in req.model_ids}
    return generate_batch_report(models, req.format)


@router.post("/model/{model_id}/perf")
def perf(model_id: str, req: PerfRequest):
    try:
        return estimate_perf(get_model(model_id), req.hardware, model_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/quant/simulate")
def quant(req: QuantRequest):
    try:
        return simulate_quantization(get_model(req.model_id), req.model_id, req.bits)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/model/{model_id}/activation")
async def activation(
    model_id: str,
    file: UploadFile = File(...),
    layer_names: str | None = Form(default=None),
):
    selected = [name.strip() for name in layer_names.split(",")] if layer_names else None
    content = await file.read()
    return collect_activations(get_model(model_id), content, model_id, selected)


@router.post("/prune/analyze")
def prune(req: PruneRequest):
    return analyze_pruning(get_model(req.model_id), req.model_id)
