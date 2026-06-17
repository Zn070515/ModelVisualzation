from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..analyzer.activation import collect_activations
from ..analyzer.attention import extract_attention
from ..analyzer.chain import trace_conversion_chain
from ..analyzer.compare import compare_models
from ..analyzer.perf import estimate_perf
from ..analyzer.prune import analyze_pruning
from ..analyzer.quant import simulate_quantization
from ..analyzer.report import generate_batch_report
from ..models import (
    ActivationResponse,
    AttentionResponse,
    BatchReportResponse,
    ChainResponse,
    CompareRequest,
    ChainRequest,
    CompareResponse,
    PerfRequest,
    PerfResponse,
    PruneRequest,
    PruneResponse,
    QuantRequest,
    QuantResponse,
    ReportRequest,
)
from .parse import _model_paths, _model_store, get_model

router = APIRouter(prefix="/api", tags=["analysis"])


@router.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    return compare_models(
        get_model(req.model_a_id),
        get_model(req.model_b_id),
        req.model_a_id,
        req.model_b_id,
    )


@router.post("/chain", response_model=ChainResponse)
def chain(req: ChainRequest):
    if len(req.model_ids) < 1:
        raise HTTPException(400, "At least 1 model_id required")
    models = [get_model(mid) for mid in req.model_ids]
    labels = req.labels if len(req.labels) == len(req.model_ids) else req.model_ids
    return trace_conversion_chain(models, labels, req.model_ids)


@router.post("/report/generate", response_model=BatchReportResponse)
def report(req: ReportRequest):
    missing = [mid for mid in req.model_ids if mid not in _model_store]
    if missing:
        raise HTTPException(404, f"Models not found: {missing}")
    models = {mid: _model_store[mid] for mid in req.model_ids}
    return generate_batch_report(models, req.format)


@router.post("/model/{model_id}/perf", response_model=PerfResponse)
def perf(model_id: str, req: PerfRequest):
    try:
        return estimate_perf(get_model(model_id), req.hardware, model_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/quant/simulate", response_model=QuantResponse)
def quant(req: QuantRequest):
    try:
        return simulate_quantization(
            get_model(req.model_id), req.model_id,
            req.bits, req.per_channel, req.unsigned,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/model/{model_id}/activation", response_model=ActivationResponse)
async def activation(
    model_id: str,
    file: UploadFile = File(...),
    layer_names: str | None = Form(default=None),
):
    selected = [name.strip() for name in layer_names.split(",")] if layer_names else None
    content = await file.read()
    model_path = _model_paths.get(model_id)
    try:
        return collect_activations(get_model(model_id), content, model_id, selected, model_path)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, f"Failed to run activation analysis: {exc}") from exc


@router.get("/model/{model_id}/attention", response_model=AttentionResponse)
def model_attention(model_id: str):
    return extract_attention(get_model(model_id), model_id)


@router.post("/prune/analyze", response_model=PruneResponse)
def prune(req: PruneRequest):
    return analyze_pruning(get_model(req.model_id), req.model_id)
