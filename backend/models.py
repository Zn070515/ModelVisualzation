"""Pydantic models for all API requests, responses, and shared types."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    warning = "warning"
    critical = "critical"


class Bound(str, Enum):
    compute = "compute"
    memory = "memory"
    balanced = "balanced"


class Hardware(str, Enum):
    i9_13900k = "i9_13900k"
    rtx_4090 = "rtx_4090"
    apple_m2 = "apple_m2"
    rpi4 = "rpi4"


class LayerDiffStatus(str, Enum):
    same = "same"
    changed = "changed"
    added = "added"
    removed = "removed"


class ActivationMethod(str, Enum):
    onnxruntime = "onnxruntime"
    pytorch_hook = "pytorch_hook"
    tflite_interpreter = "tflite_interpreter"
    synthetic = "synthetic"


# ---------------------------------------------------------------------------
# Shared / leaf models
# ---------------------------------------------------------------------------

class TensorSpecModel(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    shape: list[int]
    dtype: str


class Histogram(BaseModel):
    model_config = ConfigDict(frozen=True)
    counts: list[int]
    bin_edges: list[float]


class WeightTensorStats(BaseModel):
    model_config = ConfigDict(frozen=True)
    shape: list[int]
    mean: float
    std: float
    min: float
    max: float
    sparsity: float
    histogram: Histogram


class PerChannelError(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_abs_err: float
    mean_abs_err: float
    rmse: float
    snr_db: float


class QuantTensorStats(BaseModel):
    model_config = ConfigDict(frozen=True)
    quant_min: int
    quant_max: int
    scale: float
    zero_point: int
    per_channel_scales: list[float] | None = None
    per_channel_zero_points: list[int] | None = None
    error: PerChannelError
    per_channel_error: list[float] = Field(default_factory=list)


class ChannelImportance(BaseModel):
    model_config = ConfigDict(frozen=True)
    channel: int
    l1_norm: float
    l2_norm: float
    importance: float
    fisher_score: float
    activation_score: float
    combined_importance: float
    prune_priority: int


class HealthIssue(BaseModel):
    model_config = ConfigDict(frozen=True)
    layer: str
    severity: Severity
    type: str
    message: str
    detail: dict[str, Any]


class LayerBrief(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    op_type: str
    params: int
    input_shapes: list[list[int]]
    output_shapes: list[list[int]]


class OpTypeDiff(BaseModel):
    model_config = ConfigDict(frozen=True)
    op_type: str
    a: int
    b: int
    delta: int


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class CompareRequest(BaseModel):
    model_a_id: str
    model_b_id: str


class ChainRequest(BaseModel):
    model_ids: list[str]
    labels: list[str] = Field(default_factory=list)


class ReportRequest(BaseModel):
    model_ids: list[str]
    format: Literal["json", "html"] = "json"


class PerfRequest(BaseModel):
    hardware: str = "i9_13900k"


class QuantRequest(BaseModel):
    model_id: str
    bits: Annotated[int, Field(ge=1, le=16)] = 8
    per_channel: bool = False
    unsigned: bool = False


class PruneRequest(BaseModel):
    model_id: str


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    model_id: str


class ModelInfoResponse(BaseModel):
    id: str
    format: str
    producer: str
    opset_version: int | None
    layer_count: int
    file_size_bytes: int
    file_name: str
    inputs: list[TensorSpecModel]
    outputs: list[TensorSpecModel]


class GraphNodeData(BaseModel):
    label: str
    opType: str
    inputShapes: list[list[int]]
    outputShapes: list[list[int]]
    params: dict[str, Any]


class GraphNode(BaseModel):
    id: str
    type: str
    position: dict[str, float]
    data: GraphNodeData


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class LayerProfileItem(BaseModel):
    name: str
    op_type: str
    params_count: int
    flops: int


class ProfileResponse(BaseModel):
    total_params: int
    total_flops: int
    memory_mb: float
    layers: list[LayerProfileItem]


class WeightLayerStatsResponse(BaseModel):
    layer_name: str
    op_type: str
    weights: dict[str, WeightTensorStats]


class WeightOverviewItem(BaseModel):
    layer_name: str
    op_type: str
    param_count: int
    mean: float
    std: float
    sparsity: float


class GlobalWeightStats(BaseModel):
    total_weight_params: int
    overall_mean: float
    overall_std: float
    overall_sparsity: float


class WeightOverviewResponse(BaseModel):
    layers: list[WeightOverviewItem]
    global_stats: GlobalWeightStats


class HealthSummary(BaseModel):
    total_issues: int
    critical_count: int
    warning_count: int


class HealthResponse(BaseModel):
    model_id: str
    issues: list[HealthIssue]
    summary: HealthSummary


class CompareSummary(BaseModel):
    layer_count_a: int
    layer_count_b: int
    layer_count_delta: int
    param_count_a: int
    param_count_b: int
    param_count_delta: int
    changed_layers: int
    a_only_layers: list[str]
    b_only_layers: list[str]
    common_layer_count: int
    similarity: float


class LayerDiffItem(BaseModel):
    name: str
    status: LayerDiffStatus
    a: LayerBrief | None
    b: LayerBrief | None
    param_delta: int


class WeightDiffItem(BaseModel):
    layer_name: str
    weight_name: str
    mean_diff: float
    std_diff: float


class CompareResponse(BaseModel):
    model_a_id: str
    model_b_id: str
    summary: CompareSummary
    op_type_diff: list[OpTypeDiff]
    layers: list[LayerDiffItem]
    weight_diffs: list[WeightDiffItem]


class ChainStage(BaseModel):
    label: str
    model_id: str
    format: str
    layer_count: int
    total_params: int
    op_types: list[str]
    op_counts: dict[str, int]
    total_ops: int


class RenamedOp(BaseModel):
    old: str = Field(alias="from")
    new: str = Field(alias="to")


class ChainTransition(BaseModel):
    from_: str = Field(alias="from")
    to: str
    from_id: str
    to_id: str
    added_ops: list[str]
    removed_ops: list[str]
    renamed_ops: list[dict[str, str]]
    common_ops: list[str]
    layer_count_delta: int
    param_count_delta: int


class ChainSummary(BaseModel):
    total_steps: int
    total_op_loss: int
    preserved_op_count: int


class ChainResponse(BaseModel):
    model_ids: list[str]
    stages: list[ChainStage]
    transitions: list[dict[str, Any]]
    summary: ChainSummary


class PerfLayer(BaseModel):
    name: str
    op_type: str
    params: int
    flops: int
    est_latency_us: float
    is_bottleneck: bool
    bottleneck_score: float
    memory_read_bytes: int
    memory_write_bytes: int
    compute_latency_us: float
    memory_latency_us: float
    bound: Bound


class PerfSummary(BaseModel):
    total_latency_us: float
    total_latency_ms: float
    bottleneck_layers: list[str]
    bottleneck_count: int
    memory_total_bytes: int


class PerfResponse(BaseModel):
    model_id: str
    hardware: str
    layers: list[PerfLayer]
    summary: PerfSummary


class QuantWeightEntry(BaseModel):
    layer_name: str
    op_type: str
    weights: dict[str, QuantTensorStats]


class QuantSummary(BaseModel):
    overall_mean_abs_err: float
    overall_rmse: float
    worst_layer: str
    worst_layer_rmse: float
    sensitive_layers: list[str]


class QuantResponse(BaseModel):
    model_id: str
    bits: int
    mode: str
    per_channel: bool
    layers: list[QuantWeightEntry]
    summary: QuantSummary


class ActivationLayerItem(BaseModel):
    layer_name: str
    output_shape: list[int]
    stats: WeightTensorStats
    dead_neurons_pct: float
    dead_neuron_indices: list[int]
    saturation_pct: float
    method: str


class ActivationSummary(BaseModel):
    total_layers_analyzed: int
    layers_with_dead_neurons: int
    overall_dead_neuron_pct: float


class ActivationResponse(BaseModel):
    model_id: str
    activations: list[ActivationLayerItem]
    summary: ActivationSummary


class PruneLayerItem(BaseModel):
    layer_name: str
    op_type: str
    channel_importance: list[ChannelImportance]
    sparsity_heatmap: list[list[float]]
    total_channels: int
    prunable_channels_30pct: int
    prunable_channels_50pct: int


class PruneSummary(BaseModel):
    total_prunable_layers: int
    total_prunable_params_pct: float
    recommended_prune_ratio: float


class PruneResponse(BaseModel):
    model_id: str
    layers: list[PruneLayerItem]
    summary: PruneSummary


class BatchModelEntry(BaseModel):
    model_id: str
    format: str
    producer: str
    layer_count: int
    param_count: int
    op_types: list[str]


class BatchSummary(BaseModel):
    model_count: int
    total_params: int
    max_layers: int


class BatchReportResponse(BaseModel):
    format: str
    models: list[BatchModelEntry]
    comparisons: list[dict[str, Any]]
    summary: BatchSummary
    html: str | None = None


class AttentionHeadSlice(BaseModel):
    head_index: int
    shape: list[int]
    values: list[list[float]]


class AttentionProjection(BaseModel):
    shape: list[int]
    head_slices: list[AttentionHeadSlice]


class AttentionLayerData(BaseModel):
    layer_name: str
    op_type: str
    num_heads: int
    head_dim: int
    embed_dim: int
    projections: dict[str, AttentionProjection]


class AttentionResponse(BaseModel):
    model_id: str
    layers: list[AttentionLayerData]


class HealthStatus(BaseModel):
    status: str


class ErrorDetail(BaseModel):
    detail: str
