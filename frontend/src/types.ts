export interface TensorSpec {
  name: string
  shape: number[]
  dtype: string
}

export interface IRLayer {
  name: string
  op_type: string
  inputs: string[]
  outputs: string[]
  params: Record<string, unknown>
  input_shapes: number[][]
  output_shapes: number[][]
  weights: Record<string, number[]>
}

export interface ModelInfo {
  id: string
  format: string
  producer: string
  opset_version: number | null
  layer_count: number
  file_size_bytes: number
  file_name: string
  inputs: TensorSpec[]
  outputs: TensorSpec[]
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface GraphNode {
  id: string
  type: string
  position: { x: number; y: number }
  data: {
    label: string
    opType: string
    inputShapes: number[][]
    outputShapes: number[][]
    params: Record<string, unknown>
  }
}

export interface GraphEdge {
  id: string
  source: string
  target: string
}

export interface ProfileData {
  total_params: number
  total_flops: number
  memory_mb: number
  layers: LayerProfile[]
}

export interface LayerProfile {
  name: string
  op_type: string
  params_count: number
  flops: number
}

export interface ModelState {
  id: string
  info: ModelInfo | null
  graph: GraphData | null
  profile: ProfileData | null
  loading: boolean
  error: string | null
}

export interface WeightTensorStats {
  shape: number[]
  mean: number
  std: number
  min: number
  max: number
  sparsity: number
  histogram: {
    counts: number[]
    bin_edges: number[]
  }
}

export interface WeightLayerStats {
  layer_name: string
  op_type: string
  weights: Record<string, WeightTensorStats>
}

export interface WeightLayerSummary {
  layer_name: string
  op_type: string
  param_count: number
  mean: number
  std: number
  sparsity: number
}

export interface WeightOverview {
  layers: WeightLayerSummary[]
  global_stats: {
    total_weight_params: number
    overall_mean: number
    overall_std: number
    overall_sparsity: number
  }
}

export interface HealthIssue {
  layer: string
  severity: 'warning' | 'critical'
  type: string
  message: string
  detail: Record<string, unknown>
}

export interface HealthResult {
  model_id: string
  issues: HealthIssue[]
  summary: {
    total_issues: number
    critical_count: number
    warning_count: number
  }
}

export interface LayerBrief {
  name: string
  op_type: string
  params: number
  input_shapes: number[][]
  output_shapes: number[][]
}

export interface CompareResult {
  model_a_id: string
  model_b_id: string
  summary: {
    layer_count_a: number
    layer_count_b: number
    layer_count_delta: number
    param_count_a: number
    param_count_b: number
    param_count_delta: number
    changed_layers: number
    a_only_layers: string[]
    b_only_layers: string[]
    common_layer_count: number
    similarity: number
  }
  op_type_diff: Array<{ op_type: string; a: number; b: number; delta: number }>
  layers: Array<{
    name: string
    status: 'same' | 'changed' | 'added' | 'removed'
    a: LayerBrief | null
    b: LayerBrief | null
    param_delta: number
  }>
  weight_diffs: Array<{
    layer_name: string
    weight_name: string
    mean_diff: number
    std_diff: number
  }>
}

export interface ChainResult {
  model_ids: string[]
  stages: Array<{
    label: string
    model_id: string
    format: string
    layer_count: number
    total_params: number
    op_types: string[]
    op_counts: Record<string, number>
    total_ops: number
  }>
  transitions: Array<{
    from: string
    to: string
    from_id: string
    to_id: string
    added_ops: string[]
    removed_ops: string[]
    renamed_ops: Array<{ from: string; to: string }>
    common_ops: string[]
    layer_count_delta: number
    param_count_delta: number
  }>
  summary: { total_steps: number; total_op_loss: number; preserved_op_count: number }
}

export interface BatchReport {
  format: string
  models: Array<{
    model_id: string
    format: string
    producer: string
    layer_count: number
    param_count: number
    op_types: string[]
  }>
  comparisons: Array<Record<string, unknown>>
  summary: { model_count: number; total_params: number; max_layers: number }
  html?: string
}

export interface PerfLayer {
  name: string
  op_type: string
  params: number
  flops: number
  est_latency_us: number
  is_bottleneck: boolean
  bottleneck_score: number
  memory_read_bytes: number
  memory_write_bytes: number
  compute_latency_us: number
  memory_latency_us: number
  bound: 'compute' | 'memory' | 'balanced'
}

export interface PerfResult {
  model_id: string
  hardware: string
  layers: PerfLayer[]
  summary: {
    total_latency_us: number
    total_latency_ms: number
    bottleneck_layers: string[]
    bottleneck_count: number
    memory_total_bytes: number
  }
}

export interface QuantTensorStats {
  quant_min: number
  quant_max: number
  scale: number
  zero_point: number
  error: { max_abs_err: number; mean_abs_err: number; rmse: number; snr_db: number }
  per_channel_error: number[]
}

export interface QuantResult {
  model_id: string
  bits: number
  mode: string
  per_channel: boolean
  layers: Array<{ layer_name: string; op_type: string; weights: Record<string, QuantTensorStats> }>
  summary: {
    overall_mean_abs_err: number
    overall_rmse: number
    worst_layer: string
    worst_layer_rmse: number
    sensitive_layers: string[]
  }
}

export interface ActivationResult {
  model_id: string
  activations: Array<{
    layer_name: string
    output_shape: number[]
    stats: WeightTensorStats
    dead_neurons_pct: number
    dead_neuron_indices: number[]
    saturation_pct: number
    method: string
  }>
  summary: {
    total_layers_analyzed: number
    layers_with_dead_neurons: number
    overall_dead_neuron_pct: number
  }
}

export interface PruneResult {
  model_id: string
  layers: Array<{
    layer_name: string
    op_type: string
    channel_importance: Array<{ channel: number; l1_norm: number; l2_norm: number; importance: number; fisher_score: number; activation_score: number; combined_importance: number; prune_priority: number }>
    sparsity_heatmap: number[][]
    total_channels: number
    prunable_channels_30pct: number
    prunable_channels_50pct: number
  }>
  summary: {
    total_prunable_layers: number
    total_prunable_params_pct: number
    recommended_prune_ratio: number
  }
}
