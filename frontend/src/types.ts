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
