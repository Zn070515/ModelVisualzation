import type {
  ActivationResult,
  AttentionResponse,
  BatchReport,
  ChainResult,
  CompareResult,
  GraphData,
  HealthResult,
  ModelInfo,
  PerfResult,
  ProfileData,
  PruneResult,
  QuantResult,
  WeightLayerStats,
  WeightOverview,
} from '../types'

const BASE = '/api'

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, options)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    const msg = body.detail || body.message || `HTTP ${res.status}`
    throw new Error(msg)
  }
  return res.json()
}

export async function uploadModel(file: File): Promise<{ model_id: string }> {
  const form = new FormData()
  form.append('file', file)
  return request('/model/upload', { method: 'POST', body: form })
}

export async function getModelInfo(id: string): Promise<ModelInfo> {
  return request(`/model/${id}/info`)
}

export async function getModelGraph(id: string): Promise<GraphData> {
  return request(`/model/${id}/graph`)
}

export async function getModelProfile(id: string): Promise<ProfileData> {
  return request(`/model/${id}/profile`)
}

export async function getWeightStats(id: string, layer: string): Promise<WeightLayerStats> {
  return request(`/model/${id}/weights?layer=${encodeURIComponent(layer)}`)
}

export async function getWeightOverview(id: string): Promise<WeightOverview> {
  return request(`/model/${id}/weights/overview`)
}

export async function getHealthScan(id: string): Promise<HealthResult> {
  return request(`/model/${id}/health`)
}

export async function compareModels(modelAId: string, modelBId: string): Promise<CompareResult> {
  return request('/compare', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_a_id: modelAId, model_b_id: modelBId }),
  })
}

export async function traceChain(modelIds: string[], labels?: string[], signal?: AbortSignal): Promise<ChainResult> {
  return request('/chain', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_ids: modelIds, labels: labels ?? [] }),
    signal,
  })
}

export async function generateReport(modelIds: string[], format: 'json' | 'html' = 'json'): Promise<BatchReport> {
  return request('/report/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_ids: modelIds, format }),
  })
}

export async function estimatePerf(id: string, hardware: string): Promise<PerfResult> {
  return request(`/model/${id}/perf`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ hardware }),
  })
}

export async function simulateQuant(id: string, bits: number, perChannel?: boolean, unsigned?: boolean): Promise<QuantResult> {
  return request('/quant/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: id, bits, per_channel: perChannel ?? false, unsigned: unsigned ?? false }),
  })
}

export async function getActivation(id: string, file: File, layerNames?: string[]): Promise<ActivationResult> {
  const form = new FormData()
  form.append('file', file)
  if (layerNames?.length) form.append('layer_names', layerNames.join(','))
  return request(`/model/${id}/activation`, { method: 'POST', body: form })
}

export async function getAttention(id: string): Promise<AttentionResponse> {
  return request(`/model/${id}/attention`)
}

export async function analyzePrune(id: string): Promise<PruneResult> {
  return request('/prune/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: id }),
  })
}
