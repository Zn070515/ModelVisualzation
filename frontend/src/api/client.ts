import type { ModelInfo, GraphData, ProfileData } from '../types'

const BASE = '/api'

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, options)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    const msg = body.message || `HTTP ${res.status}`
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
