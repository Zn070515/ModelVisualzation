import { create } from 'zustand'
import type { ModelState } from '../types'
import { uploadModel, getModelInfo, getModelGraph, getModelProfile } from '../api/client'

interface AppStore {
  models: Record<string, ModelState>
  currentModelId: string | null
  addModel: (file: File) => Promise<string>
  setCurrentModel: (id: string) => void
  loadModelData: (id: string) => Promise<void>
}

export const useStore = create<AppStore>((set, get) => ({
  models: {},
  currentModelId: null,

  addModel: async (file: File) => {
    const { model_id } = await uploadModel(file)
    set((s) => ({
      models: {
        ...s.models,
        [model_id]: {
          id: model_id,
          info: null,
          graph: null,
          profile: null,
          loading: true,
          error: null,
        },
      },
      currentModelId: model_id,
    }))
    await get().loadModelData(model_id)
    return model_id
  },

  setCurrentModel: (id: string) => {
    set({ currentModelId: id })
  },

  loadModelData: async (id: string) => {
    set((s) => ({
      models: {
        ...s.models,
        [id]: { ...s.models[id], loading: true, error: null },
      },
    }))
    try {
      const [info, graph, profile] = await Promise.all([
        getModelInfo(id),
        getModelGraph(id),
        getModelProfile(id),
      ])
      set((s) => ({
        models: {
          ...s.models,
          [id]: { ...s.models[id], info, graph, profile, loading: false },
        },
      }))
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Unknown error'
      set((s) => ({
        models: {
          ...s.models,
          [id]: { ...s.models[id], loading: false, error: msg },
        },
      }))
    }
  },
}))
