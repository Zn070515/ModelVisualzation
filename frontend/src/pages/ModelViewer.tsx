import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useStore } from '../store'
import ModelLayout from '../components/ModelLayout'
import GraphCanvas from '../components/GraphCanvas'
import LayerTree from '../components/LayerTree'
import NodePanel from '../components/NodePanel'

export default function ModelViewer() {
  const { modelId } = useParams<{ modelId: string }>()
  const navigate = useNavigate()
  const model = useStore((s) => (modelId ? s.models[modelId] : undefined))
  const loadModelData = useStore((s) => s.loadModelData)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)

  useEffect(() => {
    if (!modelId) {
      navigate('/')
      return
    }
    if (!model || (!model.graph && !model.loading)) {
      loadModelData(modelId)
    }
  }, [modelId])

  if (!modelId || !model) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100vh', color: 'var(--text-muted)',
      }}>
        模型不存在
      </div>
    )
  }

  if (model.loading) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100vh', color: 'var(--text-secondary)',
      }}>
        加载中...
      </div>
    )
  }

  if (model.error) {
    return (
      <div style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        justifyContent: 'center', height: '100vh', gap: 16,
      }}>
        <div style={{ color: 'var(--accent-red)', fontSize: 14 }}>出错了</div>
        <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>{model.error}</div>
        <button onClick={() => navigate('/')}
          style={{
            padding: '8px 20px', borderRadius: 6, border: 'none',
            background: 'var(--accent)', color: '#fff', cursor: 'pointer',
          }}>
          返回首页
        </button>
      </div>
    )
  }

  return (
    <ModelLayout modelId={modelId} activeTab="viewer">
      <div style={{ display: 'flex', height: '100%', overflow: 'hidden' }}>
        {/* Left: Layer tree */}
        <div style={{
          width: 260, flexShrink: 0,
          borderRight: '1px solid var(--border)',
          background: 'var(--bg-secondary)',
          overflow: 'hidden',
        }}>
          <LayerTree
            graphData={model.graph}
            selectedNodeId={selectedNodeId}
            onSelect={setSelectedNodeId}
          />
        </div>

        {/* Center: Graph */}
        <div style={{ flex: 1, background: 'var(--bg-primary)' }}>
          <GraphCanvas
            graphData={model.graph}
            onNodeClick={setSelectedNodeId}
          />
        </div>

        {/* Right: Node panel */}
        <div style={{
          width: 280, flexShrink: 0,
          borderLeft: '1px solid var(--border)',
          background: 'var(--bg-secondary)',
          overflow: 'hidden',
        }}>
          <NodePanel
            graphData={model.graph}
            selectedNodeId={selectedNodeId}
          />
        </div>
      </div>
    </ModelLayout>
  )
}
