import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useStore } from '../store'
import GraphCanvas from '../components/GraphCanvas'
import LayerTree from '../components/LayerTree'
import NodePanel from '../components/NodePanel'
import StatusBar from '../components/StatusBar'

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
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      {/* Top bar */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '8px 16px',
        background: 'var(--bg-secondary)',
        borderBottom: '1px solid var(--border)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <button onClick={() => navigate('/')}
            style={{
              background: 'none', border: 'none', color: 'var(--text-secondary)',
              cursor: 'pointer', fontSize: 16,
            }}>
            ←
          </button>
          <span style={{ fontWeight: 600, fontSize: 14 }}>
            {model.info?.file_name || '模型查看'}
          </span>
          <span style={{
            fontSize: 10, padding: '2px 8px', borderRadius: 10,
            background: 'var(--bg-tertiary)', color: 'var(--accent)',
          }}>
            {model.info?.format?.toUpperCase()}
          </span>
        </div>
      </div>

      {/* Main content: 3-column */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
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

      {/* Bottom: Status bar */}
      <StatusBar info={model.info} profile={model.profile} />
    </div>
  )
}
