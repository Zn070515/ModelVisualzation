import { useState, useEffect, useMemo } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useStore } from '../store'
import ModelLayout from '../components/ModelLayout'
import GraphCanvas, { type NodeMetric } from '../components/GraphCanvas'
import LayerTree from '../components/LayerTree'
import NodePanel from '../components/NodePanel'
import { formatNumber, formatMemory } from '../utils'

export default function ModelViewer() {
  const { modelId } = useParams<{ modelId: string }>()
  const navigate = useNavigate()
  const model = useStore((s) => (modelId ? s.models[modelId] : undefined))
  const loadModelData = useStore((s) => s.loadModelData)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [searchText, setSearchText] = useState('')
  const [colorMetric, setColorMetric] = useState<string>('')

  const nodeMetrics: NodeMetric[] = useMemo(() => {
    if (!colorMetric || !model?.profile?.layers) return []
    const layers = model.profile.layers
    const metricMap = new Map(layers.map((l) => [l.name, l]))
    const values: NodeMetric[] = []
    for (const node of model.graph?.nodes ?? []) {
      const lp = metricMap.get(node.id)
      if (!lp) continue
      if (colorMetric === 'params') values.push({ nodeId: node.id, value: lp.params_count, label: formatNumber(lp.params_count) })
      else if (colorMetric === 'flops') values.push({ nodeId: node.id, value: lp.flops, label: formatNumber(lp.flops) })
    }
    return values
  }, [colorMetric, model?.profile, model?.graph])

  const { highlightedNodeIds, dimmedNodeIds } = useMemo(() => {
    if (!model?.graph || !searchText.trim()) return { highlightedNodeIds: undefined, dimmedNodeIds: undefined }
    const q = searchText.toLowerCase()
    const matched = new Set(
      model.graph.nodes
        .filter((n) =>
          n.id.toLowerCase().includes(q) ||
          (n.data.opType as string).toLowerCase().includes(q) ||
          (n.data.label as string).toLowerCase().includes(q),
        )
        .map((n) => n.id),
    )
    const all = new Set(model.graph.nodes.map((n) => n.id))
    const dimmed = new Set([...all].filter((id) => !matched.has(id)))
    return { highlightedNodeIds: matched, dimmedNodeIds: dimmed }
  }, [model?.graph, searchText])

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
            searchText={searchText}
            onSearchChange={setSearchText}
          />
        </div>

        {/* Center: Graph */}
        <div style={{ flex: 1, background: 'var(--bg-primary)', display: 'flex', flexDirection: 'column' }}>
          <div style={toolbar}>
            <span style={toolbarLabel}>Overlay:</span>
            <select
              value={colorMetric}
              onChange={(e) => setColorMetric(e.target.value)}
              style={select}>
              <option value="">Op type (default)</option>
              <option value="params">Param count</option>
              <option value="flops">FLOPs</option>
            </select>
          </div>
          <div style={{ flex: 1 }}>
            <GraphCanvas
              graphData={model.graph}
              onNodeClick={setSelectedNodeId}
              highlightedNodeIds={highlightedNodeIds}
              dimmedNodeIds={dimmedNodeIds}
              colorMetric={colorMetric || undefined}
              nodeMetrics={nodeMetrics.length > 0 ? nodeMetrics : undefined}
            />
          </div>
        </div>
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
            colorMetric={colorMetric || undefined}
            nodeMetrics={nodeMetrics.length > 0 ? nodeMetrics : undefined}
            profile={model.profile}
          />
        </div>
      </div>
    </ModelLayout>
  )
}

const toolbar = { display: 'flex', alignItems: 'center', gap: 10, padding: '6px 12px', borderBottom: '1px solid var(--border)', background: 'var(--bg-secondary)' }
const toolbarLabel = { fontSize: 11, color: 'var(--text-muted)', whiteSpace: 'nowrap' as const }
const select = { padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)', fontSize: 11, outline: 'none' }
