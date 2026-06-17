import { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { getAttention } from '../api/client'
import AttentionHeatmap from '../components/AttentionHeatmap'
import ModelLayout from '../components/ModelLayout'
import type { AttentionResponse, AttentionLayerData } from '../types'

export default function AttentionPage() {
  const { modelId } = useParams<{ modelId: string }>()
  const [result, setResult] = useState<AttentionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedLayerIdx, setSelectedLayerIdx] = useState(0)
  const [selectedProj, setSelectedProj] = useState('')

  useEffect(() => {
    if (!modelId) return
    setLoading(true)
    setError(null)
    getAttention(modelId)
      .then(setResult)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [modelId])

  const layer: AttentionLayerData | undefined = result?.layers?.[selectedLayerIdx]
  const projKeys = layer ? Object.keys(layer.projections) : []
  const activeProj = selectedProj || projKeys[0] || ''

  return (
    <ModelLayout modelId={modelId} activeTab="attention">
      <div style={page}>
        {loading && <div style={msg}>Loading attention data...</div>}
        {error && <div style={errStyle}>{error}</div>}

        {result && result.layers.length === 0 && !loading && (
          <div style={msg}>No attention layers found in this model</div>
        )}

        {result && result.layers.length > 0 && (
          <>
            {/* Layer selector */}
            <div style={controls}>
              <label style={lbl}>Layer:</label>
              <select value={selectedLayerIdx} onChange={(e) => { setSelectedLayerIdx(Number(e.target.value)); setSelectedProj('') }} style={sel}>
                {result.layers.map((l, i) => (
                  <option key={i} value={i}>{l.layer_name} ({l.num_heads}h×{l.head_dim}d)</option>
                ))}
              </select>
              {projKeys.length > 1 && (
                <>
                  <label style={{ ...lbl, marginLeft: 12 }}>Projection:</label>
                  <select value={activeProj} onChange={(e) => setSelectedProj(e.target.value)} style={sel}>
                    {projKeys.map((k) => <option key={k} value={k}>{k}</option>)}
                  </select>
                </>
              )}
            </div>

            {/* Layer stats */}
            {layer && (
              <div style={cards}>
                <Metric label="Heads" value={String(layer.num_heads)} />
                <Metric label="Head dim" value={String(layer.head_dim)} />
                <Metric label="Embed dim" value={String(layer.embed_dim)} />
                <Metric label="Projections" value={String(projKeys.length)} />
              </div>
            )}

            {/* Per-head heatmaps */}
            {layer && activeProj && layer.projections[activeProj] && (
              <div style={grid}>
                {layer.projections[activeProj].head_slices.map((hs, i) => (
                  <div key={i} style={heatmapCard}>
                    <AttentionHeatmap
                      values={hs.values}
                      title={`Head ${hs.head_index} ${hs.shape.join('×')}`}
                      height={140}
                    />
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </ModelLayout>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div style={metricCard}><div style={metricLbl}>{label}</div><div style={metricVal}>{value}</div></div>
}

const page = { height: '100%', overflowY: 'auto' as const, padding: 16 }
const msg = { color: 'var(--text-muted)', fontSize: 13, padding: 20 }
const errStyle = { color: 'var(--accent-red)', fontSize: 13, padding: 12 }
const controls = { display: 'flex', alignItems: 'center', marginBottom: 14, gap: 8 }
const lbl = { fontSize: 11, color: 'var(--text-muted)' }
const sel = { padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-tertiary)', color: 'var(--text-primary)', fontSize: 11, outline: 'none' }
const cards = { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginBottom: 14 }
const metricCard = { background: 'var(--bg-secondary)', borderRadius: 8, padding: 12 }
const metricLbl = { fontSize: 10, color: 'var(--text-muted)' }
const metricVal = { fontSize: 15, fontWeight: 700, color: 'var(--text-primary)' }
const grid = { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 10 }
const heatmapCard = { background: 'var(--bg-secondary)', borderRadius: 8, padding: 8 }
