import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { analyzePrune } from '../api/client'
import ImportanceChart from '../components/ImportanceChart'
import ModelLayout from '../components/ModelLayout'
import type { PruneResult } from '../types'

export default function PrunePage() {
  const { modelId } = useParams<{ modelId: string }>()
  const [result, setResult] = useState<PruneResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!modelId) return
    setError(null)
    setLoading(true)
    analyzePrune(modelId)
      .then(setResult)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [modelId])

  return (
    <ModelLayout modelId={modelId} activeTab="prune">
      <div style={page}>
        {loading && <div style={loader}>Analyzing channels...</div>}
        {error && <div style={empty}>{error}</div>}
        {result && (
          <>
            <div style={cards}>
              <Metric label="Prunable layers" value={String(result.summary.total_prunable_layers)} />
              <Metric label="Prunable params" value={`${result.summary.total_prunable_params_pct.toFixed(2)}%`} />
              <Metric label="Rec. ratio" value={`${Math.round(result.summary.recommended_prune_ratio * 100)}%`} />
            </div>
            {result.layers.length === 0 && <div style={empty}>No prunable layers found with weight tensors</div>}
            {result.layers.map((layer) => (
              <section key={layer.layer_name} style={panel}>
                <div style={title}>{layer.layer_name} · {layer.op_type}</div>
                <div style={info}>
                  {layer.total_channels} channels · {layer.prunable_channels_30pct} prunable @30% · {layer.prunable_channels_50pct} @50%
                </div>
                <ImportanceChart layer={layer} />
              </section>
            ))}
          </>
        )}
      </div>
    </ModelLayout>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div style={metric}><div style={metricLabel}>{label}</div><div style={metricValue}>{value}</div></div>
}

const page = { height: '100%', overflowY: 'auto' as const, padding: 16 }
const cards = { display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 14 }
const metric = { background: 'var(--bg-secondary)', borderRadius: 8, padding: 14 }
const metricLabel = { fontSize: 11, color: 'var(--text-muted)' }
const metricValue = { fontSize: 20, fontWeight: 700, color: 'var(--text-primary)' }
const panel = { background: 'var(--bg-secondary)', borderRadius: 8, padding: 12, marginBottom: 14 }
const title = { fontSize: 13, color: 'var(--text-primary)', fontWeight: 600, marginBottom: 4 }
const info = { fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }
const loader = { color: 'var(--accent)', fontSize: 13, padding: 12 }
const empty = { color: 'var(--text-muted)', fontSize: 13, padding: 20 }
