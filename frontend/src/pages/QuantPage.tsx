import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { simulateQuant } from '../api/client'
import ModelLayout from '../components/ModelLayout'
import QuantHeatmap from '../components/QuantHeatmap'
import type { QuantResult } from '../types'

export default function QuantPage() {
  const { modelId } = useParams<{ modelId: string }>()
  const [bits, setBits] = useState<8 | 16>(8)
  const [result, setResult] = useState<QuantResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!modelId) return
    setError(null)
    simulateQuant(modelId, bits).then(setResult).catch((err) => setError(err.message))
  }, [modelId, bits])

  return (
    <ModelLayout modelId={modelId} activeTab="quant">
      <div style={page}>
        <div style={toolbar}>
          {[8, 16].map((item) => <button key={item} onClick={() => setBits(item as 8 | 16)} style={button(bits === item)}>{item === 8 ? 'INT8' : 'FP16'}</button>)}
        </div>
        {error && <div style={empty}>{error}</div>}
        {result && (
          <>
            <div style={cards}>
              <Metric label="Mean abs err" value={result.summary.overall_mean_abs_err.toExponential(2)} />
              <Metric label="RMSE" value={result.summary.overall_rmse.toExponential(2)} />
              <Metric label="Worst layer" value={result.summary.worst_layer || '-'} />
            </div>
            <section style={panel}><QuantHeatmap result={result} /></section>
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
const toolbar = { display: 'flex', gap: 8, marginBottom: 14 }
const button = (active: boolean) => ({ padding: '6px 12px', borderRadius: 6, border: '1px solid var(--border)', background: active ? 'var(--accent)' : 'var(--bg-secondary)', color: active ? '#fff' : 'var(--text-secondary)', cursor: 'pointer' })
const cards = { display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 14 }
const metric = { background: 'var(--bg-secondary)', borderRadius: 8, padding: 14 }
const metricLabel = { fontSize: 11, color: 'var(--text-muted)' }
const metricValue = { fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }
const panel = { background: 'var(--bg-secondary)', borderRadius: 8, padding: 12 }
const empty = { color: 'var(--text-muted)', fontSize: 13, padding: 20 }
