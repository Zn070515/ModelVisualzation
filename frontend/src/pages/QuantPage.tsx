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
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!modelId) return
    setError(null)
    setLoading(true)
    simulateQuant(modelId, bits)
      .then(setResult)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [modelId, bits])

  const worstWeight = result?.layers.find((l) => l.layer_name === result.summary.worst_layer)

  return (
    <ModelLayout modelId={modelId} activeTab="quant">
      <div style={page}>
        <div style={toolbar}>
          {[8, 16].map((item) => (
            <button key={item} onClick={() => setBits(item as 8 | 16)} style={tab(bits === item)}>
              {item === 8 ? 'INT8' : 'FP16'}
            </button>
          ))}
        </div>
        {loading && <div style={loader}>Simulating...</div>}
        {error && <div style={empty}>{error}</div>}
        {result && (
          <>
            <div style={cards}>
              <Metric label="Mean abs err" value={result.summary.overall_mean_abs_err.toExponential(2)} />
              <Metric label="RMSE" value={result.summary.overall_rmse.toExponential(2)} />
              <Metric label="Worst layer" value={result.summary.worst_layer || '-'} />
              {worstWeight && <Metric label="Worst RMSE" value={worstWeight.weights ? Object.values(worstWeight.weights)[0]?.error?.rmse.toExponential(2) || '-' : '-'} />}
            </div>
            <section style={panel}><QuantHeatmap result={result} /></section>
            <section style={panel}>
              <table style={table}>
                <thead>
                  <tr>
                    <th style={th}>Layer</th>
                    <th style={th}>Type</th>
                    <th style={thR}>Max Abs Err</th>
                    <th style={thR}>RMSE</th>
                    <th style={thR}>SNR (dB)</th>
                  </tr>
                </thead>
                <tbody>
                  {result.layers.map((layer) => {
                    const firstW = Object.values(layer.weights)[0]
                    const err = firstW?.error
                    return (
                      <tr key={layer.layer_name}>
                        <td style={cell}>{layer.layer_name}</td>
                        <td style={cell}>{layer.op_type}</td>
                        <td style={right}>{err?.max_abs_err?.toExponential(2) ?? '-'}</td>
                        <td style={right}>{err?.rmse?.toExponential(2) ?? '-'}</td>
                        <td style={right}>{err?.snr_db?.toFixed(2) ?? '-'}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </section>
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
const tab = (active: boolean) => ({ padding: '6px 12px', borderRadius: 6, border: '1px solid var(--border)', background: active ? 'var(--accent)' : 'var(--bg-secondary)', color: active ? '#fff' : 'var(--text-secondary)', cursor: 'pointer', fontSize: 11, fontWeight: 600 })
const cards = { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 14 }
const metric = { background: 'var(--bg-secondary)', borderRadius: 8, padding: 14 }
const metricLabel = { fontSize: 11, color: 'var(--text-muted)' }
const metricValue = { fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }
const panel = { background: 'var(--bg-secondary)', borderRadius: 8, padding: 12, marginBottom: 14 }
const loader = { color: 'var(--accent)', fontSize: 13, padding: 12 }
const empty = { color: 'var(--text-muted)', fontSize: 13, padding: 20 }
const table = { width: '100%', borderCollapse: 'collapse' as const, fontSize: 11 }
const th = { padding: '6px 8px', borderBottom: '1px solid var(--border)', color: 'var(--text-muted)', textAlign: 'left' as const, fontSize: 10, fontWeight: 600 }
const thR = { ...th, textAlign: 'right' as const }
const cell = { padding: '5px 8px', borderBottom: '1px solid var(--border)', color: 'var(--text-secondary)' }
const right = { ...cell, textAlign: 'right' as const }
