import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { estimatePerf } from '../api/client'
import LatencyBarChart from '../components/LatencyBarChart'
import ModelLayout from '../components/ModelLayout'
import type { PerfResult } from '../types'
import { formatBytes, formatNum } from '../utils'

type Hardware = 'cpu' | 'gpu' | 'edge_tpu'

export default function PerformancePage() {
  const { modelId } = useParams<{ modelId: string }>()
  const [hardware, setHardware] = useState<Hardware>('cpu')
  const [result, setResult] = useState<PerfResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!modelId) return
    setError(null)
    estimatePerf(modelId, hardware).then(setResult).catch((err) => setError(err.message))
  }, [modelId, hardware])

  return (
    <ModelLayout modelId={modelId} activeTab="performance">
      <div style={page}>
        <div style={toolbar}>
          {(['cpu', 'gpu', 'edge_tpu'] as Hardware[]).map((item) => (
            <button key={item} onClick={() => setHardware(item)} style={button(item === hardware)}>{item.toUpperCase()}</button>
          ))}
        </div>
        {error && <div style={empty}>{error}</div>}
        {result && (
          <>
            <div style={cards}>
              <Metric label="Latency" value={`${result.summary.total_latency_ms.toFixed(3)} ms`} />
              <Metric label="Bottlenecks" value={String(result.summary.bottleneck_count)} />
              <Metric label="Memory" value={formatBytes(result.summary.memory_total_bytes)} />
            </div>
            <section style={panel}><LatencyBarChart layers={result.layers} /></section>
            <section style={panel}>
              <table style={table}>
                <tbody>
                  {result.layers.map((layer) => (
                    <tr key={layer.name}>
                      <td style={cell}>{layer.name}</td>
                      <td style={cell}>{layer.op_type}</td>
                      <td style={right}>{formatNum(layer.flops)}</td>
                      <td style={right}>{layer.est_latency_us.toFixed(3)} us</td>
                    </tr>
                  ))}
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
const button = (active: boolean) => ({ padding: '6px 12px', borderRadius: 6, border: '1px solid var(--border)', background: active ? 'var(--accent)' : 'var(--bg-secondary)', color: active ? '#fff' : 'var(--text-secondary)', cursor: 'pointer' })
const cards = { display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 14 }
const metric = { background: 'var(--bg-secondary)', borderRadius: 8, padding: 14 }
const metricLabel = { fontSize: 11, color: 'var(--text-muted)' }
const metricValue = { fontSize: 20, fontWeight: 700, color: 'var(--text-primary)' }
const panel = { background: 'var(--bg-secondary)', borderRadius: 8, padding: 12, marginBottom: 14 }
const empty = { color: 'var(--text-muted)', fontSize: 13, padding: 20 }
const table = { width: '100%', borderCollapse: 'collapse' as const, fontSize: 11 }
const cell = { padding: '6px 8px', borderBottom: '1px solid var(--border)', color: 'var(--text-secondary)' }
const right = { ...cell, textAlign: 'right' as const }
