import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { estimatePerf } from '../api/client'
import LatencyBarChart from '../components/LatencyBarChart'
import ModelLayout from '../components/ModelLayout'
import type { PerfResult } from '../types'
import { formatBytes, formatNum } from '../utils'

type Hardware = 'i9_13900k' | 'rtx_4090' | 'apple_m2' | 'rpi4'

const HW_LABELS: Record<Hardware, string> = {
  i9_13900k: 'i9-13900K',
  rtx_4090: 'RTX 4090',
  apple_m2: 'Apple M2',
  rpi4: 'RPi 4',
}

export default function PerformancePage() {
  const { modelId } = useParams<{ modelId: string }>()
  const [hardware, setHardware] = useState<Hardware>('i9_13900k')
  const [result, setResult] = useState<PerfResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!modelId) return
    setError(null)
    setLoading(true)
    estimatePerf(modelId, hardware)
      .then(setResult)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [modelId, hardware])

  return (
    <ModelLayout modelId={modelId} activeTab="performance">
      <div style={page}>
        <div style={toolbar}>
          {(Object.keys(HW_LABELS) as Hardware[]).map((item) => (
            <button key={item} onClick={() => setHardware(item)} style={tab(item === hardware)}>{HW_LABELS[item]}</button>
          ))}
        </div>
        {loading && <div style={loader}>Estimating...</div>}
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
                <thead>
                  <tr>
                    <th style={th}>Layer</th>
                    <th style={th}>Type</th>
                    <th style={thR}>Params</th>
                    <th style={thR}>FLOPs</th>
                    <th style={thR}>Latency</th>
                    <th style={thR}>Bound</th>
                    <th style={thR}>Memory R/W</th>
                  </tr>
                </thead>
                <tbody>
                  {result.layers.map((layer) => (
                    <tr key={layer.name} style={{ background: layer.is_bottleneck ? 'rgba(247,118,142,0.06)' : undefined }}>
                      <td style={cell}>{layer.name}</td>
                      <td style={cell}>{layer.op_type}</td>
                      <td style={right}>{formatNum(layer.params)}</td>
                      <td style={right}>{formatNum(layer.flops)}</td>
                      <td style={right}><span style={{ color: layer.is_bottleneck ? 'var(--red)' : undefined }}>{layer.est_latency_us.toFixed(1)} us</span></td>
                      <td style={right}><span style={{ color: boundColor(layer.bound) }}>{layer.bound}</span></td>
                      <td style={right}>{formatBytes(layer.memory_read_bytes + layer.memory_write_bytes)}</td>
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

function boundColor(bound: string): string {
  if (bound === 'compute') return 'var(--red)'
  if (bound === 'memory') return 'var(--accent)'
  return 'var(--text-secondary)'
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div style={metric}><div style={metricLabel}>{label}</div><div style={metricValue}>{value}</div></div>
}

const page = { height: '100%', overflowY: 'auto' as const, padding: 16 }
const toolbar = { display: 'flex', gap: 8, marginBottom: 14 }
const tab = (active: boolean) => ({ padding: '6px 12px', borderRadius: 6, border: '1px solid var(--border)', background: active ? 'var(--accent)' : 'var(--bg-secondary)', color: active ? '#fff' : 'var(--text-secondary)', cursor: 'pointer', fontSize: 11, fontWeight: 600 })
const cards = { display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 14 }
const metric = { background: 'var(--bg-secondary)', borderRadius: 8, padding: 14 }
const metricLabel = { fontSize: 11, color: 'var(--text-muted)' }
const metricValue = { fontSize: 20, fontWeight: 700, color: 'var(--text-primary)' }
const panel = { background: 'var(--bg-secondary)', borderRadius: 8, padding: 12, marginBottom: 14 }
const loader = { color: 'var(--accent)', fontSize: 13, padding: 12 }
const empty = { color: 'var(--text-muted)', fontSize: 13, padding: 20 }
const table = { width: '100%', borderCollapse: 'collapse' as const, fontSize: 11 }
const th = { padding: '6px 8px', borderBottom: '1px solid var(--border)', color: 'var(--text-muted)', textAlign: 'left' as const, fontSize: 10, fontWeight: 600 }
const thR = { ...th, textAlign: 'right' as const }
const cell = { padding: '5px 8px', borderBottom: '1px solid var(--border)', color: 'var(--text-secondary)' }
const right = { ...cell, textAlign: 'right' as const }
