import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { getActivation } from '../api/client'
import ActivationHistogram from '../components/ActivationHistogram'
import ModelLayout from '../components/ModelLayout'
import type { ActivationResult } from '../types'

export default function ActivationPage() {
  const { modelId } = useParams<{ modelId: string }>()
  const [result, setResult] = useState<ActivationResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function onFile(file?: File) {
    if (!file || !modelId) return
    setError(null)
    getActivation(modelId, file).then(setResult).catch((err) => setError(err.message))
  }

  return (
    <ModelLayout modelId={modelId} activeTab="activation">
      <div style={page}>
        <label style={upload}>
          <input type="file" accept=".npy,.bin" onChange={(event) => onFile(event.target.files?.[0])} style={{ display: 'none' }} />
          Select sample input
        </label>
        {error && <div style={empty}>{error}</div>}
        {result && (
          <>
            <div style={cards}>
              <Metric label="Layers" value={String(result.summary.total_layers_analyzed)} />
              <Metric label="Dead layers" value={String(result.summary.layers_with_dead_neurons)} />
              <Metric label="Dead pct" value={`${result.summary.overall_dead_neuron_pct.toFixed(2)}%`} />
            </div>
            {result.activations.map((activation) => (
              <section key={activation.layer_name} style={panel}>
                <div style={title}>{activation.layer_name} · {activation.output_shape.join('x')}</div>
                <ActivationHistogram activation={activation} />
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
const upload = { display: 'inline-block', padding: '8px 14px', borderRadius: 6, background: 'var(--accent)', color: '#fff', cursor: 'pointer', marginBottom: 14 }
const cards = { display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 14 }
const metric = { background: 'var(--bg-secondary)', borderRadius: 8, padding: 14 }
const metricLabel = { fontSize: 11, color: 'var(--text-muted)' }
const metricValue = { fontSize: 20, fontWeight: 700, color: 'var(--text-primary)' }
const panel = { background: 'var(--bg-secondary)', borderRadius: 8, padding: 12, marginBottom: 14 }
const title = { fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8 }
const empty = { color: 'var(--text-muted)', fontSize: 13, padding: 20 }
