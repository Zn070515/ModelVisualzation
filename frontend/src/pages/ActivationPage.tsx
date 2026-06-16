import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { getActivation } from '../api/client'
import ActivationHistogram from '../components/ActivationHistogram'
import ModelLayout from '../components/ModelLayout'
import type { ActivationResult } from '../types'

export default function ActivationPage() {
  const { modelId } = useParams<{ modelId: string }>()
  const [result, setResult] = useState<ActivationResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [fileName, setFileName] = useState('')

  async function onFile(file?: File) {
    if (!file || !modelId) return
    setFileName(file.name)
    setError(null)
    setLoading(true)
    try {
      setResult(await getActivation(modelId, file))
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <ModelLayout modelId={modelId} activeTab="activation">
      <div style={page}>
        <label style={upload}>
          <input type="file" accept=".npy,.bin" onChange={(event) => onFile(event.target.files?.[0])} style={{ display: 'none' }} />
          {fileName ? `Sample: ${fileName}` : 'Select sample input (.npy, .bin)'}
        </label>
        {loading && <div style={loader}>Running inference...</div>}
        {error && <div style={empty}>{error}</div>}
        {result && (
          <>
            <div style={cards}>
              <Metric label="Layers analyzed" value={String(result.summary.total_layers_analyzed)} />
              <Metric label="Dead neurons" value={String(result.summary.layers_with_dead_neurons)} />
              <Metric label="Dead neuron %" value={`${result.summary.overall_dead_neuron_pct.toFixed(2)}%`} />
              <Metric label="Method" value={result.activations[0]?.method || 'synthetic'} />
            </div>
            {result.activations.length === 0 && <div style={empty}>No activation data collected</div>}
            {result.activations.map((activation) => (
              <section key={activation.layer_name} style={panel}>
                <div style={title}>
                  {activation.layer_name} · {activation.output_shape.join('x')}
                </div>
                <div style={info}>
                  dead: {activation.dead_neurons_pct.toFixed(2)}% · sat: {activation.saturation_pct.toFixed(2)}%
                  {' · '}mean: {activation.stats.mean.toFixed(4)} · std: {activation.stats.std.toFixed(4)}
                </div>
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
const upload = { display: 'inline-block', padding: '8px 14px', borderRadius: 6, background: 'var(--accent)', color: '#fff', cursor: 'pointer', marginBottom: 14, fontSize: 12 }
const cards = { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 14 }
const metric = { background: 'var(--bg-secondary)', borderRadius: 8, padding: 14 }
const metricLabel = { fontSize: 11, color: 'var(--text-muted)' }
const metricValue = { fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }
const panel = { background: 'var(--bg-secondary)', borderRadius: 8, padding: 12, marginBottom: 14 }
const title = { fontSize: 13, color: 'var(--text-primary)', fontWeight: 600, marginBottom: 4 }
const info = { fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }
const loader = { color: 'var(--accent)', fontSize: 13, padding: 12 }
const empty = { color: 'var(--text-muted)', fontSize: 13, padding: 20 }
