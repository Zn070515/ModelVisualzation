import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { compareModels } from '../api/client'
import DiffBadge from '../components/DiffBadge'
import type { CompareResult } from '../types'
import { formatNum } from '../utils'

export default function ComparePage() {
  const { aId = '', bId = '' } = useParams<{ aId: string; bId: string }>()
  const [result, setResult] = useState<CompareResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!aId || !bId) return
    setError(null)
    compareModels(aId, bId).then(setResult).catch((err) => setError(err.message))
  }, [aId, bId])

  return (
    <div style={page}>
      {error && <div style={empty}>{error}</div>}
      {result && (
        <>
          <div style={cards}>
            <Metric label="Similarity" value={`${Math.round(result.summary.similarity * 100)}%`} />
            <Metric label="Layer delta" value={String(result.summary.layer_count_delta)} />
            <Metric label="Param delta" value={formatNum(result.summary.param_count_delta)} />
          </div>
          <section style={panel}>
            <table style={table}>
              <thead>
                <tr>
                  <th style={th}>Layer</th>
                  <th style={th}>Status</th>
                  <th style={th}>Model A</th>
                  <th style={th}>Model B</th>
                  <th style={thR}>Param Delta</th>
                </tr>
              </thead>
              <tbody>
                {result.layers.map((row) => (
                  <tr key={row.name}>
                    <td style={cell}>{row.name}</td>
                    <td style={cell}><DiffBadge status={row.status} /></td>
                    <td style={cell}>{row.a?.op_type || '-'}</td>
                    <td style={cell}>{row.b?.op_type || '-'}</td>
                    <td style={right}>{formatNum(row.param_delta)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
          {result.weight_diffs?.length > 0 && (
            <section style={panel}>
              <div style={sectionTitle}>Weight Differences</div>
              <table style={table}>
                <thead>
                  <tr>
                    <th style={th}>Layer</th>
                    <th style={th}>Weight</th>
                    <th style={thR}>Mean Diff</th>
                    <th style={thR}>Std Diff</th>
                  </tr>
                </thead>
                <tbody>
                  {result.weight_diffs.map((wd) => (
                    <tr key={`${wd.layer_name}-${wd.weight_name}`}>
                      <td style={cell}>{wd.layer_name}</td>
                      <td style={cell}>{wd.weight_name}</td>
                      <td style={right}>{wd.mean_diff}</td>
                      <td style={right}>{wd.std_diff}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}
        </>
      )}
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div style={metric}><div style={metricLabel}>{label}</div><div style={metricValue}>{value}</div></div>
}

const page = { minHeight: '100vh', background: 'var(--bg-primary)', padding: 16 }
const cards = { display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 14 }
const metric = { background: 'var(--bg-secondary)', borderRadius: 8, padding: 14 }
const metricLabel = { fontSize: 11, color: 'var(--text-muted)' }
const metricValue = { fontSize: 20, fontWeight: 700, color: 'var(--text-primary)' }
const panel = { background: 'var(--bg-secondary)', borderRadius: 8, padding: 12, marginTop: 14 }
const sectionTitle = { fontSize: 13, color: 'var(--text-primary)', fontWeight: 600, marginBottom: 8 }
const table = { width: '100%', borderCollapse: 'collapse' as const, fontSize: 11 }
const th = { padding: '6px 8px', borderBottom: '1px solid var(--border)', color: 'var(--text-muted)', textAlign: 'left' as const, fontSize: 10 }
const thR = { ...th, textAlign: 'right' as const }
const cell = { padding: '6px 8px', borderBottom: '1px solid var(--border)', color: 'var(--text-secondary)' }
const right = { ...cell, textAlign: 'right' as const }
const empty = { color: 'var(--text-muted)', fontSize: 13, padding: 20 }
