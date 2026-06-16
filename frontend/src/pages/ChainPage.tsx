import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { traceChain } from '../api/client'
import ChainTimeline from '../components/ChainTimeline'
import type { ChainResult } from '../types'

export default function ChainPage() {
  const [searchParams] = useSearchParams()
  const modelIds = searchParams.getAll('ids')
  const [result, setResult] = useState<ChainResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [inputIds, setInputIds] = useState(modelIds.join(','))

  useEffect(() => {
    if (modelIds.length < 2) return
    setError(null)
    traceChain(modelIds)
      .then(setResult)
      .catch((err) => setError(err.message))
  }, [searchParams])

  const handleTrace = () => {
    const ids = inputIds.split(',').map((s) => s.trim()).filter(Boolean)
    if (ids.length < 2) {
      setError('Enter at least 2 model IDs')
      return
    }
    const params = new URLSearchParams()
    ids.forEach((id) => params.append('ids', id))
    window.history.replaceState(null, '', `?${params.toString()}`)
    setError(null)
    traceChain(ids)
      .then(setResult)
      .catch((err) => setError(err.message))
  }

  return (
    <div style={page}>
      <h1 style={heading}>Conversion Chain</h1>
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        <input
          value={inputIds}
          onChange={(e) => setInputIds(e.target.value)}
          placeholder="model IDs, comma-separated (e.g. a1,b2,c3)"
          style={input}
        />
        <button onClick={handleTrace} style={btn}>Trace</button>
      </div>

      {error && <div style={empty}>{error}</div>}
      {result && <ChainTimeline result={result} />}
    </div>
  )
}

const page = { minHeight: '100vh', background: 'var(--bg-primary)', padding: 24 } as const
const heading = { fontSize: 20, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 16 }
const input = {
  flex: 1,
  padding: '8px 12px',
  background: 'var(--bg-secondary)',
  border: '1px solid var(--border)',
  borderRadius: 6,
  color: 'var(--text-primary)',
  fontSize: 13,
}
const btn = {
  padding: '8px 16px',
  background: 'var(--accent)',
  color: '#fff',
  border: 'none',
  borderRadius: 6,
  fontSize: 13,
  fontWeight: 600,
  cursor: 'pointer',
}
const empty = { color: 'var(--text-muted)', fontSize: 13, padding: 8 }
