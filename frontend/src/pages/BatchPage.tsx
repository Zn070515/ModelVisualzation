import { useState } from 'react'
import { generateReport } from '../api/client'
import ReportPreview from '../components/ReportPreview'
import type { BatchReport } from '../types'

export default function BatchPage() {
  const [ids, setIds] = useState('')
  const [report, setReport] = useState<BatchReport | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function run(format: 'json' | 'html') {
    const modelIds = ids.split(/[\s,]+/).map((id) => id.trim()).filter(Boolean)
    setError(null)
    generateReport(modelIds, format).then(setReport).catch((err) => setError(err.message))
  }

  return (
    <div style={page}>
      <div style={panel}>
        <textarea value={ids} onChange={(event) => setIds(event.target.value)} placeholder="model ids" style={textarea} />
        <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
          <button onClick={() => run('json')} style={button}>JSON</button>
          <button onClick={() => run('html')} style={button}>HTML</button>
        </div>
      </div>
      {error && <div style={empty}>{error}</div>}
      {report && <ReportPreview report={report} />}
      {report?.html && <pre style={pre}>{report.html}</pre>}
    </div>
  )
}

const page = { minHeight: '100vh', background: 'var(--bg-primary)', padding: 16 }
const panel = { background: 'var(--bg-secondary)', borderRadius: 8, padding: 14, marginBottom: 14 }
const textarea = { width: '100%', minHeight: 90, boxSizing: 'border-box' as const, background: 'var(--bg-tertiary)', color: 'var(--text-primary)', border: '1px solid var(--border)', borderRadius: 6, padding: 10 }
const button = { padding: '6px 12px', borderRadius: 6, border: '1px solid var(--border)', background: 'var(--accent)', color: '#fff', cursor: 'pointer' }
const empty = { color: 'var(--text-muted)', fontSize: 13, padding: 8 }
const pre = { background: 'var(--bg-secondary)', borderRadius: 8, padding: 14, overflow: 'auto', fontSize: 11, textAlign: 'left' as const }
