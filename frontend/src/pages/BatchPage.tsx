import { useState } from 'react'
import { generateReport } from '../api/client'
import ReportPreview from '../components/ReportPreview'
import type { BatchReport } from '../types'

export default function BatchPage() {
  const [ids, setIds] = useState('')
  const [report, setReport] = useState<BatchReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function run(format: 'json' | 'html') {
    const modelIds = ids.split(/[\s,]+/).map((id) => id.trim()).filter(Boolean)
    if (!modelIds.length) {
      setError('Enter at least one model ID')
      return
    }
    setError(null)
    setLoading(true)
    try {
      setReport(await generateReport(modelIds, format))
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={page}>
      <h1 style={heading}>Batch Report</h1>
      <div style={panel}>
        <textarea value={ids} onChange={(event) => setIds(event.target.value)} placeholder="model IDs (comma or space separated)" style={textarea} />
        <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
          <button onClick={() => run('json')} style={button} disabled={loading}>JSON</button>
          <button onClick={() => run('html')} style={button} disabled={loading}>HTML</button>
        </div>
      </div>
      {loading && <div style={loadingMsg}>Generating report...</div>}
      {error && <div style={errorMsg}>{error}</div>}
      {report && <ReportPreview report={report} />}
      {report?.html && <pre style={pre}>{report.html.slice(0, 4000)}</pre>}
    </div>
  )
}

const page = { minHeight: '100vh', background: 'var(--bg-primary)', padding: 24 }
const heading = { fontSize: 20, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 16 }
const panel = { background: 'var(--bg-secondary)', borderRadius: 8, padding: 14, marginBottom: 14 }
const textarea = { width: '100%', minHeight: 90, boxSizing: 'border-box' as const, background: 'var(--bg-tertiary)', color: 'var(--text-primary)', border: '1px solid var(--border)', borderRadius: 6, padding: 10, resize: 'vertical' as const }
const button = { padding: '6px 12px', borderRadius: 6, border: '1px solid var(--border)', background: 'var(--accent)', color: '#fff', cursor: 'pointer', fontSize: 12 }
const loadingMsg = { color: 'var(--accent)', fontSize: 13, padding: 12 }
const errorMsg = { color: 'var(--red)', fontSize: 13, padding: 8 }
const pre = { background: 'var(--bg-secondary)', borderRadius: 8, padding: 14, overflow: 'auto', fontSize: 11, maxHeight: 400 }
