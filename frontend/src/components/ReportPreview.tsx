import type { BatchReport } from '../types'
import { formatNum } from '../utils'

export default function ReportPreview({ report }: { report: BatchReport }) {
  return (
    <div style={{ background: 'var(--bg-secondary)', borderRadius: 8, padding: 14 }}>
      <div style={{ display: 'flex', gap: 18, marginBottom: 12, fontSize: 12, color: 'var(--text-secondary)' }}>
        <span>Models {report.summary.model_count}</span>
        <span>Params {formatNum(report.summary.total_params)}</span>
        <span>Max layers {report.summary.max_layers}</span>
      </div>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
        <thead>
          <tr style={{ color: 'var(--text-muted)', textAlign: 'left' }}>
            <th style={cell}>ID</th>
            <th style={cell}>Format</th>
            <th style={cell}>Layers</th>
            <th style={cell}>Params</th>
            <th style={cell}>Ops</th>
          </tr>
        </thead>
        <tbody>
          {report.models.map((model) => (
            <tr key={model.model_id}>
              <td style={cell}>{model.model_id}</td>
              <td style={cell}>{model.format}</td>
              <td style={cell}>{model.layer_count}</td>
              <td style={cell}>{formatNum(model.param_count)}</td>
              <td style={cell}>{model.op_types.join(', ')}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const cell = { padding: '6px 8px', borderBottom: '1px solid var(--border)' }
