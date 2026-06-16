import type { ChainResult } from '../types'

export default function ChainTimeline({ result }: { result: ChainResult }) {
  return (
    <div>
      <div style={{ display: 'flex', gap: 12, alignItems: 'stretch', marginBottom: 16 }}>
        {result.stages.map((stage, index) => (
          <div key={stage.label} style={{ display: 'flex', gap: 12, flex: 1 }}>
            <div style={{ flex: 1, background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 8, padding: 14 }}>
              <div style={{ fontSize: 12, color: 'var(--accent)', fontWeight: 700 }}>{stage.label}</div>
              <div style={{ fontSize: 16, color: 'var(--text-primary)', marginTop: 4 }}>{stage.format}</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>
                {stage.layer_count} layers · {stage.total_params.toLocaleString()} params · {stage.total_ops} ops
              </div>
            </div>
            {index < result.stages.length - 1 && <div style={{ alignSelf: 'center', color: 'var(--text-muted)', fontSize: 18 }}>→</div>}
          </div>
        ))}
      </div>

      {result.transitions.length > 0 && (
        <div style={{ background: 'var(--bg-secondary)', borderRadius: 8, padding: 14 }}>
          <div style={{ fontSize: 13, color: 'var(--text-primary)', fontWeight: 600, marginBottom: 10 }}>Transitions</div>
          {result.transitions.map((t) => (
            <div key={`${t.from}-${t.to}`} style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 12, color: 'var(--accent)', fontWeight: 700 }}>{t.from} → {t.to}</div>
              <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 4 }}>
                Layers: {t.layer_count_delta > 0 ? '+' : ''}{t.layer_count_delta} &nbsp;|&nbsp;
                Params: {t.param_count_delta > 0 ? '+' : ''}{t.param_count_delta.toLocaleString()}
              </div>
              {t.added_ops.length > 0 && <div style={{ fontSize: 11, color: 'var(--green)', marginTop: 2 }}>+ {t.added_ops.join(', ')}</div>}
              {t.removed_ops.length > 0 && <div style={{ fontSize: 11, color: 'var(--red)', marginTop: 2 }}>- {t.removed_ops.join(', ')}</div>}
              {t.renamed_ops.length > 0 && <div style={{ fontSize: 11, color: 'var(--accent)', marginTop: 2 }}>
                ~ {t.renamed_ops.map((r) => `${r.from}→${r.to}`).join(', ')}
              </div>}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
