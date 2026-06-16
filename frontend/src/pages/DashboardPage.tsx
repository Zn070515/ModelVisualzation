import { useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { useStore } from '../store'
import ModelLayout from '../components/ModelLayout'
import LayerTypeCharts from '../components/LayerTypeCharts'
import { formatNum, formatBytes } from '../utils'

export default function DashboardPage() {
  const { modelId } = useParams<{ modelId: string }>()
  const model = useStore((s) => (modelId ? s.models[modelId] : undefined))
  const loadModelData = useStore((s) => s.loadModelData)
  const profile = model?.profile
  const info = model?.info

  useEffect(() => {
    if (!modelId) return
    if (!model || (!model.graph && !model.loading)) {
      loadModelData(modelId)
    }
  }, [modelId])

  if (!profile) {
    return (
      <ModelLayout modelId={modelId} activeTab="dashboard">
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          height: '100%', color: 'var(--text-muted)', fontSize: 13,
        }}>
          {model?.loading ? '加载中...' : '暂无模型数据'}
        </div>
      </ModelLayout>
    )
  }

  const cards = [
    { label: '总参数量', value: formatNum(profile.total_params), color: '#7aa2f7' },
    { label: '总 FLOPs', value: formatNum(profile.total_flops), color: '#bb9af7' },
    { label: '显存估算', value: profile.memory_mb.toFixed(2) + ' MB', color: '#73daca' },
    { label: '总层数', value: String(info?.layer_count || profile.layers.length), color: '#e0af68' },
    { label: '文件大小', value: info ? formatBytes(info.file_size_bytes) : '-', color: '#f7768e' },
    { label: '格式', value: info?.format?.toUpperCase() || '-', color: '#ff9e64' },
  ]

  return (
    <ModelLayout modelId={modelId} activeTab="dashboard">
      <div style={{ overflowY: 'auto', height: '100%', padding: 16 }}>
        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 10, marginBottom: 16,
        }}>
          {cards.map((c) => (
            <div key={c.label} style={{
              background: 'var(--bg-secondary)', borderRadius: 8, padding: '12px 16px',
              borderLeft: `3px solid ${c.color}`,
            }}>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4 }}>{c.label}</div>
              <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>{c.value}</div>
            </div>
          ))}
        </div>

        <LayerTypeCharts profile={profile} />

        <div style={{
          background: 'var(--bg-secondary)', borderRadius: 8, padding: 12, marginTop: 16,
        }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8 }}>
            各层详情
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
            <thead>
              <tr style={{ color: 'var(--text-muted)', textAlign: 'left' }}>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>#</th>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>名称</th>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)' }}>类型</th>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)', textAlign: 'right' }}>参数量</th>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)', textAlign: 'right' }}>FLOPs</th>
                <th style={{ padding: '4px 8px', borderBottom: '1px solid var(--border)', textAlign: 'right' }}>占比</th>
              </tr>
            </thead>
            <tbody>
              {profile.layers.map((l, i) => (
                <tr key={l.name} style={{ color: 'var(--text-secondary)' }}>
                  <td style={{ padding: '4px 8px', color: 'var(--text-muted)' }}>{i + 1}</td>
                  <td style={{
                    padding: '4px 8px', fontFamily: "'JetBrains Mono', monospace", fontSize: 10,
                    maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  }}>
                    {l.name}
                  </td>
                  <td style={{ padding: '4px 8px', color: 'var(--accent)' }}>{l.op_type}</td>
                  <td style={{ padding: '4px 8px', textAlign: 'right' }}>{formatNum(l.params_count)}</td>
                  <td style={{ padding: '4px 8px', textAlign: 'right' }}>{formatNum(l.flops)}</td>
                  <td style={{ padding: '4px 8px', textAlign: 'right' }}>
                    {profile.total_params > 0
                      ? ((l.params_count / profile.total_params) * 100).toFixed(1) + '%'
                      : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </ModelLayout>
  )
}
