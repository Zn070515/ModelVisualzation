import { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { useStore } from '../store'
import { getHealthScan } from '../api/client'
import type { HealthResult } from '../types'
import ModelLayout from '../components/ModelLayout'

const SEVERITY_COLORS: Record<string, string> = {
  critical: 'var(--accent-red)',
  warning: 'var(--accent-gold)',
}

const TYPE_LABELS: Record<string, string> = {
  weight_outlier: '权重离群值',
  high_sparsity: '高稀疏度',
  bn_anomaly: 'BN 参数异常',
  vanishing_gradient_risk: '梯度消失风险',
  dead_relu_risk: 'ReLU 死亡风险',
}

export default function HealthPage() {
  const { modelId } = useParams<{ modelId: string }>()
  const model = useStore((s) => (modelId ? s.models[modelId] : undefined))

  const [health, setHealth] = useState<HealthResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expandedIssue, setExpandedIssue] = useState<number | null>(null)

  useEffect(() => {
    if (!modelId) return
    setLoading(true)
    setError(null)
    getHealthScan(modelId)
      .then(setHealth)
      .catch((e) => setError(e instanceof Error ? e.message : 'Unknown error'))
      .finally(() => setLoading(false))
  }, [modelId])

  const content = () => {
    if (loading) {
      return (
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          height: '100%', color: 'var(--text-secondary)', fontSize: 13,
        }}>
          正在扫描模型健康状态...
        </div>
      )
    }

    if (error) {
      return (
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          height: '100%', color: 'var(--accent-red)', fontSize: 13,
        }}>
          扫描失败: {error}
        </div>
      )
    }

    if (!health) return null

    return (
      <div style={{ overflowY: 'auto', height: '100%', padding: 16 }}>
        <div style={{ display: 'flex', gap: 12, marginBottom: 20 }}>
          <SummaryCard label="总问题数" value={health.summary.total_issues} color="var(--accent)" />
          <SummaryCard label="严重" value={health.summary.critical_count} color="var(--accent-red)" />
          <SummaryCard label="警告" value={health.summary.warning_count} color="var(--accent-gold)" />
          <SummaryCard label="健康层" value={Math.max(0, (model?.info?.layer_count || 0) - health.summary.total_issues)} color="var(--accent-green)" />
        </div>

        {health.issues.length === 0 ? (
          <div style={{
            display: 'flex', flexDirection: 'column', alignItems: 'center',
            justifyContent: 'center', padding: 60, gap: 12,
          }}>
            <div style={{ fontSize: 40 }}>✓</div>
            <div style={{ color: 'var(--accent-green)', fontSize: 16, fontWeight: 600 }}>
              模型健康状态良好
            </div>
            <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>
              未检测到明显异常
            </div>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {health.issues.map((issue, i) => (
              <div
                key={i}
                onClick={() => setExpandedIssue(expandedIssue === i ? null : i)}
                style={{
                  background: 'var(--bg-secondary)',
                  borderRadius: 8,
                  borderLeft: `3px solid ${SEVERITY_COLORS[issue.severity] || 'var(--text-muted)'}`,
                  cursor: 'pointer',
                  transition: 'background 0.15s',
                }}
              >
                <div style={{ padding: '10px 14px', display: 'flex', alignItems: 'center', gap: 10 }}>
                  <span style={{
                    fontSize: 10, padding: '2px 8px', borderRadius: 10,
                    background: issue.severity === 'critical' ? 'rgba(247,118,142,0.2)' : 'rgba(224,175,104,0.2)',
                    color: SEVERITY_COLORS[issue.severity],
                    fontWeight: 600,
                  }}>
                    {issue.severity === 'critical' ? '严重' : '警告'}
                  </span>
                  <span style={{
                    fontSize: 10, padding: '2px 8px', borderRadius: 10,
                    background: 'var(--bg-tertiary)', color: 'var(--text-muted)',
                  }}>
                    {TYPE_LABELS[issue.type] || issue.type}
                  </span>
                  <span style={{
                    fontSize: 11, fontFamily: "'JetBrains Mono', monospace",
                    color: 'var(--accent)',
                  }}>
                    {issue.layer}
                  </span>
                  <span style={{ fontSize: 11, color: 'var(--text-secondary)', flex: 1 }}>
                    {issue.message}
                  </span>
                </div>
                {expandedIssue === i && (
                  <div style={{
                    padding: '8px 14px 12px', borderTop: '1px solid var(--border)',
                    fontSize: 10, color: 'var(--text-muted)',
                    fontFamily: "'JetBrains Mono', monospace",
                  }}>
                    {Object.entries(issue.detail).map(([k, v]) => (
                      <div key={k} style={{ marginBottom: 2 }}>
                        <span style={{ color: 'var(--text-secondary)' }}>{k}:</span>{' '}
                        {JSON.stringify(v)}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    )
  }

  return (
    <ModelLayout modelId={modelId} activeTab="health">
      {content()}
    </ModelLayout>
  )
}

function SummaryCard({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div style={{
      flex: 1, background: 'var(--bg-secondary)', borderRadius: 8, padding: '12px 16px',
      borderLeft: `3px solid ${color}`,
    }}>
      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 700, color }}>{value}</div>
    </div>
  )
}
