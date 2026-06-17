import type { GraphData } from '../types'
import type { NodeMetric } from './GraphCanvas'

interface Props {
  graphData: GraphData | null
  selectedNodeId: string | null
  colorMetric?: string
  nodeMetrics?: NodeMetric[]
  profile?: { layers: Array<{ name: string; params_count: number; flops: number }> } | null
}

const METRIC_LABELS: Record<string, string> = {
  params: 'Param count',
  flops: 'FLOPs',
}

export default function NodePanel({ graphData, selectedNodeId, colorMetric, nodeMetrics, profile }: Props) {
  if (!selectedNodeId || !graphData) {
    return (
      <div style={{
        padding: 12, color: 'var(--text-muted)', fontSize: 12,
        textAlign: 'center', marginTop: 40,
      }}>
        {graphData ? '点击图中的一个节点查看详情' : '暂无数据'}
      </div>
    )
  }

  const node = graphData.nodes.find((n) => n.id === selectedNodeId)
  if (!node) {
    return <div style={{ padding: 12, color: 'var(--text-muted)', fontSize: 12 }}>
      节点未找到
    </div>
  }

  const { data } = node

  // Look up profile data for this layer
  const layerProfile = profile?.layers?.find((l) => l.name === node.id)
  const metricValue = nodeMetrics?.find((m) => m.nodeId === node.id)

  return (
    <div style={{ padding: 12, overflowY: 'auto', height: '100%' }}>
      <h3 style={{
        fontSize: 13, fontWeight: 700, color: 'var(--accent)',
        marginBottom: 12, textTransform: 'uppercase',
      }}>
        {data.opType as string}
      </h3>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>
        {node.id}
      </div>

      {(layerProfile || metricValue) && (
        <div style={metricBox}>
          {layerProfile && <MetricRow label="Params" value={(layerProfile.params_count).toLocaleString()} />}
          {layerProfile && <MetricRow label="FLOPs" value={(layerProfile.flops).toLocaleString()} />}
          {colorMetric && metricValue && (
            <MetricRow
              label={METRIC_LABELS[colorMetric] || colorMetric}
              value={metricValue.label}
              highlight
            />
          )}
        </div>
      )}

      {Object.keys(data.params).length > 0 && (
        <>
          <div style={{
            fontSize: 10, fontWeight: 700, color: 'var(--text-secondary)',
            marginTop: 12, marginBottom: 6, textTransform: 'uppercase',
          }}>
            参数
          </div>
          {Object.entries(data.params).map(([k, v]) => (
            <div key={k} style={{
              display: 'flex', justifyContent: 'space-between',
              padding: '3px 0', fontSize: 11,
            }}>
              <span style={{ color: 'var(--text-secondary)' }}>{k}</span>
              <span style={{ color: 'var(--accent-green)', fontFamily: "'JetBrains Mono', monospace" }}>
                {Array.isArray(v) ? `[${v.join(', ')}]` : String(v)}
              </span>
            </div>
          ))}
        </>
      )}

      {data.inputShapes.length > 0 && (
        <>
          <div style={{
            fontSize: 10, fontWeight: 700, color: 'var(--text-secondary)',
            marginTop: 12, marginBottom: 6, textTransform: 'uppercase',
          }}>
            输入形状
          </div>
          {data.inputShapes.map((s, i) => (
            <div key={i} style={{ fontSize: 11, color: 'var(--accent-teal)',
              fontFamily: "'JetBrains Mono', monospace", marginBottom: 2 }}>
              [{s.join(', ')}]
            </div>
          ))}
        </>
      )}

      {data.outputShapes.length > 0 && (
        <>
          <div style={{
            fontSize: 10, fontWeight: 700, color: 'var(--text-secondary)',
            marginTop: 12, marginBottom: 6, textTransform: 'uppercase',
          }}>
            输出形状
          </div>
          {data.outputShapes.map((s, i) => (
            <div key={i} style={{ fontSize: 11, color: 'var(--accent-teal)',
              fontFamily: "'JetBrains Mono', monospace", marginBottom: 2 }}>
              [{s.join(', ')}]
            </div>
          ))}
        </>
      )}
    </div>
  )
}

function MetricRow({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', padding: '3px 0', fontSize: 11,
    }}>
      <span style={{ color: highlight ? 'var(--accent)' : 'var(--text-secondary)' }}>{label}</span>
      <span style={{
        color: highlight ? 'var(--accent)' : 'var(--accent-green)',
        fontFamily: "'JetBrains Mono', monospace",
        fontWeight: highlight ? 700 : 400,
      }}>
        {value}
      </span>
    </div>
  )
}

const metricBox = { background: 'var(--bg-tertiary)', borderRadius: 6, padding: '8px 10px', marginBottom: 10 }
