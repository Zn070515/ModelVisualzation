import type { GraphData } from '../types'

interface Props {
  graphData: GraphData | null
  selectedNodeId: string | null
}

export default function NodePanel({ graphData, selectedNodeId }: Props) {
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
