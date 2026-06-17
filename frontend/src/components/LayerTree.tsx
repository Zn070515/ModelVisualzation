import { useMemo } from 'react'
import type { GraphData } from '../types'

interface Props {
  graphData: GraphData | null
  selectedNodeId: string | null
  onSelect: (nodeId: string) => void
  searchText: string
  onSearchChange: (text: string) => void
}

export default function LayerTree({ graphData, selectedNodeId, onSelect, searchText, onSearchChange }: Props) {
  const filteredNodes = useMemo(() => {
    if (!graphData) return []
    if (!searchText.trim()) return graphData.nodes
    const q = searchText.toLowerCase()
    return graphData.nodes.filter(
      (n) =>
        n.id.toLowerCase().includes(q) ||
        (n.data.opType as string).toLowerCase().includes(q) ||
        (n.data.label as string).toLowerCase().includes(q),
    )
  }, [graphData, searchText])

  if (!graphData) {
    return (
      <div style={{ padding: 12, color: 'var(--text-muted)', fontSize: 12 }}>
        暂无模型数据
      </div>
    )
  }

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: 8 }}>
        <input
          placeholder="搜索层..."
          value={searchText}
          onChange={(e) => onSearchChange(e.target.value)}
          style={{
            width: '100%', padding: '6px 10px',
            background: 'var(--bg-tertiary)', border: '1px solid var(--border)',
            borderRadius: 6, color: 'var(--text-primary)', fontSize: 12,
            outline: 'none',
          }}
        />
      </div>
      <div style={{ flex: 1, overflowY: 'auto', padding: '0 8px 8px' }}>
        {filteredNodes.map((node, i) => (
          <div
            key={node.id}
            onClick={() => onSelect(node.id)}
            style={{
              padding: '6px 10px',
              marginBottom: 2,
              borderRadius: 6,
              cursor: 'pointer',
              fontSize: 11,
              fontFamily: "'JetBrains Mono', monospace",
              color: 'var(--text-secondary)',
              background: selectedNodeId === node.id
                ? 'rgba(122,162,247,0.15)'
                : 'transparent',
              borderLeft: selectedNodeId === node.id
                ? '2px solid var(--accent)'
                : '2px solid transparent',
              transition: 'background 0.15s',
            }}
          >
            <span style={{ fontWeight: 600, marginRight: 8, color: 'var(--text-muted)', fontSize: 10 }}>
              {i + 1}
            </span>
            <span style={{ color: 'var(--accent)', fontWeight: 600 }}>
              {node.data.opType as string}
            </span>
            <span style={{ color: 'var(--text-muted)', marginLeft: 8 }}>
              {node.id.length > 36 ? node.id.slice(0, 36) + '...' : node.id}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
