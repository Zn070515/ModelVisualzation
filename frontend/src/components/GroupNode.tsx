import { useState } from 'react'
import type { NodeProps } from '@xyflow/react'

export default function GroupNode({ data }: NodeProps) {
  const [collapsed, setCollapsed] = useState(false)
  const opType = data.opType as string
  const count = (data.params as Record<string, unknown>)?.count as number || 0

  return (
    <div
      onClick={() => setCollapsed(!collapsed)}
      style={{
        padding: '6px 14px',
        borderRadius: 10,
        background: 'rgba(86,95,137,0.12)',
        border: '2px dashed #3b4261',
        color: 'var(--text-secondary)',
        fontSize: 11,
        fontFamily: "'JetBrains Mono', monospace",
        fontWeight: 600,
        textAlign: 'center',
        whiteSpace: 'nowrap',
        cursor: 'pointer',
        userSelect: 'none',
        minWidth: 120,
        transition: 'background 0.15s',
      }}
      onMouseEnter={(e) => { (e.currentTarget as HTMLDivElement).style.background = 'rgba(86,95,137,0.22)' }}
      onMouseLeave={(e) => { (e.currentTarget as HTMLDivElement).style.background = 'rgba(86,95,137,0.12)' }}
    >
      <span style={{ marginRight: 6 }}>{collapsed ? '▶' : '▼'}</span>
      <span style={{ color: 'var(--text-primary)' }}>{count}x</span>
      <span style={{ marginLeft: 6 }}>{opType}</span>
    </div>
  )
}

export function getCollapsedState() {
  let state: Record<string, boolean> = {}
  try {
    state = JSON.parse(localStorage.getItem('mv_group_collapsed') || '{}')
  } catch { /* ignore */ }
  return state
}

export function setCollapsedState(groupId: string, collapsed: boolean) {
  const state = getCollapsedState()
  state[groupId] = collapsed
  localStorage.setItem('mv_group_collapsed', JSON.stringify(state))
}
