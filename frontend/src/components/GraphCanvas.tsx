import { useCallback, useEffect, useMemo } from 'react'
import {
  ReactFlow,
  Controls,
  MiniMap,
  Background,
  BackgroundVariant,
  useNodesState,
  useEdgesState,
  type NodeProps,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import type { GraphData } from '../types'
import { getOpColor, heatColor } from '../constants'
import { buildGroups } from '../utils/graphGroups'
import GroupNode from './GroupNode'

function ModelNode({ data }: NodeProps) {
  const color = (data.metricColor as string) || getOpColor(data.opType as string)
  const metricLabel = data.metricLabel as string | undefined
  return (
    <div style={{
      padding: '8px 16px',
      borderRadius: 8,
      background: color,
      border: '2px solid #1a1b2e',
      color: '#1e1e2e',
      fontSize: 10,
      fontFamily: "'JetBrains Mono', monospace",
      fontWeight: 600,
      textAlign: 'center',
      whiteSpace: 'pre-line',
      minWidth: 80,
    }}>
      {(data.label as string) || (data.opType as string)}
      {metricLabel && <div style={{ fontSize: 8, opacity: 0.7, marginTop: 2 }}>{metricLabel}</div>}
    </div>
  )
}

const nodeTypes = { modelNode: ModelNode, groupNode: GroupNode }

export interface NodeMetric {
  nodeId: string
  value: number
  label: string
}

interface Props {
  graphData: GraphData | null
  onNodeClick: (nodeId: string) => void
  highlightedNodeIds?: Set<string>
  dimmedNodeIds?: Set<string>
  colorMetric?: string
  nodeMetrics?: NodeMetric[]
}

export default function GraphCanvas({ graphData, onNodeClick, highlightedNodeIds, dimmedNodeIds, colorMetric, nodeMetrics }: Props) {
  const initialNodes = useMemo(() => {
    if (!graphData) return []
    const grouped = buildGroups(graphData)
    return grouped.nodes.map((n, i) => ({
      ...n,
      position: n.position || { x: 80, y: i * 70 },
      type: n.type || 'modelNode',
    }))
  }, [graphData])

  const initialEdges = useMemo(() => {
    if (!graphData) return []
    const grouped = buildGroups(graphData)
    return grouped.edges
  }, [graphData])

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)

  // Reset when graphData changes
  useEffect(() => {
    setNodes(initialNodes)
    setEdges(initialEdges)
  }, [initialNodes, initialEdges, setNodes, setEdges])

  // Apply filter opacity when highlighted/dimmed sets change
  useEffect(() => {
    if (!highlightedNodeIds || highlightedNodeIds.size === 0) {
      setNodes((nds) =>
        nds.map((n) => ({ ...n, style: { ...n.style, opacity: 1 } })),
      )
      setEdges((eds) =>
        eds.map((e) => ({ ...e, style: { ...e.style, opacity: 1 } })),
      )
      return
    }
    setNodes((nds) =>
      nds.map((n) => ({
        ...n,
        style: { ...n.style, opacity: highlightedNodeIds.has(n.id) ? 1 : dimmedNodeIds?.has(n.id) ? 0.15 : 1 },
      })),
    )
    setEdges((eds) =>
      eds.map((e) => ({
        ...e,
        style: {
          ...e.style,
          opacity: highlightedNodeIds.has(e.source) && highlightedNodeIds.has(e.target) ? 1 : 0.08,
          stroke: highlightedNodeIds.has(e.source) && highlightedNodeIds.has(e.target) ? '#3b4261' : '#252540',
        },
      })),
    )
  }, [highlightedNodeIds, dimmedNodeIds, setNodes, setEdges])

  // Apply metric colors when overlay changes
  useEffect(() => {
    if (!colorMetric || !nodeMetrics || nodeMetrics.length === 0) {
      setNodes((nds) =>
        nds.map((n) => {
          const { metricColor, metricLabel, ...rest } = n.data as Record<string, unknown>
          return { ...n, data: rest, style: { ...n.style, background: undefined } }
        }),
      )
      return
    }
    const metricMap = new Map(nodeMetrics.map((m) => [m.nodeId, m]))
    const values = nodeMetrics.map((m) => m.value)
    const min = Math.min(...values)
    const max = Math.max(...values)
    setNodes((nds) =>
      nds.map((n) => {
        const m = metricMap.get(n.id)
        if (!m) return { ...n, data: { ...n.data, metricColor: undefined, metricLabel: undefined } }
        const color = heatColor(m.value, min, max)
        return {
          ...n,
          data: { ...n.data, metricColor: color, metricLabel: m.label },
          style: { ...n.style, background: color },
        }
      }),
    )
  }, [colorMetric, nodeMetrics, setNodes])

  const handleNodeClick = useCallback(
    (_: React.MouseEvent, node: { id: string }) => onNodeClick(node.id),
    [onNodeClick],
  )

  if (!graphData) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100%', color: 'var(--text-muted)', fontSize: 14,
      }}>
        加载模型后这里将显示网络结构图
      </div>
    )
  }

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onNodeClick={handleNodeClick}
      nodeTypes={nodeTypes}
      fitView
      fitViewOptions={{ padding: 0.3 }}
      minZoom={0.2}
      maxZoom={2}
      defaultEdgeOptions={{
        style: { stroke: '#3b4261', strokeWidth: 1.5 },
        type: 'smoothstep',
      }}
    >
      <Controls />
      <MiniMap
        nodeColor={(n) => {
          if (colorMetric && n.data?.metricColor) return n.data.metricColor as string
          return getOpColor((n.data?.opType as string) || '')
        }}
        style={{ background: 'var(--bg-tertiary)' }}
      />
      <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#252540" />
    </ReactFlow>
  )
}
