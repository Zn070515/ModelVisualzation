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
import { getOpColor } from '../constants'

function ModelNode({ data }: NodeProps) {
  const color = getOpColor(data.opType as string)
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
    </div>
  )
}

const nodeTypes = { modelNode: ModelNode }

interface Props {
  graphData: GraphData | null
  onNodeClick: (nodeId: string) => void
}

export default function GraphCanvas({ graphData, onNodeClick }: Props) {
  const initialNodes = useMemo(() => {
    if (!graphData) return []
    return graphData.nodes.map((n, i) => ({
      ...n,
      position: n.position || { x: 80, y: i * 70 },
      type: n.type || 'modelNode',
    }))
  }, [graphData])

  const initialEdges = useMemo(() => {
    if (!graphData) return []
    return graphData.edges
  }, [graphData])

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)

  // Reset when graphData changes
  useEffect(() => {
    setNodes(initialNodes)
    setEdges(initialEdges)
  }, [initialNodes, initialEdges, setNodes, setEdges])

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
        nodeColor={(n) => getOpColor((n.data?.opType as string) || '')}
        style={{ background: 'var(--bg-tertiary)' }}
      />
      <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#252540" />
    </ReactFlow>
  )
}
