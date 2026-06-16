import { useCallback, useEffect, useMemo } from 'react'
import {
  ReactFlow,
  Node,
  Edge,
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

const OP_COLORS: Record<string, string> = {
  Conv2d: '#7aa2f7', Conv1d: '#7aa2f7', ConvTranspose2d: '#7aa2f7',
  BatchNorm2d: '#e0af68', BatchNorm1d: '#e0af68',
  ReLU: '#bb9af7', LeakyReLU: '#bb9af7', Sigmoid: '#bb9af7',
  Tanh: '#bb9af7', Softmax: '#bb9af7', ReLU6: '#bb9af7',
  MaxPool2d: '#73daca', AvgPool2d: '#73daca', AdaptiveAvgPool2d: '#73daca',
  Linear: '#f7768e', Flatten: '#f7768e',
  Concat: '#ff9e64', Add: '#ff9e64', Mul: '#ff9e64',
}

function getColor(opType: string): string {
  for (const [key, color] of Object.entries(OP_COLORS)) {
    if (opType.toLowerCase().includes(key.toLowerCase())) return color
  }
  return '#565f89'
}

function ModelNode({ data }: NodeProps) {
  const color = getColor(data.opType as string)
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

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes as Node[])
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges as Edge[])

  // Reset when graphData changes
  useEffect(() => {
    setNodes(initialNodes as Node[])
    setEdges(initialEdges as Edge[])
  }, [initialNodes, initialEdges, setNodes, setEdges])

  const handleNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => onNodeClick(node.id),
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
        nodeColor={(n) => getColor((n.data?.opType as string) || '')}
        style={{ background: 'var(--bg-tertiary)' }}
      />
      <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#252540" />
    </ReactFlow>
  )
}
