import type { GraphData, GraphNode, GraphEdge } from '../types'

export interface GroupedGraph {
  nodes: GraphNode[]
  edges: GraphEdge[]
  groupIds: Set<string>
}

/** Group 3+ consecutive layers of the same op_type into collapsible block nodes. */
export function buildGroups(graph: GraphData): GroupedGraph {
  if (!graph.nodes.length) return { nodes: [], edges: [], groupIds: new Set() }

  // Sort nodes by Y position to preserve sequential order
  const sorted = [...graph.nodes].sort((a, b) => a.position.y - b.position.y)

  // Detect runs of 3+ nodes with the same op_type
  const runs: { start: number; end: number; opType: string }[] = []
  let runStart = 0
  for (let i = 1; i <= sorted.length; i++) {
    if (i < sorted.length && sorted[i].data.opType === sorted[runStart].data.opType) continue
    const len = i - runStart
    if (len >= 3) runs.push({ start: runStart, end: i - 1, opType: sorted[runStart].data.opType as string })
    runStart = i
  }

  if (runs.length === 0) return { nodes: graph.nodes, edges: graph.edges, groupIds: new Set() }

  // Build grouped nodes and edges
  const groupedNodes: GraphNode[] = []
  const groupedEdges: GraphEdge[] = []
  const groupIds = new Set<string>()
  const childToGroup = new Map<string, string>()

  for (let blockIdx = 0; blockIdx < runs.length; blockIdx++) {
    const run = runs[blockIdx]
    const groupId = `group_${blockIdx}`
    groupIds.add(groupId)

    // Compute centroid position for group
    const children = sorted.slice(run.start, run.end + 1)
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
    for (const child of children) {
      childToGroup.set(child.id, groupId)
      minX = Math.min(minX, child.position.x)
      minY = Math.min(minY, child.position.y)
      maxX = Math.max(maxX, child.position.x + 180)
      maxY = Math.max(maxY, child.position.y + 50)
    }

    // Create group node (type: 'group' is React Flow v12 built-in)
    groupIds.add(groupId)
    const groupNode: GraphNode = {
      id: groupId,
      type: 'groupNode',
      position: { x: minX - 12, y: minY - 20 },
      data: {
        label: run.opType,
        opType: run.opType,
        inputShapes: [],
        outputShapes: [],
        params: { count: run.end - run.start + 1 } as Record<string, unknown>,
      },
    }
    groupedNodes.push(groupNode)

    // Add children with parentId
    for (const child of children) {
      groupedNodes.push({
        ...child,
        parentId: groupId,
        extent: 'parent' as const,
        position: { x: child.position.x - (minX - 12), y: child.position.y - (minY - 20) },
      })
    }
  }

  // Add non-grouped nodes
  const groupedChildIds = new Set(runs.flatMap((r) => sorted.slice(r.start, r.end + 1).map((n) => n.id)))
  for (const node of sorted) {
    if (!groupedChildIds.has(node.id)) {
      groupedNodes.push(node)
    }
  }

  // Rewire edges: replace child→child edges within same group with group→group edges
  const edgesBySource = new Map<string, GraphEdge[]>()
  const edgesByTarget = new Map<string, GraphEdge[]>()
  for (const edge of graph.edges) {
    if (!edgesBySource.has(edge.source)) edgesBySource.set(edge.source, [])
    edgesBySource.get(edge.source)!.push(edge)
    if (!edgesByTarget.has(edge.target)) edgesByTarget.set(edge.target, [])
    edgesByTarget.get(edge.target)!.push(edge)
  }

  const seenEdgeKeys = new Set<string>()
  for (const edge of graph.edges) {
    const srcGroup = childToGroup.get(edge.source)
    const tgtGroup = childToGroup.get(edge.target)

    let finalSource = edge.source
    let finalTarget = edge.target

    if (srcGroup && tgtGroup && srcGroup === tgtGroup) {
      // Internal edge within a group — skip
      continue
    }
    if (srcGroup) finalSource = srcGroup
    if (tgtGroup) finalTarget = tgtGroup

    const key = `${finalSource}→${finalTarget}`
    if (seenEdgeKeys.has(key)) continue
    seenEdgeKeys.add(key)

    // If source node is inside a group and target is outside, or vice versa
    if (srcGroup && !tgtGroup) {
      // Connect from group to external node
      const nextNodes = edgesBySource.get(edge.source)?.map((e) => e.target) ?? []
      const hasDirectChain = nextNodes.some((t) => t === edge.target)
      if (hasDirectChain) {
        groupedEdges.push({
          id: `ge_${finalSource}_${finalTarget}`,
          source: finalSource,
          target: finalTarget,
        })
      }
    } else if (!srcGroup && tgtGroup) {
      const prevNodes = edgesByTarget.get(edge.target)?.map((e) => e.source) ?? []
      const hasDirectChain = prevNodes.some((s) => s === edge.source)
      if (hasDirectChain) {
        groupedEdges.push({
          id: `ge_${finalSource}_${finalTarget}`,
          source: finalSource,
          target: finalTarget,
        })
      }
    } else if (!srcGroup && !tgtGroup) {
      // Regular edge between non-grouped nodes
      groupedEdges.push({ ...edge, id: `ge_${edge.id}` })
    } else if (srcGroup && tgtGroup) {
      groupedEdges.push({
        id: `ge_${finalSource}_${finalTarget}`,
        source: finalSource,
        target: finalTarget,
      })
    }
  }

  return { nodes: groupedNodes, edges: groupedEdges, groupIds }
}
