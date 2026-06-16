import ReactECharts from 'echarts-for-react'
import type { PruneResult } from '../types'

export default function ImportanceChart({ layer }: { layer: PruneResult['layers'][number] }) {
  const top = layer.channel_importance.slice(0, 32)
  const option = {
    grid: { left: 56, right: 20, top: 20, bottom: 48 },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: top.map((item) => String(item.channel)), axisLabel: { color: '#9ca3af' } },
    yAxis: { type: 'value', axisLabel: { color: '#9ca3af' } },
    series: [{ type: 'bar', data: top.map((item) => item.importance), itemStyle: { color: '#bb9af7' } }],
  }
  return <ReactECharts option={option} style={{ height: 280, width: '100%' }} />
}
