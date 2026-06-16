import ReactECharts from 'echarts-for-react'
import type { ActivationResult } from '../types'

export default function ActivationHistogram({ activation }: { activation: ActivationResult['activations'][number] }) {
  const edges = activation.stats.histogram.bin_edges
  const option = {
    grid: { left: 56, right: 20, top: 20, bottom: 40 },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: edges.slice(0, -1).map((edge) => edge.toFixed(2)), axisLabel: { color: '#9ca3af' } },
    yAxis: { type: 'value', axisLabel: { color: '#9ca3af' } },
    series: [{ type: 'bar', data: activation.stats.histogram.counts, itemStyle: { color: '#73daca' } }],
  }
  return <ReactECharts option={option} style={{ height: 260, width: '100%' }} />
}
