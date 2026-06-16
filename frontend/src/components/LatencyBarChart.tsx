import ReactECharts from 'echarts-for-react'
import type { PerfLayer } from '../types'

export default function LatencyBarChart({ layers }: { layers: PerfLayer[] }) {
  const option = {
    grid: { left: 56, right: 20, top: 20, bottom: 70 },
    tooltip: { trigger: 'axis' },
    xAxis: {
      type: 'category',
      data: layers.map((layer) => layer.name),
      axisLabel: { rotate: 45, color: '#9ca3af', fontSize: 10 },
    },
    yAxis: { type: 'value', name: 'us', axisLabel: { color: '#9ca3af' } },
    series: [{
      type: 'bar',
      data: layers.map((layer) => ({
        value: layer.est_latency_us,
        itemStyle: { color: layer.is_bottleneck ? '#f7768e' : '#7aa2f7' },
      })),
    }],
  }
  return <ReactECharts option={option} style={{ height: 320, width: '100%' }} />
}
