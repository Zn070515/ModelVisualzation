import ReactECharts from 'echarts-for-react'
import type { QuantResult } from '../types'

export default function QuantHeatmap({ result }: { result: QuantResult }) {
  const rows = result.layers
  const maxChannels = Math.max(1, ...rows.map((layer) => firstErrors(layer).length))
  const data = rows.flatMap((layer, y) => firstErrors(layer).map((value, x) => [x, y, value]))
  const option = {
    tooltip: { position: 'top' },
    grid: { left: 120, right: 20, top: 20, bottom: 40 },
    xAxis: { type: 'category', data: Array.from({ length: maxChannels }, (_, i) => String(i)), axisLabel: { color: '#9ca3af' } },
    yAxis: { type: 'category', data: rows.map((layer) => layer.layer_name), axisLabel: { color: '#9ca3af', fontSize: 10 } },
    visualMap: { min: 0, max: Math.max(1e-9, ...data.map((item) => item[2] as number)), calculable: true, orient: 'horizontal', left: 'center', bottom: 0 },
    series: [{ type: 'heatmap', data }],
  }
  return <ReactECharts option={option} style={{ height: 360, width: '100%' }} />
}

function firstErrors(layer: QuantResult['layers'][number]) {
  const tensor = Object.values(layer.weights)[0]
  return tensor?.per_channel_error || []
}
