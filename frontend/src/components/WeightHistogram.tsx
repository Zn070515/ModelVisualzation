import ReactECharts from 'echarts-for-react'
import type { WeightTensorStats } from '../types'

interface Props {
  stats: WeightTensorStats
  weightName: string
  color: string
}

export default function WeightHistogram({ stats, weightName, color }: Props) {
  const option = {
    grid: { top: 10, right: 16, bottom: 32, left: 48 },
    xAxis: {
      type: 'category',
      data: stats.histogram.bin_edges.slice(0, -1).map((v) => v.toFixed(4)),
      axisLabel: { fontSize: 9, color: '#565f89', rotate: 45, interval: 9 },
      name: 'value',
      nameTextStyle: { fontSize: 9, color: '#565f89' },
    },
    yAxis: {
      type: 'value',
      axisLabel: { fontSize: 9, color: '#565f89' },
      name: 'count',
      nameTextStyle: { fontSize: 9, color: '#565f89' },
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#1a1b2e',
      borderColor: '#252540',
      textStyle: { color: '#e0e0e0', fontSize: 10 },
    },
    series: [{
      type: 'bar',
      data: stats.histogram.counts,
      itemStyle: { color },
      emphasis: { itemStyle: { color } },
    }],
  }

  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{
        fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)',
        marginBottom: 4,
      }}>
        {weightName}
        <span style={{ color: 'var(--text-muted)', fontWeight: 400, marginLeft: 8, fontSize: 10 }}>
          shape: [{stats.shape.join(', ')}]
        </span>
      </div>
      <ReactECharts option={option} style={{ height: 180 }} />
      <div style={{
        display: 'flex', gap: 16, flexWrap: 'wrap',
        fontSize: 10, color: 'var(--text-muted)', marginTop: 4,
      }}>
        <span>μ={stats.mean.toFixed(4)}</span>
        <span>σ={stats.std.toFixed(4)}</span>
        <span>min={stats.min.toFixed(4)}</span>
        <span>max={stats.max.toFixed(4)}</span>
        <span>稀疏度={(stats.sparsity * 100).toFixed(1)}%</span>
      </div>
    </div>
  )
}
