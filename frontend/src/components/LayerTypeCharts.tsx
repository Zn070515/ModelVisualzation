import ReactECharts from 'echarts-for-react'
import type { ProfileData } from '../types'

const TYPE_COLORS: Record<string, string> = {
  Conv2d: '#7aa2f7', Conv1d: '#7aa2f7', ConvTranspose2d: '#7aa2f7',
  BatchNorm2d: '#e0af68', BatchNorm1d: '#e0af68',
  ReLU: '#bb9af7', LeakyReLU: '#bb9af7', Sigmoid: '#bb9af7', Tanh: '#bb9af7', Softmax: '#bb9af7',
  MaxPool2d: '#73daca', AvgPool2d: '#73daca', AdaptiveAvgPool2d: '#73daca',
  Linear: '#f7768e', Flatten: '#f7768e', Dropout: '#f7768e',
  Concat: '#ff9e64', Add: '#ff9e64', Mul: '#ff9e64',
}

function getTypeColor(opType: string): string {
  for (const [key, color] of Object.entries(TYPE_COLORS)) {
    if (opType.toLowerCase().includes(key.toLowerCase())) return color
  }
  return '#565f89'
}

interface Props {
  profile: ProfileData
}

export default function LayerTypeCharts({ profile }: Props) {
  const typeMap = new Map<string, { count: number; params: number; flops: number }>()
  for (const l of profile.layers) {
    const entry = typeMap.get(l.op_type) || { count: 0, params: 0, flops: 0 }
    entry.count++
    entry.params += l.params_count
    entry.flops += l.flops
    typeMap.set(l.op_type, entry)
  }
  const types = Array.from(typeMap.entries()).sort((a, b) => b[1].params - a[1].params)

  const pieOption = {
    tooltip: {
      trigger: 'item' as const,
      backgroundColor: '#1a1b2e',
      borderColor: '#252540',
      textStyle: { color: '#e0e0e0', fontSize: 10 },
    },
    legend: {
      orient: 'vertical' as const,
      right: 8,
      top: 'center',
      textStyle: { color: '#a9b1d6', fontSize: 10 },
    },
    series: [{
      type: 'pie' as const,
      radius: ['35%', '65%'],
      center: ['35%', '50%'],
      data: types.map(([name, d]) => ({
        name,
        value: d.count,
        itemStyle: { color: getTypeColor(name) },
      })),
      label: { show: false },
      emphasis: { label: { show: true, fontSize: 10 } },
    }],
  }

  const barOption = {
    grid: { top: 16, right: 16, bottom: 40, left: 60 },
    tooltip: {
      trigger: 'axis' as const,
      backgroundColor: '#1a1b2e',
      borderColor: '#252540',
      textStyle: { color: '#e0e0e0', fontSize: 10 },
    },
    xAxis: {
      type: 'category' as const,
      data: types.map(([n]) => n),
      axisLabel: { fontSize: 9, color: '#565f89', rotate: 30 },
    },
    yAxis: {
      type: 'value' as const,
      axisLabel: {
        fontSize: 9, color: '#565f89',
        formatter: (v: number) => v >= 1e6 ? (v / 1e6).toFixed(1) + 'M' : v >= 1e3 ? (v / 1e3).toFixed(1) + 'K' : String(v),
      },
    },
    series: [{
      name: '参数量',
      type: 'bar' as const,
      data: types.map(([, d]) => ({
        value: d.params,
        itemStyle: { color: getTypeColor(types[0][0]) },
      })),
      itemStyle: { borderRadius: [4, 4, 0, 0] },
    }],
  }

  const flopsBarOption = {
    ...barOption,
    series: [{
      name: 'FLOPs',
      type: 'bar' as const,
      data: types.map(([, d]) => ({
        value: d.flops,
        itemStyle: { color: getTypeColor(types[0][0]) },
      })),
      itemStyle: { borderRadius: [4, 4, 0, 0] },
    }],
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'flex', gap: 16 }}>
        <div style={{ flex: 1, background: 'var(--bg-secondary)', borderRadius: 8, padding: 12 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8 }}>
            按层类型分布（层数）
          </div>
          <ReactECharts option={pieOption} style={{ height: 220 }} />
        </div>
      </div>
      <div style={{ display: 'flex', gap: 16 }}>
        <div style={{ flex: 1, background: 'var(--bg-secondary)', borderRadius: 8, padding: 12 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8 }}>
            每层参数量（按类型聚合）
          </div>
          <ReactECharts option={barOption} style={{ height: 220 }} />
        </div>
        <div style={{ flex: 1, background: 'var(--bg-secondary)', borderRadius: 8, padding: 12 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8 }}>
            每层 FLOPs（按类型聚合）
          </div>
          <ReactECharts option={flopsBarOption} style={{ height: 220 }} />
        </div>
      </div>
    </div>
  )
}
