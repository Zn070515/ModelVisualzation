import { useMemo } from 'react'
import ReactEChartsCore from 'echarts-for-react/lib/core'
import * as echarts from 'echarts/core'
import { HeatmapChart } from 'echarts/charts'
import { TooltipComponent, VisualMapComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

echarts.use([HeatmapChart, TooltipComponent, VisualMapComponent, CanvasRenderer])

interface Props {
  values: number[][]
  title?: string
  height?: number
}

export default function AttentionHeatmap({ values, title, height = 200 }: Props) {
  const option = useMemo(() => {
    if (!values.length) return {}
    const data: [number, number, number][] = []
    let maxVal = 0
    for (let r = 0; r < values.length; r++) {
      for (let c = 0; c < (values[r]?.length ?? 0); c++) {
        const v = Math.abs(values[r][c])
        if (v > maxVal) maxVal = v
        data.push([c, r, values[r][c]])
      }
    }

    return {
      tooltip: {
        position: 'top' as const,
        formatter: (p: { data: [number, number, number] }) =>
          `(${p.data[0]}, ${p.data[1]}) = ${p.data[2].toFixed(6)}`,
      },
      grid: { left: 0, right: 0, top: title ? 18 : 0, bottom: 0 },
      xAxis: { type: 'category' as const, show: false, splitArea: { show: false } },
      yAxis: { type: 'category' as const, show: false, splitArea: { show: false } },
      visualMap: {
        min: -maxVal,
        max: maxVal,
        calculable: false,
        orient: 'horizontal' as const,
        left: 'center',
        bottom: 0,
        inRange: { color: ['#7aa2f7', '#1a1b2e', '#f7768e'] },
        show: title != null,
      },
      series: [
        {
          type: 'heatmap',
          data,
          label: { show: false },
          emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' } },
          itemStyle: { borderWidth: 0 },
        },
      ],
    }
  }, [values, title])

  if (!values.length) {
    return <div style={{ color: 'var(--text-muted)', fontSize: 11, padding: 12 }}>No data</div>
  }

  return (
    <div>
      {title && <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginBottom: 4 }}>{title}</div>}
      <ReactEChartsCore echarts={echarts} option={option} style={{ height, width: '100%' }} notMerge />
    </div>
  )
}
