import { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { useStore } from '../store'
import { getWeightOverview, getWeightStats } from '../api/client'
import type { WeightLayerSummary, WeightLayerStats } from '../types'
import WeightHistogram from '../components/WeightHistogram'
import ModelLayout from '../components/ModelLayout'
import ReactECharts from 'echarts-for-react'

const HISTO_COLORS = ['#7aa2f7', '#e0af68', '#bb9af7', '#73daca', '#f7768e', '#ff9e64']

export default function WeightsPage() {
  const { modelId } = useParams<{ modelId: string }>()
  const model = useStore((s) => (modelId ? s.models[modelId] : undefined))
  const loadModelData = useStore((s) => s.loadModelData)

  const [overview, setOverview] = useState<WeightLayerSummary[] | null>(null)
  const [selectedLayer, setSelectedLayer] = useState<string | null>(null)
  const [layerStats, setLayerStats] = useState<WeightLayerStats | null>(null)
  const [loading, setLoading] = useState(false)
  const [overviewError, setOverviewError] = useState<string | null>(null)
  const [layerError, setLayerError] = useState<string | null>(null)

  useEffect(() => {
    if (!modelId) return
    if (!model || (!model.graph && !model.loading)) {
      loadModelData(modelId)
    }
  }, [modelId])

  useEffect(() => {
    if (!modelId) return
    setOverviewError(null)
    getWeightOverview(modelId).then((o) => {
      setOverview(o.layers)
      if (o.layers.length > 0) {
        setSelectedLayer(o.layers[0].layer_name)
      }
    }).catch((err: Error) => setOverviewError(err.message))
  }, [modelId])

  useEffect(() => {
    if (!modelId || !selectedLayer) return
    setLoading(true)
    setLayerError(null)
    getWeightStats(modelId, selectedLayer)
      .then(setLayerStats)
      .catch((err: Error) => { setLayerStats(null); setLayerError(err.message) })
      .finally(() => setLoading(false))
  }, [modelId, selectedLayer])

  const comparisonOption = overview && overview.length > 0 ? {
    grid: { top: 16, right: 16, bottom: 32, left: 48 },
    legend: { data: ['mean', 'std'], textStyle: { color: '#a9b1d6', fontSize: 10 } },
    xAxis: {
      type: 'category',
      data: overview.map((l) => l.layer_name.length > 20 ? l.layer_name.slice(0, 20) + '...' : l.layer_name),
      axisLabel: { fontSize: 9, color: '#565f89', rotate: 45 },
    },
    yAxis: { type: 'value', axisLabel: { fontSize: 9, color: '#565f89' } },
    tooltip: { backgroundColor: '#1a1b2e', borderColor: '#252540', textStyle: { color: '#e0e0e0', fontSize: 10 } },
    series: [
      { name: 'mean', type: 'bar', data: overview.map((l) => l.mean), itemStyle: { color: '#7aa2f7' } },
      { name: 'std', type: 'bar', data: overview.map((l) => l.std), itemStyle: { color: '#bb9af7' } },
    ],
  } : null

  return (
    <ModelLayout modelId={modelId} activeTab="weights">
      <div style={{ display: 'flex', height: '100%', overflow: 'hidden' }}>
        <div style={{
          width: 250, flexShrink: 0, borderRight: '1px solid var(--border)',
          background: 'var(--bg-secondary)', overflowY: 'auto',
        }}>
          <div style={{
            padding: '10px 12px', fontSize: 11, fontWeight: 600,
            color: 'var(--text-secondary)', borderBottom: '1px solid var(--border)',
          }}>
            层列表
            {overview && (
              <span style={{ color: 'var(--text-muted)', fontWeight: 400, marginLeft: 8 }}>
                ({overview.length})
              </span>
            )}
          </div>
          {overview?.map((l) => (
            <div
              key={l.layer_name}
              onClick={() => setSelectedLayer(l.layer_name)}
              style={{
                padding: '6px 12px', cursor: 'pointer', fontSize: 11,
                fontFamily: "'JetBrains Mono', monospace",
                borderLeft: selectedLayer === l.layer_name ? '2px solid var(--accent)' : '2px solid transparent',
                background: selectedLayer === l.layer_name ? 'rgba(122,162,247,0.1)' : 'transparent',
                color: 'var(--text-secondary)',
              }}
            >
              <div style={{ fontWeight: 600, color: 'var(--accent)', fontSize: 10 }}>{l.op_type}</div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                {l.layer_name.length > 32 ? l.layer_name.slice(0, 32) + '...' : l.layer_name}
              </div>
              <div style={{ fontSize: 9, color: 'var(--text-muted)' }}>
                params: {l.param_count.toLocaleString()} | μ={l.mean.toFixed(4)} | sp={l.sparsity.toFixed(2)}
              </div>
            </div>
          ))}
          {overviewError && (
            <div style={{ padding: 12, color: 'var(--red)', fontSize: 11 }}>
              {overviewError}
            </div>
          )}
          {(!overview || overview.length === 0) && !overviewError && (
            <div style={{ padding: 12, color: 'var(--text-muted)', fontSize: 11 }}>
              暂无有权重的层
            </div>
          )}
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: 16 }}>
          {!selectedLayer && (
            <div style={{ color: 'var(--text-muted)', fontSize: 13, textAlign: 'center', paddingTop: 40 }}>
              选择左侧层查看权重分布
            </div>
          )}

          {loading && (
            <div style={{ color: 'var(--text-secondary)', fontSize: 13, textAlign: 'center', paddingTop: 40 }}>
              加载中...
            </div>
          )}

          {layerError && !loading && (
            <div style={{ color: 'var(--red)', fontSize: 13, textAlign: 'center', paddingTop: 40 }}>
              {layerError}
            </div>
          )}

          {layerStats && !loading && (
            <div>
              <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>
                {layerStats.layer_name}
                <span style={{
                  fontSize: 10, marginLeft: 8, padding: '1px 8px', borderRadius: 10,
                  background: 'var(--bg-tertiary)', color: 'var(--accent)',
                }}>
                  {layerStats.op_type}
                </span>
              </div>

              {Object.entries(layerStats.weights).map(([wname, wstats], i) => (
                <WeightHistogram
                  key={wname}
                  stats={wstats}
                  weightName={wname}
                  color={HISTO_COLORS[i % HISTO_COLORS.length]}
                />
              ))}

              {comparisonOption && overview && overview.length > 1 && (
                <div style={{ marginTop: 24 }}>
                  <div style={{
                    fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)',
                    marginBottom: 8, borderTop: '1px solid var(--border)', paddingTop: 16,
                  }}>
                    跨层对比 (mean / std)
                  </div>
                  <ReactECharts option={comparisonOption} style={{ height: 250 }} />
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </ModelLayout>
  )
}
