import type { ModelInfo, ProfileData } from '../types'
import { formatNum, formatBytes } from '../utils'

interface Props {
  info: ModelInfo | null
  profile: ProfileData | null
}

export default function StatusBar({ info, profile }: Props) {
  if (!info) {
    return (
      <div style={{
        display: 'flex', justifyContent: 'space-between',
        padding: '6px 16px',
        background: 'var(--bg-secondary)',
        borderTop: '1px solid var(--border)',
        fontSize: 11, color: 'var(--text-muted)',
      }}>
        <span>未加载模型</span>
      </div>
    )
  }

  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', gap: 24,
      padding: '6px 16px',
      background: 'var(--bg-secondary)',
      borderTop: '1px solid var(--border)',
      fontSize: 11, color: 'var(--text-secondary)',
    }}>
      <span>{info.file_name || 'model'} · {info.format}</span>
      <span>{formatBytes(info.file_size_bytes)}</span>
      {profile && (
        <>
          <span>参数: {formatNum(profile.total_params)}</span>
          <span>FLOPs: {formatNum(profile.total_flops)}</span>
          <span>显存估算: {profile.memory_mb}MB</span>
        </>
      )}
      <span>{info.layer_count} 层</span>
    </div>
  )
}
