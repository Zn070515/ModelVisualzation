import type { ReactNode } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useStore } from '../store'
import ModelTabs from './ModelTabs'
import StatusBar from './StatusBar'

interface Props {
  modelId?: string
  activeTab: string
  children: ReactNode
}

export default function ModelLayout({ modelId: propId, activeTab, children }: Props) {
  const paramsId = useParams<{ modelId: string }>().modelId
  const modelId = propId || paramsId || ''
  const navigate = useNavigate()
  const model = useStore((s) => (modelId ? s.models[modelId] : undefined))

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      {/* Top bar */}
      <div style={{
        display: 'flex', alignItems: 'center',
        padding: '6px 16px',
        background: 'var(--bg-secondary)',
        borderBottom: '1px solid var(--border)',
      }}>
        <button onClick={() => navigate('/')}
          style={{
            background: 'none', border: 'none', color: 'var(--text-secondary)',
            cursor: 'pointer', fontSize: 16, marginRight: 12,
          }}>
          ←
        </button>
        <span style={{ fontWeight: 600, fontSize: 13, marginRight: 10 }}>
          {model?.info?.file_name || '模型查看'}
        </span>
        <span style={{
          fontSize: 10, padding: '2px 8px', borderRadius: 10,
          background: 'var(--bg-tertiary)', color: 'var(--accent)',
          marginRight: 16,
        }}>
          {model?.info?.format?.toUpperCase()}
        </span>
        <ModelTabs modelId={modelId} activeTab={activeTab} />
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: 'hidden' }}>
        {children}
      </div>

      {/* Bottom status bar */}
      <StatusBar info={model?.info || null} profile={model?.profile || null} />
    </div>
  )
}
