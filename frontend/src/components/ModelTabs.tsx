import { useNavigate } from 'react-router-dom'

const TABS = [
  { key: 'viewer', label: '结构图', path: (id: string) => `/viewer/${id}` },
  { key: 'weights', label: '权重分析', path: (id: string) => `/weights/${id}` },
  { key: 'dashboard', label: '模型画像', path: (id: string) => `/dashboard/${id}` },
  { key: 'health', label: '健康扫描', path: (id: string) => `/health/${id}` },
]

interface Props {
  modelId: string
  activeTab: string
}

export default function ModelTabs({ modelId, activeTab }: Props) {
  const navigate = useNavigate()

  return (
    <div style={{ display: 'flex', gap: 0 }}>
      {TABS.map((tab) => {
        const isActive = tab.key === activeTab
        return (
          <button
            key={tab.key}
            onClick={() => navigate(tab.path(modelId))}
            style={{
              padding: '6px 16px',
              border: 'none',
              borderBottom: isActive ? '2px solid var(--accent)' : '2px solid transparent',
              background: isActive ? 'rgba(122,162,247,0.08)' : 'transparent',
              color: isActive ? 'var(--accent)' : 'var(--text-muted)',
              fontSize: 12,
              fontWeight: isActive ? 600 : 400,
              cursor: 'pointer',
              transition: 'all 0.15s',
            }}
          >
            {tab.label}
          </button>
        )
      })}
    </div>
  )
}
