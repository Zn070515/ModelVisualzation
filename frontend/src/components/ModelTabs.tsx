import { useNavigate } from 'react-router-dom'

const TABS = [
  { key: 'viewer', label: 'Graph', path: (id: string) => `/viewer/${id}` },
  { key: 'weights', label: 'Weights', path: (id: string) => `/weights/${id}` },
  { key: 'dashboard', label: 'Profile', path: (id: string) => `/dashboard/${id}` },
  { key: 'health', label: 'Health', path: (id: string) => `/health/${id}` },
  { key: 'chain', label: 'Chain', path: (_id: string) => '/chain' },
  { key: 'performance', label: 'Perf', path: (id: string) => `/performance/${id}` },
  { key: 'quant', label: 'Quant', path: (id: string) => `/quant/${id}` },
  { key: 'activation', label: 'Act', path: (id: string) => `/activation/${id}` },
  { key: 'prune', label: 'Prune', path: (id: string) => `/prune/${id}` },
]

interface Props {
  modelId: string
  activeTab: string
}

export default function ModelTabs({ modelId, activeTab }: Props) {
  const navigate = useNavigate()

  return (
    <div style={{ display: 'flex', gap: 0, flexWrap: 'wrap' }}>
      {TABS.map((tab) => {
        const isActive = tab.key === activeTab
        return (
          <button
            key={tab.key}
            onClick={() => navigate(tab.path(modelId))}
            style={{
              padding: '6px 12px',
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
