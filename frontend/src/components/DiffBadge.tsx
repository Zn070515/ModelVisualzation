export default function DiffBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    same: '#73daca',
    changed: '#e0af68',
    added: '#7aa2f7',
    removed: '#f7768e',
  }
  return (
    <span style={{
      color: colors[status] || 'var(--text-muted)',
      border: `1px solid ${colors[status] || 'var(--border)'}`,
      borderRadius: 6,
      padding: '1px 6px',
      fontSize: 10,
      textTransform: 'uppercase',
    }}>
      {status}
    </span>
  )
}
