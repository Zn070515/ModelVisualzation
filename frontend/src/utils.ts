export function formatNum(n: number): string {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'G'
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M'
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K'
  return String(n)
}

export function formatBytes(n: number): string {
  if (n >= 1e6) return (n / 1e6).toFixed(2) + ' MB'
  if (n >= 1e3) return (n / 1e3).toFixed(1) + ' KB'
  return n + ' B'
}
