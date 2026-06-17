export const OP_COLORS: Record<string, string> = {
  Conv2d: '#7aa2f7', Conv1d: '#7aa2f7', ConvTranspose2d: '#7aa2f7',
  BatchNorm2d: '#e0af68', BatchNorm1d: '#e0af68',
  ReLU: '#bb9af7', LeakyReLU: '#bb9af7', Sigmoid: '#bb9af7',
  Tanh: '#bb9af7', Softmax: '#bb9af7', ReLU6: '#bb9af7',
  MaxPool2d: '#73daca', AvgPool2d: '#73daca', AdaptiveAvgPool2d: '#73daca',
  Linear: '#f7768e', Flatten: '#f7768e',
  Concat: '#ff9e64', Add: '#ff9e64', Mul: '#ff9e64',
}

export function getOpColor(opType: string): string {
  for (const [key, color] of Object.entries(OP_COLORS)) {
    if (opType.toLowerCase().includes(key.toLowerCase())) return color
  }
  return '#565f89'
}

/** Map a numeric value to a coolwarm heat color (blue→green→yellow→red). */
export function heatColor(value: number, min: number, max: number): string {
  if (max <= min) return '#565f89'
  const t = Math.max(0, Math.min(1, (value - min) / (max - min)))
  // 6 gradient stops: #7aa2f7 (blue, 0.0) → #73daca (green, 0.33) → #e0af68 (yellow, 0.66) → #f7768e (red, 1.0)
  const stops = [
    { pos: 0.0, r: 0x7a, g: 0xa2, b: 0xf7 },
    { pos: 0.33, r: 0x73, g: 0xda, b: 0xca },
    { pos: 0.66, r: 0xe0, g: 0xaf, b: 0x68 },
    { pos: 1.0, r: 0xf7, g: 0x76, b: 0x8e },
  ]
  let lo = stops[0], hi = stops[stops.length - 1]
  for (let i = 0; i < stops.length - 1; i++) {
    if (t >= stops[i].pos && t <= stops[i + 1].pos) { lo = stops[i]; hi = stops[i + 1]; break }
  }
  const span = hi.pos - lo.pos || 1
  const f = (t - lo.pos) / span
  const r = Math.round(lo.r + (hi.r - lo.r) * f)
  const g = Math.round(lo.g + (hi.g - lo.g) * f)
  const b = Math.round(lo.b + (hi.b - lo.b) * f)
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`
}
