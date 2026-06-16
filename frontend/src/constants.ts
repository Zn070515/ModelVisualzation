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
