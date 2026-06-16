import { useState, useRef, type DragEvent, type ChangeEvent } from 'react'

const SUPPORTED = ['.pt', '.pth', '.onnx', '.tflite', '.pb', '.h5', '.keras']

interface Props {
  onUpload: (file: File) => void
  uploading: boolean
}

export default function FileUploader({ onUpload, uploading }: Props) {
  const [dragover, setDragover] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const folderRef = useRef<HTMLInputElement>(null)

  const handleDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragover(false)
    const files = Array.from(e.dataTransfer.files)
    for (const f of files) {
      const ext = '.' + f.name.split('.').pop()?.toLowerCase()
      if (SUPPORTED.includes(ext)) {
        onUpload(f)
        return
      }
    }
  }

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) onUpload(f)
    e.target.value = ''
  }

  const handleFolderChange = (e: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    for (const f of files) {
      const ext = '.' + f.name.split('.').pop()?.toLowerCase()
      if (SUPPORTED.includes(ext)) {
        onUpload(f)
        e.target.value = ''
        return
      }
    }
    e.target.value = ''
  }

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragover(true) }}
      onDragLeave={() => setDragover(false)}
      onDrop={handleDrop}
      style={{
        border: `2px dashed ${dragover ? 'var(--accent)' : 'var(--border)'}`,
        borderRadius: 12,
        padding: '60px 40px',
        textAlign: 'center',
        transition: 'border-color 0.2s, background 0.2s',
        background: dragover ? 'rgba(122,162,247,0.08)' : 'var(--bg-secondary)',
        cursor: 'pointer',
      }}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept={SUPPORTED.join(',')}
        onChange={handleChange}
        style={{ display: 'none' }}
      />
      <input
        ref={folderRef}
        type="file"
        /* @ts-expect-error webkitdirectory is not in React types */
        webkitdirectory=""
        onChange={handleFolderChange}
        style={{ display: 'none' }}
      />

      {uploading ? (
        <>
          <div style={{
            width: 40, height: 40, border: '3px solid var(--border)',
            borderTopColor: 'var(--accent)', borderRadius: '50%',
            animation: 'spin 0.8s linear infinite', margin: '0 auto 16px',
          }} />
          <p style={{ color: 'var(--text-secondary)' }}>正在解析模型...</p>
        </>
      ) : (
        <>
          <p style={{ fontSize: 18, fontWeight: 600, marginBottom: 8 }}>
            拖入模型文件
          </p>
          <p style={{ color: 'var(--text-secondary)', fontSize: 14, marginBottom: 16 }}>
            支持 .pt / .pth / .onnx / .tflite / .pb / .h5 / .keras
          </p>
          <div style={{ display: 'flex', gap: 12, justifyContent: 'center' }}>
            <button
              onClick={(e) => { e.stopPropagation(); inputRef.current?.click() }}
              style={{
                padding: '8px 20px', borderRadius: 8, border: 'none',
                background: 'var(--accent)', color: '#fff', cursor: 'pointer',
                fontWeight: 600,
              }}
            >
              选择文件
            </button>
            <button
              onClick={(e) => { e.stopPropagation(); folderRef.current?.click() }}
              style={{
                padding: '8px 20px', borderRadius: 8, border: '1px solid var(--border)',
                background: 'transparent', color: 'var(--text-secondary)',
                cursor: 'pointer',
              }}
            >
              打开文件夹
            </button>
          </div>
        </>
      )}
    </div>
  )
}
