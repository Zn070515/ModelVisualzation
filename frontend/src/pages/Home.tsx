import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import FileUploader from '../components/FileUploader'
import { useStore } from '../store'

export default function Home() {
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const addModel = useStore((s) => s.addModel)
  const navigate = useNavigate()

  const handleUpload = async (file: File) => {
    setUploading(true)
    setError(null)
    try {
      const id = await addModel(file)
      navigate(`/viewer/${id}`)
    } catch (e) {
      setError(e instanceof Error ? e.message : '上传失败')
    } finally {
      setUploading(false)
    }
  }

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      justifyContent: 'center', minHeight: '100vh', padding: 40,
    }}>
      <h1 style={{ fontSize: 28, fontWeight: 700, marginBottom: 8 }}>
        ModelViz
      </h1>
      <p style={{ color: 'var(--text-secondary)', marginBottom: 32, fontSize: 14 }}>
        AI 模型可视化工具 — 拖入模型文件，即刻查看网络结构
      </p>

      <div style={{ width: 520, maxWidth: '90vw' }}>
        <FileUploader onUpload={handleUpload} uploading={uploading} />
      </div>

      {error && (
        <div style={{
          marginTop: 16, padding: '12px 20px',
          background: 'rgba(247,118,142,0.1)',
          border: '1px solid rgba(247,118,142,0.3)',
          borderRadius: 8, color: 'var(--accent-red)', fontSize: 13,
        }}>
          {error}
        </div>
      )}
    </div>
  )
}
