import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Home from './pages/Home'
import ModelViewer from './pages/ModelViewer'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/viewer/:modelId" element={<ModelViewer />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
