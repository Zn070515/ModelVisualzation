import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Home from './pages/Home'
import ModelViewer from './pages/ModelViewer'
import WeightsPage from './pages/WeightsPage'
import DashboardPage from './pages/DashboardPage'
import HealthPage from './pages/HealthPage'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/viewer/:modelId" element={<ModelViewer />} />
        <Route path="/weights/:modelId" element={<WeightsPage />} />
        <Route path="/dashboard/:modelId" element={<DashboardPage />} />
        <Route path="/health/:modelId" element={<HealthPage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
