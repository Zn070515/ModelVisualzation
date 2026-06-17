import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Home from './pages/Home'
import ModelViewer from './pages/ModelViewer'
import WeightsPage from './pages/WeightsPage'
import DashboardPage from './pages/DashboardPage'
import HealthPage from './pages/HealthPage'
import ActivationPage from './pages/ActivationPage'
import BatchPage from './pages/BatchPage'
import ChainPage from './pages/ChainPage'
import ComparePage from './pages/ComparePage'
import PerformancePage from './pages/PerformancePage'
import PrunePage from './pages/PrunePage'
import QuantPage from './pages/QuantPage'
import AttentionPage from './pages/AttentionPage'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/viewer/:modelId" element={<ModelViewer />} />
        <Route path="/weights/:modelId" element={<WeightsPage />} />
        <Route path="/dashboard/:modelId" element={<DashboardPage />} />
        <Route path="/health/:modelId" element={<HealthPage />} />
        <Route path="/compare/:aId/:bId" element={<ComparePage />} />
        <Route path="/chain" element={<ChainPage />} />
        <Route path="/batch" element={<BatchPage />} />
        <Route path="/performance/:modelId" element={<PerformancePage />} />
        <Route path="/quant/:modelId" element={<QuantPage />} />
        <Route path="/activation/:modelId" element={<ActivationPage />} />
        <Route path="/prune/:modelId" element={<PrunePage />} />
        <Route path="/attention/:modelId" element={<AttentionPage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
