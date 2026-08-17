import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import './App.css';

// Pages
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import IncidentDetails from './pages/IncidentDetails';
import Technicians from './pages/Technicians';
import ResolutionHistory from './pages/ResolutionHistory';

function App() {
  return (
    <Router>
      <Routes>
        {/* Login Route */}
        <Route path="/" element={<Login />} />
        <Route path="/login" element={<Login />} />

        {/* Dashboard and Main Routes */}
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/incident/:id" element={<IncidentDetails />} />
        <Route path="/technicians" element={<Technicians />} />
        <Route path="/resolution-history" element={<ResolutionHistory />} />

        {/* Catch all - redirect to login */}
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    </Router>
  );
}

export default App;
