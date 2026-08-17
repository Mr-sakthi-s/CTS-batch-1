import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Dashboard.css';
import API_BASE_URL from '../services/api';

const severityTone = {
  'High Severity': 'high',
  'Medium Severity': 'medium',
  'Low Severity': 'low',
};

export default function Dashboard() {
  const navigate = useNavigate();
  const [incident, setIncident] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [agentResult, setAgentResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showRca, setShowRca] = useState(false);

  const fetchLatestIncident = async () => {
    try {
      setLoading(true);
            const response = await fetch(
        `${API_BASE_URL}/api/incident/latest-with-prediction`
      );
      const result = await response.json();

      if (!response.ok || !result.success || !result.data) {
        setIncident(null);
        setPrediction(null);
        setError('No incident data available');
        setLoading(false);
        return;
      }

      const latestIncident = result.data;
      const predictionResult = result.prediction || null;
      const rcaReport = result.agent_result || null;

      setIncident(latestIncident);
      setPrediction(predictionResult);
      setAgentResult(rcaReport);

      if (!predictionResult) {
        setError(result.message || 'Waiting for model prediction');
      } else {
        setError('');
      }
      setLoading(false);
    } catch (err) {
      setError(err.message || 'Failed to fetch');
      setIncident(null);
      setPrediction(null);
      setLoading(false);
    }
  };

  useEffect(() => {
    // Only fetch once on page load
    fetchLatestIncident();
  }, []);

  const severity = prediction?.severity || 'Waiting for model';
  const severityClass = severityTone[severity] || 'medium';

const handleRootCauseAction = () => {
  if (!incident) {
    return;
  }

  navigate(`/incident/${incident.id}`);
};

  return (
    <div className="noc-dashboard">
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-badge">CTS</div>
          <div>
            <p className="eyebrow">Incident Response Console</p>
            <h1>NOC Dashboard</h1>
          </div>
        </div>

        <div className="topbar-actions">
          <button className="ghost-button">Live Feed</button>
          <button className="primary-button" onClick={fetchLatestIncident} disabled={loading}>
            {loading ? 'Fetching...' : 'Send Data'}
          </button>
        </div>
      </header>

      <div className="dashboard-grid">
        <section className="panel incident-panel">
          <div className="panel-header">
            <span className="panel-label">NOC</span>
            <span className="panel-sub">Incoming incident</span>
          </div>

          {incident ? (
            <div className="incident-card">
              <div className="incident-meta">
                <div>
                  <p className="meta-label">Ticket ID</p>
                  <h3>{incident.id}</h3>
                </div>
                <div className="status-chip">Active</div>
              </div>

              <div className="detail-grid">
                <div className="detail-row">
                  <span>Location</span>
                  <strong>{incident.location}</strong>
                </div>
                <div className="detail-row">
                  <span>Severity Type</span>
                  <strong>{incident.severity_type}</strong>
                </div>
                <div className="detail-row">
                  <span>Resource Type</span>
                  <strong>{incident.resource_type}</strong>
                </div>
                <div className="detail-row">
                  <span>Event Types</span>
                  <strong>{incident.event_types.join(', ')}</strong>
                </div>
                <div className="detail-row">
                  <span>Log Features</span>
                  <strong>{incident.log_features.join(', ')}</strong>
                </div>
                <div className="detail-row">
                  <span>Total Log Volume</span>
                  <strong>{incident.total_log_volume}</strong>
                </div>
                <div className="detail-row">
                  <span>Mean Log Volume</span>
                  <strong>{incident.mean_log_volume}</strong>
                </div>
                <div className="detail-row">
                  <span>Max Log Volume</span>
                  <strong>{incident.max_log_volume}</strong>
                </div>
              </div>
            </div>
          ) : (
            <div className="idle-state">Waiting for data...</div>
          )}
        </section>

        <section className="panel analysis-panel">
          <div className="panel-header">
            <span className="panel-label">Model Output</span>
            <span className="panel-sub">Fault severity</span>
          </div>

          <div className="severity-box">
            <p className="meta-label">Detected Severity</p>
            <div className={`severity-badge ${severityClass}`}>{severity}</div>
            {prediction && (
              <div className="severity-stats">
                <div>
                  <span>Class</span>
                  <strong>{prediction.fault_severity}</strong>
                </div>
                <div>
                  <span>Confidence</span>
                  <strong>{Number(prediction.confidence * 100).toFixed(2)}%</strong>
                </div>
              </div>
            )}
          </div>

          <div className="rca-action">
            <button className="root-cause-button" onClick={handleRootCauseAction} disabled={!prediction}>
              {showRca ? 'Review Root Cause' : 'Root Cause'}
            </button>
          </div>

          {showRca && prediction && (
            <div className="rca-panel">
              <h4>Root Cause Analysis</h4>
              <p>Severity has been classified by the ML pipeline and is now ready for RCA review.</p>
              <ul>
                <li>Fault Class: {prediction.fault_severity}</li>
                <li>Severity: {prediction.severity}</li>
                <li>Confidence: {Number(prediction.confidence * 100).toFixed(2)}%</li>
              </ul>
            </div>
          )}

          {error && <div className="error-box">{error}</div>}
        </section>
      </div>
    </div>
  );
}
