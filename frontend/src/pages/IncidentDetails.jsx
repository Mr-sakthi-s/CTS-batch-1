import React, { useMemo, useState } from 'react';
import { useLocation,useNavigate} from 'react-router-dom';
import './IncidentDetails.css';

export default function IncidentDetails() {
  const location = useLocation();
  const navigate = useNavigate();
  const payload = location.state || {};

  const incident = payload.incident || null;
  const prediction = payload.prediction || null;
  const agentResult = payload.agentResult || null;

  const rankedCauses = useMemo(() => {
    if (agentResult?.ranked_causes?.length) {
      return agentResult.ranked_causes;
    }
    if (agentResult?.rca?.ranked_causes?.length) {
      return agentResult.rca.ranked_causes;
    }
    return [];
  }, [agentResult]);

  const [currentIndex, setCurrentIndex] = useState(0);
  const [answers, setAnswers] = useState({});

  const currentCause = rankedCauses[currentIndex] || null;
  const isComplete = rankedCauses.length > 0 && Object.keys(answers).length >= rankedCauses.length;

  const handleAnswer = (fixed) => {
    if (!currentCause) return;

    setAnswers((prev) => ({
      ...prev,
      [currentIndex]: fixed,
    }));

    if (currentIndex < rankedCauses.length - 1) {
      setCurrentIndex((prev) => prev + 1);
    }
  };
console.log('IncidentDetails payload:', { incident, prediction, agentResult, rankedCauses });
  if (!incident && !prediction && !agentResult) {
    return (
      <div className="incident-details empty-state">
        <h1>No RCA data available</h1>
        <button className="back-button" onClick={() => navigate('/dashboard')}>
          Back to dashboard
        </button>
      </div>
    );
  }

  return (
    <div className="incident-details-page">
      <div className="incident-header-row">
        <div>
          <p className="eyebrow">Incident Review</p>
          <h1>Ticket #{incident?.id || 'Unknown'}</h1>
        </div>
        <button className="back-button" onClick={() => navigate('/dashboard')}>
          Back to Dashboard
        </button>
      </div>

      <div className="incident-details-grid">
        <section className="incident-summary-card">
          <h3>Incident Summary</h3>
          <div className="summary-grid">
            <div>
              <span>Location</span>
              <strong>{incident?.location || 'Not available'}</strong>
            </div>
            <div>
              <span>Severity</span>
              <strong>{prediction?.severity || 'Waiting for prediction'}</strong>
            </div>
            <div>
              <span>Resource</span>
              <strong>{incident?.resource_type || 'Not available'}</strong>
            </div>
            <div>
              <span>Fault Class</span>
              <strong>{prediction?.fault_severity ?? 'N/A'}</strong>
            </div>
            <div>
              <span>Confidence</span>
              <strong>
                {prediction?.confidence ? `${(prediction.confidence * 100).toFixed(2)}%` : 'N/A'}
              </strong>
            </div>
            <div>
              <span>Risk Level</span>
              <strong>{agentResult?.risk_level || 'UNKNOWN'}</strong>
            </div>
          </div>

          {agentResult?.technical_summary && (
            <div className="technical-summary">
              <h4>Technical Summary</h4>
              <p>{agentResult.technical_summary}</p>
            </div>
          )}
        </section>

        <section className="rca-review-card">
          <div className="review-header">
            <div>
              <span className="meta-label">RCA Review</span>
              <h3>
                {rankedCauses.length > 0
                  ? `Candidate ${currentIndex + 1} of ${rankedCauses.length}`
                  : 'No RCA candidates'}
              </h3>
            </div>
            {rankedCauses.length > 0 && (
              <span className="progress-chip">{Object.keys(answers).length}/{rankedCauses.length} answered</span>
            )}
          </div>

          {currentCause ? (
            <>
              <div className="rca-card-item">
                <div className="rca-line">
                  <span>Root Cause</span>
                  <strong>{currentCause.root_cause}</strong>
                </div>
                <div className="rca-line">
                  <span>Confidence</span>
                  <strong>{Number(currentCause.confidence || 0) * 100}%</strong>
                </div>
                <div className="rca-line">
                  <span>Prevention / Recommended action</span>
                  <strong>{currentCause.resolution || 'Review the recommended mitigation plan.'}</strong>
                </div>
                <div className="rca-line">
                  <span>Evidence</span>
                  <strong>{currentCause.evidence || 'Historical and knowledge-based evidence.'}</strong>
                </div>
              </div>

              <div className="feedback-box">
                <p>Issue Fixed?</p>
                <div className="feedback-controls">
                  <button
                    className="feedback-button yes"
                    onClick={() => handleAnswer(true)}
                  >
                    Yes
                  </button>
                  <button
                    className="feedback-button no"
                    onClick={() => handleAnswer(false)}
                  >
                    No
                  </button>
                </div>
              </div>
            </>
          ) : (
            <div className="empty-msg">No ranked RCA data available for this incident.</div>
          )}

          {rankedCauses.length > 0 && isComplete && (
            <div className="all-answered-banner">
              RCA review completed for all top candidates.
            </div>
          )}
        </section>
      </div>

      {rankedCauses.length > 0 && (
        <section className="candidate-list-card">
          <h3>Top 3 RCA candidates</h3>
          <div className="candidate-list">
            {rankedCauses.map((candidate, index) => (
              <div
                key={`${candidate.root_cause}-${index}`}
                className={`candidate-item ${index === currentIndex ? 'active' : ''}`}
              >
                <div className="candidate-head">
                  <span className="candidate-rank">#{index + 1}</span>
                  <strong>{candidate.root_cause}</strong>
                </div>
                <p>{candidate.resolution || 'Recommended mitigation unavailable.'}</p>
                <div className="candidate-meta">
                  <span>Confidence: {(Number(candidate.confidence || 0) * 100).toFixed(2)}%</span>
                  <span>
                    {answers[index] === undefined ? 'Pending' : answers[index] ? 'Fixed: Yes' : 'Fixed: No'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
