import React, { useState, useEffect } from 'react';

function Header() {
  const [status, setStatus] = useState('checking');

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch('http://localhost:5001/api/health');
        setStatus(response.ok ? 'connected' : 'error');
      } catch (error) {
        setStatus('error');
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const statusLabel =
    status === 'connected' ? 'Model online' : status === 'checking' ? 'Connecting' : 'Model offline';

  return (
    <header className="site-header">
      <div className="header-inner">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            </svg>
          </div>
          <span className="brand-name">SecBERT NER</span>
        </div>

        <div className={`status-pill status-${status}`} role="status">
          <span className="status-dot" aria-hidden="true" />
          {statusLabel}
        </div>
      </div>
    </header>
  );
}

export default Header;
