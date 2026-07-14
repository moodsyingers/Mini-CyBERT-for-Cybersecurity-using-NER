import React, { useState, useEffect } from 'react';

function Header() {
  const [status, setStatus] = useState('checking');

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch('http://localhost:5001/api/health');
        if (response.ok) {
          setStatus('connected');
        } else {
          setStatus('error');
        }
      } catch (error) {
        setStatus('error');
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header style={{
      background: '#ffffff',
      borderBottom: '1px solid rgba(226, 232, 240, 0.8)',
      height: '88px',
      position: 'sticky',
      top: 0,
      zIndex: 100,
      boxShadow: '0 1px 3px rgba(0, 0, 0, 0.02), 0 4px 12px rgba(0, 0, 0, 0.03)',
      backdropFilter: 'blur(12px)',
      backgroundColor: 'rgba(255, 255, 255, 0.95)'
    }}>
      <div style={{ 
        height: '100%', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between', 
        maxWidth: '1200px', 
        margin: '0 auto', 
        padding: '0 2rem' 
      }}>
        
        {/* Logo Section */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '18px' }}>
          <div style={{
            width: 52,
            height: 52,
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            borderRadius: '14px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 4px 16px rgba(102, 126, 234, 0.25), 0 8px 32px rgba(102, 126, 234, 0.15)',
            position: 'relative'
          }}>
            <div style={{
              position: 'absolute',
              inset: 0,
              borderRadius: '14px',
              background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.2) 0%, rgba(255, 255, 255, 0) 100%)'
            }}></div>
            <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
          </div>
          <div>
            <h1 style={{
              fontSize: '26px',
              fontWeight: '900',
              color: '#0f172a',
              margin: 0,
              letterSpacing: '-1px',
              lineHeight: '1'
            }}>
              SecBERT <span style={{
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                fontWeight: '900'
              }}>NER</span>
            </h1>
            <span style={{
              fontSize: '13px',
              color: '#64748b',
              fontWeight: '700',
              letterSpacing: '0.3px',
              marginTop: '4px',
              display: 'block'
            }}>
              Advanced Threat Intelligence Platform
            </span>
          </div>
        </div>

        {/* Status Section */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '14px', 
            background: 'linear-gradient(135deg, #fafbfc 0%, #f8fafc 100%)', 
            padding: '12px 24px', 
            borderRadius: '14px',
            border: '1px solid #e2e8f0',
            boxShadow: '0 1px 3px rgba(0, 0, 0, 0.02)'
          }}>
            <div style={{ textAlign: 'right' }}>
              <div style={{ 
                fontSize: '13px', 
                fontWeight: '800', 
                color: '#334155',
                letterSpacing: '-0.2px'
              }}>
                Model Engine
              </div>
              <div style={{ 
                fontSize: '12px', 
                fontWeight: '800',
                color: status === 'connected' ? '#10b981' : '#ef4444',
                letterSpacing: '0.3px',
                textTransform: 'uppercase',
                marginTop: '2px'
              }}>
                {status === 'connected' ? 'Online' : status === 'checking' ? 'Connecting' : 'Offline'}
              </div>
            </div>
            <div style={{ position: 'relative' }}>
              <div style={{
                width: 14,
                height: 14,
                background: status === 'connected' ? '#10b981' : status === 'checking' ? '#f59e0b' : '#ef4444',
                borderRadius: '50%',
                boxShadow: status === 'connected' 
                  ? '0 0 16px rgba(16, 185, 129, 0.6), 0 0 32px rgba(16, 185, 129, 0.3)' 
                  : 'none',
                border: '2px solid white'
              }}></div>
              {status === 'connected' && (
                <>
                  <div style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: 14,
                    height: 14,
                    background: '#10b981',
                    borderRadius: '50%',
                    opacity: 0.5,
                    animation: 'ping 1.5s cubic-bezier(0, 0, 0.2, 1) infinite'
                  }}></div>
                  <div style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: 14,
                    height: 14,
                    background: '#10b981',
                    borderRadius: '50%',
                    opacity: 0.3,
                    animation: 'ping 1.5s cubic-bezier(0, 0, 0.2, 1) infinite 0.75s'
                  }}></div>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
      <style>{`
        @keyframes ping {
          75%, 100% {
            transform: scale(2.5);
            opacity: 0;
          }
        }
      `}</style>
    </header>
  );
}

export default Header;
