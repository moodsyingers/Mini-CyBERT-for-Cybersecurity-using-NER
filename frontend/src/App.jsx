import { useState, useEffect } from 'react';
import './App.css';
import Header from './components/Header';

function App() {
  const [inputText, setInputText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const API_URL = 'http://localhost:5001';

  useEffect(() => {
    console.log('🚀 App Component Mounted!');
    console.log('API URL:', API_URL);
  }, []);

  const handleAnalyze = async () => {
    console.log('=== ANALYZE BUTTON CLICKED ===');
    console.log('Input Text:', inputText);

    if (!inputText.trim()) {
      console.log('❌ Error: Empty text');
      setError('Please enter some text');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    console.log('🔄 Loading started...');

    try {
      const endpoint = `${API_URL}/api/ner/analyze`;
      const requestBody = { text: inputText };

      console.log('📡 Making request to:', endpoint);
      console.log('📦 Request body:', requestBody);

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      console.log('📨 Response status:', response.status);

      if (!response.ok) {
        console.log('❌ Response not OK');
        throw new Error('Analysis failed');
      }

      const data = await response.json();
      console.log('✅ Response data:', data);

      // Filter out two-letter words from NER results
      if (data.entities) {
        // First filter by length
        data.entities = data.entities.filter(entity => {
          const word = (entity.word || '').trim().replace(/[.,;:!?]/g, '');
          return word.length > 2;
        });

        // Remove duplicates - keep only unique word + entity_type combinations
        const seen = new Set();
        data.entities = data.entities.filter(entity => {
          const key = `${entity.word.toLowerCase().trim()}|${entity.entity_type}`;
          if (seen.has(key)) {
            console.log('🔄 Removing duplicate:', entity.word, entity.entity_type);
            return false;
          }
          seen.add(key);
          return true;
        });

        data.entity_count = data.entities.length;
        console.log('✅ After deduplication, entity count:', data.entity_count);

        // Also filter sentence entities
        if (data.sentences) {
          data.sentences = data.sentences.map(sent => ({
            ...sent,
            entities: (sent.entities || []).filter(entity => {
              const word = (entity.word || '').trim().replace(/[.,;:!?]/g, '');
              return word.length > 2;
            })
          }));
        }
      }

      console.log('✅ Setting result state');
      setResult(data);
    } catch (err) {
      console.error('❌ Error occurred:', err);
      setError(err.message);
    } finally {
      setLoading(false);
      console.log('✅ Loading finished');
    }
  };

  const getEntityColor = (entityType) => {
    const colors = {
      APT: '#e03131',
      THREAT_ACTOR: '#ff6b6b',
      MALWARE: '#ee5a6f',
      VULNERABILITY: '#f06595',
      TOOL: '#845ef7',
      EXPLOIT: '#cc5de8',
      METHOD: '#7950f2',
      CAMPAIGN: '#5c7cfa',
      INDICATOR: '#339af0',
      HASH: '#228be6',
      IP: '#15aabf',
      URL: '#12b886',
      FILE: '#40c057',
      SOFTWARE: '#82c91e',
      INFRASTRUCTURE: '#fab005',
    };
    return colors[entityType] || '#868e96';
  };

  const exampleTexts = [
    'APT28, also known as Fancy Bear, used phishing to exploit CVE-2023-12345.',
    'The ransomware Conti targeted healthcare using Microsoft Exchange vulnerabilities.',
    'Attackers exploited CVE-2021-44228 (Log4Shell) to gain network access.',
  ];

  return (
    <div className="app">
      <Header />

      <div className="container">
        <div className="main-content">
          <div className="card">
            <h2>Cybersecurity Named Entity Recognition</h2>
            <p className="subtitle">
              Extract cybersecurity entities with SecBERT fine-tuned for NER
            </p>

            {/* Input Section */}
            <div className="input-section">
              <label>Enter cybersecurity text:</label>
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="e.g., APT28 exploited CVE-2023-12345 in a phishing campaign"
                rows={4}
              />

              <button
                onClick={handleAnalyze}
                disabled={loading}
                className="analyze-btn"
              >
                {loading ? 'Analyzing...' : 'Analyze Text'}
              </button>
            </div>

            {/* Error Message */}
            {error && (
              <div className="error-message">
                {error}
              </div>
            )}

            {/* Results Section */}
            {result && (
              <div className="results-section">
                <h3>Detected Entities ({result.entity_count})</h3>
                {result.entities && result.entities.length > 0 ? (
                  <div className="entities-grid">
                    {result.entities.map((entity, index) => (
                      <div
                        key={index}
                        className="entity-card"
                        style={{ borderLeftColor: getEntityColor(entity.entity_type) }}
                      >
                        <div className="entity-word">{entity.word}</div>
                        <div
                          className="entity-type"
                          style={{ backgroundColor: getEntityColor(entity.entity_type) }}
                        >
                          {entity.entity_type}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p>No entities detected</p>
                )}
              </div>
            )}

            {/* Example Texts */}
            <div className="examples-section">
              <h4>Example Texts:</h4>
              <div className="examples-grid">
                {exampleTexts.map((example, index) => (
                  <button
                    key={index}
                    onClick={() => setInputText(example)}
                    className="example-btn"
                  >
                    Example {index + 1}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
