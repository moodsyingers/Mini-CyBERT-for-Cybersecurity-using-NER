import { useState } from 'react';
import './App.css';
import Header from './components/Header';

const API_URL = 'http://localhost:5001';

// Color encodes the semantic family (8 validated hues); the visible label
// always carries the exact entity type, so identity never relies on color alone.
const ENTITY_FAMILIES = {
  APT: 'actor',
  THREAT_ACTOR: 'actor',
  CAMPAIGN: 'actor',
  MALWARE: 'malware',
  VULNERABILITY: 'weakness',
  EXPLOIT: 'weakness',
  METHOD: 'technique',
  TOOL: 'tool',
  SOFTWARE: 'system',
  INFRASTRUCTURE: 'system',
  FILE: 'artifact',
  HASH: 'artifact',
  IP: 'network',
  URL: 'network',
  INDICATOR: 'network',
};

const FAMILY_COLORS = {
  actor: '#9085e9',
  malware: '#e66767',
  weakness: '#d95926',
  technique: '#d55181',
  tool: '#3987e5',
  system: '#c98500',
  artifact: '#199e70',
  network: '#008300',
  other: '#8b93a1',
};

const EXAMPLE_TEXTS = [
  'APT28, also known as Fancy Bear, used phishing to exploit CVE-2023-12345.',
  'The ransomware Conti targeted healthcare using Microsoft Exchange vulnerabilities.',
  'Attackers exploited CVE-2021-44228 (Log4Shell) to gain network access.',
];

const entityColor = (type) => FAMILY_COLORS[ENTITY_FAMILIES[type] || 'other'];

const tint = (hex, alpha) => {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
};

const isSubstantive = (entity) => {
  const word = (entity.word || '').trim().replace(/[.,;:!?]/g, '');
  return word.length > 2;
};

// Non-overlapping entities sorted by position, for inline highlighting.
const spanEntities = (entities) => {
  const sorted = [...entities].filter(isSubstantive).sort((a, b) => a.start - b.start);
  const spans = [];
  let cursor = 0;
  for (const e of sorted) {
    if (e.start >= cursor && e.end > e.start) {
      spans.push(e);
      cursor = e.end;
    }
  }
  return spans;
};

// Entities grouped by type with per-word counts, for the summary panel.
const groupByType = (entities) => {
  const groups = new Map();
  for (const e of entities.filter(isSubstantive)) {
    if (!groups.has(e.entity_type)) groups.set(e.entity_type, new Map());
    const words = groups.get(e.entity_type);
    const key = e.word.toLowerCase().trim();
    if (!words.has(key)) words.set(key, { word: e.word, count: 0 });
    words.get(key).count += 1;
  }
  return [...groups.entries()]
    .map(([type, words]) => ({
      type,
      color: entityColor(type),
      words: [...words.values()],
      total: [...words.values()].reduce((sum, w) => sum + w.count, 0),
    }))
    .sort((a, b) => b.total - a.total);
};

function AnnotatedText({ text, entities }) {
  const spans = spanEntities(entities);
  if (spans.length === 0) return <p className="annotated-text">{text}</p>;

  const parts = [];
  let cursor = 0;
  spans.forEach((e, i) => {
    if (e.start > cursor) parts.push(<span key={`t${i}`}>{text.slice(cursor, e.start)}</span>);
    const color = entityColor(e.entity_type);
    parts.push(
      <mark
        key={`e${i}`}
        className="entity-span"
        style={{ backgroundColor: tint(color, 0.16), borderColor: tint(color, 0.45) }}
      >
        {text.slice(e.start, e.end)}
        <span className="entity-tag" style={{ color }}>{e.entity_type}</span>
      </mark>
    );
    cursor = e.end;
  });
  if (cursor < text.length) parts.push(<span key="tail">{text.slice(cursor)}</span>);

  return <p className="annotated-text">{parts}</p>;
}

function App() {
  const [inputText, setInputText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!inputText.trim()) {
      setError('Enter some text to analyze.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/api/ner/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText }),
      });
      if (!response.ok) throw new Error(`Analysis failed (HTTP ${response.status})`);
      setResult(await response.json());
    } catch (err) {
      setError(
        err.message === 'Failed to fetch'
          ? 'Cannot reach the API — make sure the backend is running on port 5001.'
          : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleAnalyze();
  };

  const groups = result ? groupByType(result.entities || []) : [];
  const totalEntities = groups.reduce((sum, g) => sum + g.total, 0);

  return (
    <div className="app">
      <Header />

      <main className="container">
        <section className="card input-card">
          <div className="card-heading">
            <h1>Analyze cybersecurity text</h1>
            <p className="card-caption">
              Entities are extracted by SecBERT fine-tuned for token classification on the CyberNER dataset.
            </p>
          </div>

          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Paste a threat report, vulnerability description, or incident summary…"
            rows={5}
            spellCheck={false}
          />

          <div className="input-actions">
            <div className="examples">
              <span className="examples-label">Try:</span>
              {EXAMPLE_TEXTS.map((example, i) => (
                <button
                  key={i}
                  type="button"
                  className="example-chip"
                  onClick={() => setInputText(example)}
                  title={example}
                >
                  {example.length > 44 ? `${example.slice(0, 44)}…` : example}
                </button>
              ))}
            </div>

            <button className="analyze-btn" onClick={handleAnalyze} disabled={loading}>
              {loading && <span className="spinner" aria-hidden="true" />}
              {loading ? 'Analyzing' : 'Analyze'}
            </button>
          </div>

          {error && (
            <div className="error-banner" role="alert">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden="true">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              {error}
            </div>
          )}
        </section>

        {result && (
          <>
            <section className="stat-row">
              <div className="stat-tile">
                <span className="stat-value">{totalEntities}</span>
                <span className="stat-label">Entities detected</span>
              </div>
              <div className="stat-tile">
                <span className="stat-value">{groups.length}</span>
                <span className="stat-label">Entity types</span>
              </div>
              <div className="stat-tile">
                <span className="stat-value">{(result.sentences || []).length}</span>
                <span className="stat-label">Sentences</span>
              </div>
            </section>

            <section className="card">
              <h2 className="section-title">Annotated text</h2>
              <AnnotatedText text={result.text} entities={result.entities || []} />
            </section>

            <section className="card">
              <h2 className="section-title">Entity summary</h2>
              {groups.length > 0 ? (
                <div className="summary-list">
                  {groups.map((g) => (
                    <div className="summary-row" key={g.type}>
                      <div className="summary-type">
                        <span className="type-dot" style={{ backgroundColor: g.color }} aria-hidden="true" />
                        <span className="type-name">{g.type}</span>
                        <span className="type-count">{g.total}</span>
                      </div>
                      <div className="summary-words">
                        {g.words.map((w) => (
                          <span
                            className="word-chip"
                            key={w.word}
                            style={{ borderColor: tint(g.color, 0.4), backgroundColor: tint(g.color, 0.1) }}
                          >
                            {w.word}
                            {w.count > 1 && <span className="word-count">×{w.count}</span>}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="empty-note">No cybersecurity entities were detected in this text.</p>
              )}
            </section>
          </>
        )}
      </main>

      <footer className="site-footer">
        SecBERT fine-tuned for cybersecurity NER · CyberNER dataset · Flask + React demo
      </footer>
    </div>
  );
}

export default App;
