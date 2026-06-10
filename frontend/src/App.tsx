import { useState, useRef } from 'react'
import { analyze, type AnalyzeResponse } from './api'
import './App.css'

/* ── Icon helpers ── */
const Icon = {
  Search: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
    </svg>
  ),
  Alert: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
    </svg>
  ),
  Sparkle: () => (
    <svg viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2l1.6 6.4L20 10l-6.4 1.6L12 18l-1.6-6.4L4 10l6.4-1.6L12 2z"/>
    </svg>
  ),
  Star: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
    </svg>
  ),
  Brain: () => (
    <svg viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 14H9V8h2v8zm4 0h-2V8h2v8z"/>
    </svg>
  ),
}

/* ── Confidence Bar ── */
function ConfidenceBar({ value, type }: { value: number; type: 'clickbait' | 'safe' }) {
  return (
    <div className="confidence-section">
      <div className="confidence-label">
        <span>Confidence</span>
        <span>{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="confidence-track">
        <div
          className={`confidence-fill ${type}`}
          style={{ width: `${value * 100}%` }}
        />
      </div>
    </div>
  )
}

/* ── Model Card ── */
function ModelCard({
  name,
  result,
  cascadeUsed,
}: {
  name: string
  result: { label: number; is_clickbait: boolean; confidence: number }
  cascadeUsed?: boolean
}) {
  const isClickbait = result.is_clickbait
  const type = isClickbait ? 'clickbait' : 'safe'

  return (
    <div className={`model-card ${type}`}>
      <div className="card-header">
        <span className="card-model-name">{name}</span>
        {cascadeUsed && (
          <span className="cascade-badge">
            <Icon.Sparkle />
            Cascade
          </span>
        )}
      </div>

      <div className="verdict">
        <div className={`verdict-chip ${type}`}>
          <span className="verdict-icon">{isClickbait ? '⚠️' : '✅'}</span>
          {isClickbait ? 'Clickbait' : 'Legitimate'}
        </div>
      </div>

      <ConfidenceBar value={result.confidence} type={type} />
    </div>
  )
}

/* ── Spoiler Card (Gemini colours) ── */
function SpoilerCard({ spoiler }: { spoiler: string }) {
  return (
    <div className="spoiler-card">
      <div className="spoiler-header">
        <div className="gemini-logo">
          {/* Gemini star */}
          <svg viewBox="0 0 24 24" fill="white">
            <path d="M12 2 L13.8 9.2 L21 11 L13.8 12.8 L12 20 L10.2 12.8 L3 11 L10.2 9.2 Z"/>
          </svg>
        </div>
        <span className="spoiler-title">Gemini AI Spoiler</span>
      </div>
      <p className="spoiler-text">"{spoiler}"</p>
    </div>
  )
}

/* ════════════════════════════════════
   Main App
   ════════════════════════════════════ */
export default function App() {
  const [title, setTitle] = useState('')
  const [content, setContent] = useState('')
  const [cascade, setCascade] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AnalyzeResponse | null>(null)

  const resultsRef = useRef<HTMLDivElement>(null)

  async function handleAnalyze(e: React.SyntheticEvent) {
    e.preventDefault()
    if (!title.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await analyze({ title: title.trim(), content: content.trim(), cascade })
      setResult(data)
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  const spoiler = result?.transformer.spoiler

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo-mark">
            <Icon.Search />
          </div>
          <span className="logo-text">ClickSight</span>
          <span className="logo-tag">Clickbait Detection</span>
          <div className="header-spacer" />
        </div>
      </header>

      {/* ── Main ── */}
      <main className="main">
        <h1 className="page-title">Is this clickbait?</h1>
        <p className="page-sub">
          Paste an article title and content — two models analyse it instantly.
        </p>

        {/* ── Input form ── */}
        <form onSubmit={handleAnalyze}>
          <div className="input-card">
            <div>
              <div className="field-label">
                <span className="dot" />
                Headline / Title
              </div>
              <textarea
                className="text-input"
                rows={2}
                placeholder="e.g. 你絕對猜不到這個人做了什麼…"
                value={title}
                onChange={e => setTitle(e.target.value)}
                required
              />
            </div>

            <div style={{ marginTop: 16 }}>
              <div className="field-label">
                <span className="dot" style={{ background: 'var(--cl-purple)' }} />
                Article Content
                <span style={{ fontSize: 11, fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>
                  (optional — improves accuracy)
                </span>
              </div>
              <textarea
                className="text-input"
                rows={5}
                placeholder="Paste the article body here…"
                value={content}
                onChange={e => setContent(e.target.value)}
              />
            </div>

            <div style={{ marginTop: 20, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <label className="cascade-toggle">
                <div
                  className={`toggle-track ${cascade ? 'on' : 'off'}`}
                  onClick={() => setCascade(v => !v)}
                  role="switch"
                  aria-checked={cascade}
                >
                  <div className="toggle-thumb" />
                </div>
                <span className="toggle-label">
                  層級輔助偵測
                  <span className="toggle-sub">{cascade ? '已開啟 (Gemini 複核)' : '已關閉 (僅 XLM-RoBERTa)'}</span>
                </span>
              </label>

              <button
                type="submit"
                className="analyze-btn"
                disabled={loading || !title.trim()}
              >
                {loading ? (
                  <>
                    <span className="spinner" />
                    Analysing…
                  </>
                ) : (
                  <>
                    <Icon.Search />
                    Analyse
                  </>
                )}
              </button>
            </div>
          </div>
        </form>

        {/* ── Error ── */}
        {error && (
          <div className="error-banner">
            <Icon.Alert />
            <span>{error}</span>
          </div>
        )}

        {/* ── Results ── */}
        <div ref={resultsRef}>
          {result ? (
            <div className="results">
              <div className="results-heading">Analysis Results</div>

              <div className="cards-row">
                <ModelCard
                  name="TF-IDF + Logistic Regression"
                  result={result.baseline}
                />
                <ModelCard
                  name={result.transformer.gemini_used ? "XLM-RoBERTa + Gemini (輔助)" : "XLM-RoBERTa"}
                  result={result.transformer}
                  cascadeUsed={result.transformer.gemini_used}
                />
              </div>

              {spoiler && (
                <>
                  <div className="divider" />
                  <SpoilerCard spoiler={spoiler} />
                </>
              )}
            </div>
          ) : !loading && (
            <div className="hint-box">
              <p className="hint-text">
                Enter a headline above to detect clickbait and reveal the story.
              </p>
            </div>
          )}
        </div>
      </main>

      {/* ── Footer ── */}
      <footer className="footer">
        ClickSight · NLP Final Project · XLM-RoBERTa + Gemini Cascade
      </footer>
    </div>
  )
}
