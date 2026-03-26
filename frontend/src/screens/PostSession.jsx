import { useState, useEffect } from 'react'

const API = ''

const verdictStyles = {
  SUCCESSFUL: 'bg-success/10 text-success border-success/30',
  PARTIAL: 'bg-warning-mild/10 text-warning-mild border-warning-mild/30',
  DISPUTE: 'bg-error/10 text-error border-error/30',
  PENDING: 'bg-outline/10 text-outline border-outline/30',
}

const classStyles = {
  correct: 'text-success',
  weakly_correct: 'text-warning-mild',
  incorrect: 'text-error',
  out_of_scope: 'text-warning-severe',
}

const severityStyles = {
  mild: 'border-warning-mild/40 bg-warning-mild/5 text-warning-mild',
  strong: 'border-warning-strong/40 bg-warning-strong/5 text-warning-strong',
  severe: 'border-warning-severe/40 bg-warning-severe/5 text-warning-severe',
}

export default function PostSession({ barterId, onReset }) {
  const [verdict, setVerdict] = useState(null)
  const [transcript, setTranscript] = useState([])
  const [windows, setWindows] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [tab, setTab] = useState('verdict')

  useEffect(() => { load() }, [barterId])

  async function load() {
    setLoading(true)
    setError('')
    try {
      await fetch(`${API}/verdict/${barterId}/generate`, { method: 'POST' })
      await fetch(`${API}/trust/${barterId}/update`, { method: 'POST' })
      const [vRes, tRes, wRes] = await Promise.all([
        fetch(`${API}/verdict/${barterId}`),
        fetch(`${API}/session/${barterId}/transcript`),
        fetch(`${API}/session/${barterId}/windows`),
      ])
      if (!vRes.ok) throw new Error(await vRes.text())
      setVerdict(await vRes.json())
      setTranscript(tRes.ok ? await tRes.json() : [])
      setWindows(wRes.ok ? await wRes.json() : [])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  if (loading) return (
    <div>
      <section className="mb-12 border-l-4 border-primary pl-8">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tighter text-primary uppercase">Results</h1>
        <p className="text-on-surface-variant text-sm font-medium mt-2">Generating verdict...</p>
      </section>
    </div>
  )

  if (error) return (
    <div>
      <section className="mb-12 border-l-4 border-primary pl-8">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tighter text-primary uppercase">Results</h1>
        <p className="text-error text-sm font-medium mt-2 mb-4">{error}</p>
        <button
          className="border border-primary text-primary px-8 py-3 font-bold uppercase tracking-[0.2em] text-sm hover:bg-primary hover:text-on-primary transition-all duration-150 bg-transparent"
          onClick={load}
        >
          Retry
        </button>
      </section>
    </div>
  )

  const drift = verdict.drift_summary
  const engagement = drift?.engagement
  const d1 = verdict.trust_delta_user1
  const d2 = verdict.trust_delta_user2

  function delta(v) {
    const color = v > 0 ? 'text-success' : v < 0 ? 'text-error' : 'text-outline'
    const str = v > 0 ? `+${v.toFixed(4)}` : v.toFixed(4)
    return <span className={`font-bold ${color}`}>{str}</span>
  }

  const tabs = ['verdict', 'windows', 'transcript', 'engagement']

  return (
    <>
      {/* Header */}
      <section className="mb-10 border-l-4 border-primary pl-8">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tighter text-primary uppercase">
          Session #{barterId}
        </h1>
        <p className="text-on-surface-variant text-sm font-medium mt-2 uppercase tracking-widest">Complete</p>
      </section>

      {/* Tab bar */}
      <div className="grid grid-cols-4 border border-primary mb-10">
        {tabs.map(t => (
          <button
            key={t}
            className={`py-4 text-center font-bold uppercase tracking-widest text-xs transition-all duration-200 ${
              tab === t
                ? 'bg-primary text-surface'
                : 'bg-transparent text-primary hover:bg-surface-container-high'
            } ${t !== tabs[tabs.length - 1] ? 'border-r border-primary' : ''}`}
            onClick={() => setTab(t)}
          >
            {t}
          </button>
        ))}
      </div>

      {/* VERDICT TAB */}
      {tab === 'verdict' && (
        <>
          <div className="border border-outline-variant p-8 mb-6">
            {/* Verdict badge */}
            <span className={`inline-block px-5 py-2 text-xs font-bold uppercase tracking-[0.2em] border mb-6 ${verdictStyles[verdict.verdict] || verdictStyles.PENDING}`}>
              {verdict.verdict}
            </span>

            {/* Check pills */}
            <div className="flex gap-3 mb-8">
              <span className={`px-4 py-1.5 text-xs font-bold uppercase tracking-widest ${
                verdict.duration_check ? 'bg-success/10 text-success' : 'bg-error/10 text-error'
              }`}>
                Duration {verdict.duration_check ? 'ok' : 'short'}
              </span>
              <span className={`px-4 py-1.5 text-xs font-bold uppercase tracking-widest ${
                verdict.confirmation_check ? 'bg-success/10 text-success' : 'bg-error/10 text-error'
              }`}>
                {verdict.confirmation_check ? 'Both confirmed' : 'Not confirmed'}
              </span>
            </div>

            {/* Stats grid */}
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div className="border border-outline-variant p-5">
                <div className="text-[10px] font-bold uppercase tracking-[0.2em] text-on-surface-variant mb-2">On-topic</div>
                <div className="text-3xl font-extrabold text-primary">{(verdict.on_topic_percentage ?? 0).toFixed(1)}%</div>
              </div>
              <div className="border border-outline-variant p-5">
                <div className="text-[10px] font-bold uppercase tracking-[0.2em] text-on-surface-variant mb-2">Warnings</div>
                <div className="text-3xl font-extrabold text-primary">{verdict.warning_count}</div>
              </div>
              {drift && (
                <>
                  <div className="border border-outline-variant p-5">
                    <div className="text-[10px] font-bold uppercase tracking-[0.2em] text-on-surface-variant mb-2">Windows</div>
                    <div className="text-3xl font-extrabold text-primary">{drift.total_windows}</div>
                  </div>
                  <div className="border border-outline-variant p-5">
                    <div className="text-[10px] font-bold uppercase tracking-[0.2em] text-on-surface-variant mb-2">Max consec. off</div>
                    <div className="text-3xl font-extrabold text-primary">{drift.max_consecutive_incorrect}</div>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Trust scores */}
          <div className="border border-outline-variant p-8 mb-6">
            <h2 className="text-[10px] font-bold uppercase tracking-[0.2em] text-primary mb-6">Trust Score Changes</h2>
            <div className="flex justify-between items-center py-3 border-b border-outline-variant text-sm">
              <span className="font-bold uppercase tracking-widest text-xs">Alice</span>
              {delta(d1)}
            </div>
            <div className="flex justify-between items-center py-3 text-sm">
              <span className="font-bold uppercase tracking-widest text-xs">Bob</span>
              {delta(d2)}
            </div>
          </div>

          {/* Warning log */}
          {drift?.warnings?.length > 0 && (
            <div className="border border-outline-variant p-8 mb-6">
              <h2 className="text-[10px] font-bold uppercase tracking-[0.2em] text-primary mb-4">Warning Log</h2>
              <div className="flex flex-col gap-2">
                {drift.warnings.map((w, i) => (
                  <div
                    key={i}
                    className={`flex justify-between items-baseline gap-3 px-4 py-3 border text-sm font-medium ${severityStyles[w.severity] || 'border-outline'}`}
                  >
                    <span>{w.reason}</span>
                    <span className="text-xs opacity-50 whitespace-nowrap">
                      {w.timestamp ? new Date(w.timestamp).toLocaleTimeString() : ''}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {/* WINDOWS TAB */}
      {tab === 'windows' && (
        <div className="border border-outline-variant p-8">
          <h2 className="text-[10px] font-bold uppercase tracking-[0.2em] text-primary mb-6">
            Window Results ({windows.length})
          </h2>
          {windows.length === 0 ? (
            <p className="text-on-surface-variant text-sm">No windows recorded yet.</p>
          ) : (
            <div className="flex flex-col">
              {windows.map((w, i) => (
                <div key={i} className="flex flex-col gap-1 py-4 border-b border-outline-variant last:border-b-0">
                  <div className="flex justify-between items-center">
                    <span className="font-bold text-sm text-primary">Window #{w.window_id}</span>
                    <span className={`font-bold text-xs uppercase tracking-widest ${classStyles[w.classification] || 'text-outline'}`}>
                      {w.classification.replace('_', ' ')}
                    </span>
                    <span className="text-xs text-on-surface-variant">sim {w.similarity}</span>
                  </div>
                  {w.text_preview && (
                    <p className="text-xs text-on-surface-variant mt-1">{w.text_preview}</p>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* TRANSCRIPT TAB */}
      {tab === 'transcript' && (
        <div className="border border-outline-variant p-8">
          <h2 className="text-[10px] font-bold uppercase tracking-[0.2em] text-primary mb-6">
            Full Transcript ({transcript.length} segments)
          </h2>
          {transcript.length === 0 ? (
            <p className="text-on-surface-variant text-sm">No transcript recorded.</p>
          ) : (
            <div className="flex flex-col gap-4">
              {transcript.map((s, i) => (
                <div key={i} className="flex gap-3 items-start">
                  <span className={`font-bold text-xs min-w-[44px] pt-0.5 ${s.user_id === 1 ? 'text-blue-600' : 'text-emerald-600'}`}>
                    {s.speaker}
                  </span>
                  <p className="text-sm leading-relaxed text-on-surface m-0">{s.text}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ENGAGEMENT TAB */}
      {tab === 'engagement' && (
        <div className="border border-outline-variant p-8">
          <h2 className="text-[10px] font-bold uppercase tracking-[0.2em] text-primary mb-6">
            Learner Engagement
          </h2>
          {!engagement ? (
            <p className="text-on-surface-variant text-sm">No engagement data recorded.</p>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="border border-outline-variant p-5">
                <div className="text-[10px] font-bold uppercase tracking-[0.2em] text-on-surface-variant mb-2">Engagement score</div>
                <div className="text-3xl font-extrabold text-primary">{(engagement.learner_engagement_score * 100).toFixed(0)}%</div>
              </div>
              <div className="border border-outline-variant p-5">
                <div className="text-[10px] font-bold uppercase tracking-[0.2em] text-on-surface-variant mb-2">Learner speaking</div>
                <div className="text-3xl font-extrabold text-primary">{engagement.learner_speaking_seconds?.toFixed(0)}s</div>
              </div>
              <div className="border border-outline-variant p-5">
                <div className="text-[10px] font-bold uppercase tracking-[0.2em] text-on-surface-variant mb-2">Teacher speaking</div>
                <div className="text-3xl font-extrabold text-primary">{engagement.teacher_speaking_seconds?.toFixed(0)}s</div>
              </div>
              <div className="border border-outline-variant p-5">
                <div className="text-[10px] font-bold uppercase tracking-[0.2em] text-on-surface-variant mb-2">Questions asked</div>
                <div className="text-3xl font-extrabold text-primary">{engagement.learner_question_count}</div>
              </div>
              <div className="border border-outline-variant p-5">
                <div className="text-[10px] font-bold uppercase tracking-[0.2em] text-on-surface-variant mb-2">Acknowledgments</div>
                <div className="text-3xl font-extrabold text-primary">{engagement.learner_acknowledgment_count}</div>
              </div>
              <div className="border border-outline-variant p-5">
                <div className="text-[10px] font-bold uppercase tracking-[0.2em] text-on-surface-variant mb-2">Learner segments</div>
                <div className="text-3xl font-extrabold text-primary">{engagement.learner_segment_count}</div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Reset */}
      <div className="mt-10">
        <button
          className="border border-primary text-primary px-10 py-4 font-bold uppercase tracking-[0.3em] text-sm hover:bg-primary hover:text-on-primary transition-all duration-150 bg-transparent"
          onClick={onReset}
        >
          New Session
        </button>
      </div>
    </>
  )
}
