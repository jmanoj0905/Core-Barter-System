import { useState, useEffect } from 'react'
import { getEscrow } from '../api/resource'
import SettlementBreakdown from '../components/SettlementBreakdown'

const API = ''

const verdictConfig = {
  SUCCESSFUL: { bg: 'bg-tertiary-container', text: 'text-tertiary',   stamp: 'VALID',    stampColor: 'text-tertiary   border-tertiary'   },
  PARTIAL:    { bg: 'bg-[#fef9c3]',          text: 'text-on-background', stamp: 'PARTIAL', stampColor: 'text-warning-mild border-warning-mild' },
  DISPUTE:    { bg: 'bg-error-container',    text: 'text-error',      stamp: 'DISPUTE',  stampColor: 'text-error      border-error'      },
  PENDING:    { bg: 'bg-surface-container',  text: 'text-outline',    stamp: 'PENDING',  stampColor: 'text-outline    border-outline'    },
}

const classColor = {
  correct:        'text-tertiary',
  weakly_correct: 'text-warning-mild',
  incorrect:      'text-error',
  out_of_scope:   'text-warning-severe',
}

const severityBg = {
  mild:   'bg-[#fef9c3]',
  strong: 'bg-secondary-container',
  severe: 'bg-error-container',
}

export default function PostSession({ barterId, onReset, settlement }) {
  const [verdict, setVerdict]     = useState(null)
  const [transcript, setTranscript] = useState([])
  const [windows, setWindows]     = useState([])
  const [escrows, setEscrows]     = useState([])
  const [loading, setLoading]     = useState(true)
  const [error, setError]         = useState('')
  const [tab, setTab]             = useState('verdict')

  useEffect(() => { load() }, [barterId])

  async function load() {
    setLoading(true)
    setError('')
    try {
      await fetch(`${API}/verdict/${barterId}/generate`, { method: 'POST' })
      await fetch(`${API}/trust/${barterId}/update`,    { method: 'POST' })
      const [vRes, tRes, wRes, eRes] = await Promise.all([
        fetch(`${API}/verdict/${barterId}`),
        fetch(`${API}/session/${barterId}/transcript`),
        fetch(`${API}/session/${barterId}/windows`),
        // resource_agent's GET /resource/escrow/{id} returns
        // { session_id, escrows: [...] }, not a bare array like the old
        // backend /escrow endpoint did -- unwrap it here so the array-typed
        // consumers below (`escrows.find`/`.reduce`/`.length`) keep working.
        getEscrow(barterId).then(d => ({ ok: true, json: async () => d.escrows })),
      ])
      if (!vRes.ok) throw new Error(await vRes.text())
      setVerdict(await vRes.json())
      setTranscript(tRes.ok ? await tRes.json() : [])
      setWindows(wRes.ok   ? await wRes.json() : [])
      setEscrows(eRes.ok   ? await eRes.json() : [])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  if (loading) return (
    <div className="p-6 md:p-12">
      <div className="max-w-5xl mx-auto">
        <span className="font-headline text-xs font-bold uppercase tracking-[0.2em] text-primary block mb-2">Core Barter System</span>
        <h1 className="font-headline text-5xl font-bold tracking-tighter text-on-background mb-4">Post Session Results</h1>
        <div className="border-4 border-on-background p-8 neo-shadow bg-white">
          <p className="font-headline font-bold uppercase text-sm text-on-surface-variant animate-blink">
            Generating verdict...
          </p>
        </div>
      </div>
    </div>
  )

  if (error) return (
    <div className="p-6 md:p-12">
      <div className="max-w-5xl mx-auto">
        <h1 className="font-headline text-5xl font-bold tracking-tighter text-on-background mb-6">Post Session Results</h1>
        <div className="border-4 border-error bg-error-container p-8 neo-shadow mb-6">
          <p className="text-on-error-container font-bold">{error}</p>
        </div>
        <button
          className="border-4 border-on-background px-8 py-4 font-headline font-bold uppercase tracking-widest text-sm neo-shadow hover:translate-x-0.5 hover:translate-y-0.5 hover:shadow-none transition-all bg-white"
          onClick={load}
        >
          Retry
        </button>
      </div>
    </div>
  )

  const drift       = verdict.drift_summary
  const engagement  = drift?.engagement
  const d1          = verdict.trust_delta_user1
  const d2          = verdict.trust_delta_user2
  const vc          = verdictConfig[verdict.verdict] || verdictConfig.PENDING

  function Delta({ v }) {
    const color = v > 0 ? 'text-tertiary' : v < 0 ? 'text-error' : 'text-outline'
    const str   = v > 0 ? `+${v.toFixed(4)}` : v.toFixed(4)
    return <span className={`font-headline font-bold text-xl ${color}`}>{str}</span>
  }

  const tabs = ['verdict', 'windows', 'transcript', 'engagement']

  // Escrow summary
  const teacherEscrow = escrows.find(e => e.user_id === 1)
  const learnerEscrow = escrows.find(e => e.user_id === 2)

  return (
    <div className="p-6 md:p-12">
      <div className="max-w-5xl mx-auto">

        {/* ── Page header ── */}
        <section className="mb-12">
          <span className="font-headline uppercase text-xs font-bold tracking-[0.2em] text-primary mb-2 block">
            Core Barter System
          </span>
          <h1 className="font-headline text-5xl md:text-7xl font-bold tracking-tighter text-on-background mb-4">
            Post Session Results
          </h1>
          <div className="h-2 w-32 bg-on-background" />
        </section>

        {/* ── Bento grid ── */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-8">

          {/* Verdict hero */}
          <div className="md:col-span-8 bg-surface-container-lowest border-4 border-on-background p-8 relative overflow-hidden neo-shadow-lg">
            <div className="flex flex-col h-full justify-between">
              <div>
                <h2 className="font-headline text-2xl font-bold mb-2">Verdict Status</h2>
                <p className="text-on-surface-variant max-w-md text-sm">
                  Session #{barterId} analyzed against the compliance framework. Final state recorded below.
                </p>
              </div>
              <div className="mt-10 flex items-center justify-between flex-wrap gap-4">
                <div className="space-y-1">
                  <p className="font-headline text-sm font-bold uppercase">Session ID</p>
                  <p className="text-xl font-bold">#{barterId}</p>
                </div>
                <div className={`border-8 font-headline font-black uppercase tracking-tighter px-8 py-3 text-5xl md:text-7xl opacity-90 -rotate-6 ${vc.stampColor}`}>
                  {vc.stamp}
                </div>
              </div>
            </div>
            <div className="absolute -top-4 -right-4 w-20 h-20 bg-tertiary-container border-4 border-on-background rounded-full flex items-center justify-center">
              <span className="material-symbols-outlined text-3xl" style={{ fontVariationSettings: "'FILL' 1" }}>verified</span>
            </div>
          </div>

          {/* Engagement summary */}
          <div className="md:col-span-4 bg-secondary-container border-4 border-on-background p-6 neo-shadow-lg flex flex-col justify-between">
            <h2 className="font-headline text-xl font-bold mb-6">Session Summary</h2>
            <div className="space-y-5">
              <div className="flex justify-between items-end">
                <div>
                  <p className="font-headline text-[10px] font-bold uppercase text-on-secondary-container">Duration</p>
                  <p className="text-2xl font-bold">{verdict.actual_duration_seconds
                    ? `${Math.floor(verdict.actual_duration_seconds / 60)}m ${Math.round(verdict.actual_duration_seconds % 60)}s`
                    : '—'
                  }</p>
                </div>
                <span className="material-symbols-outlined text-3xl">timer</span>
              </div>
              <div className="flex justify-between items-end">
                <div>
                  <p className="font-headline text-[10px] font-bold uppercase text-on-secondary-container">Windows</p>
                  <p className="text-2xl font-bold">{drift?.total_windows ?? windows.length}</p>
                </div>
                <span className="material-symbols-outlined text-3xl">grid_view</span>
              </div>
              <div className="flex justify-between items-end">
                <div>
                  <p className="font-headline text-[10px] font-bold uppercase text-on-secondary-container">Warnings</p>
                  <p className="text-2xl font-bold">{verdict.warning_count}</p>
                </div>
                <span className="material-symbols-outlined text-3xl">warning</span>
              </div>
            </div>
          </div>

          {/* On-topic metric */}
          <div className="md:col-span-4 bg-primary-container border-4 border-on-background p-6 neo-shadow-lg">
            <div className="flex justify-between items-start mb-4">
              <span className="material-symbols-outlined text-4xl">topic</span>
              <span className="font-headline text-4xl font-bold">
                {(verdict.on_topic_percentage ?? 0).toFixed(0)}%
              </span>
            </div>
            <p className="font-headline font-bold text-sm uppercase mb-3">On-Topic Accuracy</p>
            <div className="w-full bg-white/50 h-4 border-2 border-on-background">
              <div className="bg-primary h-full transition-all" style={{ width: `${verdict.on_topic_percentage ?? 0}%` }} />
            </div>
            {drift && (
              <p className="text-xs mt-3 text-on-primary-container">
                Max {drift.max_consecutive_incorrect} consecutive off-topic windows.
              </p>
            )}
          </div>

          {/* Trust delta metric */}
          <div className="md:col-span-4 bg-surface-container-high border-4 border-on-background p-6 neo-shadow-lg">
            <div className="flex justify-between items-start mb-4">
              <span className="material-symbols-outlined text-4xl">trending_up</span>
              <div className="text-right">
                <Delta v={d1} />
              </div>
            </div>
            <p className="font-headline font-bold text-sm uppercase mb-4">Trust Deltas</p>
            <div className="space-y-2">
              <div className="flex justify-between items-center py-2 border-b border-outline-variant">
                <span className="font-bold text-xs uppercase tracking-widest">Alice</span>
                <Delta v={d1} />
              </div>
              <div className="flex justify-between items-center py-2">
                <span className="font-bold text-xs uppercase tracking-widest">Bob</span>
                <Delta v={d2} />
              </div>
            </div>
          </div>

          {/* Escrow settlement metric */}
          <div className="md:col-span-4 bg-[#fef9c3] border-4 border-on-background p-6 neo-shadow-lg">
            <div className="flex justify-between items-start mb-4">
              <span className="material-symbols-outlined text-4xl">account_balance_wallet</span>
              <span className="font-headline text-2xl font-bold">
                {escrows.length > 0
                  ? escrows.reduce((sum, e) => sum + e.amount, 0) + ' cr'
                  : '—'
                }
              </span>
            </div>
            <p className="font-headline font-bold text-sm uppercase mb-4">Escrow Settlement</p>
            <div className="space-y-2">
              {teacherEscrow && (
                <div className="flex justify-between items-center py-2 border-b border-on-background/20">
                  <span className="text-xs font-bold uppercase">Alice (Teacher)</span>
                  <span className={`text-xs font-bold uppercase px-2 py-1 border border-on-background ${
                    teacherEscrow.state === 'SETTLED' ? 'bg-tertiary-container text-tertiary' :
                    teacherEscrow.state === 'VOIDED' ? 'bg-primary-container text-primary' :
                    teacherEscrow.state === 'HELD' ? 'bg-error-container text-error' :
                    'bg-surface-container text-outline'
                  }`}>
                    {teacherEscrow.state} · {teacherEscrow.amount} cr
                  </span>
                </div>
              )}
              {learnerEscrow && (
                <div className="flex justify-between items-center py-2">
                  <span className="text-xs font-bold uppercase">Bob (Learner)</span>
                  <span className={`text-xs font-bold uppercase px-2 py-1 border border-on-background ${
                    learnerEscrow.state === 'SETTLED' ? 'bg-tertiary-container text-tertiary' :
                    learnerEscrow.state === 'VOIDED' ? 'bg-primary-container text-primary' :
                    learnerEscrow.state === 'HELD' ? 'bg-error-container text-error' :
                    'bg-surface-container text-outline'
                  }`}>
                    {learnerEscrow.state} · {learnerEscrow.amount} cr
                  </span>
                </div>
              )}
              {escrows.length === 0 && (
                <p className="text-xs text-outline">No escrow records found.</p>
              )}
            </div>
          </div>

          {/* Settlement breakdown */}
          {settlement && settlement.breakdown && (
            <div className="md:col-span-12">
              <SettlementBreakdown
                settlement={settlement}
                names={{ 1: 'Alice (Teacher)', 2: 'Bob (Learner)' }}
              />
            </div>
          )}

          {/* Warning log */}
          {drift?.warnings?.length > 0 && (
            <div className="md:col-span-12 bg-white border-4 border-on-background neo-shadow-lg">
              <div className="p-6 border-b-4 border-on-background bg-error-container flex items-center gap-3">
                <span className="material-symbols-outlined">history</span>
                <h2 className="font-headline text-xl font-bold">Violation &amp; Warning Log</h2>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left">
                  <thead className="bg-surface-container font-headline text-xs uppercase font-bold border-b-2 border-on-background">
                    <tr>
                      <th className="p-4">Time</th>
                      <th className="p-4">Severity</th>
                      <th className="p-4">Message</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y-2 divide-outline-variant">
                    {drift.warnings.map((w, i) => (
                      <tr key={i} className="hover:bg-surface-container-low transition-colors">
                        <td className="p-4 font-mono text-sm">
                          {w.timestamp ? new Date(w.timestamp).toLocaleTimeString() : '—'}
                        </td>
                        <td className="p-4">
                          <span className={`px-2 py-1 border border-on-background text-[10px] font-bold uppercase ${severityBg[w.severity] || 'bg-surface-container'}`}>
                            {w.severity}
                          </span>
                        </td>
                        <td className="p-4 text-sm italic">{w.reason}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Detail tabs */}
          <div className="md:col-span-12">
            {/* Tab bar */}
            <div className="grid grid-cols-4 border-4 border-on-background mb-0 neo-shadow">
              {tabs.map((t, i) => (
                <button
                  key={t}
                  className={`py-4 text-center font-headline font-bold uppercase tracking-widest text-xs transition-all duration-200 border-on-background ${
                    tab === t ? 'bg-on-background text-white' : 'bg-white text-on-background hover:bg-surface-container-high'
                  } ${i < tabs.length - 1 ? 'border-r-4' : ''}`}
                  onClick={() => setTab(t)}
                >
                  {t}
                </button>
              ))}
            </div>

            <div className="bg-white border-4 border-t-0 border-on-background p-8 neo-shadow">

              {/* Verdict tab */}
              {tab === 'verdict' && (
                <div className="space-y-6">
                  <div className="flex gap-3 flex-wrap">
                    <span className={`px-4 py-2 border-4 border-on-background text-xs font-bold uppercase tracking-widest ${vc.bg} ${vc.text}`}>
                      {verdict.verdict}
                    </span>
                    <span className={`px-4 py-2 border-4 border-on-background text-xs font-bold uppercase tracking-widest ${verdict.duration_check ? 'bg-tertiary-container text-tertiary' : 'bg-error-container text-error'}`}>
                      Duration {verdict.duration_check ? 'OK' : 'Short'}
                    </span>
                    <span className={`px-4 py-2 border-4 border-on-background text-xs font-bold uppercase tracking-widest ${verdict.confirmation_check ? 'bg-tertiary-container text-tertiary' : 'bg-error-container text-error'}`}>
                      {verdict.confirmation_check ? 'Both Confirmed' : 'Not Confirmed'}
                    </span>
                  </div>
                  {engagement && (
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-4">
                      {[
                        { label: 'Engagement',         val: `${(engagement.learner_engagement_score * 100).toFixed(0)}%` },
                        { label: 'Learner Speaking',   val: `${engagement.learner_speaking_seconds?.toFixed(0)}s` },
                        { label: 'Teacher Speaking',   val: `${engagement.teacher_speaking_seconds?.toFixed(0)}s` },
                        { label: 'Questions Asked',    val: engagement.learner_question_count },
                        { label: 'Acknowledgments',    val: engagement.learner_acknowledgment_count },
                        { label: 'Learner Segments',   val: engagement.learner_segment_count },
                      ].map(({ label, val }) => (
                        <div key={label} className="border border-outline-variant p-4">
                          <div className="text-[10px] font-bold uppercase tracking-widest text-on-surface-variant mb-1">{label}</div>
                          <div className="text-2xl font-extrabold text-primary font-headline">{val}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Windows tab */}
              {tab === 'windows' && (
                <div>
                  <h2 className="font-headline font-bold text-sm uppercase tracking-widest mb-4">
                    Window Results ({windows.length})
                  </h2>
                  {windows.length === 0 ? (
                    <p className="text-on-surface-variant text-sm">No windows recorded.</p>
                  ) : (
                    <div className="flex flex-col divide-y-2 divide-outline-variant">
                      {windows.map((w, i) => (
                        <div key={i} className="flex justify-between items-center py-3 gap-4">
                          <span className="font-bold text-sm">#{w.window_id}</span>
                          <span className={`text-xs font-bold uppercase ${classColor[w.classification] || 'text-outline'}`}>
                            {w.classification.replace('_', ' ')}
                          </span>
                          <span className="text-xs text-on-surface-variant">sim {w.similarity}</span>
                          <span className="text-xs text-on-surface-variant max-w-[200px] overflow-hidden text-ellipsis whitespace-nowrap">
                            {w.text_preview}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Transcript tab */}
              {tab === 'transcript' && (
                <div>
                  <h2 className="font-headline font-bold text-sm uppercase tracking-widest mb-4">
                    Full Transcript ({transcript.length} segments)
                  </h2>
                  {transcript.length === 0 ? (
                    <p className="text-on-surface-variant text-sm">No transcript recorded.</p>
                  ) : (
                    <div className="flex flex-col gap-4">
                      {transcript.map((s, i) => (
                        <div key={i} className="flex gap-3 items-start">
                          <span className={`font-bold text-xs min-w-[52px] pt-0.5 font-headline ${s.user_id === 1 ? 'text-primary' : 'text-tertiary'}`}>
                            {s.speaker}
                          </span>
                          <p className="text-sm leading-relaxed text-on-surface">{s.text}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Engagement tab */}
              {tab === 'engagement' && (
                <div>
                  <h2 className="font-headline font-bold text-sm uppercase tracking-widest mb-4">
                    Learner Engagement
                  </h2>
                  {!engagement ? (
                    <p className="text-on-surface-variant text-sm">No engagement data recorded.</p>
                  ) : (
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                      {[
                        { label: 'Engagement Score',   val: `${(engagement.learner_engagement_score * 100).toFixed(0)}%` },
                        { label: 'Learner Speaking',   val: `${engagement.learner_speaking_seconds?.toFixed(0)}s` },
                        { label: 'Teacher Speaking',   val: `${engagement.teacher_speaking_seconds?.toFixed(0)}s` },
                        { label: 'Questions Asked',    val: engagement.learner_question_count },
                        { label: 'Acknowledgments',    val: engagement.learner_acknowledgment_count },
                        { label: 'Learner Segments',   val: engagement.learner_segment_count },
                      ].map(({ label, val }) => (
                        <div key={label} className="border border-outline-variant p-5">
                          <div className="text-[10px] font-bold uppercase tracking-widest text-on-surface-variant mb-2">{label}</div>
                          <div className="text-3xl font-extrabold text-primary font-headline">{val}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Footer actions */}
          <div className="md:col-span-12 flex flex-col md:flex-row gap-6">
            <button
              className="flex-1 bg-primary-container border-4 border-on-background py-4 px-8 neo-shadow active:translate-x-0.5 active:translate-y-0.5 active:shadow-none transition-all flex items-center justify-center gap-3"
              onClick={onReset}
            >
              <span className="material-symbols-outlined">add</span>
              <span className="font-headline font-bold uppercase text-sm">New Session</span>
            </button>
            <button
              className="flex-1 bg-tertiary-container border-4 border-on-background py-4 px-8 neo-shadow active:translate-x-0.5 active:translate-y-0.5 active:shadow-none transition-all flex items-center justify-center gap-3"
              onClick={() => window.print()}
            >
              <span className="material-symbols-outlined">file_download</span>
              <span className="font-headline font-bold uppercase text-sm">Export Report</span>
            </button>
          </div>

        </div>
      </div>
    </div>
  )
}
