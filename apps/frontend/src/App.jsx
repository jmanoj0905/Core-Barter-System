import { useState, useEffect } from 'react'
import './App.css'
import Setup from './screens/Setup'
import LiveSession from './screens/LiveSession'
import PostSession from './screens/PostSession'
import { getAccount } from './api/resource'

const NAV = [
  { icon: 'event_note', label: 'Sessions', key: 'sessions' },
  { icon: 'analytics',  label: 'Reports',  key: 'reports' },
  { icon: 'settings',   label: 'Settings', key: 'settings' },
]

const STT_OPTIONS = [
  { key: 'whisper',  label: 'Whisper',  sub: 'Local · Offline', icon: 'computer' },
  { key: 'deepgram', label: 'Deepgram', sub: 'Nova-2 · Cloud',  icon: 'cloud' },
  { key: 'aws',      label: 'AWS Transcribe', sub: 'Nova-2 · Cloud', icon: 'cloud' },
]

function SttToggle() {
  const [backend, setBackend]   = useState('whisper')
  const [loading, setLoading]   = useState(false)

  useEffect(() => {
    fetch('/stt/config')
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (d?.backend) setBackend(d.backend) })
      .catch(() => {})
  }, [])

  async function switchTo(key) {
    if (key === backend || loading) return
    setLoading(true)
    try {
      const res = await fetch('/stt/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ backend: key }),
      })
      const data = await res.json()
      if (res.ok) setBackend(data.backend)
    } catch (_) {}
    finally { setLoading(false) }
  }

  return (
    <div className="px-4 py-4 border-t-4 border-zinc-900">
      <div className="flex items-center gap-2 mb-3">
        <span className="material-symbols-outlined text-sm text-outline">mic</span>
        <span className="font-headline text-[10px] font-bold uppercase tracking-widest text-outline">
          STT Engine
        </span>
        {loading && (
          <span className="ml-auto text-[9px] font-bold uppercase text-primary animate-pulse">
            switching
          </span>
        )}
      </div>
      <div className="flex flex-col gap-1.5">
        {STT_OPTIONS.map(({ key, label, sub, icon, disabled }) => {
          const active = backend === key
          return (
            <button
              key={key}
              type="button"
              disabled={disabled || loading}
              onClick={() => switchTo(key)}
              className={`flex items-center gap-3 px-3 py-2.5 border-2 border-zinc-900 transition-all text-left w-full
                ${active
                  ? 'bg-primary-container neo-shadow-sm'
                  : disabled
                    ? 'bg-white opacity-35 cursor-not-allowed'
                    : 'bg-white hover:bg-secondary-container cursor-pointer active:translate-x-px active:translate-y-px'
                }`}
            >
              <span className={`material-symbols-outlined text-base ${active ? 'text-primary' : 'text-on-surface-variant'}`}>
                {icon}
              </span>
              <div className="flex-1 min-w-0">
                <p className={`font-headline font-bold text-xs ${active ? 'text-on-background' : 'text-on-surface-variant'}`}>
                  {label}
                </p>
                <p className="text-[9px] uppercase font-bold text-outline leading-none mt-0.5">{sub}</p>
              </div>
              {active && (
                <span className="w-2 h-2 rounded-full bg-primary flex-shrink-0" />
              )}
            </button>
          )
        })}
      </div>
    </div>
  )
}

function WalletPanel() {
  const [wallets, setWallets] = useState({ 1: null, 2: null })

  useEffect(() => {
    function fetchWallets() {
      Promise.all([
        getAccount(1).catch(() => null),
        getAccount(2).catch(() => null),
      ]).then(([w1, w2]) => setWallets({ 1: w1, 2: w2 }))
    }
    fetchWallets()
    const id = setInterval(fetchWallets, 10_000)
    return () => clearInterval(id)
  }, [])

  const users = [
    { id: 1, name: 'Alice', role: 'Teacher' },
    { id: 2, name: 'Bob',   role: 'Learner' },
  ]

  return (
    <div className="mt-auto border-t-4 border-zinc-900 p-4">
      <span className="font-headline text-[10px] font-bold uppercase tracking-widest text-outline block mb-3">
        Wallets
      </span>
      <div className="space-y-3">
        {users.map(u => {
          const w      = wallets[u.id]
          // Account summaries don't echo trust_score (it's an input to
          // regen_rate, not part of the balance record) — show the regen
          // rate that trust produces instead of fabricating a percentage.
          const regen  = w ? `+${w.regen_rate}/day` : '—'
          const avail  = w ? w.available : '—'
          const locked = w ? w.locked : 0
          return (
            <div key={u.id} className="p-3 border-2 border-zinc-900 bg-surface-container-lowest">
              <div className="flex justify-between items-center mb-1">
                <span className="font-headline font-bold text-xs">{u.name}</span>
                <span className="text-[10px] font-bold text-primary">{regen}</span>
              </div>
              <div className="flex justify-between text-[10px] font-bold text-on-surface-variant">
                <span>{avail} cr</span>
                {locked > 0 && (
                  <span className="text-warning-mild">{locked} locked</span>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function AppShell({ children, screen, onNavigate }) {
  return (
    <div className="bg-surface font-body text-on-background min-h-screen selection:bg-primary-container selection:text-on-primary-container">
      <div className="flex">
        {/* ── Sidebar ── */}
        <aside className="hidden md:flex flex-col w-64 fixed left-0 top-0 h-screen bg-white border-r-4 border-zinc-900 z-40">
          {/* Brand */}
          <div className="p-6 pb-5 border-b-4 border-zinc-900">
            <span className="font-headline text-[10px] font-bold uppercase tracking-widest text-outline block mb-0.5">
              Protocol Layer
            </span>
            <span className="text-xl font-black tracking-tighter uppercase font-headline text-zinc-900">
              Core Barter
            </span>
          </div>

          {/* Nav */}
          <nav className="flex flex-col px-2 gap-1 pt-4">
            {NAV.map(n => (
              <button
                key={n.key}
                onClick={() => onNavigate(n.key)}
                className={`flex items-center gap-3 p-3 m-1 font-headline uppercase text-xs font-bold transition-all duration-100 ${
                  n.key === screen
                    ? 'bg-tertiary-container border-4 border-zinc-900 neo-shadow'
                    : 'bg-transparent border-2 border-transparent hover:bg-secondary-container hover:border-zinc-900'
                }`}
              >
                <span className="material-symbols-outlined text-base">{n.icon}</span>
                <span>{n.label}</span>
              </button>
            ))}
          </nav>

          {/* STT Toggle */}
          <SttToggle />

          {/* Wallet panel pinned to bottom */}
          <WalletPanel />
        </aside>

        {/* ── Main content ── */}
        <main className="flex-1 md:ml-64 dot-grid min-h-screen pb-20 md:pb-0">
          {children}
        </main>
      </div>

      {/* ── Mobile bottom nav ── */}
      <nav className="md:hidden fixed bottom-0 left-0 w-full z-50 flex justify-around items-center h-16 bg-white border-t-4 border-zinc-900 px-4">
        {NAV.map(n => (
          <button
            key={n.key}
            onClick={() => onNavigate(n.key)}
            className={`flex flex-col items-center justify-center font-headline text-[10px] font-bold py-1 px-4 transition-all ${
              n.key === screen
                ? 'bg-[#fef9c3] border-2 border-zinc-900 neo-shadow-sm'
                : 'text-zinc-500 hover:bg-zinc-100 active:scale-95'
            }`}
          >
            <span className="material-symbols-outlined text-xl">{n.icon}</span>
            <span>{n.label}</span>
          </button>
        ))}
      </nav>
    </div>
  )
}

function Reports() {
  return (
    <div className="p-6 md:p-12">
      <div className="max-w-5xl mx-auto">
        <div className="mb-12">
          <span className="font-headline text-xs font-bold uppercase tracking-[0.2em] text-primary block mb-2">
            Protocol Layer 02
          </span>
          <h1 className="font-headline text-5xl md:text-7xl font-bold tracking-tight text-on-background">
            Reports
          </h1>
        </div>
        <div className="bg-surface-container-lowest border-4 border-on-background p-12 neo-shadow text-center">
          <span className="material-symbols-outlined text-6xl text-outline mb-4">analytics</span>
          <p className="font-headline text-lg font-bold text-on-surface-variant">
            Session reports coming soon
          </p>
        </div>
      </div>
    </div>
  )
}

function Settings() {
  return (
    <div className="p-6 md:p-12">
      <div className="max-w-5xl mx-auto">
        <div className="mb-12">
          <span className="font-headline text-xs font-bold uppercase tracking-[0.2em] text-primary block mb-2">
            Protocol Layer 03
          </span>
          <h1 className="font-headline text-5xl md:text-7xl font-bold tracking-tight text-on-background">
            Settings
          </h1>
        </div>
        <div className="bg-surface-container-lowest border-4 border-on-background p-12 neo-shadow text-center">
          <span className="material-symbols-outlined text-6xl text-outline mb-4">settings</span>
          <p className="font-headline text-lg font-bold text-on-surface-variant">
            Settings coming soon
          </p>
        </div>
      </div>
    </div>
  )
}

function MainContent({ screen, onSessionCreated, onComplete, onReset, barterId, agreedMinutes, userId, settlement }) {
  switch (screen) {
    case 'setup':
      return <Setup onSessionCreated={onSessionCreated} />
    case 'live':
      return <LiveSession barterId={barterId} agreedMinutes={agreedMinutes} userId={userId} onComplete={onComplete} />
    case 'post':
      return <PostSession barterId={barterId} onReset={onReset} settlement={settlement} />
    case 'reports':
      return <Reports />
    case 'settings':
      return <Settings />
    default:
      return <Setup onSessionCreated={onSessionCreated} />
  }
}

export default function App() {
  const [screen, setScreen]               = useState('setup')
  const [barterId, setBarterId]           = useState(null)
  const [agreedMinutes, setAgreedMinutes] = useState(5)
  const [userId, setUserId]               = useState(1)
  const [settlement, setSettlement]       = useState(null)

  function handleSessionCreated(id, minutes, uid) {
    setBarterId(id)
    setAgreedMinutes(minutes)
    setUserId(uid)
    setScreen('live')
  }

  // `settlementData` is only present when this browser's own confirm call is
  // the one that completed the session (resource_agent's settle response
  // rides back on that HTTP call, not on the WebSocket both_confirmed
  // broadcast or the status poll) -- see LiveSession's handleConfirm.
  function handleComplete(id, settlementData) {
    setBarterId(id)
    setSettlement(settlementData || null)
    setScreen('post')
  }

  function handleReset() {
    setBarterId(null)
    setScreen('setup')
  }

  function handleNavigate(key) {
    if (key === 'sessions') {
      setScreen('setup')
    } else {
      setScreen(key)
    }
  }

  return (
    <AppShell screen={screen} onNavigate={handleNavigate}>
      <MainContent
        screen={screen}
        onSessionCreated={handleSessionCreated}
        onComplete={handleComplete}
        onReset={handleReset}
        barterId={barterId}
        agreedMinutes={agreedMinutes}
        userId={userId}
        settlement={settlement}
      />
    </AppShell>
  )
}
