import { useState, useEffect } from 'react'
import { getAccount } from '../api/resource'

const API = ''

function calcEscrow(trustScore) {
  return Math.max(5, Math.floor(40 * (1 - trustScore)))
}

export default function Setup({ onSessionCreated }) {
  const [userId, setUserId]         = useState(1)
  const [bobBarterId, setBobBarterId] = useState('')
  const [form, setForm]             = useState({ topic: '', scope: '', agreed_duration_minutes: 5 })
  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState('')
  const [wallets, setWallets]       = useState({ 1: null, 2: null })

  function set(field, value) {
    setForm(f => ({ ...f, [field]: value }))
  }

  useEffect(() => {
    // The resource API's account summary doesn't echo trust_score back (it's
    // an input to it, not part of the balance record), so it's fetched
    // separately from /users and merged in here for the escrow preview.
    Promise.all([
      fetch(`${API}/users`).then(r => r.ok ? r.json() : []).catch(() => []),
    ]).then(([users]) => {
      const trustById = Object.fromEntries(users.map(u => [u.id, u.trust_score]))
      Promise.all([
        getAccount(1, trustById[1] ?? 0.5).catch(() => null),
        getAccount(2, trustById[2] ?? 0.5).catch(() => null),
      ]).then(([w1, w2]) => setWallets({
        1: w1 && { ...w1, trust_score: trustById[1] ?? 1.0 },
        2: w2 && { ...w2, trust_score: trustById[2] ?? 1.0 },
      }))
    }).catch(() => {})
  }, [])

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      if (userId === 2) {
        const id = Number(bobBarterId)
        if (!id) throw new Error('Enter a valid Barter ID.')
        onSessionCreated(id, 5, 2)
      } else {
        const res = await fetch(`${API}/session/create`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            topic: form.topic,
            scope: form.scope,
            agreed_duration_minutes: Number(form.agreed_duration_minutes),
          }),
        })
        if (!res.ok) throw new Error(await res.text())
        const data = await res.json()
        onSessionCreated(data.barter_id, Number(form.agreed_duration_minutes), 1)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const activeWallet = wallets[userId]
  const trustScore   = activeWallet?.trust_score ?? 1.0
  const escrowAmt    = calcEscrow(trustScore)
  const available    = activeWallet?.available ?? 999999
  const canAfford    = available >= escrowAmt

  return (
    <div className="p-6 md:p-12">
      <div className="max-w-5xl mx-auto">

        {/* ── Page header ── */}
        <div className="mb-12">
          <span className="font-headline text-xs font-bold uppercase tracking-[0.2em] text-primary block mb-2">
            Protocol Layer 01
          </span>
          <h1 className="font-headline text-5xl md:text-7xl font-bold tracking-tight text-on-background">
            Session Setup
          </h1>
          <p className="text-lg text-on-surface-variant mt-4 max-w-2xl font-medium">
            Define the parameters of exchange. Establish core boundaries before the barter begins.
          </p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

            {/* ── Left: form ── */}
            <section className="lg:col-span-7 space-y-8">

              {/* Identity toggle */}
              <div className="bg-surface-container-lowest border-4 border-on-background p-6 neo-shadow">
                <h3 className="font-headline text-sm font-bold uppercase mb-6 flex items-center gap-2">
                  <span className="material-symbols-outlined text-sm">fingerprint</span>
                  Active Identity
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  {[
                    { id: 1, name: 'Alice', role: 'Teacher' },
                    { id: 2, name: 'Bob',   role: 'Learner' },
                  ].map(u => {
                    const w      = wallets[u.id]
                    const ts     = w?.trust_score ?? 1.0
                    const active = userId === u.id
                    return (
                      <button
                        key={u.id}
                        type="button"
                        onClick={() => setUserId(u.id)}
                        className={`flex flex-col items-center justify-center p-6 border-4 border-on-background transition-all active:translate-x-0.5 active:translate-y-0.5 ${
                          active ? 'bg-primary-container neo-shadow' : 'bg-white hover:bg-secondary-container'
                        }`}
                      >
                        <div className={`w-16 h-16 border-2 border-on-background rounded-full mb-3 flex items-center justify-center ${
                          active ? 'bg-primary' : 'bg-surface-container-high'
                        }`}>
                          <span className={`font-headline font-black text-2xl ${active ? 'text-white' : 'text-on-surface-variant'}`}>
                            {u.name[0]}
                          </span>
                        </div>
                        <span className="font-headline font-bold text-lg">{u.name}</span>
                        <span className="text-[10px] uppercase font-bold tracking-widest mt-1 text-outline">
                          {u.role}
                        </span>
{w && (
                          <span className="text-[10px] font-bold mt-2 text-primary">
                            Trust {(ts * 100).toFixed(0)}%
                          </span>
                        )}
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* Form inputs */}
              <div className="bg-surface-container-lowest border-4 border-on-background p-8 neo-shadow space-y-8">
                {userId === 2 ? (
                  <div className="space-y-2">
                    <label className="font-headline text-xs font-bold uppercase tracking-widest flex items-center gap-2">
                      <span className="material-symbols-outlined text-base">tag</span>
                      Barter ID
                    </label>
                    <input
                      className="w-full border-4 border-on-background p-4 font-headline font-bold text-xl focus:bg-primary-fixed focus:outline-none transition-all"
                      required
                      type="number"
                      min="1"
                      placeholder="ID from Alice's session"
                      value={bobBarterId}
                      onChange={e => setBobBarterId(e.target.value)}
                    />
                  </div>
                ) : (
                  <>
                    <div className="space-y-2">
                      <label className="font-headline text-xs font-bold uppercase tracking-widest flex items-center gap-2">
                        <span className="material-symbols-outlined text-base">label</span>
                        Exchange Topic
                      </label>
                      <input
                        className="w-full border-4 border-on-background p-4 font-headline font-bold text-xl focus:bg-primary-fixed focus:outline-none transition-all"
                        required
                        placeholder="e.g. Machine Learning Fundamentals"
                        value={form.topic}
                        onChange={e => set('topic', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="font-headline text-xs font-bold uppercase tracking-widest flex items-center gap-2">
                        <span className="material-symbols-outlined text-base">format_list_bulleted</span>
                        Scope of Exchange
                      </label>
                      <textarea
                        className="w-full border-4 border-on-background p-4 font-body text-base focus:bg-primary-fixed focus:outline-none transition-all resize-none"
                        rows="3"
                        required
                        placeholder="Detail the deliverables and expectations..."
                        value={form.scope}
                        onChange={e => set('scope', e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="font-headline text-xs font-bold uppercase tracking-widest flex items-center gap-2">
                        <span className="material-symbols-outlined text-base">timer</span>
                        Duration (minutes)
                      </label>
                      <input
                        className="w-full border-4 border-on-background p-4 font-headline font-bold text-xl focus:bg-primary-fixed focus:outline-none transition-all"
                        type="number"
                        min="1"
                        max="60"
                        value={form.agreed_duration_minutes}
                        onChange={e => set('agreed_duration_minutes', e.target.value)}
                      />
                    </div>
                  </>
                )}
              </div>

              {error && (
                <div className="border-4 border-error bg-error-container p-4">
                  <p className="text-on-error-container font-bold text-sm">{error}</p>
                </div>
              )}
            </section>

            {/* ── Right: contract summary ── */}
            <aside className="lg:col-span-5 space-y-6">
              <div className="bg-[#fef9c3] border-4 border-on-background p-8 neo-shadow-lg sticky top-24">
                <h3 className="font-headline text-sm font-bold uppercase mb-8 border-b-2 border-on-background pb-4 tracking-tighter">
                  Contract Summary
                </h3>

                <div className="space-y-6 mb-8">
                  {/* Topic preview */}
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="text-[10px] uppercase font-bold text-on-primary-container">Exchange Topic</p>
                      <p className="font-headline font-bold text-xl mt-1">
                        {form.topic || <span className="text-outline font-normal italic">Not set</span>}
                      </p>
                    </div>
                    <span className="material-symbols-outlined text-3xl text-primary">token</span>
                  </div>

                  {/* Trust + escrow panel */}
                  <div className="p-4 bg-white border-2 border-on-background">
                    <p className="text-[10px] uppercase font-bold text-outline mb-3">
                      {userId === 1 ? 'Alice' : 'Bob'} — Escrow Preview
                    </p>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <p className="text-[10px] font-bold uppercase text-on-surface-variant">Trust Score</p>
                        <p className="font-headline font-bold text-2xl">{(trustScore * 100).toFixed(0)}%</p>
                      </div>
                      <div>
                        <p className="text-[10px] font-bold uppercase text-on-surface-variant">Will Lock</p>
                        <p className={`font-headline font-bold text-2xl ${canAfford ? '' : 'text-error'}`}>
                          {escrowAmt}
                          <span className="text-sm font-normal ml-1">cr</span>
                        </p>
                      </div>
                      <div>
                        <p className="text-[10px] font-bold uppercase text-on-surface-variant">Available</p>
                        <p className="font-headline font-bold text-xl">{available} cr</p>
                      </div>
                      <div>
                        <p className="text-[10px] font-bold uppercase text-on-surface-variant">After Lock</p>
                        <p className="font-headline font-bold text-xl">{available - escrowAmt} cr</p>
                      </div>
                    </div>
                    {!canAfford && (
                      <p className="text-error text-xs font-bold mt-3">
                        Insufficient balance. Need {escrowAmt} cr, have {available} cr.
                      </p>
                    )}
                  </div>

                  {/* Duration + status row */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-tertiary-container border-2 border-on-background">
                      <p className="text-[10px] uppercase font-bold text-on-tertiary-container">Duration</p>
                      <p className="font-headline font-bold text-lg">
                        {userId === 2 ? '—' : `${form.agreed_duration_minutes} min`}
                      </p>
                    </div>
                    <div className="p-4 bg-secondary-container border-2 border-on-background">
                      <p className="text-[10px] uppercase font-bold text-on-secondary-container">Status</p>
                      <p className="font-headline font-bold text-lg">Drafting</p>
                    </div>
                  </div>
                </div>

                <button
                  className="w-full bg-on-background text-white p-5 font-headline font-bold text-lg uppercase tracking-widest hover:bg-primary transition-all active:translate-x-0.5 active:translate-y-0.5 neo-shadow disabled:opacity-40 disabled:cursor-not-allowed"
                  type="submit"
                  disabled={loading}
                >
                  {loading ? 'Please wait...' : userId === 2 ? 'Join Session' : 'Finalize Session'}
                </button>
                <p className="text-center text-[10px] uppercase font-bold mt-4 text-outline tracking-widest">
                  Escrow locked on session start
                </p>
              </div>
            </aside>
          </div>
        </form>
      </div>
    </div>
  )
}
