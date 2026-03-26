import { useState } from 'react'

const API = ''

export default function Setup({ onSessionCreated }) {
  const [userId, setUserId] = useState(1)
  const [bobBarterId, setBobBarterId] = useState('')
  const [form, setForm] = useState({
    topic: '',
    scope: '',
    agreed_duration_minutes: 5,
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  function set(field, value) {
    setForm(f => ({ ...f, [field]: value }))
  }

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

  return (
    <>


      {/* Form */}
      <div className="flex justify-center">
        <div className="w-full max-w-2xl">
          <form className="space-y-12" onSubmit={handleSubmit}>
            {/* User identity toggle */}
            <div>
              <label className="block text-[10px] font-bold uppercase tracking-[0.2em] text-primary mb-6">
                You are
              </label>
              <div className="grid grid-cols-2 gap-0 border border-primary">
                <label className="relative cursor-pointer group">
                  <input
                    className="peer sr-only"
                    type="radio"
                    name="user_identity"
                    value="Alice"
                    checked={userId === 1}
                    onChange={() => setUserId(1)}
                  />
                  <div className="py-6 text-center border-r border-primary peer-checked:bg-primary peer-checked:text-surface transition-all duration-200 font-bold uppercase tracking-widest text-sm">
                    Alice
                  </div>
                </label>
                <label className="relative cursor-pointer group">
                  <input
                    className="peer sr-only"
                    type="radio"
                    name="user_identity"
                    value="Bob"
                    checked={userId === 2}
                    onChange={() => setUserId(2)}
                  />
                  <div className="py-6 text-center peer-checked:bg-primary peer-checked:text-surface transition-all duration-200 font-bold uppercase tracking-widest text-sm">
                    Bob
                  </div>
                </label>
              </div>
            </div>

            {/* Form fields */}
            {userId === 2 ? (
              <div className="space-y-8">
                <div className="group">
                  <label className="block text-[10px] font-bold uppercase tracking-[0.2em] text-primary mb-2">
                    Barter ID
                  </label>
                  <input
                    className="w-full bg-surface-container-low border border-outline px-4 py-4 focus:outline-none focus:border-primary transition-colors duration-200 text-primary font-medium"
                    required
                    type="number"
                    min="1"
                    placeholder="ID from Alice's session"
                    value={bobBarterId}
                    onChange={e => setBobBarterId(e.target.value)}
                  />
                </div>
              </div>
            ) : (
              <div className="space-y-8">
                <div className="group">
                  <label className="block text-[10px] font-bold uppercase tracking-[0.2em] text-primary mb-2">
                    Topic
                  </label>
                  <input
                    className="w-full bg-surface-container-low border border-outline px-4 py-4 focus:outline-none focus:border-primary transition-colors duration-200 text-primary font-medium"
                    required
                    placeholder="e.g., Agricultural Exchange Rates"
                    value={form.topic}
                    onChange={e => set('topic', e.target.value)}
                  />
                </div>
                <div className="group">
                  <label className="block text-[10px] font-bold uppercase tracking-[0.2em] text-primary mb-2">
                    Scope
                  </label>
                  <input
                    className="w-full bg-surface-container-low border border-outline px-4 py-4 focus:outline-none focus:border-primary transition-colors duration-200 text-primary font-medium"
                    required
                    placeholder="e.g., Global Market Q3"
                    value={form.scope}
                    onChange={e => set('scope', e.target.value)}
                  />
                </div>
                <div className="group">
                  <label className="block text-[10px] font-bold uppercase tracking-[0.2em] text-primary mb-2">
                    Duration (minutes)
                  </label>
                  <input
                    className="w-full bg-surface-container-low border border-outline px-4 py-4 focus:outline-none focus:border-primary transition-colors duration-200 text-primary font-medium"
                    type="number"
                    min="1"
                    max="60"
                    placeholder="45"
                    value={form.agreed_duration_minutes}
                    onChange={e => set('agreed_duration_minutes', e.target.value)}
                  />
                </div>
              </div>
            )}

            {error && (
              <p className="text-error text-sm font-medium">{error}</p>
            )}

            <div className="pt-4">
              <button
                className="w-full md:w-auto bg-primary text-on-primary px-12 py-5 font-bold uppercase tracking-[0.3em] text-sm hover:opacity-90 active:scale-95 transition-all duration-150 disabled:opacity-40 disabled:cursor-not-allowed"
                type="submit"
                disabled={loading}
              >
                {loading ? 'Please wait...' : userId === 2 ? 'Join Session' : 'Create Session'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </>
  )
}
