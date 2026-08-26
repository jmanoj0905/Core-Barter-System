import { useEffect, useState } from 'react'
import { getLedger } from '../api/resource'

const LABELS = {
  grant: 'Initial grant',
  regen: 'Regeneration',
  floor_topup: 'Floor top-up',
  escrow_reserve: 'Escrow locked',
  escrow_settle: 'Settlement',
  escrow_void: 'Escrow returned',
}

export default function Ledger({ userId, onBack }) {
  const [entries, setEntries] = useState([])
  const [error, setError] = useState(null)

  useEffect(() => {
    getLedger(userId)
      .then(data => setEntries(data.entries))
      .catch(err => setError(err.message))
  }, [userId])

  if (error) return <div className="p-6 text-red-400">Ledger unavailable: {error}</div>

  return (
    <div className="mx-auto max-w-2xl p-6">
      <button onClick={onBack} className="mb-4 text-sm text-slate-400 hover:text-slate-200">
        ← Back
      </button>
      <h2 className="mb-4 text-lg font-semibold text-slate-100">
        Credit history · User {userId}
      </h2>

      {entries.length === 0 && <p className="text-sm text-slate-400">No transactions yet.</p>}

      <ul className="divide-y divide-slate-800">
        {entries.map(entry => (
          <li key={`${entry.id}-${entry.amount}`} className="flex items-center justify-between py-2">
            <div>
              <div className="text-sm text-slate-200">
                {LABELS[entry.entry_type] || entry.entry_type}
              </div>
              <div className="text-xs text-slate-500">
                {entry.session_id ? `Session ${entry.session_id} · ` : ''}
                {new Date(entry.created_at).toLocaleString()}
              </div>
            </div>
            <span
              className={`text-sm font-medium ${
                entry.amount >= 0 ? 'text-emerald-400' : 'text-red-400'
              }`}
            >
              {entry.amount >= 0 ? '+' : ''}{entry.amount} cr
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}
