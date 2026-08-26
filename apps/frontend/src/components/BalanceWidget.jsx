import { useEffect, useState } from 'react'
import { getAccount } from '../api/resource'

export default function BalanceWidget({ userId, label, trustScore = 0.5 }) {
  const [account, setAccount] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    getAccount(userId, trustScore)
      .then(data => { if (!cancelled) setAccount(data) })
      .catch(err => { if (!cancelled) setError(err.message) })
    return () => { cancelled = true }
  }, [userId, trustScore])

  if (error) return <div className="text-xs text-red-400">wallet unavailable</div>
  if (!account) return <div className="text-xs text-slate-400">…</div>

  return (
    <div className="flex items-center gap-3 rounded-lg bg-slate-800 px-3 py-2">
      <span className="text-xs font-medium text-slate-300">{label}</span>
      <span className="text-sm font-semibold text-emerald-400">
        {account.available} cr
      </span>
      {account.locked > 0 && (
        <span className="text-xs text-amber-400">{account.locked} locked</span>
      )}
    </div>
  )
}
