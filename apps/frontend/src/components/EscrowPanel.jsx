export default function EscrowPanel({ escrows, names = {} }) {
  if (!escrows || escrows.length === 0) return null

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900 p-4">
      <h3 className="mb-2 text-sm font-semibold text-slate-200">Escrow locked</h3>
      <p className="mb-3 text-xs text-slate-400">
        Stake scales inversely with trust — less trust means more skin in the game.
      </p>
      <ul className="space-y-1">
        {escrows.map(item => (
          <li key={item.user_id} className="flex justify-between text-sm">
            <span className="text-slate-300">
              {names[item.user_id] || `User ${item.user_id}`}
            </span>
            <span className="font-medium text-amber-400">{item.amount} cr</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
