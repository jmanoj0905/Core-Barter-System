const ROWS = [
  ['stake_returned', 'Stake returned'],
  ['teaching_bonus', 'Teaching bonus'],
  ['engagement_bonus', 'Engagement bonus'],
  ['compensation', 'Compensation'],
  ['penalty', 'Penalty'],
  ['fee_share', 'Platform fee'],
]

export default function SettlementBreakdown({ settlement, names = {} }) {
  if (!settlement || !settlement.breakdown) return null

  const negative = new Set(['penalty', 'fee_share'])

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900 p-4">
      <h3 className="mb-1 text-sm font-semibold text-slate-200">Settlement</h3>
      <p className="mb-3 text-xs text-slate-400">{settlement.mode}</p>

      {Object.entries(settlement.breakdown).map(([userId, row]) => (
        <div key={userId} className="mb-4">
          <div className="mb-1 text-xs font-medium text-slate-300">
            {names[userId] || `User ${userId}`}
          </div>
          <table className="w-full text-sm">
            <tbody>
              {ROWS.filter(([key]) => row[key]).map(([key, label]) => (
                <tr key={key}>
                  <td className="py-0.5 text-slate-400">{label}</td>
                  <td className="py-0.5 text-right text-slate-200">
                    {negative.has(key) ? '−' : '+'}{row[key]} cr
                  </td>
                </tr>
              ))}
              <tr className="border-t border-slate-700">
                <td className="py-1 font-medium text-slate-300">Net</td>
                <td
                  className={`py-1 text-right font-semibold ${
                    row.net >= 0 ? 'text-emerald-400' : 'text-red-400'
                  }`}
                >
                  {row.net >= 0 ? '+' : ''}{row.net} cr
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      ))}
    </div>
  )
}
