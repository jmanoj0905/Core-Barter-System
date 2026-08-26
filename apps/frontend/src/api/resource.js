const RESOURCE = '/resource'

async function json(res) {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export async function getAccount(userId, trustScore = 0.5) {
  return json(await fetch(`${RESOURCE}/accounts/${userId}?trust_score=${trustScore}`))
}

export async function getEscrow(sessionId) {
  return json(await fetch(`${RESOURCE}/escrow/${sessionId}`))
}

export async function getLedger(userId, limit = 50, offset = 0) {
  return json(await fetch(`${RESOURCE}/ledger/${userId}?limit=${limit}&offset=${offset}`))
}
