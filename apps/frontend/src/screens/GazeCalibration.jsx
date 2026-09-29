import { useEffect, useRef, useState } from 'react'

const FRAME_COUNT = 5
const FRAME_INTERVAL_MS = 1000
const DEADLINE_MS = 6000

function captureFrame(videoEl) {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas')
    canvas.width = videoEl.videoWidth
    canvas.height = videoEl.videoHeight
    canvas.getContext('2d').drawImage(videoEl, 0, 0)
    canvas.toBlob((blob) => resolve(blob), 'image/jpeg', 0.7)
  })
}

export default function GazeCalibration({ videoEl, barterId, userId, onDone }) {
  const [status, setStatus] = useState('capturing') // capturing | submitting
  const onDoneRef = useRef(onDone)
  useEffect(() => { onDoneRef.current = onDone }, [onDone])

  useEffect(() => {
    let cancelled = false
    let submitted = false
    let ticks = 0
    const pending = [] // promises of blobs

    async function submit() {
      if (submitted || cancelled) return
      submitted = true
      clearInterval(interval)
      clearTimeout(deadline)
      setStatus('submitting')
      const blobs = (await Promise.all(pending)).filter(Boolean)
      if (cancelled) return
      let result
      try {
        const form = new FormData()
        blobs.forEach((blob, i) => form.append('frames', blob, `frame${i}.jpg`))
        const res = await fetch(`/video/${barterId}/${userId}/calibrate`, { method: 'POST', body: form })
        result = await res.json()
      } catch {
        result = { calibrated: false, reason: 'request_failed' }
      }
      if (!cancelled) onDoneRef.current(result)
    }

    const interval = setInterval(() => {
      if (submitted || cancelled || !videoEl || videoEl.videoWidth === 0) return
      pending.push(captureFrame(videoEl))
      ticks += 1
      if (ticks >= FRAME_COUNT) submit()
    }, FRAME_INTERVAL_MS)
    const deadline = setTimeout(submit, DEADLINE_MS)

    return () => { cancelled = true; clearInterval(interval); clearTimeout(deadline) }
  }, [videoEl, barterId, userId])

  return (
    <div className="w-full mt-2 p-3 border-4 border-on-background bg-primary-container font-headline font-bold text-xs uppercase text-center">
      {status === 'capturing' ? 'Look at your screen naturally for a few seconds...' : 'Calibrating...'}
    </div>
  )
}
