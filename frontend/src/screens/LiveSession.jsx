import { useState, useEffect, useRef } from 'react'

const API = ''
const WS = location.protocol === 'https:' ? 'wss' : 'ws'
const AUDIO_WS = `${WS}://${location.host}`
const WARNINGS_WS = `${WS}://${location.host}`

function fmt(s) {
  const m = Math.floor(s / 60).toString().padStart(2, '0')
  const sc = Math.floor(s % 60).toString().padStart(2, '0')
  return `${m}:${sc}`
}

const severityStyles = {
  mild: 'border-warning-mild/40 bg-warning-mild/5 text-warning-mild',
  strong: 'border-warning-strong/40 bg-warning-strong/5 text-warning-strong',
  severe: 'border-warning-severe/40 bg-warning-severe/5 text-warning-severe',
}

const classStyles = {
  correct: 'text-success',
  weakly_correct: 'text-warning-mild',
  incorrect: 'text-error',
  out_of_scope: 'text-warning-severe',
}

export default function LiveSession({ barterId, agreedMinutes, userId, onComplete }) {
  const [started, setStarted] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [recording, setRecording] = useState(false)
  const [confirmed, setConfirmed] = useState(false)
  const [warnings, setWarnings] = useState([])
  const [windows, setWindows] = useState([])
  const [liveTranscripts, setLiveTranscripts] = useState([])
  const [terminated, setTerminated] = useState(false)
  const [error, setError] = useState('')

  const mrRef = useRef(null)
  const audioWsRef = useRef(null)
  const timerRef = useRef(null)
  const pollRef = useRef(null)
  const localVideoElRef = useRef(null)
  const remoteVideoElRef = useRef(null)
  const pcRef = useRef(null)
  const signalWsRef = useRef(null)
  const frameIntervalRef = useRef(null)

  const agreedSeconds = agreedMinutes * 60
  const name = userId === 1 ? 'Alice' : 'Bob'
  const remoteName = userId === 1 ? 'Bob' : 'Alice'

  // Warnings + transcripts WebSocket
  useEffect(() => {
    let opened = false
    const ws = new WebSocket(`${WARNINGS_WS}/ws/warnings/${barterId}`)
    ws.onopen = () => { opened = true; setError('') }
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data)
      if (data.type === 'window') {
        setWindows(prev => [data, ...prev])
      } else if (data.type === 'transcript') {
        setLiveTranscripts(prev => [data, ...prev].slice(0, 30))
      } else {
        setWarnings(prev => [data, ...prev])
        if (data.severity === 'severe') {
          setTerminated(true)
          halt()
        }
      }
    }
    ws.onerror = () => { if (!opened) setError('Warning connection failed — is the backend running?') }
    return () => ws.close()
  }, [barterId])

  // Timer
  useEffect(() => {
    if (!started || terminated) return
    timerRef.current = setInterval(() => setElapsed(e => e + 1), 1000)
    return () => clearInterval(timerRef.current)
  }, [started, terminated])

  // Poll for both_confirmed
  useEffect(() => {
    if (!started) return
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API}/session/${barterId}/status`)
        const data = await res.json()
        if (data.both_confirmed) {
          clearInterval(pollRef.current)
          onComplete(barterId)
        }
      } catch (_) {}
    }, 10_000)
    return () => clearInterval(pollRef.current)
  }, [started, barterId, onComplete])

  function halt() {
    clearInterval(timerRef.current)
    clearInterval(pollRef.current)
    clearInterval(frameIntervalRef.current)
    mrRef.current?.stop()
    audioWsRef.current?.close()
    pcRef.current?.close()
    signalWsRef.current?.close()
    if (localVideoElRef.current?.srcObject) {
      localVideoElRef.current.srcObject.getTracks().forEach(t => t.stop())
    }
    setRecording(false)
  }

  function setupWebRTC(stream) {
    const pc = new RTCPeerConnection({
      iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' },
      ]
    })
    pcRef.current = pc
    stream.getTracks().forEach(t => pc.addTrack(t, stream))

    pc.ontrack = (e) => {
      if (remoteVideoElRef.current && e.streams[0]) {
        remoteVideoElRef.current.srcObject = e.streams[0]
        remoteVideoElRef.current.play().catch(() => {})
      }
    }

    pc.onconnectionstatechange = () => {
      if (pc.connectionState === 'failed')
        setError('Peer connection failed — make sure both devices are on the same Wi-Fi network.')
    }

    const iceQueue = []
    let remoteReady = false

    async function flushIce() {
      remoteReady = true
      for (const c of iceQueue.splice(0)) {
        try { await pc.addIceCandidate(new RTCIceCandidate(c)) } catch (_) {}
      }
    }

    const signalWs = new WebSocket(`${WARNINGS_WS}/ws/signal/${barterId}/${userId}`)
    signalWsRef.current = signalWs

    const signal = (msg) => {
      if (signalWs.readyState === WebSocket.OPEN) signalWs.send(JSON.stringify(msg))
    }

    pc.onicecandidate = (e) => {
      if (e.candidate) signal({ type: 'ice', candidate: e.candidate })
    }

    signalWs.onmessage = async (e) => {
      const msg = JSON.parse(e.data)
      if (msg.type === 'peer_joined' && userId === 1) {
        const offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        signal({ type: 'offer', sdp: pc.localDescription })
      } else if (msg.type === 'offer') {
        await pc.setRemoteDescription(new RTCSessionDescription(msg.sdp))
        await flushIce()
        const answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        signal({ type: 'answer', sdp: pc.localDescription })
      } else if (msg.type === 'answer') {
        await pc.setRemoteDescription(new RTCSessionDescription(msg.sdp))
        await flushIce()
      } else if (msg.type === 'ice') {
        if (remoteReady) {
          try { await pc.addIceCandidate(new RTCIceCandidate(msg.candidate)) } catch (_) {}
        } else {
          iceQueue.push(msg.candidate)
        }
      }
    }
  }

  async function handleStart() {
    setError('')
    try {
      const res = await fetch(`${API}/session/${barterId}/start`, { method: 'POST' })
      if (!res.ok) throw new Error(await res.text())
      setStarted(true)

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true })

      if (localVideoElRef.current) {
        localVideoElRef.current.srcObject = stream
        localVideoElRef.current.play().catch(() => {})
      }

      setupWebRTC(stream)

      const ws = new WebSocket(`${AUDIO_WS}/audio/${barterId}/${userId}`)
      audioWsRef.current = ws

      await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject })

      const audioStream = new MediaStream(stream.getAudioTracks())
      const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4']
        .find(t => MediaRecorder.isTypeSupported(t)) || ''
      const mr = mimeType ? new MediaRecorder(audioStream, { mimeType }) : new MediaRecorder(audioStream)
      mrRef.current = mr
      mr.ondataavailable = (e) => {
        if (e.data.size > 0 && ws.readyState === WebSocket.OPEN) ws.send(e.data)
      }
      mr.start(5000)
      setRecording(true)

      frameIntervalRef.current = setInterval(() => {
        if (!localVideoElRef.current || localVideoElRef.current.videoWidth === 0) return
        const canvas = document.createElement('canvas')
        canvas.width = localVideoElRef.current.videoWidth
        canvas.height = localVideoElRef.current.videoHeight
        canvas.getContext('2d').drawImage(localVideoElRef.current, 0, 0)
        const base64 = canvas.toDataURL('image/jpeg', 0.5).split(',')[1]
        fetch(`${API}/safety/check-frame`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ barter_id: barterId, user_id: userId, image_base64: base64 }),
        }).catch(() => {})
      }, 10_000)
    } catch (err) {
      setError(err.message)
    }
  }

  async function handleConfirm() {
    setError('')
    try {
      const res = await fetch(`${API}/session/${barterId}/confirm`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId }),
      })
      if (!res.ok) throw new Error(await res.text())
      setConfirmed(true)
      clearInterval(frameIntervalRef.current)
      mrRef.current?.stop()
      audioWsRef.current?.close()
      pcRef.current?.close()
      signalWsRef.current?.close()
      if (localVideoElRef.current?.srcObject) {
        localVideoElRef.current.srcObject.getTracks().forEach(t => t.stop())
      }
      setRecording(false)
    } catch (err) {
      setError(err.message)
    }
  }

  async function handleTerminate() {
    await fetch(`${API}/session/${barterId}/terminate`, { method: 'POST' }).catch(() => {})
    setTerminated(true)
    halt()
  }

  return (
    <>
      {/* Session header */}
      <div className="flex justify-between items-start mb-8 border-b border-outline-variant pb-6">
        <div>
          <h1 className="text-3xl md:text-4xl font-extrabold tracking-tighter text-primary uppercase">
            Session #{barterId}
          </h1>
          <p className="text-sm text-on-surface-variant font-medium tracking-tight mt-1">
            {name} &middot; {agreedMinutes} min agreed
          </p>
        </div>
        <div className={`text-4xl font-bold tabular-nums tracking-tight ${elapsed > agreedSeconds ? 'text-warning-strong' : 'text-primary'}`}>
          {fmt(elapsed)}
        </div>
      </div>

      {/* Video feeds */}
      <div className="grid grid-cols-2 gap-4 mb-8">
        <div className="relative bg-primary aspect-[4/3] overflow-hidden">
          <video ref={localVideoElRef} muted playsInline autoPlay className="w-full h-full object-cover block" />
          <span className="absolute bottom-2 left-3 text-xs text-white bg-black/60 px-2 py-0.5">
            {name} (you)
          </span>
        </div>
        <div className="relative bg-primary aspect-[4/3] overflow-hidden">
          <video ref={remoteVideoElRef} playsInline autoPlay className="w-full h-full object-cover block" />
          <span className="absolute bottom-2 left-3 text-xs text-white bg-black/60 px-2 py-0.5">
            {remoteName}
          </span>
        </div>
      </div>

      {/* Pre-start card */}
      {!started && !terminated && (
        <div className="border border-outline-variant p-8 mb-8">
          <p className="text-on-surface-variant text-sm mb-6">
            Both participants ready? Click Start to open the microphone.
          </p>
          <button
            className="bg-primary text-on-primary px-10 py-4 font-bold uppercase tracking-[0.3em] text-sm hover:opacity-90 active:scale-95 transition-all duration-150"
            onClick={handleStart}
          >
            Start Session
          </button>
        </div>
      )}

      {started && (
        <>
          {/* Mic status */}
          <div className="flex items-center gap-3 mb-6 text-on-surface-variant text-sm font-medium">
            <div className="relative">
              <div className={`w-2.5 h-2.5 rounded-full ${recording ? 'bg-error animate-blink' : 'bg-outline'}`} />
              {recording && (
                <div className="absolute inset-0 w-2.5 h-2.5 rounded-full bg-error animate-pulse-ring" />
              )}
            </div>
            {name} — {recording ? 'live' : 'mic off'}
          </div>

          {/* Live transcript */}
          {liveTranscripts.length > 0 && (
            <div className="border border-outline-variant p-6 mb-6">
              <h2 className="text-[10px] font-bold uppercase tracking-[0.2em] text-primary mb-4">
                Live Transcript
              </h2>
              <div className="flex flex-col gap-3 max-h-56 overflow-y-auto">
                {liveTranscripts.map((t, i) => (
                  <div key={i} className="flex gap-3 items-start">
                    <span className={`font-bold text-xs min-w-[44px] pt-0.5 ${t.user_id === 1 ? 'text-blue-600' : 'text-emerald-600'}`}>
                      {t.speaker}
                    </span>
                    <span className="text-sm leading-relaxed text-on-surface">{t.text}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Warnings */}
          <div className="border border-outline-variant p-6 mb-6">
            <h2 className="text-[10px] font-bold uppercase tracking-[0.2em] text-primary mb-4">
              Warnings {warnings.length > 0 && `(${warnings.length})`}
            </h2>
            {warnings.length === 0 ? (
              <p className="text-on-surface-variant text-sm">None yet.</p>
            ) : (
              <div className="flex flex-col gap-2">
                {warnings.map((w, i) => (
                  <div
                    key={i}
                    className={`flex justify-between items-baseline gap-3 px-4 py-3 border text-sm font-medium ${severityStyles[w.severity] || 'border-outline'}`}
                  >
                    <span>{w.reason}</span>
                    <span className="text-xs opacity-50 whitespace-nowrap">
                      {w.timestamp ? new Date(w.timestamp).toLocaleTimeString() : ''}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Windows */}
          {windows.length > 0 && (
            <div className="border border-outline-variant p-6 mb-6">
              <h2 className="text-[10px] font-bold uppercase tracking-[0.2em] text-primary mb-4">
                Windows ({windows.length})
              </h2>
              <div className="flex flex-col gap-2">
                {windows.map((w, i) => (
                  <div
                    key={i}
                    className={`flex justify-between items-baseline gap-3 px-4 py-3 border border-outline-variant text-sm ${w.classification === 'correct' ? 'opacity-60' : ''}`}
                  >
                    <span>
                      <strong className="font-bold">#{w.window_id}</strong>{' '}
                      <span className={classStyles[w.classification] || ''}>{w.classification.replace('_', ' ')}</span>
                      {' '}&middot; sim {w.similarity}
                    </span>
                    <span className="text-xs text-on-surface-variant max-w-[200px] overflow-hidden text-ellipsis whitespace-nowrap">
                      {w.text_preview}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Actions */}
          {terminated ? (
            <div className="border border-warning-severe/40 bg-warning-severe/5 text-warning-severe px-6 py-4 font-bold uppercase tracking-widest text-sm text-center">
              Session terminated.
            </div>
          ) : (
            <div className="flex gap-4 flex-wrap mt-2">
              <button
                className="bg-primary text-on-primary px-10 py-4 font-bold uppercase tracking-[0.3em] text-sm hover:opacity-90 active:scale-95 transition-all duration-150 disabled:opacity-40 disabled:cursor-not-allowed"
                onClick={handleConfirm}
                disabled={confirmed}
              >
                {confirmed ? 'Marked complete' : 'Mark Complete'}
              </button>
              <button
                className="border border-error text-error px-10 py-4 font-bold uppercase tracking-[0.3em] text-sm hover:bg-error hover:text-on-error transition-all duration-150 bg-transparent"
                onClick={handleTerminate}
              >
                Terminate
              </button>
            </div>
          )}

          {confirmed && !terminated && (
            <p className="text-on-surface-variant text-sm mt-4 font-medium">
              Waiting for other participant...
            </p>
          )}
        </>
      )}

      {error && (
        <p className="text-error text-sm font-medium mt-4">{error}</p>
      )}
    </>
  )
}
