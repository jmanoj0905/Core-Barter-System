import { useState, useEffect, useRef } from 'react'

const API        = ''
const WS         = location.protocol === 'https:' ? 'wss' : 'ws'
const AUDIO_WS   = `${WS}://${location.host}`
const WARNINGS_WS = `${WS}://${location.host}`

function fmt(s) {
  const h  = Math.floor(s / 3600)
  const m  = Math.floor((s % 3600) / 60).toString().padStart(2, '0')
  const sc = Math.floor(s % 60).toString().padStart(2, '0')
  return h > 0 ? `${h}:${m}:${sc}` : `${m}:${sc}`
}

const severityBg = {
  mild:   'bg-[#fef9c3] border-on-background',
  strong: 'bg-secondary-container border-on-background',
  severe: 'bg-error-container border-on-background',
}

const severityIcon = {
  mild:   'info',
  strong: 'warning',
  severe: 'priority_high',
}

const classColor = {
  correct:        'text-tertiary font-bold',
  weakly_correct: 'text-warning-mild font-bold',
  incorrect:      'text-error font-bold',
  out_of_scope:   'text-warning-severe font-bold',
}

export default function LiveSession({ barterId, agreedMinutes, userId, onComplete }) {
  const [started, setStarted]             = useState(false)
  const [elapsed, setElapsed]             = useState(0)
  const [recording, setRecording]         = useState(false)
  const [confirmed, setConfirmed]         = useState(false)
  const [warnings, setWarnings]           = useState([])
  const [windows, setWindows]             = useState([])
  const [liveTranscripts, setLiveTranscripts] = useState([])
  const [terminated, setTerminated]       = useState(false)
  const [error, setError]                 = useState('')
  const [escrowData, setEscrowData]       = useState(null)
  const [isMicMuted, setIsMicMuted]       = useState(false)
  const [isVideoOff, setIsVideoOff]       = useState(false)
  const [isRemoteMuted, setIsRemoteMuted] = useState(false)
  const [isRemoteHidden, setIsRemoteHidden] = useState(false)

  const mrRef           = useRef(null)
  const audioWsRef      = useRef(null)
  const timerRef        = useRef(null)
  const pollRef         = useRef(null)
  const localVideoElRef = useRef(null)
  const remoteVideoElRef = useRef(null)
  const pcRef           = useRef(null)
  const signalWsRef     = useRef(null)
  const frameIntervalRef = useRef(null)

  const agreedSeconds = agreedMinutes * 60
  const name          = userId === 1 ? 'Alice' : 'Bob'
  const remoteName    = userId === 1 ? 'Bob' : 'Alice'
  const overTime      = elapsed > agreedSeconds

  // Warnings + transcripts + confirmation WebSocket
  useEffect(() => {
    let opened = false
    const ws = new WebSocket(`${WARNINGS_WS}/ws/warnings/${barterId}`)
    ws.onopen  = () => { opened = true; setError('') }
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data)
      if (data.type === 'window') {
        setWindows(prev => [data, ...prev])
      } else if (data.type === 'transcript') {
        setLiveTranscripts(prev => [data, ...prev].slice(0, 30))
      } else if (data.type === 'both_confirmed') {
        // Real-time notification when both users confirm - end session immediately
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
        onComplete(barterId)
      } else if (data.type === 'peer_confirmed') {
        // Show notification that the other user confirmed
        setWarnings(prev => [{ ...data, severity: 'mild' }, ...prev])
      } else {
        setWarnings(prev => [data, ...prev])
        if (data.severity === 'severe') { setTerminated(true); halt() }
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
        const res  = await fetch(`${API}/session/${barterId}/status`)
        const data = await res.json()
        if (data.both_confirmed) { clearInterval(pollRef.current); onComplete(barterId) }
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
      ],
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
        setError('Peer connection failed — ensure both devices share the same network.')
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
    const signal = (msg) => { if (signalWs.readyState === WebSocket.OPEN) signalWs.send(JSON.stringify(msg)) }
    pc.onicecandidate = (e) => { if (e.candidate) signal({ type: 'ice', candidate: e.candidate }) }
    signalWs.onmessage = async (e) => {
      const msg = JSON.parse(e.data)
      console.log('[Signal] Received:', msg.type, 'from:', msg.from)
      if (msg.type === 'peer_joined' && userId === 1) {
        console.log('[WebRTC] Creating offer as Alice...')
        const offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        signal({ type: 'offer', sdp: pc.localDescription })
        console.log('[WebRTC] Offer sent')
      } else if (msg.type === 'offer') {
        console.log('[WebRTC] Received offer, creating answer...')
        await pc.setRemoteDescription(new RTCSessionDescription(msg.sdp))
        await flushIce()
        const answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        signal({ type: 'answer', sdp: pc.localDescription })
        console.log('[WebRTC] Answer sent')
      } else if (msg.type === 'answer') {
        console.log('[WebRTC] Received answer, connecting...')
        await pc.setRemoteDescription(new RTCSessionDescription(msg.sdp))
        await flushIce()
      } else if (msg.type === 'ice') {
        console.log('[WebRTC] ICE candidate received')
        if (remoteReady) { try { await pc.addIceCandidate(new RTCIceCandidate(msg.candidate)) } catch (_) {} }
        else iceQueue.push(msg.candidate)
      }
    }
    signalWs.onopen = () => { console.log('[Signal] WebSocket connected for', userId === 1 ? 'Alice' : 'Bob') }
    signalWs.onerror = (e) => { console.error('[Signal] WebSocket error:', e) }
  }

  function toggleMic() {
    if (!localVideoElRef.current?.srcObject) return
    const track = localVideoElRef.current.srcObject.getAudioTracks()[0]
    if (!track) return
    track.enabled = !track.enabled
    setIsMicMuted(!track.enabled)
  }

  function toggleLocalVideo() {
    if (!localVideoElRef.current?.srcObject) return
    const track = localVideoElRef.current.srcObject.getVideoTracks()[0]
    if (!track) return
    track.enabled = !track.enabled
    setIsVideoOff(!track.enabled)
  }

  function toggleRemoteMute() {
    if (!remoteVideoElRef.current) return
    remoteVideoElRef.current.muted = !remoteVideoElRef.current.muted
    setIsRemoteMuted(remoteVideoElRef.current.muted)
  }

  function toggleRemoteVideo() {
    setIsRemoteHidden(h => !h)
  }

  async function handleStart() {
    setError('')
    try {
      const res = await fetch(`${API}/session/${barterId}/start`, { method: 'POST' })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setEscrowData(data)
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
      mr.ondataavailable = (e) => { if (e.data.size > 0 && ws.readyState === WebSocket.OPEN) ws.send(e.data) }
      mr.start(5000)
      setRecording(true)

      frameIntervalRef.current = setInterval(() => {
        if (!localVideoElRef.current || localVideoElRef.current.videoWidth === 0) return
        const canvas = document.createElement('canvas')
        canvas.width  = localVideoElRef.current.videoWidth
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

  const teacherEscrow = escrowData?.teacher_escrow?.amount ?? 0
  const learnerEscrow = escrowData?.learner_escrow?.amount ?? 0

  return (
    <div className="p-6 md:p-10">
      {/* ── Header ── */}
      <div className="mb-10 flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div>
          <span className="font-headline font-bold text-sm tracking-widest uppercase text-primary block mb-2">
            Live Negotiation
          </span>
          <h1 className="text-5xl font-headline font-black tracking-tighter text-on-background">
            Session #{barterId}
          </h1>
          <p className="text-on-surface-variant mt-1 font-medium">
            {name} &middot; {agreedMinutes} min agreed
          </p>
        </div>
        {/* Timer */}
        <div className={`border-4 border-on-background p-5 neo-shadow-lg flex flex-col items-center min-w-[160px] ${overTime ? 'bg-error-container' : 'bg-primary-container'}`}>
          <span className="font-headline font-bold text-xs uppercase mb-1 text-on-surface-variant">
            Session Time
          </span>
          <span className={`text-5xl font-headline font-black tracking-tight tabular-nums ${overTime ? 'text-error' : ''}`}>
            {fmt(elapsed)}
          </span>
          {overTime && (
            <span className="text-[10px] font-bold uppercase text-error mt-1">Over time</span>
          )}
        </div>
      </div>

      {/* ── Video panels ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-10">
        {[
          { label: name,       ref: localVideoElRef,  bg: 'bg-tertiary-container',  muted: true },
          { label: remoteName, ref: remoteVideoElRef, bg: 'bg-secondary-container', muted: false },
        ].map((p, i) => {
          const isLocal = i === 0
          const micMuted  = isLocal ? isMicMuted  : isRemoteMuted
          const videoOff  = isLocal ? isVideoOff  : isRemoteHidden
          const onMic     = isLocal ? toggleMic   : toggleRemoteMute
          const onVideo   = isLocal ? toggleLocalVideo : toggleRemoteVideo
          return (
          <div key={i} className="bg-white border-4 border-on-background neo-shadow-lg overflow-hidden">
            <div className="aspect-video relative bg-surface-container-highest">
              <video
                ref={p.ref}
                muted={p.muted}
                playsInline
                autoPlay
                className={`w-full h-full object-cover transition-opacity ${videoOff ? 'opacity-0' : 'opacity-100'}`}
              />
              {videoOff && (
                <div className="absolute inset-0 flex items-center justify-center bg-surface-container-highest">
                  <span className="material-symbols-outlined text-5xl text-on-surface-variant">videocam_off</span>
                </div>
              )}
              <div className={`absolute top-3 left-3 ${p.bg} border-2 border-on-background px-3 py-1 font-headline font-bold text-xs uppercase`}>
                {p.label}
              </div>
              <div className="absolute bottom-3 right-3 flex gap-2">
                <button
                  onClick={onMic}
                  disabled={!started}
                  className={`p-2 border-2 border-on-background cursor-pointer disabled:opacity-40 ${micMuted ? 'bg-error text-white' : 'bg-on-background text-white'}`}
                >
                  <span className="material-symbols-outlined text-sm">
                    {micMuted ? 'mic_off' : 'mic'}
                  </span>
                </button>
                <button
                  onClick={onVideo}
                  disabled={!started}
                  className={`p-2 border-2 border-on-background cursor-pointer disabled:opacity-40 ${videoOff ? 'bg-error text-white' : 'bg-on-background text-white'}`}
                >
                  <span className="material-symbols-outlined text-sm">
                    {videoOff ? 'videocam_off' : 'videocam'}
                  </span>
                </button>
              </div>
            </div>
          </div>
          )
        })}
      </div>

      {/* ── Lower: transcript + console ── */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-8 items-start">

        {/* Live transcript */}
        <div className="xl:col-span-2 bg-white border-4 border-on-background neo-shadow p-8">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-2xl font-headline font-black uppercase tracking-tight">
              Live Transcript
            </h3>
            <span className="flex items-center gap-2 text-xs font-bold font-headline uppercase text-on-surface-variant">
              <span className={`w-2 h-2 rounded-full ${recording ? 'bg-error animate-blink' : 'bg-outline'}`} />
              {recording ? 'Real-time analysis' : 'Not recording'}
            </span>
          </div>
          <div className="space-y-4 max-h-72 overflow-y-auto pr-2 font-body">
            {liveTranscripts.length === 0 ? (
              <p className="text-on-surface-variant text-sm italic">
                {started ? 'Waiting for speech...' : 'Session not started.'}
              </p>
            ) : liveTranscripts.map((t, i) => (
              <div key={i} className={`flex gap-3 ${t.user_id === 1 ? '' : 'border-l-4 border-tertiary-container pl-3'}`}>
                <span className={`font-headline font-black text-xs min-w-[56px] pt-0.5 ${t.user_id === 1 ? 'text-primary' : 'text-tertiary'}`}>
                  {t.speaker}
                </span>
                <p className="text-sm leading-relaxed text-on-surface">{t.text}</p>
              </div>
            ))}
          </div>

          {/* Windows feed */}
          {windows.length > 0 && (
            <div className="mt-8 pt-6 border-t-2 border-outline-variant">
              <h4 className="font-headline font-bold text-xs uppercase tracking-widest mb-3 text-on-surface-variant">
                Analysis Windows ({windows.length})
              </h4>
              <div className="flex flex-col gap-2 max-h-40 overflow-y-auto">
                {windows.map((w, i) => (
                  <div key={i} className="flex justify-between items-center px-3 py-2 border border-outline-variant text-xs">
                    <span className="font-bold">#{w.window_id}</span>
                    <span className={classColor[w.classification] || ''}>
                      {w.classification.replace('_', ' ')}
                    </span>
                    <span className="text-on-surface-variant">sim {w.similarity}</span>
                    <span className="text-on-surface-variant max-w-[140px] overflow-hidden text-ellipsis whitespace-nowrap">
                      {w.text_preview}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Referee console */}
        <div className="bg-surface-container-low border-4 border-on-background neo-shadow p-8 space-y-4">
          <h3 className="text-xl font-headline font-black uppercase tracking-tight mb-2">
            Referee Console
          </h3>

          {/* Escrow status */}
          {started && (
            <div className="p-4 bg-primary-container border-2 border-on-background">
              <p className="text-[10px] uppercase font-bold text-on-primary-container mb-3">
                Escrow Locked
              </p>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <p className="text-[10px] font-bold uppercase text-on-surface-variant">Alice (Teacher)</p>
                  <p className="font-headline font-bold text-lg">{teacherEscrow} cr</p>
                </div>
                <div>
                  <p className="text-[10px] font-bold uppercase text-on-surface-variant">Bob (Learner)</p>
                  <p className="font-headline font-bold text-lg">{learnerEscrow} cr</p>
                </div>
              </div>
              <p className="text-[10px] font-bold text-outline mt-2 uppercase">
                Total: {teacherEscrow + learnerEscrow} cr locked
              </p>
            </div>
          )}

          {/* Fairness meter */}
          {windows.length > 0 && (() => {
            const correct = windows.filter(w => w.classification === 'correct' || w.classification === 'weakly_correct').length
            const pct = Math.round((correct / windows.length) * 100)
            return (
              <div className="p-4 bg-tertiary-container border-2 border-on-background flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="material-symbols-outlined">gavel</span>
                  <span className="font-headline font-bold text-sm">On-Topic</span>
                </div>
                <span className="font-black text-xl">{pct}%</span>
              </div>
            )
          })()}

          {/* Warnings */}
          {warnings.length === 0 ? (
            <div className="p-4 bg-[#c4f5d3] border-2 border-on-background">
              <div className="flex items-center gap-3">
                <span className="material-symbols-outlined text-tertiary">check_circle</span>
                <span className="font-headline font-bold text-sm">No Warnings</span>
              </div>
            </div>
          ) : warnings.slice(0, 3).map((w, i) => (
            <div key={i} className={`p-4 border-2 ${severityBg[w.severity] || 'bg-white border-on-background'}`}>
              <div className="flex items-center gap-3 mb-1">
                <span className="material-symbols-outlined text-on-background">
                  {severityIcon[w.severity] || 'warning'}
                </span>
                <span className="font-headline font-bold text-sm capitalize">{w.severity} Warning</span>
              </div>
              <p className="text-xs text-on-surface-variant leading-relaxed">{w.reason}</p>
            </div>
          ))}
          {warnings.length > 3 && (
            <p className="text-xs font-bold text-outline text-center">
              +{warnings.length - 3} more warnings
            </p>
          )}

          {/* Pre-start */}
          {!started && !terminated && (
            <button
              className="w-full mt-4 py-4 bg-on-background text-white font-headline font-black uppercase tracking-widest text-sm neo-shadow hover:translate-x-0.5 hover:translate-y-0.5 hover:shadow-none transition-all"
              onClick={handleStart}
            >
              Start Session
            </button>
          )}

          {/* Actions */}
          {started && !terminated && (
            <>
              <button
                className="w-full mt-2 py-4 bg-on-background text-white font-headline font-black uppercase tracking-widest text-sm neo-shadow hover:translate-x-0.5 hover:translate-y-0.5 hover:shadow-none transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                onClick={handleConfirm}
                disabled={confirmed}
              >
                {confirmed ? 'Marked Complete' : 'Mark Complete'}
              </button>
              <button
                className="w-full py-4 bg-error-container border-4 border-on-background text-on-error-container font-headline font-black uppercase tracking-widest text-sm neo-shadow hover:translate-x-0.5 hover:translate-y-0.5 hover:shadow-none transition-all"
                onClick={handleTerminate}
              >
                Terminate
              </button>
            </>
          )}

          {terminated && (
            <div className="p-4 bg-error-container border-4 border-on-background text-center">
              <span className="material-symbols-outlined text-error block mb-2">block</span>
              <p className="font-headline font-bold uppercase text-sm text-on-error-container">
                Session Terminated
              </p>
            </div>
          )}

          {confirmed && !terminated && (
            <p className="text-center text-xs font-bold text-outline uppercase tracking-widest">
              Waiting for other participant...
            </p>
          )}
        </div>
      </div>

      {error && (
        <div className="mt-6 border-4 border-error bg-error-container p-4">
          <p className="text-on-error-container font-bold text-sm">{error}</p>
        </div>
      )}
    </div>
  )
}
