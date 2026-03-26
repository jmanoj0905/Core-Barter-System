import { useState } from 'react'
import './App.css'
import Setup from './screens/Setup'
import LiveSession from './screens/LiveSession'
import PostSession from './screens/PostSession'

function Header() {
  return (
    <header className="bg-surface/80 backdrop-blur-md text-primary fixed top-0 w-full z-50 border-b border-primary flex justify-between items-center px-6 h-16">
      <div className="text-xl font-black uppercase tracking-widest">
        Barter Monitor
      </div>
      <div className="flex items-center gap-4" />
    </header>
  )
}

export default function App() {
  const [screen, setScreen] = useState('setup')
  const [barterId, setBarterId] = useState(null)
  const [agreedMinutes, setAgreedMinutes] = useState(5)
  const [userId, setUserId] = useState(1)

  function handleSessionCreated(id, minutes, uid) {
    setBarterId(id)
    setAgreedMinutes(minutes)
    setUserId(uid)
    setScreen('live')
  }

  function handleComplete(id) {
    setBarterId(id)
    setScreen('post')
  }

  function handleReset() {
    setBarterId(null)
    setScreen('setup')
  }

  return (
    <>
      <Header />
      <main className="pt-32 px-6 md:px-24 lg:px-48 max-w-7xl mx-auto pb-12">
        {screen === 'setup' && <Setup onSessionCreated={handleSessionCreated} />}
        {screen === 'live' && (
          <LiveSession
            barterId={barterId}
            agreedMinutes={agreedMinutes}
            userId={userId}
            onComplete={handleComplete}
          />
        )}
        {screen === 'post' && <PostSession barterId={barterId} onReset={handleReset} />}
      </main>
    </>
  )
}
