import React, { useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import {
  sendChat,
  uploadDocument,
  uploadAudio,
  uploadImage,
} from './api'
import { Upload, Mic, Image as ImageIcon, Send, Trash2 } from 'lucide-react'

const STORAGE_KEY = 'mindbot_chat_v1'

function loadInitialState() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved) return JSON.parse(saved)
  } catch (e) {}
  return {
    conversationId: null,
    messages: [
      {
        role: 'assistant',
        content: 'Hi, I am MindBot. Upload a document or start chatting.',
      },
    ],
  }
}

function App() {
  const [state, setState] = useState(loadInitialState)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState('')
  const [recording, setRecording] = useState(false)
  const [recordedBlob, setRecordedBlob] = useState(null)
  const messagesEndRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
  }, [state])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [state.messages, loading])

  const conversationLabel = useMemo(
    () => (state.conversationId ? `Conversation #${state.conversationId}` : 'New session'),
    [state.conversationId]
  )

  const pushMessage = (role, content) => {
    setState((prev) => ({
      ...prev,
      messages: [...prev.messages, { role, content }],
    }))
  }

  const handleSend = async () => {
    const text = input.trim()
    if (!text || loading) return

    setInput('')
    setLoading(true)
    setStatus('Sending message...')
    pushMessage('user', text)

    try {
      const res = await sendChat(text, state.conversationId)
      setState((prev) => ({
        ...prev,
        conversationId: res.conversation_id,
        messages: [
          ...prev.messages,
          { role: 'assistant', content: res.reply },
        ],
      }))
      setStatus(res.context_used ? 'Context retrieved from uploaded docs.' : 'Reply generated.')
    } catch (err) {
      pushMessage('system', err.message || 'Something went wrong')
      setStatus('Error')
    } finally {
      setLoading(false)
    }
  }

  const handleDocumentUpload = async (file) => {
    if (!file) return
    setLoading(true)
    setStatus('Uploading document...')
    try {
      const res = await uploadDocument(file)
      pushMessage('system', `Document indexed: ${res.detail}`)
      setStatus('Document ready for Q&A.')
    } catch (err) {
      pushMessage('system', err.message)
      setStatus('Upload failed')
    } finally {
      setLoading(false)
    }
  }

  const handleImageUpload = async (file) => {
    if (!file) return
    setLoading(true)
    setStatus('Uploading image...')
    try {
      const res = await uploadImage(file, 'Analyze this image and describe what is visible.')
      pushMessage('system', `Image analysis: ${res.result}`)
      setStatus('Image processed.')
    } catch (err) {
      pushMessage('system', err.message)
      setStatus('Image upload failed')
    } finally {
      setLoading(false)
    }
  }

  const handleAudioUpload = async (file) => {
    if (!file) return
    setLoading(true)
    setStatus('Transcribing audio...')
    try {
      const res = await uploadAudio(file)
      pushMessage('user', res.transcript)
      setState((prev) => ({ ...prev, messages: [...prev.messages] }))
      setStatus('Audio transcribed.')
    } catch (err) {
      pushMessage('system', err.message)
      setStatus('Audio upload failed')
    } finally {
      setLoading(false)
    }
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      mediaRecorderRef.current = recorder
      chunksRef.current = []

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        setRecordedBlob(blob)
        stream.getTracks().forEach((track) => track.stop())
      }

      recorder.start()
      setRecording(true)
      setStatus('Recording...')
    } catch (err) {
      pushMessage('system', 'Microphone access denied or unavailable.')
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop()
      setRecording(false)
      setStatus('Recording saved.')
    }
  }

  const uploadRecordedAudio = async () => {
    if (!recordedBlob) return
    const file = new File([recordedBlob], 'recording.webm', { type: 'audio/webm' })
    await handleAudioUpload(file)
    setRecordedBlob(null)
  }

  const clearChat = () => {
    setState({
      conversationId: null,
      messages: [
        {
          role: 'assistant',
          content: 'Chat cleared. Start a new conversation with MindBot.',
        },
      ],
    })
    setStatus('Chat cleared.')
  }

  return (
    <div className="chat-shell">
      <aside className="panel sidebar">
        <h2 className="brand">MindBot</h2>
        <div className="muted small">
          Intelligent Conversational AI Assistant
        </div>

        <div className="section-title">Session</div>
        <div className="meta">
          <span className="tag">{conversationLabel}</span>
          <span className="tag">{status || 'Idle'}</span>
        </div>

        <div className="section-title">Upload Files</div>

        <label className="action-btn" style={{ display: 'block', textAlign: 'center' }}>
          <Upload size={16} style={{ verticalAlign: 'middle', marginRight: 8 }} />
          Upload PDF / TXT
          <input
            className="file-input"
            type="file"
            accept=".pdf,.txt,.md,.json,.csv"
            onChange={(e) => handleDocumentUpload(e.target.files?.[0])}
          />
        </label>

        <label className="action-btn secondary" style={{ display: 'block', textAlign: 'center' }}>
          <ImageIcon size={16} style={{ verticalAlign: 'middle', marginRight: 8 }} />
          Upload Image
          <input
            className="file-input"
            type="file"
            accept="image/*"
            onChange={(e) => handleImageUpload(e.target.files?.[0])}
          />
        </label>

        <label className="action-btn secondary" style={{ display: 'block', textAlign: 'center' }}>
          <Mic size={16} style={{ verticalAlign: 'middle', marginRight: 8 }} />
          Upload Audio
          <input
            className="file-input"
            type="file"
            accept="audio/*"
            onChange={(e) => handleAudioUpload(e.target.files?.[0])}
          />
        </label>

        <div className="section-title">Voice Recorder</div>
        <button
          className="action-btn secondary"
          onClick={recording ? stopRecording : startRecording}
        >
          {recording ? 'Stop Recording' : 'Start Recording'}
        </button>

        {recordedBlob && (
          <button className="action-btn" onClick={uploadRecordedAudio}>
            Upload Recorded Audio
          </button>
        )}

        <div className="section-title">Utilities</div>
        <button className="action-btn secondary" onClick={clearChat}>
          <Trash2 size={16} style={{ verticalAlign: 'middle', marginRight: 8 }} />
          Clear Chat
        </button>

        <div className="section-title">Project Notes</div>
        <div className="muted small">
          MindBot supports document RAG, voice transcription, image analysis, and persistent conversation history.
        </div>
      </aside>

      <main className="panel main">
        <div className="header">
          <h1>Chat Interface</h1>
          <div className="muted small">
            Ask questions about your uploaded files or use it as a general assistant.
          </div>
        </div>

        <div className="messages">
          {state.messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              {msg.role === 'assistant' || msg.role === 'user' ? (
                <ReactMarkdown>{msg.content}</ReactMarkdown>
              ) : (
                <div>{msg.content}</div>
              )}
            </div>
          ))}

          {loading && (
            <div className="message assistant">
              <em>MindBot is typing...</em>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="composer">
          <textarea
            value={input}
            placeholder="Type your message here..."
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                handleSend()
              }
            }}
          />

          <div className="row">
            <button className="send-btn" onClick={handleSend} disabled={loading}>
              <Send size={16} style={{ verticalAlign: 'middle', marginRight: 8 }} />
              Send
            </button>
            <div className="muted small">
              Tip: upload a PDF first, then ask questions about it.
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App