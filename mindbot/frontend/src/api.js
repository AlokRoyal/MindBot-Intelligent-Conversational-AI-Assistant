const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export async function sendChat(message, conversationId) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, conversation_id: conversationId || null }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Failed to send message')
  }
  return res.json()
}

export async function uploadDocument(file) {
  const form = new FormData()
  form.append('file', file)

  const res = await fetch(`${API_BASE}/upload/document`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Document upload failed')
  }
  return res.json()
}

export async function uploadAudio(file) {
  const form = new FormData()
  form.append('file', file)

  const res = await fetch(`${API_BASE}/upload/audio`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Audio upload failed')
  }
  return res.json()
}

export async function uploadImage(file, prompt = 'Describe this image in detail.') {
  const form = new FormData()
  form.append('file', file)
  form.append('prompt', prompt)

  const res = await fetch(`${API_BASE}/upload/image`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Image upload failed')
  }
  return res.json()
}

export default API_BASE