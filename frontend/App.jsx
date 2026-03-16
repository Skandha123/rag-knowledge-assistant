import { useState } from 'react'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import StatusBar from './components/StatusBar'
import { useDocuments } from './hooks/useDocuments'
import { useChat } from './hooks/useChat'
import { useToast } from './components/Toast'

export default function App() {
  const toast = useToast()
  const { documents, loading, upload, remove, refresh } = useDocuments()
  const { messages, streaming, sendMessage, clearHistory } = useChat()
  const [selectedDocs, setSelectedDocs] = useState([])

  const toggleDoc = (docId) => {
    setSelectedDocs(prev =>
      prev.includes(docId) ? prev.filter(id => id !== docId) : [...prev, docId]
    )
  }

  const handleUpload = async (file, onProgress) => {
    try {
      const result = await upload(file, onProgress)
      toast(`"${file.name}" ingested — ${result.chunks_stored} chunks indexed`, 'success')
      return result
    } catch (err) {
      toast(`Upload failed: ${err.message}`, 'error')
      throw err
    }
  }

  const handleDelete = async (docId) => {
    const doc = documents.find(d => d.doc_id === docId)
    try {
      await remove(docId)
      setSelectedDocs(prev => prev.filter(id => id !== docId))
      toast(`"${doc?.filename || docId}" removed from knowledge base`, 'info')
    } catch (err) {
      toast(`Delete failed: ${err.message}`, 'error')
    }
  }

  const handleSend = (question) => {
    const filterIds = selectedDocs.length > 0 ? selectedDocs : null
    sendMessage(question, filterIds)
  }

  return (
    <div className="flex flex-col h-dvh overflow-hidden">
      <StatusBar />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar
          documents={documents}
          loading={loading}
          onUpload={handleUpload}
          onDelete={handleDelete}
          onRefresh={refresh}
          selectedDocs={selectedDocs}
          onToggleDoc={toggleDoc}
        />
        <main className="flex-1 overflow-hidden">
          <ChatWindow
            messages={messages}
            streaming={streaming}
            onSend={handleSend}
            onClear={clearHistory}
            hasDocuments={documents.length > 0}
            selectedDocCount={selectedDocs.length}
          />
        </main>
      </div>
    </div>
  )
}
