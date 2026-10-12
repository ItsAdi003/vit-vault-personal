import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { supabase } from '../lib/supabase'
import { useAuth } from '../lib/AuthContext'
import { MESSAGES } from '../lib/utils'
import toast from 'react-hot-toast'
import { isAdminEmail } from '../lib/config'

export default function UploadModal({ isOpen, onClose, folderId = null }) {
  const { user } = useAuth()
  const fileInputRef = useRef(null)
  
  const [uploadType, setUploadType] = useState('DIRECT')
  const [eventType, setEventType] = useState('INSTITUTIONAL')
  const [eventName, setEventName] = useState('')
  const [eventDate, setEventDate] = useState('')
  const [imageName, setImageName] = useState('')
  const [description, setDescription] = useState('')
  const [files, setFiles] = useState([])
  
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [errorMsg, setErrorMsg] = useState('')

  useEffect(() => {
    if (isOpen) {
      setErrorMsg('')
      setFiles([])
      setDescription('')
      setUploadProgress(0)
      if (folderId) {
        fetchFolderDetails()
      } else {
        setUploadType('DIRECT')
        setEventName('')
        setEventDate('')
        setImageName('')
        setEventType('INSTITUTIONAL')
      }
    }
  }, [folderId, isOpen])

  const fetchFolderDetails = async () => {
    let folderData = null;
    
    // Try to fetch with collaborators first
    const { data, error } = await supabase
      .from('events')
      .select('title, type, created_by, collaborators')
      .eq('id', folderId)
      .single()
      
    if (error && error.message.includes('collaborators')) {
      const fallback = await supabase
        .from('events')
        .select('title, type, created_by')
        .eq('id', folderId)
        .single()
      folderData = fallback.data;
    } else if (data) {
      folderData = data;
    }
    
    if (folderData) {
      setEventName(folderData.title)
      setEventType(folderData.type)
      
      const isCreator = folderData.created_by === user.id;
      const isAdmin = isAdminEmail(user.email);
      const isCollaborator = folderData.collaborators?.includes(user?.email?.toLowerCase());

      if (!isCreator && !isAdmin && !isCollaborator) {
        setErrorMsg(MESSAGES.ERRORS.NO_PERMISSION_ADD_IMAGES)
      }
    }
  }

  if (!isOpen) return null

  const handleFileSelect = (e) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files))
    }
  }

  const handleUpload = async () => {
    const finalEventName = (!folderId && uploadType === 'DIRECT') 
      ? `Standalone Uploads (${eventType === 'COMMITTEE' ? 'Committee' : 'Institutional'})` 
      : eventName.trim();
    const finalEventType = eventType;

    if (!finalEventName) {
      setErrorMsg(MESSAGES.ERRORS.EMPTY_FOLDER_NAME)
      return
    }
    if (files.length === 0) {
      setErrorMsg(MESSAGES.ERRORS.NO_FILES_SELECTED)
      return
    }

    if (eventDate) {
      const selectedDate = new Date(eventDate);
      const now = new Date();
      // Reset hours to only compare calendar days
      selectedDate.setHours(0, 0, 0, 0);
      now.setHours(0, 0, 0, 0);
      
      if (selectedDate > now) {
        setErrorMsg(MESSAGES.ERRORS.INVALID_DATE_FUTURE);
        return;
      }
    }

    setIsUploading(true)
    setErrorMsg('')
    
    try {
      // 1. Find or create event
      let eventId = folderId;
      
      if (!eventId) {
        let existing = null;
        const { data: extData, error: extErr } = await supabase
          .from('events')
          .select('id, created_by, collaborators')
          .eq('title', finalEventName)
          .eq('type', finalEventType)
          .limit(1)
          
        if (extErr && extErr.message.includes('collaborators')) {
          const fb = await supabase.from('events').select('id, created_by').eq('title', finalEventName).eq('type', finalEventType).limit(1)
          existing = fb.data;
        } else if (extData) {
          existing = extData;
        }

        if (existing && existing.length > 0) {
          const f = existing[0];
          const isCollaborator = f.collaborators?.includes(user?.email?.toLowerCase());
          const isDirectUpload = uploadType === 'DIRECT';
          if (!isDirectUpload && f.created_by !== user.id && !isAdminEmail(user.email) && !isCollaborator) {
            throw new Error(MESSAGES.ERRORS.FOLDER_EXISTS_OTHER_USER)
          }
          eventId = existing[0].id
        } else {
          const { data: newEvent, error: eventErr } = await supabase
            .from('events')
            .insert({
              title: finalEventName,
              type: finalEventType,
              event_date: eventDate ? new Date(eventDate).toISOString() : new Date().toISOString(),
              organizer: 'VIT Admin',
              created_by: user.id
            })
            .select()
            .single()
            
          if (eventErr) throw eventErr
          eventId = newEvent.id
        }
      } else {
        // Double check ownership for existing folderId
        let doubleCheck = null;
        const { data: folderInfo, error: dcErr } = await supabase
          .from('events')
          .select('created_by, collaborators')
          .eq('id', eventId)
          .single()
          
        if (dcErr && dcErr.message.includes('collaborators')) {
           const dcFall = await supabase.from('events').select('created_by').eq('id', eventId).single()
           doubleCheck = dcFall.data;
        } else if (folderInfo) {
           doubleCheck = folderInfo;
        }
        
        if (doubleCheck) {
          const isCollaborator = doubleCheck.collaborators?.includes(user?.email?.toLowerCase());
          if (doubleCheck.created_by !== user.id && !isAdminEmail(user.email) && !isCollaborator) {
            throw new Error(MESSAGES.ERRORS.NO_PERMISSION_FOLDER)
          }
        }
      }

      // 2. Upload files and create media records
      for (let i = 0; i < files.length; i++) {
        const file = files[i]
        const fileExt = file.name.split('.').pop()
        const fileName = `${Date.now()}_${Math.random().toString(36).substring(7)}.${fileExt}`
        const filePath = `${user.id}/${fileName}`

        // Upload to storage
        const { error: uploadError } = await supabase.storage
          .from('vault-media')
          .upload(filePath, file)

        if (uploadError) throw uploadError

        // Insert into media table
        const { data: mediaRecord, error: dbError } = await supabase
          .from('media')
          .insert({
            event_id: eventId,
            uploader_id: user.id,
            file_path: filePath,
            status: 'APPROVED', // Auto-approving for now so it shows in the gallery
            metadata: { 
              description,
              custom_name: files.length > 1 && imageName.trim() ? `${imageName.trim()} ${i + 1}` : imageName.trim(),
              custom_date: eventDate ? new Date(eventDate).toISOString() : null
            }
          })
          .select('id')
          .single()
          
        if (dbError) throw dbError

        // 🧠 Send to AI Microservice for auto-tagging + deduplication + embedding
        // This is fire-and-forget — the upload finishes immediately while AI processes in background
        if (mediaRecord?.id) {
          const AI_URL = import.meta.env.VITE_AI_SERVICE_URL || 'http://localhost:5000';
          fetch(`${AI_URL}/api/process`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              media_id: mediaRecord.id,
              file_path: filePath
            })
          })
          .then(res => res.json())
          .then(result => {
            if (result.status === 'duplicate') {
              toast.error(`"${file.name}" was rejected. ${MESSAGES.ERRORS.IMAGE_ALREADY_EXISTS}`, { duration: 5000 });
              window.dispatchEvent(new Event('mediaUploaded'));
            } else if (result.status === 'error' || result.error) {
              console.error(`[AI] Processing failed for ${file.name}`, result);
              toast.error(`AI tagging failed for "${file.name}": ${result.error || 'unknown error'}`);
            } else {
              console.log(`[AI] Processing complete for ${file.name}:`, result);
              window.dispatchEvent(new Event('mediaUploaded'));
            }
          })
          .catch(err => {
            console.warn(`[AI] AI processing skipped or failed for ${file.name}:`, err.message || err);
            toast.error(`Couldn't reach the AI service for "${file.name}" — is app.py running on port 5000?`);
          })
        }

        // Update progress
        setUploadProgress(Math.round(((i + 1) / files.length) * 100))
      }

      // Reset and close
      setFiles([])
      setEventName('')
      setDescription('')
      setUploadProgress(0)
      onClose()
      
      // Temporary hack to force gallery refresh if we are on it
      window.dispatchEvent(new Event('mediaUploaded'))

    } catch (error) {
      console.error('Upload failed:', error)
      setErrorMsg(error.message || "Failed to upload files.")
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-[100] flex items-center justify-center p-4 md:p-8"
        onClick={!isUploading ? onClose : undefined}
      >
        {/* Backdrop */}
        <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />

        {/* Modal */}
        <motion.div
          initial={{ scale: 0.9, opacity: 0, y: 20 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.9, opacity: 0, y: 20 }}
          transition={{ type: 'spring', damping: 25, stiffness: 300 }}
          className="relative w-full max-w-5xl glass-effect rounded-xl shadow-2xl overflow-hidden flex flex-col bg-[#0a0a14]/90 border border-[#3b3bed]/20"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <header className="flex items-center justify-between px-8 py-6 border-b border-white/10">
            <div className="flex items-center gap-3">
              <div className="bg-[#3b3bed] p-2 rounded-lg flex items-center justify-center shadow-lg shadow-[#3b3bed]/20">
                <span className="material-symbols-outlined text-white text-2xl">cloud_upload</span>
              </div>
              <div>
                <h1 className="text-xl font-bold tracking-tight text-white">Upload to Vault</h1>
                <p className="text-slate-400 text-xs uppercase tracking-widest font-semibold">Organize into Folders</p>
              </div>
            </div>
            {!isUploading && (
              <button className="text-slate-400 hover:text-white transition-colors" onClick={onClose}>
                <span className="material-symbols-outlined">close</span>
              </button>
            )}
          </header>

          <div className="flex flex-col md:flex-row min-h-[500px]">
            {/* Left Column: Form Details */}
            <div className="w-full md:w-5/12 p-8 border-r border-white/5 bg-white/[0.02] flex flex-col gap-6">
              
              {!folderId && (
                <div className="flex bg-[#1e1e38]/80 p-1.5 rounded-xl border border-white/10 shadow-inner">
                  <button
                    onClick={() => setUploadType('DIRECT')}
                    disabled={isUploading}
                    className={`flex-1 flex items-center justify-center gap-2 py-2.5 text-xs font-bold rounded-lg transition-all ${uploadType === 'DIRECT' ? 'bg-[#3b3bed] text-white shadow-md' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}
                  >
                    <span className="material-symbols-outlined text-sm">imagesmode</span> Direct Upload
                  </button>
                  <button
                    onClick={() => setUploadType('FOLDER')}
                    disabled={isUploading}
                    className={`flex-1 flex items-center justify-center gap-2 py-2.5 text-xs font-bold rounded-lg transition-all ${uploadType === 'FOLDER' ? 'bg-[#3b3bed] text-white shadow-md' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}
                  >
                    <span className="material-symbols-outlined text-sm">create_new_folder</span> Create Folder
                  </button>
                </div>
              )}

              {/* Event Type Configuration */}
              {(!folderId) && (
                <div className="flex flex-col gap-2 animate-in fade-in slide-in-from-top-4 duration-300">
                  <label className="text-xs font-bold uppercase tracking-wider text-slate-400">Event Type</label>
                  <div className="flex gap-2 p-1 bg-[#1e1e38]/50 border border-white/10 rounded-lg">
                    <button
                      className={`flex-1 py-3 text-xs font-bold rounded-md transition-all ${eventType === 'INSTITUTIONAL' ? 'bg-[#3b3bed] text-white shadow-lg' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}
                      onClick={() => setEventType('INSTITUTIONAL')}
                      disabled={isUploading}
                    >Institutional Folder</button>
                    <button
                      className={`flex-1 py-3 text-xs font-bold rounded-md transition-all ${eventType === 'COMMITTEE' ? 'bg-[#3b3bed] text-white shadow-lg' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}
                      onClick={() => setEventType('COMMITTEE')}
                      disabled={isUploading}
                    >Committee Folder</button>
                  </div>
                </div>
              )}

              {/* Folder / Event Name */}
              {!folderId ? (
                uploadType === 'FOLDER' && (
                  <div className="flex flex-col gap-4 animate-in fade-in slide-in-from-top-4 duration-300">
                    <div className="flex flex-col gap-2">
                      <label className="text-xs font-bold uppercase tracking-wider text-slate-400">Folder / Event Name</label>
                      <input 
                        type="text" 
                        value={eventName}
                        onChange={(e) => setEventName(e.target.value)}
                        disabled={isUploading}
                        className="w-full bg-[#1e1e38]/50 border border-white/10 rounded-lg px-4 py-3 focus:ring-2 focus:ring-[#3b3bed] outline-none text-white placeholder:text-slate-600 transition-all"
                        placeholder="e.g., Verve Annual Fest 2024"
                      />
                    </div>
                  </div>
                )
              ) : (
                <div className="flex flex-col gap-2">
                  <label className="text-xs font-bold uppercase tracking-wider text-slate-400">Target Folder</label>
                  <div className="w-full bg-[#3b3bed]/10 border border-[#3b3bed]/30 rounded-lg px-4 py-3 focus:ring-2 focus:ring-[#3b3bed] outline-none text-white transition-all flex items-center gap-3">
                    <span className="material-symbols-outlined text-[#3b3bed]">folder</span>
                    <span className="font-bold text-lg">{eventName}</span>
                  </div>
                  <p className="text-[10px] text-slate-500 italic mt-1">Classification and name are locked for existing folders.</p>
                </div>
              )}

              {/* Detail Inputs */}
              <div className="flex flex-col gap-4 flex-1">
                <div className="flex gap-4 w-full">
                  <div className="flex flex-col gap-2 flex-1">
                    <label className="text-xs font-bold uppercase tracking-wider text-slate-400">Image Name (Optional)</label>
                    <input 
                      type="text" 
                      value={imageName}
                      onChange={(e) => setImageName(e.target.value)}
                      disabled={isUploading}
                      className="w-full bg-[#1e1e38]/50 border border-white/10 rounded-lg px-4 py-3 focus:ring-2 focus:ring-[#3b3bed] outline-none text-white transition-all placeholder:text-slate-600"
                      placeholder="e.g., Award Ceremony"
                    />
                  </div>
                  <div className="flex flex-col gap-2 flex-1">
                    <label className="text-xs font-bold uppercase tracking-wider text-slate-400">Captured Date (Optional)</label>
                    <input 
                      type="date" 
                      value={eventDate}
                      onChange={(e) => setEventDate(e.target.value)}
                      disabled={isUploading}
                      className="w-full bg-[#1e1e38]/50 border border-white/10 rounded-lg px-4 py-3 focus:ring-2 focus:ring-[#3b3bed] outline-none text-white transition-all [&::-webkit-calendar-picker-indicator]:invert"
                    />
                  </div>
                </div>
                <div className="flex flex-col gap-2 flex-1">
                  <label className="text-xs font-bold uppercase tracking-wider text-slate-400">Asset Description (Optional)</label>
                  <textarea 
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    disabled={isUploading}
                    className="w-full h-full min-h-[80px] bg-[#1e1e38]/50 border border-white/10 rounded-lg px-4 py-3 focus:ring-2 focus:ring-[#3b3bed] outline-none text-white placeholder:text-slate-600 resize-none transition-all"
                    placeholder="Provide context or a caption for these uploads..."
                  />
                </div>
              </div>

            </div>

            {/* Right Column: Dropzone & Queue */}
            <div className="w-full md:w-7/12 p-8 flex flex-col">
              
              {/* Dropzone */}
              <div 
                onClick={() => !isUploading && fileInputRef.current?.click()}
                className={`flex-1 min-h-[200px] flex flex-col items-center justify-center border-2 border-dashed rounded-xl px-8 py-10 transition-all ${
                  isUploading ? 'border-white/10 bg-white/5 cursor-not-allowed opacity-50' : 'border-[#3b3bed]/40 bg-[#3b3bed]/5 hover:border-[#3b3bed] hover:bg-[#3b3bed]/10 cursor-pointer group'
                }`}
              >
                <div className={`w-16 h-16 rounded-full flex items-center justify-center mb-4 transition-all ${
                  isUploading ? 'bg-white/10' : 'bg-[#3b3bed]/20 group-hover:scale-110'
                }`}>
                  <span className={`material-symbols-outlined text-4xl ${isUploading ? 'text-white/30' : 'text-[#3b3bed]'}`}>
                    {files.length > 0 ? 'library_add' : 'upload_file'}
                  </span>
                </div>
                <h3 className="text-lg font-bold mb-1 text-white">
                  {files.length > 0 ? 'Add more files' : 'Select media to upload'}
                </h3>
                <p className="text-slate-400 text-sm text-center">Supports images and videos</p>
                <input 
                  type="file" 
                  multiple 
                  className="hidden" 
                  ref={fileInputRef}
                  onChange={handleFileSelect}
                  accept="image/*,video/*"
                />
              </div>

              {/* File Info / Error */}
              {errorMsg && (
                <div className="mt-4 p-3 bg-red-500/10 border border-red-500/20 text-red-500 rounded-lg text-sm font-medium">
                  {errorMsg}
                </div>
              )}

              {files.length > 0 && !isUploading && (
                <div className="mt-6">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3 flex justify-between">
                    Selected Files 
                    <span className="text-[#3b3bed]">{files.length} items</span>
                  </h4>
                  <div className="max-h-32 overflow-y-auto space-y-2 pr-2 custom-scrollbar">
                    {files.map((f, i) => (
                      <div key={i} className="flex justify-between items-center bg-white/5 px-3 py-2 rounded-lg border border-white/5">
                        <span className="text-sm text-slate-300 truncate max-w-[80%]">{f.name}</span>
                        <span className="text-xs text-slate-500">{(f.size / (1024*1024)).toFixed(1)} MB</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Progress Bar during upload */}
              {isUploading && (
                <div className="mt-6 space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-[#3b3bed] font-medium animate-pulse">Uploading securely to Vault...</span>
                    <span className="text-white font-bold">{uploadProgress}%</span>
                  </div>
                  <div className="w-full bg-white/10 h-2 rounded-full overflow-hidden">
                    <motion.div 
                      className="h-full bg-gradient-to-r from-[#3b3bed] to-indigo-400"
                      initial={{ width: 0 }}
                      animate={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                </div>
              )}

            </div>
          </div>

          {/* Footer Actions */}
          <footer className="px-8 py-5 border-t border-white/10 flex items-center justify-end gap-4 bg-black/20">
            <button
              onClick={onClose}
              disabled={isUploading}
              className="px-6 py-2.5 rounded-lg font-bold text-sm text-slate-300 hover:text-white hover:bg-white/5 transition-all disabled:opacity-50"
            >
              Cancel
            </button>
            <button 
              onClick={handleUpload}
              disabled={isUploading || files.length === 0 || !!errorMsg.includes("permission")}
              className="bg-[#3b3bed] hover:bg-[#3b3bed]/80 disabled:opacity-50 disabled:hover:bg-[#3b3bed] text-white px-8 py-2.5 rounded-lg font-bold text-sm shadow-xl shadow-[#3b3bed]/30 transition-all flex items-center justify-center gap-2"
            >
              {isUploading ? (
                <> <span className="material-symbols-outlined text-sm animate-spin">refresh</span> Processing... </>
              ) : (
                <> {folderId ? 'Add to Folder' : 'Start Ingestion'} <span className="material-symbols-outlined text-sm">rocket_launch</span> </>
              )}
            </button>
          </footer>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}
