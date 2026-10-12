import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { Link } from 'react-router-dom'

const AI_URL = import.meta.env.VITE_AI_SERVICE_URL || 'http://localhost:5000'
const CLUSTER_COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899']

async function api(path, options) {
  const res = await fetch(`${AI_URL}${path}`, options)
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`)
  return data
}

const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`

function Panel({ title, subtitle, right, children }) {
  return (
    <section className="glass-card rounded-2xl p-5 md:p-6">
      <div className="flex flex-wrap items-start justify-between gap-3 mb-4">
        <div>
          <h3 className="text-base font-bold text-white">{title}</h3>
          {subtitle && <p className="text-xs text-slate-400 mt-1 max-w-2xl leading-relaxed">{subtitle}</p>}
        </div>
        {right}
      </div>
      {children}
    </section>
  )
}

function Bar({ label, value, max = 1, color = '#3b3bed', right }) {
  return (
    <div className="flex items-center gap-3 text-xs">
      <span className="w-32 shrink-0 truncate text-slate-300" title={label}>{label}</span>
      <div className="flex-1 h-2.5 rounded-full bg-white/5 overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${Math.max(0, Math.min(1, value / max)) * 100}%`, background: color }} />
      </div>
      <span className="w-16 text-right tabular-nums text-slate-400">{right ?? pct(value)}</span>
    </div>
  )
}

function Stat({ label, value }) {
  return (
    <div className="rounded-xl bg-white/5 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className="text-sm font-semibold text-white tabular-nums">{value}</div>
    </div>
  )
}

/* ------------------------------------------------------------ report card */
function ReportCard({ report }) {
  if (!report) return null
  const { title, description, meta, table, curve, confusion, notes } = report
  const maxC = confusion ? Math.max(...confusion.matrix.flat(), 1) : 1
  return (
    <div className="rounded-xl border border-white/10 p-4 space-y-3">
      <div>
        <h4 className="text-sm font-bold text-white">{title}</h4>
        {description && <p className="text-xs text-slate-400 mt-1 leading-relaxed">{description}</p>}
      </div>
      {meta && (
        <div className="flex flex-wrap gap-2">
          {Object.entries(meta).map(([k, v]) => (
            <span key={k} className="text-[10px] px-2 py-1 rounded-full bg-white/5 text-slate-300">{k}: <b className="text-white">{String(v)}</b></span>
          ))}
        </div>
      )}
      {table && (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-slate-500 border-b border-white/10">
                {table.columns.map(c => <th key={c} className="py-1.5 pr-4 font-medium">{c}</th>)}
              </tr>
            </thead>
            <tbody>
              {table.rows.map((row, i) => (
                <tr key={i} className="border-b border-white/5 text-slate-200">
                  {row.map((cell, j) => <td key={j} className="py-1.5 pr-4 tabular-nums">{typeof cell === 'number' ? (Number.isInteger(cell) ? cell : cell.toFixed(4)) : cell}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {curve && curve.y.length > 1 && (() => {
        const ys = curve.y, lo = Math.min(...ys), hi = Math.max(...ys), W = 360, H = 90
        const pts = ys.map((y, i) => `${(i / (ys.length - 1)) * (W - 10) + 5},${H - 8 - ((y - lo) / (hi - lo || 1)) * (H - 16)}`).join(' ')
        return (
          <div>
            <div className="text-[10px] text-slate-500 mb-1">{curve.label}</div>
            <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-md"><polyline points={pts} fill="none" stroke="#818cf8" strokeWidth="2" /></svg>
          </div>
        )
      })()}
      {confusion && (
        <div className="overflow-x-auto">
          <div className="text-[10px] text-slate-500 mb-1">Confusion matrix (rows = truth, columns = prediction)</div>
          <table className="text-[10px] tabular-nums">
            <tbody>
              {confusion.matrix.map((row, i) => (
                <tr key={i}>
                  <td className="pr-2 text-slate-400 whitespace-nowrap">{confusion.classes[i]}</td>
                  {row.map((v, j) => (
                    <td key={j} className="w-9 h-7 text-center text-white" style={{ background: `rgba(99,102,241,${0.08 + 0.85 * (v / maxC)})` }}>{v}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {notes && <p className="text-xs text-slate-400 leading-relaxed">{notes}</p>}
    </div>
  )
}

function Reports({ benchmarks, keys, emptyHint }) {
  const reports = keys.map(k => benchmarks?.[k]).filter(Boolean)
  if (!reports.length) {
    return <p className="text-xs text-slate-500">{emptyHint}</p>
  }
  return <div className="space-y-4">{reports.map(r => <ReportCard key={r.title} report={r} />)}</div>
}

/* --------------------------------------------------------- corpus manager */
function CorpusPanel({ corpus, onChange }) {
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const input = useRef(null)

  const add = async (files) => {
    if (!files.length) return
    const fd = new FormData()
    files.forEach(f => fd.append('images', f))
    setBusy(true); setErr('')
    try {
      await api('/api/lab/corpus', { method: 'POST', body: fd })
      await onChange()
    } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }
  const clear = async () => {
    setBusy(true)
    try { await api('/api/lab/corpus', { method: 'DELETE' }); await onChange() } finally { setBusy(false) }
  }

  return (
    <Panel
      title={`Photo corpus (${corpus.length})`}
      subtitle="Drop a folder's worth of photos here. They stay in memory on the AI service only (no database) and are used for the clustering map and the search comparison."
      right={
        <div className="flex gap-2">
          <button onClick={() => input.current?.click()} disabled={busy}
            className="px-3 py-1.5 rounded-lg bg-[#3b3bed] text-white text-xs font-bold disabled:opacity-50">
            {busy ? 'Working...' : 'Add photos'}
          </button>
          {corpus.length > 0 && (
            <button onClick={clear} disabled={busy} className="px-3 py-1.5 rounded-lg bg-white/5 text-slate-300 text-xs font-bold">Clear</button>
          )}
          <input ref={input} type="file" accept="image/*" multiple hidden
            onChange={e => { add([...e.target.files]); e.target.value = '' }} />
        </div>
      }
    >
      {err && <p className="text-xs text-red-400 mb-2">{err}</p>}
      {corpus.length === 0 ? (
        <p className="text-xs text-slate-500">Empty. Add at least 4 photos to see the clustering map; any number for search.</p>
      ) : (
        <div className="grid grid-cols-3 sm:grid-cols-5 md:grid-cols-8 gap-2">
          {corpus.map(c => (
            <div key={c.id} className="relative rounded-lg overflow-hidden bg-black/30" title={`${c.name}\n${c.tags.join(', ')}`}>
              <img src={c.thumb} alt={c.name} className="w-full h-16 object-cover" />
              <div className="px-1.5 py-1 text-[9px] text-slate-400 truncate">{c.tags.slice(0, 2).join(', ') || '-'}</div>
            </div>
          ))}
        </div>
      )}
    </Panel>
  )
}

/* ---------------------------------------------------------------- AML tab */
function NaiveBayesPanel({ info, analysis }) {
  // User edits are remembered per analysis; a new analysis starts again from its detected tags.
  const [edit, setEdit] = useState({ forAnalysis: null, tags: [] })
  const [result, setResult] = useState(null)
  const [err, setErr] = useState('')
  const tags = useMemo(
    () => (edit.forAnalysis === analysis ? edit.tags : (analysis?.tags ?? [])),
    [edit, analysis],
  )

  useEffect(() => {
    let cancelled = false
    const t = setTimeout(async () => {
      try {
        const r = await api('/api/lab/nb', {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ tags }),
        })
        if (!cancelled) { setResult(r); setErr('') }
      } catch (e) { if (!cancelled) setErr(e.message) }
    }, 150)
    return () => { cancelled = true; clearTimeout(t) }
  }, [tags])

  const toggle = (t) => setEdit({
    forAnalysis: analysis,
    tags: tags.includes(t) ? tags.filter(x => x !== t) : [...tags, t],
  })
  const hand = result?.handcrafted
  const learned = result?.learned

  return (
    <Panel
      title="Naive Bayes: a Bayesian network over CLIP tags"
      subtitle="The category is the parent node, the observed tags are its children, assumed conditionally independent given the category. Toggle tags to see the posterior P(category | tags) update."
    >
      <div className="grid lg:grid-cols-2 gap-6">
        <div>
          <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-2">Observed evidence (click to toggle)</div>
          <div className="flex flex-wrap gap-1.5">
            {(info?.labels ?? []).map(l => (
              <button key={l} onClick={() => toggle(l)}
                className={`px-2 py-1 rounded-full text-[11px] transition-colors ${tags.includes(l) ? 'bg-[#3b3bed] text-white' : 'bg-white/5 text-slate-400 hover:text-white'}`}>
                {l}
              </button>
            ))}
          </div>
          <p className="text-[11px] text-slate-500 mt-3 font-mono leading-relaxed">
            P(C | t₁…tₙ) ∝ P(C) · Π P(tᵢ | C)<br />
            log-space + log-sum-exp for stability; prediction = MAP.
          </p>
        </div>

        <div className="space-y-5">
          {err && <p className="text-xs text-red-400">{err}</p>}
          {hand && (
            <div>
              <div className="text-xs font-semibold text-white mb-2">Hand-crafted CPT (used in the live app)</div>
              <div className="space-y-1.5">
                {Object.entries(hand.posterior).map(([c, p]) => (
                  <Bar key={c} label={c} value={p} color={c === hand.predicted ? '#3b3bed' : '#475569'} />
                ))}
              </div>
            </div>
          )}
          {learned ? (
            <div>
              <div className="text-xs font-semibold text-white mb-2">Learned from labelled events (MLE + Laplace)</div>
              <div className="space-y-1.5">
                {Object.entries(learned.posterior).map(([c, p]) => (
                  <Bar key={c} label={c} value={p} color={c === learned.predicted ? '#10b981' : '#475569'} />
                ))}
              </div>
            </div>
          ) : (
            <p className="text-[11px] text-slate-500">No learned model yet. Train one with <code className="text-slate-300">python experiments/train_nb.py</code> once labelled events are available.</p>
          )}
        </div>
      </div>

      {hand?.evidence?.length > 0 && (
        <div className="mt-5">
          <div className="text-xs font-semibold text-white mb-2">Weight of evidence per tag (log P(tag | COMMITTEE) − log P(tag | INSTITUTIONAL))</div>
          <div className="space-y-1.5">
            {[...hand.evidence].sort((a, b) => Math.abs(b.llr) - Math.abs(a.llr)).map(e => (
              <div key={e.tag} className="flex items-center gap-3 text-xs">
                <span className="w-28 shrink-0 text-slate-300 truncate">{e.tag}</span>
                <div className="flex-1 h-2.5 relative bg-white/5 rounded-full">
                  <div className="absolute top-0 bottom-0 left-1/2 w-px bg-white/20" />
                  <div className="absolute top-0 bottom-0 rounded-full"
                    style={{
                      background: e.llr >= 0 ? '#10b981' : '#6366f1',
                      width: `${Math.min(Math.abs(e.llr) / 1.5, 1) * 50}%`,
                      left: e.llr >= 0 ? '50%' : `${50 - Math.min(Math.abs(e.llr) / 1.5, 1) * 50}%`,
                    }} />
                </div>
                <span className="w-14 text-right tabular-nums text-slate-400">{e.llr.toFixed(2)}</span>
              </div>
            ))}
          </div>
          <div className="flex justify-between text-[10px] text-slate-500 mt-1 px-[7.5rem] pr-14"><span>towards INSTITUTIONAL</span><span>towards COMMITTEE</span></div>
        </div>
      )}
    </Panel>
  )
}

function ClusterPanel({ corpusSize }) {
  const [fetched, setFetched] = useState(null)
  const [fetchErr, setFetchErr] = useState('')
  const [k, setK] = useState('auto')
  const data = corpusSize >= 4 ? fetched : null
  const err = corpusSize >= 4 ? fetchErr : ''

  useEffect(() => {
    if (corpusSize < 4) return
    let cancelled = false
    api(`/api/lab/embedding-map?k=${k}`)
      .then(d => { if (!cancelled) { setFetched(d); setFetchErr('') } })
      .catch(e => { if (!cancelled) { setFetched(null); setFetchErr(e.message) } })
    return () => { cancelled = true }
  }, [corpusSize, k])

  const W = 560, H = 340, pad = 28
  let plot = null
  if (data) {
    const xs = data.points.map(p => p.x), ys = data.points.map(p => p.y)
    const sx = v => pad + ((v - Math.min(...xs)) / (Math.max(...xs) - Math.min(...xs) || 1)) * (W - 2 * pad)
    const sy = v => H - pad - ((v - Math.min(...ys)) / (Math.max(...ys) - Math.min(...ys) || 1)) * (H - 2 * pad)
    plot = data.points.map(p => (
      <g key={p.id}>
        <circle cx={sx(p.x)} cy={sy(p.y)} r="7" fill={CLUSTER_COLORS[p.cluster % 6]} opacity="0.85"><title>{`${p.name}\n${p.tags.join(', ')}`}</title></circle>
        <text x={sx(p.x) + 10} y={sy(p.y) + 3} fontSize="9" fill="#94a3b8">{p.name.length > 16 ? p.name.slice(0, 15) + '…' : p.name}</text>
      </g>
    ))
  }

  return (
    <Panel
      title="Unsupervised theme discovery: Gaussian mixture fitted with EM"
      subtitle="CLIP embeddings (512-D) are projected with PCA; a diagonal-covariance GMM is fitted by our own EM implementation and the number of clusters is chosen by BIC. No labels are used."
      right={
        <select value={k} onChange={e => setK(e.target.value)} className="bg-[#1e1e38] text-xs text-white rounded-lg px-2 py-1.5 border border-white/10">
          <option value="auto">k by BIC</option>
          {[1, 2, 3, 4, 5, 6].map(n => <option key={n} value={n}>k = {n}</option>)}
        </select>
      }
    >
      {corpusSize < 4 && <p className="text-xs text-slate-500">Add at least 4 photos to the corpus above.</p>}
      {err && <p className="text-xs text-red-400">{err}</p>}
      {data && (
        <div className="grid lg:grid-cols-[1.4fr_1fr] gap-6">
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-xl bg-black/20">{plot}</svg>
          <div className="space-y-4 text-xs">
            <div className="grid grid-cols-3 gap-2">
              <Stat label="k chosen" value={`${data.chosen_k}${data.chosen_k === data.auto_k ? ' (BIC)' : ''}`} />
              <Stat label="EM iterations" value={data.em_iterations} />
              <Stat label="PC1 + PC2 var" value={pct(data.explained_variance.reduce((a, b) => a + b, 0), 0)} />
            </div>
            <div>
              <div className="text-slate-500 mb-1">BIC per k (lower is better)</div>
              <div className="flex flex-wrap gap-1.5">
                {Object.entries(data.bic).map(([kk, v]) => (
                  <span key={kk} className={`px-2 py-1 rounded-md tabular-nums ${Number(kk) === data.auto_k ? 'bg-emerald-500/20 text-emerald-300' : 'bg-white/5 text-slate-400'}`}>k={kk}: {v.toFixed(0)}</span>
                ))}
              </div>
            </div>
            <div>
              <div className="text-slate-500 mb-1">Clusters</div>
              {data.clusters.map(c => (
                <div key={c.cluster} className="flex items-center gap-2 py-0.5 text-slate-300">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ background: CLUSTER_COLORS[c.cluster % 6] }} />
                  <span>{c.size} photos</span>
                  <span className="text-slate-500">{c.top_tags.join(', ') || '-'}</span>
                </div>
              ))}
            </div>
            <ReportCard report={{ title: 'EM log-likelihood per iteration', curve: { label: 'Average log-likelihood never decreases', y: data.log_likelihood_curve } }} />
          </div>
        </div>
      )}
    </Panel>
  )
}

/* ----------------------------------------------------------------- DL tab */
function DeepLearningTab({ info, analysis, template, setTemplate, reanalyze }) {
  const [showAll, setShowAll] = useState(false)
  const labels = analysis?.labels ?? []
  const shown = showAll ? labels : labels.slice(0, 10)
  const v = info?.vision, t = info?.text

  return (
    <div className="space-y-6">
      <Panel title="CLIP ViT-B/32 architecture" subtitle="Two transformers trained contrastively so that an image and its caption land close together in one 512-D space.">
        {info ? (
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-2">
            <Stat label="Parameters" value={`${(info.parameters.total / 1e6).toFixed(1)} M`} />
            <Stat label="Image encoder" value={`ViT: ${v.layers}L · ${v.heads}H · d=${v.hidden}`} />
            <Stat label="Patches" value={`${v.patch}×${v.patch} px → ${(v.image_size / v.patch) ** 2} + [CLS]`} />
            <Stat label="Text encoder" value={`Transformer: ${t.layers}L · ${t.heads}H · d=${t.hidden}`} />
            <Stat label="Joint space" value={`${info.embedding_dim}-D, L2-normalised`} />
            <Stat label="Logit scale s" value={info.logit_scale.toFixed(1)} />
            <Stat label="Text context" value={`${t.context} tokens`} />
            <Stat label="Vision / text params" value={`${(info.parameters.vision / 1e6).toFixed(0)} M / ${(info.parameters.text / 1e6).toFixed(0)} M`} />
          </div>
        ) : <p className="text-xs text-slate-500">Loading...</p>}
      </Panel>

      {!analysis ? (
        <p className="text-sm text-slate-400">Analyse a photo above to see its zero-shot scores and what the Vision Transformer attends to.</p>
      ) : (
        <div className="grid lg:grid-cols-2 gap-6">
          <Panel
            title="Zero-shot classification"
            subtitle="p(label) = softmax( s · cosine(image, text(label)) ). No label was ever used to train the model."
            right={
              <select value={template} onChange={e => { setTemplate(e.target.value); reanalyze(e.target.value) }}
                className="bg-[#1e1e38] text-xs text-white rounded-lg px-2 py-1.5 border border-white/10">
                {Object.keys(info?.templates ?? { bare: 1 }).map(k => <option key={k} value={k}>prompt: {k}</option>)}
              </select>
            }
          >
            <div className="space-y-1.5">
              {shown.map(l => <Bar key={l.label} label={l.label} value={l.prob} color={analysis.tags.includes(l.label) ? '#3b3bed' : '#475569'} right={`${pct(l.prob)} · cos ${l.cosine.toFixed(3)}`} />)}
            </div>
            <button onClick={() => setShowAll(s => !s)} className="mt-3 text-[11px] text-indigo-300 hover:text-white">
              {showAll ? 'Show top 10 only' : `Show all ${labels.length} labels`}
            </button>
            <p className="text-[11px] text-slate-500 mt-2">Blue bars are the tags saved by the app (top 5 above 5 %).</p>
          </Panel>

          <Panel title="Where the Vision Transformer looks" subtitle="Attention rollout: head-averaged attention of all 12 layers multiplied together (with residual connections), [CLS] → each of the 49 patches.">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <div className="text-[10px] text-slate-500 mb-1">Input</div>
                <img src={analysis.thumb} alt="input" className="w-full rounded-lg" />
              </div>
              <div>
                <div className="text-[10px] text-slate-500 mb-1">Attention rollout (red = high)</div>
                {analysis.attention?.overlay
                  ? <img src={analysis.attention.overlay} alt="attention" className="w-full rounded-lg" />
                  : <p className="text-xs text-red-400">{analysis.attention?.error || 'Not available'}</p>}
              </div>
            </div>
            <p className="text-[11px] text-slate-500 mt-3">Attention is a plausible explanation, not a proof of causality; mention this in the viva.</p>
          </Panel>

          <Panel title="The embedding" subtitle="The image is now one point on a 512-D unit sphere. This is what gets stored in pgvector.">
            <div className="grid grid-cols-3 gap-2 mb-3">
              <Stat label="Dimensions" value={analysis.embedding.dims} />
              <Stat label="L2 norm" value={analysis.embedding.norm.toFixed(4)} />
              <Stat label="Encode time" value={`${analysis.timings.image_encoder_ms} ms`} />
            </div>
            <div className="font-mono text-[11px] text-slate-400 break-words">[{analysis.embedding.head.join(', ')}, …]</div>
          </Panel>

          <Panel title="Perceptual hash (duplicate detection)" subtitle="pHash: DCT of a 32×32 grey image, keep the low-frequency 8×8, threshold at the median → 64 bits. Duplicate if Hamming distance < 10.">
            <div className="font-mono text-[11px] break-all text-slate-300 mb-2">{analysis.phash.hex}</div>
            <div className="grid grid-cols-16 gap-px" style={{ gridTemplateColumns: 'repeat(16, minmax(0, 1fr))' }}>
              {analysis.phash.bits.split('').map((b, i) => <span key={i} className={`aspect-square rounded-sm ${b === '1' ? 'bg-indigo-400' : 'bg-white/10'}`} />)}
            </div>
            {analysis.phash.nearest_in_corpus && (
              <p className="text-xs text-slate-400 mt-3">
                Nearest photo in corpus: <b className="text-white">{analysis.phash.nearest_in_corpus.name}</b> at Hamming distance{' '}
                <b className={analysis.phash.nearest_in_corpus.distance < 10 ? 'text-red-400' : 'text-emerald-400'}>{analysis.phash.nearest_in_corpus.distance}</b>
                {analysis.phash.nearest_in_corpus.distance < 10 ? ' → duplicate' : ' → distinct'}
              </p>
            )}
          </Panel>
        </div>
      )}
    </div>
  )
}

/* ---------------------------------------------------------------- NLP tab */
function TokenizerPanel() {
  const [text, setText] = useState('students dancing on stage at the cultural fest')
  const [fetched, setOut] = useState(null)
  const [err, setErr] = useState('')
  const out = text.trim() ? fetched : null

  useEffect(() => {
    if (!text.trim()) return
    const t = setTimeout(async () => {
      try {
        setOut(await api('/api/lab/tokenize', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) })); setErr('')
      } catch (e) { setErr(e.message) }
    }, 250)
    return () => clearTimeout(t)
  }, [text])

  return (
    <Panel title="From words to vectors" subtitle="CLIP's text side: byte-pair-encoding tokens → 12-layer transformer → one 512-D vector. The sparse side only sees stemmed words.">
      <input value={text} onChange={e => setText(e.target.value)}
        className="w-full bg-[#3b3bed]/10 rounded-xl px-4 py-2 text-sm text-white outline-none focus:ring-2 focus:ring-[#3b3bed]/50" />
      {err && <p className="text-xs text-red-400 mt-2">{err}</p>}
      {out && (
        <div className="mt-4 space-y-3 text-xs">
          <div>
            <div className="text-slate-500 mb-1">BPE tokens ({out.num_tokens} of {out.context}{out.truncated ? ', truncated' : ''})</div>
            <div className="flex flex-wrap gap-1">
              {out.tokens.map((tk, i) => (
                <span key={i} className="px-1.5 py-0.5 rounded bg-white/5 text-slate-200 font-mono" title={`id ${out.ids[i]}`}>{tk}<span className="text-slate-500"> {out.ids[i]}</span></span>
              ))}
            </div>
          </div>
          <div>
            <div className="text-slate-500 mb-1">Sparse view (lower-cased, stop-words removed, stemmed)</div>
            <div className="flex flex-wrap gap-1">
              {out.word_tokens.map((w, i) => <span key={i} className="px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-200 font-mono">{w}</span>)}
            </div>
          </div>
          <div>
            <div className="text-slate-500 mb-1">Dense view: {out.embedding.dims}-D, norm {out.embedding.norm.toFixed(3)}</div>
            <div className="font-mono text-[11px] text-slate-400 break-words">[{out.embedding.head.join(', ')}, …]</div>
          </div>
        </div>
      )}
    </Panel>
  )
}

const MODES = [
  { key: 'keyword', title: 'Keyword (old search bar)', note: 'substring match on tags / name' },
  { key: 'bm25', title: 'BM25 (sparse)', note: 'term overlap, idf-weighted' },
  { key: 'dense', title: 'CLIP dense', note: 'cosine of embeddings' },
  { key: 'hybrid', title: 'Hybrid (RRF)', note: 'BM25 + dense fused' },
]

function SearchComparePanel({ corpus }) {
  const [query, setQuery] = useState('a garden with colourful flowers')
  const [res, setRes] = useState(null)
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)
  const byId = Object.fromEntries(corpus.map(c => [c.id, c]))

  const run = async () => {
    if (!query.trim()) return
    setBusy(true)
    try {
      setRes(await api('/api/lab/search', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query, k: 6 }) })); setErr('')
    } catch (e) { setRes(null); setErr(e.message) } finally { setBusy(false) }
  }

  return (
    <Panel title="Sparse vs dense retrieval, side by side" subtitle="Same query, same photos, four rankers. Try a phrase that shares no word with any tag and watch the sparse columns go empty while the dense column still ranks the right photo first.">
      <div className="flex gap-2">
        <input value={query} onChange={e => setQuery(e.target.value)} onKeyDown={e => e.key === 'Enter' && run()}
          placeholder="Describe a photo..."
          className="flex-1 bg-[#3b3bed]/10 rounded-xl px-4 py-2 text-sm text-white outline-none focus:ring-2 focus:ring-[#3b3bed]/50" />
        <button onClick={run} disabled={busy || corpus.length === 0}
          className="px-4 rounded-xl bg-[#3b3bed] text-white text-sm font-bold disabled:opacity-40">{busy ? '...' : 'Search'}</button>
      </div>
      {corpus.length === 0 && <p className="text-xs text-slate-500 mt-3">Add photos to the corpus first.</p>}
      {err && corpus.length > 0 && <p className="text-xs text-red-400 mt-3">{err}</p>}
      {res && (
        <>
          <p className="text-[11px] text-slate-500 mt-3 font-mono">query tokens → [{res.query_tokens.join(', ')}]</p>
          <div className="grid md:grid-cols-2 xl:grid-cols-4 gap-4 mt-3">
            {MODES.map(m => (
              <div key={m.key} className="rounded-xl bg-black/20 p-3">
                <div className="text-xs font-bold text-white">{m.title}</div>
                <div className="text-[10px] text-slate-500 mb-2">{m.note}</div>
                {res[m.key].length === 0 && <p className="text-[11px] text-amber-300/80">No matches</p>}
                <div className="space-y-2">
                  {res[m.key].map(r => (
                    <div key={r.id} className="flex items-center gap-2">
                      <span className="text-[10px] text-slate-500 w-3">{r.rank}</span>
                      {byId[r.id] && <img src={byId[r.id].thumb} alt="" className="w-10 h-10 rounded object-cover" />}
                      <div className="min-w-0 flex-1">
                        <div className="text-[11px] text-slate-200 truncate">{r.name}</div>
                        <div className="text-[10px] text-slate-500 tabular-nums">{m.key === 'keyword' ? 'match' : r.score.toFixed(m.key === 'hybrid' ? 4 : 3)}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </Panel>
  )
}

/* ------------------------------------------------------------------- page */
const TABS = [
  { key: 'aml', label: 'AML', sub: 'Probabilistic models' },
  { key: 'dl', label: 'Deep Learning', sub: 'Representations' },
  { key: 'nlp', label: 'NLP', sub: 'Retrieval & language' },
]

export default function LabPage() {
  const [tab, setTab] = useState('aml')
  const [info, setInfo] = useState(null)
  const [serviceError, setServiceError] = useState('')
  const [analysis, setAnalysis] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [analyzeError, setAnalyzeError] = useState('')
  const [template, setTemplate] = useState('bare')
  const [corpus, setCorpus] = useState([])
  const [benchmarks, setBenchmarks] = useState(null)
  const lastFile = useRef(null)
  const fileInput = useRef(null)

  const refreshCorpus = useCallback(async () => {
    try { setCorpus((await api('/api/lab/corpus')).items) } catch { /* shown by banner */ }
  }, [])

  useEffect(() => {
    document.documentElement.classList.add('dark')
    ;(async () => {
      try {
        setInfo(await api('/api/lab/info'))
        setServiceError('')
        refreshCorpus()
        api('/api/lab/benchmarks').then(setBenchmarks).catch(() => {})
      } catch (e) {
        setServiceError(e.message)
      }
    })()
  }, [refreshCorpus])

  const analyze = async (file, tpl = template) => {
    if (!file) return
    lastFile.current = file
    setAnalyzing(true); setAnalyzeError('')
    const fd = new FormData()
    fd.append('image', file)
    fd.append('template', tpl)
    try {
      setAnalysis(await api('/api/lab/analyze', { method: 'POST', body: fd }))
    } catch (e) { setAnalyzeError(e.message) } finally { setAnalyzing(false) }
  }

  return (
    <div className="min-h-screen bg-[#0a0a14] text-slate-100">
      <header className="sticky top-0 z-40 glass px-4 md:px-8 py-3 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-3">
          <div className="bg-[#3b3bed] p-1.5 rounded-lg"><span className="material-symbols-outlined text-white text-2xl">science</span></div>
          <div>
            <h1 className="text-lg font-bold leading-tight text-white">VIT Media Vault · AI Lab</h1>
            <p className="text-[10px] text-slate-500 uppercase tracking-wider">Every model in the project, live</p>
          </div>
        </Link>
        <Link to="/gallery" className="text-sm text-slate-300 hover:text-white">Gallery</Link>
      </header>

      <main className="max-w-6xl mx-auto px-4 md:px-8 py-8 space-y-6">
        {serviceError && (
          <div className="rounded-xl border border-red-500/40 bg-red-500/10 p-4 text-sm">
            <b className="text-red-300">AI service not reachable at {AI_URL}.</b>{' '}
            <span className="text-slate-300">Start it with <code>cd ai-service</code> then <code>python app.py</code>. ({serviceError})</span>
          </div>
        )}

        <Panel
          title="1 · Analyse a photo"
          subtitle="Everything below is computed by the running models: no database, no mock data."
          right={
            <button onClick={() => fileInput.current?.click()} disabled={analyzing}
              className="px-4 py-2 rounded-xl bg-[#3b3bed] text-white text-sm font-bold glow-button disabled:opacity-50">
              {analyzing ? 'Analysing...' : analysis ? 'Choose another photo' : 'Choose a photo'}
            </button>
          }
        >
          <input ref={fileInput} type="file" accept="image/*" hidden onChange={e => { analyze(e.target.files[0]); e.target.value = '' }} />
          {analyzeError && <p className="text-xs text-red-400">{analyzeError}</p>}
          {analysis ? (
            <div className="flex flex-wrap items-center gap-4">
              <img src={analysis.thumb} alt="analysed" className="h-24 rounded-lg" />
              <div className="space-y-2">
                <div className="flex flex-wrap gap-1.5">
                  {analysis.tags.map(t => <span key={t} className="px-2.5 py-1 rounded-full bg-[#3b3bed]/30 text-indigo-100 text-xs font-medium">{t}</span>)}
                </div>
                <p className="text-xs text-slate-400">
                  {analysis.image_size.join(' × ')} px · image encoder {analysis.timings.image_encoder_ms} ms · category{' '}
                  <b className="text-white">{analysis.naive_bayes.handcrafted.predicted}</b> ({pct(analysis.naive_bayes.handcrafted.confidence)})
                </p>
              </div>
            </div>
          ) : (
            <p className="text-xs text-slate-500">Pick any photo. The AML, DL and NLP views below all react to it.</p>
          )}
        </Panel>

        <CorpusPanel corpus={corpus} onChange={refreshCorpus} />

        <div className="flex gap-2 border-b border-white/10">
          {TABS.map(t => (
            <button key={t.key} onClick={() => setTab(t.key)}
              className={`px-4 py-2.5 text-sm font-bold border-b-2 -mb-px transition-colors ${tab === t.key ? 'border-[#3b3bed] text-white' : 'border-transparent text-slate-500 hover:text-slate-300'}`}>
              {t.label}<span className="hidden sm:inline text-[10px] font-normal text-slate-500 ml-2">{t.sub}</span>
            </button>
          ))}
        </div>

        {tab === 'aml' && (
          <div className="space-y-6">
            <NaiveBayesPanel info={info} analysis={analysis} />
            <ClusterPanel corpusSize={corpus.length} />
            <Panel title="Experiments (saved results)">
              <Reports benchmarks={benchmarks} keys={['aml_nb_recovery', 'nb_learned_eval', 'aml_em_gmm']}
                emptyHint="No experiment results yet. Run the scripts in ai-service/experiments (see the notes) and refresh." />
            </Panel>
          </div>
        )}

        {tab === 'dl' && (
          <div className="space-y-6">
            <DeepLearningTab info={info} analysis={analysis} template={template} setTemplate={setTemplate}
              reanalyze={tpl => lastFile.current && analyze(lastFile.current, tpl)} />
            <Panel title="Experiments (saved results)">
              <Reports benchmarks={benchmarks} keys={['dl_cifar10']}
                emptyHint="No experiment results yet. Run experiments/dl_cifar10_benchmark.py and refresh." />
            </Panel>
          </div>
        )}

        {tab === 'nlp' && (
          <div className="space-y-6">
            <TokenizerPanel />
            <SearchComparePanel corpus={corpus} />
            <Panel title="Experiments (saved results)">
              <Reports benchmarks={benchmarks} keys={['nlp_paraphrase', 'nlp_crossmodal']}
                emptyHint="No experiment results yet. Run the nlp_* scripts in ai-service/experiments and refresh." />
            </Panel>
          </div>
        )}
      </main>
    </div>
  )
}
