const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:5000'

export interface AnalyzeRequest {
  title: string
  content: string
  cascade: boolean
}

export interface ModelResult {
  label: number
  is_clickbait: boolean
  confidence: number
}

export interface TransformerResult extends ModelResult {
  gemini_used: boolean
  spoiler: string | null
}

export interface AnalyzeResponse {
  baseline: ModelResult
  transformer: TransformerResult
}

export async function analyze(req: AnalyzeRequest): Promise<AnalyzeResponse> {
  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error((err as { error?: string }).error ?? `HTTP ${res.status}`)
  }
  return res.json()
}
