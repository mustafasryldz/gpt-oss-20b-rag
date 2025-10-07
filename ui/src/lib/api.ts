// src/lib/api.ts
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export type RRFMatch = {
  id?: string;
  doc_id?: string;
  chunk_id?: number;
  source?: string;
  doc_title?: string;
  text: string;
  rrf?: number;
  bm25_score?: number;
  rerank?: number;
  score?: number;
};

export type AnswerResponse = {
  answer: string;
  used_context: RRFMatch[];
  model?: string;
  tokens_per_s?: number | null;
  from_cache?: boolean | null;
};

const headers = { "Content-Type": "application/json; charset=utf-8" };

export async function searchRRF(query: string, k_final = 12): Promise<RRFMatch[]> {
  const r = await fetch(`${API_BASE}/search/rrf`, {
    method: "POST",
    headers,
    body: JSON.stringify({ query, k_final }),
  });
  if (!r.ok) throw new Error(await r.text());
  const j = await r.json();
  return j.matches ?? [];
}

export async function answer(
  query: string,
  useCache = true,
  k_final = 12
): Promise<AnswerResponse> {
  const r = await fetch(`${API_BASE}/answer`, {
    method: "POST",
    headers,
    body: JSON.stringify({ query, k_final, use_cache: useCache }),
  });
  if (!r.ok) throw new Error(await r.text());
  const j = await r.json();

  // alan adları backend’de snake_case; yine de her iki ihtimali karşılayalım
  return {
    answer: j.answer,
    used_context: j.used_context ?? [],
    model: j.model,
    tokens_per_s: j.tokens_per_s ?? j.tokensPerS ?? null,
    from_cache: j.from_cache ?? j.fromCache ?? null,
  };
}

export const Api = { searchRRF, answer };
