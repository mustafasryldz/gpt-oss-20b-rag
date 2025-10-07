// src/App.tsx
import { useState } from "react";
import { Api, type RRFMatch } from "./lib/api";
import SearchBox from "./components/SearchBox";
import Results from "./components/Results";
import TracePanel from "./components/TracePanel";

export default function App() {
  const [hits, setHits] = useState<RRFMatch[]>([]);
  const [answer, setAnswer] = useState<string | undefined>();
  const [loading, setLoading] = useState(false);
  const [fromCache, setFromCache] = useState<boolean | null>(null);
  const [tokensPerS, setTokensPerS] = useState<number | null>(null);

  async function onSearch(q: string) {
    setLoading(true);
    setAnswer(undefined);
    setFromCache(null);
    setTokensPerS(null);
    try {
      const res = await Api.searchRRF(q, 12);
      setHits(res);
    } finally {
      setLoading(false);
    }
  }

  async function onAnswer(q: string, useCache: boolean) {
    setLoading(true);
    try {
      const r = await Api.answer(q, useCache, 12);
      setAnswer(r.answer);
      setHits(r.used_context ?? []);
      setFromCache(r.from_cache ?? null);
      setTokensPerS(typeof r.tokens_per_s === "number" ? r.tokens_per_s : null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-[#1e1e1e] text-[#e5e5e5]">
      <div className="p-6 max-w-6xl mx-auto text-[16px] leading-7 md:text-[17px]">
        <header className="mb-8 text-center">
          <h1 className="text-4xl font-extrabold tracking-tight text-[#d0d0d0]">
            GPT-OSS-20B RAG
          </h1>
          <p className="mt-2 mb-4 text-[#9b9b9b]">
            Retrieval • Yanıt Üretimi • Cache • Prometheus Metrikleri • Hız Bilgisi
          </p>
        </header>

        {/* Arama kutusu */}
        <div className="rounded-2xl border border-[#2a2a2b] bg-[#2a2a2a] p-4 mb-6">
          <SearchBox onSearch={onSearch} onAnswer={onAnswer} />
        </div>

        {/* Alt satır: solda Kaynaklar(+LLM Yanıtı Results içinde), sağda Trace */}
        <div className="grid md:grid-cols-[2fr_1fr] gap-5 items-start">
          <Results
            hits={hits}
            answer={answer}
            loading={loading}
            fromCache={fromCache ?? undefined}
            tokensPerS={tokensPerS ?? undefined}
          />
          <TracePanel hits={hits} />
        </div>
      </div>
    </div>
  );
}
