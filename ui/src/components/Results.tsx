import type { RRFMatch } from "../lib/api";

export default function Results({
  hits,
  answer,
  loading,
  fromCache,
  tokensPerS,
}: {
  hits: RRFMatch[];
  answer?: string;
  loading?: boolean;
  fromCache?: boolean | null;
  tokensPerS?: number | null;
}) {
  const total = hits?.length ?? 0;

  return (
    <div className="space-y-6">
      {loading && <div className="text-[#cfcfcf]">Cevaplanıyor…</div>}

      {answer && (
        <div className="p-5 border border-[#2a2a2b] rounded-xl bg-[#1e1e1e]">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between text-sm text-[#9b9b9b] mb-3 gap-1">
            <div>
              Cache: <b>{fromCache ? "Evet" : "Hayır"}</b>
            </div>
            {typeof tokensPerS === "number" && (
              <div>
                tokens/s: <b>{tokensPerS.toFixed(2)}</b>
              </div>
            )}
          </div>

          <div className="mt-4">
            <div className="font-bold text-[#9b9b9b] text-lg mb-2">LLM Yanıtı:</div>
            <div className="whitespace-pre-wrap leading-relaxed text-[17px] text-slate-100">
              {answer}
            </div>
          </div>
        </div>
      )}

      {/* Kaynaklar */}
      <div>
        <div className="mt-2">
          <div className="font-semibold text-[#cfcfcf] mb-2">
            Kaynaklar {total > 0 && <span className="text-[#7a7a7a]">({total})</span>}
          </div>
        </div>

        {total === 0 ? (
          <div className="text-sm text-[#8a8a8a]">— Sonuç yok.</div>
        ) : (
          <div className="space-y-4">
            {hits.map((h, i) => {
              const score = h.score ?? h.rrf ?? h.bm25_score ?? h.rerank;
              const key = h.id ?? `${h.source}-${h.chunk_id}-${i}`;
              return (
                <div
                  key={key}
                  className="p-4 border border-[#2a2a2b] rounded-xl bg-[#1e1e1e]"
                >
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <div className="text-xs text-[#9b9b9b]">
                      {h.source}
                      {typeof h.chunk_id === "number" && (
                        <span className="ml-1 text-[#7a7a7a]">#{h.chunk_id}</span>
                      )}
                    </div>
                    <div className="text-xs text-[#9b9b9b]">
                      skor: {typeof score === "number" ? score.toFixed(3) : "-"}
                    </div>
                  </div>
                  <div className="whitespace-pre-wrap leading-relaxed text-[#e0e0e0]">
                    {h.text}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
