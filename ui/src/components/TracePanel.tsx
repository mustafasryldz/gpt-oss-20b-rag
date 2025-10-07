// src/TracePanel.tsx
import type { RRFMatch } from "../lib/api";

export default function TracePanel({ hits }: { hits: RRFMatch[] }) {
  return (
    <div className="rounded-2xl border border-[#2a2a2b] bg-[#242424] p-4 shadow-lg">
      <div className="text-sm font-semibold text-[#c8c8c8] mb-3">Trace</div>
      <div className="text-xs text-[#9b9b9b] mb-2">
        Toplam pasaj: <b>{hits.length}</b>
      </div>
      <ol className="space-y-2 list-decimal ml-5">
        {hits.slice(0, 10).map((h, idx) => (
          <li key={h.id ?? `${h.source}-${idx}`} className="text-xs text-[#d3d3d3] hover:text-[#e5e5e5] cursor-default transition">
            <div className="truncate">
              {h.source}{" "}
              <span className="text-[#7a7a7a]">
                (#{typeof h.chunk_id === "number" ? h.chunk_id : "?"})
              </span>
            </div>
          </li>
        ))}
        {hits.length === 0 && <div className="text-xs text-[#7a7a7a] italic">— Sonuç yok.</div>}
      </ol>
    </div>
  );
}
