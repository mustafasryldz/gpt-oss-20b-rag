import { useState } from "react";

type Props = {
  onSearch: (q: string) => void | Promise<void>;
  onAnswer: (q: string, useCache: boolean) => void | Promise<void>;
  placeholder?: string;
  defaultCache?: boolean;
};

export default function SearchBox({
  onSearch,
  onAnswer,
  placeholder = "Soru veya arama terimi...",
  defaultCache = true,
}: Props) {
  const [q, setQ] = useState("");
  const [useCache, setUseCache] = useState<boolean>(defaultCache);

  const runSearch = () => {
    if (!q.trim()) return;
    onSearch(q.trim());
  };

  const runAnswer = () => {
    if (!q.trim()) return;
    onAnswer(q.trim(), useCache);
  };

  return (
    <div className="flex flex-col md:flex-row md:items-center gap-3">
      <input
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder={placeholder}
        onKeyDown={(e) => e.key === "Enter" && runAnswer()}
        className="flex-1 rounded-lg border border-[#2a2a2b] bg-[#1b1b1c]/70
                   px-4 py-3 text-[18px] text-[#e5e5e5] placeholder-[#777]
                   focus:outline-none focus:ring-2 focus:ring-[#3a3a3a]"
      />

      <div className="flex items-center flex-wrap gap-3">
        <button
          onClick={runSearch}
          className="px-5 py-2 rounded-md bg-[#2f2f2f] border border-[#3a3a3a]
                     hover:bg-[#3a3a3a] transition text-sm md:text-base"
        >
        🔍  Ara
        </button>

        <button
          onClick={runAnswer}
          className="px-5 py-2 rounded-md bg-[#2f2f2f] border border-[#3a3a3a]
                     hover:bg-[#3a3a3a] transition text-sm md:text-base"
        >
        💬  Sor
        </button>

        <label className="inline-flex items-center gap-2 text-sm md:text-base text-[#cfcfcf]">
          <input
            type="checkbox"
            checked={useCache}
            onChange={(e) => setUseCache(e.target.checked)}
            className="h-5 w-5 accent-text-[#4a4a4a]"
          />
          Cache Kullan
        </label>
      </div>
    </div>
  );
}
