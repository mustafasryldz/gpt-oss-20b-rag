from typing import List, Dict
import math

def recall_at_k(pred_sources: List[str], gold_any: List[str], k: int) -> float:
    topk = pred_sources[:k]
    return 1.0 if any(src in topk for src in gold_any) else 0.0

def mrr_at_k(pred_sources: List[str], gold_any: List[str], k: int) -> float:
    topk = pred_sources[:k]
    for idx, src in enumerate(topk, start=1):
        if src in gold_any:
            return 1.0 / idx
    return 0.0
