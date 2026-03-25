from collections import defaultdict
from typing import List, Dict


RRF_K = 60  # Constant from the original RRF paper (Cormack et al. 2009)


def reciprocal_rank_fusion(result_lists: List[List[Dict]], k: int = RRF_K) -> List[Dict]:
    """
    Merge multiple ranked result lists into one via Reciprocal Rank Fusion.

    Parameters
    ----------
    result_lists : Each inner list is a ranked list of result dicts.
                   Each dict must have a "file_id" key.
    k            : RRF smoothing constant.

    Returns
    -------
    Merged and re-ranked list of result dicts, sorted by fused score.
    """
    rrf_scores: Dict[int, float] = defaultdict(float)
    meta:       Dict[int, Dict]  = {}           # file_id → metadata

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            fid = result["file_id"]
            rrf_scores[fid] += 1.0 / (k + rank)
            if fid not in meta:
                meta[fid] = result  # keep first-seen metadata

    fused = []
    for fid, rrf_score in rrf_scores.items():
        entry = dict(meta[fid])
        entry["score"]    = round(rrf_score, 6)
        entry["strategy"] = "fused"
        fused.append(entry)

    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused


def deduplicate(results: List[Dict]) -> List[Dict]:
    """Remove duplicate file_ids, keeping the highest-scoring occurrence."""
    seen: set = set()
    unique   = []
    for r in sorted(results, key=lambda x: x.get("score", 0), reverse=True):
        if r["file_id"] not in seen:
            seen.add(r["file_id"])
            unique.append(r)
    return unique


def rank_results(
    keyword_results: List[Dict],
    tfidf_results:   List[Dict],
    fuzzy_results:   List[Dict],
    semantic_results: List[Dict],
) -> List[Dict]:
    """
    High-level convenience function: fuse all four result lists and return
    a single deduplicated, ranked list.
    """
    all_lists = [l for l in [keyword_results, tfidf_results, fuzzy_results, semantic_results] if l]
    if not all_lists:
        return []

    fused = reciprocal_rank_fusion(all_lists)
    return deduplicate(fused)