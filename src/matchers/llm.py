"""LLM-based taxonomy matcher — LLMs4OM framework (arXiv 2404.10317).

Binary classification prompting (yes/no per candidate pair) with
three concept representations (C, CP, CC) and cardinality filtering.

One API call per source node (k candidate pairs per call).
"""

from __future__ import annotations

import json
import os
import string
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from openai import OpenAI

from src.matchers.embedding import EmbeddingMatcher
from src.taxonomy import Alignment, TaxonMatch, Taxonomy

KODIKROUTER_BASE = "https://api.kodikrouter.ru/v1"


@dataclass
class LLMMatchResult:
    """Parsed match result for one source node."""

    source_id: str
    target_id: str  # Empty if no match found.
    confidence: float
    reasoning: str = ""


@dataclass
class PairJudgement:
    """LLM judgement for one (source, candidate) pair."""

    source_id: str
    candidate_id: str
    match: bool
    confidence: float


class LLMMatcher:
    """LLM-based ontology matching via LLMs4OM framework.

    1. Concept representation (C, CP, CC) with preprocessing.
    2. Embedding retrieval (top-k candidates).
    3. Binary classification per source node (one API call per node).
    4. Post-processing: confidence threshold + cardinality filtering.

    Constructor accepts model, api_key, base_url for flexibility.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4.1-mini",
        api_key: str | None = None,
        base_url: str = KODIKROUTER_BASE,
        top_k: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_workers: int = 10,
        use_structured_output: bool = True,
    ):
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("KODIKROUTER_API_KEY", "not-needed"),
            base_url=base_url,
        )
        self.top_k = top_k
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        self.use_structured_output = use_structured_output

        self._embed_matcher = EmbeddingMatcher(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self._embed_cache: dict[tuple[str, str], dict[str, list[tuple[str, float]]]] = {}

        self.last_timing: dict[str, float] = {}

    # ── Concept representations (LLMs4OM §3, step 1) ─────────────────────

    @staticmethod
    def _describe_concept(
        tax: Taxonomy,
        node_id: str,
        mode: str = "concept",
    ) -> str:
        """Build a preprocessed text description for one LLMs4OM input type."""
        node = tax.nodes.get(node_id)
        if node is None:
            return f"[Unknown: {node_id}]"

        parts: list[str] = [f"Name: {node.name}"]

        if node.attributes:
            attr_str = "; ".join(
                f"{k}: {v}" for k, v in sorted(node.attributes.items())
            )
            parts.append(f"Attributes: {attr_str}")
        if node.comment:
            parts.append(f"Comment: {node.comment}")
        if node.disjoint_with:
            dnames = [
                tax.nodes[d].name if d in tax.nodes else d
                for d in node.disjoint_with
            ]
            parts.append(f"Disjoint with: {', '.join(dnames)}")

        if mode == "concept-parent":
            path = tax.get_path(node_id)
            if len(path) > 1:
                parts.append(f"Parent chain: {' -> '.join(path)}")
        elif mode == "concept-children":
            if node.children:
                cnames = [
                    tax.nodes[c].name if c in tax.nodes else c
                    for c in node.children
                ]
                parts.append(f"Children: {', '.join(cnames)}")

        text = "\n".join(parts)
        # LLMs4OM preprocessing: lowercase + remove punctuation.
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text

    # ── Retrieval (LLMs4OM §3, step 2) ───────────────────────────────────

    def _get_top_k_targets(
        self, source: Taxonomy, target: Taxonomy
    ) -> dict[str, list[tuple[str, float]]]:
        """Pre-compute top-k target candidates via embedding cosine similarity."""
        cache_key = (source.name, target.name)
        if cache_key in self._embed_cache:
            return self._embed_cache[cache_key]

        src_ids, src_embs = self._embed_matcher.encode_taxonomy(
            source, description_mode="full"
        )
        tgt_ids, tgt_embs = self._embed_matcher.encode_taxonomy(
            target, description_mode="full"
        )

        import numpy as np
        sim = np.dot(src_embs, tgt_embs.T)

        result: dict[str, list[tuple[str, float]]] = {}
        for i, src_id in enumerate(src_ids):
            top_idx = np.argsort(sim[i])[::-1][: self.top_k]
            result[src_id] = [(tgt_ids[j], float(sim[i, j])) for j in top_idx]

        self._embed_cache[cache_key] = result
        return result

    # ── Binary classification per source node (LLMs4OM §3, step 3) ───────

    RETRY_BASE_DELAY = 1.0  # seconds.
    RETRY_MAX_DELAY = 15.0  # cap for exponential backoff.

    # JSON Schema for structured output (per source node).
    _CLASSIFY_SCHEMA = {
        "type": "json_schema",
        "json_schema": {
            "name": "pair_decisions",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "decisions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "pair": {"type": "integer"},
                                "match": {"type": "boolean"},
                                "confidence": {"type": "number"},
                                "reasoning": {"type": "string"},
                            },
                            "required": ["pair", "match", "confidence", "reasoning"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["decisions"],
                "additionalProperties": False,
            },
        },
    }

    CLASSIFY_PROMPT = """Source concept:
{source_desc}

Classify each candidate pair: do the source and candidate refer to the same entity?

Candidates:
{candidates}"""

    CLASSIFY_PROMPT_JSON = """Source concept:
{source_desc}

Classify each candidate pair: do the source and candidate refer to the same entity?

Candidates:
{candidates}

Output ONLY a JSON object with a "decisions" array. Each entry: {{"pair": <index>, "match": bool, "confidence": 0.0-1.0, "reasoning": "short"}}. No other text."""

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        """Check if an error is retryable (rate limit or transient network)."""
        msg = str(error).lower()
        # 429 rate limit or connection errors.
        if "429" in msg or "rate" in msg:
            return True
        if "connection" in msg or "timeout" in msg or "timed out" in msg:
            return True
        # 500 model errors, 400 bad request, etc. — not retryable.
        return False

    def _classify_with_retry(
        self,
        source: Taxonomy,
        source_id: str,
        target: Taxonomy,
        candidate_ids: list[str],
        mode: str = "concept",
    ) -> tuple[list[PairJudgement], float]:
        """Classify with exponential backoff retry on transient errors.

        Retries only on rate limits (429) and connection errors.
        Server/model errors (500) raise immediately for fallback.

        Returns (judgements, api_seconds) — wall time of the successful call.
        """
        attempt = 0
        while True:
            try:
                t0 = time.perf_counter()
                judgements = self._classify_source_candidates(
                    source, source_id, target, candidate_ids, mode
                )
                dt = time.perf_counter() - t0
                return judgements, dt
            except Exception as e:
                if not self._is_retryable(e):
                    raise
                delay = min(self.RETRY_BASE_DELAY * (2 ** attempt), self.RETRY_MAX_DELAY)
                attempt += 1
                if attempt <= 5 or attempt % 5 == 0:
                    print(f"    Retry {attempt} for {source.nodes[source_id].name} "
                          f"in {delay:.0f}s: {e}", flush=True)
                time.sleep(delay)

    def _classify_source_candidates(
        self,
        source: Taxonomy,
        source_id: str,
        target: Taxonomy,
        candidate_ids: list[str],
        mode: str = "concept",
    ) -> list[PairJudgement]:
        """Classify k candidate pairs for one source node in a single API call."""
        source_desc = self._describe_concept(source, source_id, mode)

        pair_texts: list[str] = []
        for idx, cid in enumerate(candidate_ids):
            cand_desc = self._describe_concept(target, cid, mode)
            pair_texts.append(f"[{idx}] {cand_desc}")

        prompt_template = self.CLASSIFY_PROMPT if self.use_structured_output else self.CLASSIFY_PROMPT_JSON
        prompt = prompt_template.format(
            source_desc=source_desc,
            candidates="\n\n".join(pair_texts),
        )

        kwargs: dict = dict(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an ontology matching expert. Output valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if self.use_structured_output:
            kwargs["response_format"] = self._CLASSIFY_SCHEMA

        response = self.client.chat.completions.create(**kwargs)

        raw = response.choices[0].message.content or "{}"
        # Strip markdown code fences if present (common in non-structured mode).
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}

        entries = parsed.get("decisions", []) if isinstance(parsed, dict) else []

        judgements: list[PairJudgement] = []
        for entry in entries:
            pair_idx = int(entry.get("pair", -1))
            if pair_idx < 0 or pair_idx >= len(candidate_ids):
                continue
            match_val = entry.get("match", False)
            is_match = match_val if isinstance(match_val, bool) else False
            confidence = float(entry.get("confidence", 0.0))
            judgements.append(PairJudgement(
                source_id=source_id,
                candidate_id=candidate_ids[pair_idx],
                match=is_match,
                confidence=confidence,
            ))
        return judgements

    # ── Parallel execution ───────────────────────────────────────────────

    def _classify_all_parallel(
        self,
        source: Taxonomy,
        target: Taxonomy,
        top_k_map: dict[str, list[tuple[str, float]]],
        mode: str,
        desc: str = "",
        position: int = 0,
    ) -> tuple[list[PairJudgement], float]:
        """Classify all source nodes in parallel with ThreadPoolExecutor.

        Returns:
            (all_judgements, wall_clock_api_time).
        """
        from tqdm import tqdm

        src_ids_sorted = sorted(source.nodes.keys())
        all_judgements: list[PairJudgement] = []

        # Build task list: (source_id, candidate_ids).
        tasks: list[tuple[str, list[str]]] = []
        for src_id in src_ids_sorted:
            candidates = top_k_map.get(src_id, [])
            if not candidates:
                continue
            tasks.append((src_id, [cid for cid, _ in candidates]))

        api_t0 = time.perf_counter()

        pbar = tqdm(total=len(tasks), desc=desc, position=position, leave=False, unit="node")

        # Early abort: if the first N failures share the same error, stop.
        first_failures: list[str] = []
        any_success = False
        EARLY_ABORT_THRESHOLD = 3

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_node: dict = {
                executor.submit(
                    self._classify_with_retry,
                    source, src_id, target, cand_ids, mode,
                ): (src_id, cand_ids)
                for src_id, cand_ids in tasks
            }

            for future in as_completed(future_to_node):
                src_id, cand_ids = future_to_node[future]
                try:
                    judgements, dt = future.result()
                except Exception as e:
                    err_msg = str(e)
                    pbar.write(f"  {source.nodes[src_id].name}: FALLBACK ({e})")

                    # Early abort: track first failures for pattern detection.
                    if not any_success:
                        first_failures.append(err_msg)
                        if len(first_failures) >= EARLY_ABORT_THRESHOLD:
                            unique = set(first_failures)
                            if len(unique) == 1:
                                pbar.close()
                                raise RuntimeError(
                                    f"Early abort: first {EARLY_ABORT_THRESHOLD} nodes all failed "
                                    f"with the same error. First error: {first_failures[0]}"
                                ) from e

                    # Fallback: top embedding candidate with low confidence.
                    if cand_ids:
                        best_id = cand_ids[0]
                        judgements = [PairJudgement(
                            source_id=src_id, candidate_id=best_id,
                            match=True, confidence=0.5,
                        )]
                    else:
                        judgements = []
                    all_judgements.extend(judgements)
                    pbar.update(1)
                    continue
                any_success = True
                yes_count = sum(1 for j in judgements if j.match)
                pbar.set_postfix_str(f"{source.nodes[src_id].name} {yes_count}/{len(judgements)} yes {dt:.1f}s")
                pbar.update(1)
                all_judgements.extend(judgements)

        pbar.close()

        api_time = time.perf_counter() - api_t0
        return all_judgements, api_time

    def match(
        self,
        source: Taxonomy,
        target: Taxonomy,
        mode: str = "concept",
        pbar_desc: str = "",
        pbar_position: int = 0,
    ) -> tuple[Alignment, list[LLMMatchResult]]:
        """Match all source nodes using LLMs4OM pipeline.

        One API call per source node, then cardinality filtering across all.
        """
        t_start = time.perf_counter()

        # Step 1: retrieval.
        top_k_map = self._get_top_k_targets(source, target)
        embed_time = time.perf_counter() - t_start
        self.last_timing["embed_time"] = embed_time

        # Step 2: classify all source nodes in parallel.
        all_judgements, api_time = self._classify_all_parallel(
            source, target, top_k_map, mode,
            desc=pbar_desc, position=pbar_position,
        )
        self.last_timing["api_time"] = api_time

        # Step 3: post-processing.
        results = self._postprocess(all_judgements, source)

        alignment = Alignment(
            source=source.name,
            target=target.name,
            matches=[
                TaxonMatch(
                    source_id=r.source_id, target_id=r.target_id,
                    confidence=r.confidence,
                )
                for r in results if r.target_id
            ],
        )

        self.last_timing["total"] = time.perf_counter() - t_start
        return alignment, results

    def _postprocess(
        self,
        judgements: list[PairJudgement],
        source: Taxonomy,
    ) -> list[LLMMatchResult]:
        """LLMs4OM post-processing: confidence threshold + cardinality 1:1."""
        yes_pairs = [j for j in judgements if j.match and j.confidence > 0.7]
        yes_pairs.sort(key=lambda j: j.confidence, reverse=True)

        used_sources: set[str] = set()
        used_targets: set[str] = set()
        assigned: dict[str, tuple[str, float]] = {}

        for j in yes_pairs:
            if j.source_id in used_sources or j.candidate_id in used_targets:
                continue
            used_sources.add(j.source_id)
            used_targets.add(j.candidate_id)
            assigned[j.source_id] = (j.candidate_id, j.confidence)

        results: list[LLMMatchResult] = []
        for src_id in sorted(source.nodes.keys()):
            if src_id in assigned:
                tgt_id, conf = assigned[src_id]
                results.append(LLMMatchResult(
                    source_id=src_id, target_id=tgt_id, confidence=conf,
                ))
            else:
                results.append(LLMMatchResult(
                    source_id=src_id, target_id="", confidence=0.0,
                ))

        return results

    def match_ensemble(
        self,
        source: Taxonomy,
        target: Taxonomy,
    ) -> tuple[Alignment, list[LLMMatchResult]]:
        """Ensemble of three representations (C, CP, CC) — majority vote."""
        modes = ["concept", "concept-parent", "concept-children"]
        all_results: dict[str, list[LLMMatchResult]] = {}
        total_api = 0.0

        for i, mode in enumerate(modes):
            if i > 0:
                # Cooldown between modes to avoid rate limiting.
                time.sleep(3.0)
            _, results = self.match(source, target, mode=mode)
            all_results[mode] = results
            total_api += self.last_timing.get("api_time", 0.0)

        aggregated: list[LLMMatchResult] = []
        for src_id in sorted(source.nodes.keys()):
            votes: dict[str, tuple[int, float]] = {}
            for mode in modes:
                r = next((r for r in all_results[mode] if r.source_id == src_id), None)
                if r and r.target_id:
                    prev = votes.get(r.target_id, (0, 0.0))
                    votes[r.target_id] = (prev[0] + 1, prev[1] + r.confidence)

            if votes:
                best_tgt = max(votes, key=lambda k: (votes[k][0], votes[k][1]))
                count, sum_conf = votes[best_tgt]
                aggregated.append(LLMMatchResult(
                    source_id=src_id, target_id=best_tgt,
                    confidence=sum_conf / count,
                ))
            else:
                aggregated.append(LLMMatchResult(
                    source_id=src_id, target_id="", confidence=0.0,
                ))

        alignment = Alignment(
            source=source.name,
            target=target.name,
            matches=[
                TaxonMatch(
                    source_id=r.source_id, target_id=r.target_id,
                    confidence=r.confidence,
                )
                for r in aggregated if r.target_id
            ],
        )
        self.last_timing = {
            "embed_time": self.last_timing.get("embed_time", 0.0),
            "api_time": total_api,
            "total": total_api + self.last_timing.get("embed_time", 0.0),
        }
        return alignment, aggregated
