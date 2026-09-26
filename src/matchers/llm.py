"""LLM-based taxonomy matcher — LLMs4OM framework (arXiv 2404.10317).

Binary classification prompting (yes/no per candidate pair) with three concept
representations (C, CP, CC) and cardinality filtering.

Cost model.  One API request carries all k candidate judgements of one source
node (a JSON array, see `_classify_source_candidates`), so a pair costs |S|
requests per representation and 3·|S| requests in total; k does not enter the
request count.  The token count is O(k·|S|), because every request contains k
candidate descriptions.  `_classify_all_parallel` records the exact request
count in `last_timing["requests"]`.

Deliberate deviations from the paper:

* The paper verbalises one (source, candidate) pair per request; here the k
  pairs of a source node are batched into one request.
* The paper's high-precision step `S_ir > 0.9` (force a match when the
  retrieval similarity is high) is NOT implemented: `retrieve_candidates`
  computes the similarities, `_postprocess` ignores them.
* Retrieval uses the full textual description for both sides, while the prompt
  uses the C/CP/CC representation; the paper applies one representation to
  both stages.
* Every representation starts from name + attributes + comment + `Disjoint
  with`; CP adds one shortest ancestor chain, CC the child names.  C is
  therefore not name-only.

An ensemble over the three representations is this work's own addition
(`ensemble_candidates`); LLMs4OM itself evaluates single representations and
has no ensemble.
"""

from __future__ import annotations

import json
import os
import string
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from statistics import mean

from openai import OpenAI

from src.matching_rules import Candidate, greedy_injective
from src.taxonomy import Alignment, TaxonMatch, Taxonomy

KODIKROUTER_BASE = "https://api.kodikrouter.ru/v1"

# The three concept representations of LLMs4OM §3, step 1.
ENSEMBLE_MODES: tuple[str, ...] = ("concept", "concept-parent", "concept-children")

# A target must be voted for by at least this many representations to be kept.
ENSEMBLE_MIN_VOTES = 2

# LLMs4OM step 1: the confidence cut on the "yes" class.
CONFIDENCE_THRESHOLD = 0.7

# Confidence of the embedding fallback used when a request fails.  It must
# stay above CONFIDENCE_THRESHOLD, otherwise the fallback candidate is dropped
# again by `_postprocess` and the failure silently becomes an empty cell.
FALLBACK_CONFIDENCE = 0.75


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
    fallback: bool = False  # True when the judgement came from the embedding fallback.


def ensemble_candidates(
    per_mode: dict[str, Iterable[Candidate]],
    min_votes: int = ENSEMBLE_MIN_VOTES,
) -> list[Candidate]:
    """Combine per-representation predictions into one 1:1 candidate list.

    A target is accepted for a source node only when at least `min_votes` of
    the representations vote for it.  The accepted candidates carry the mean
    confidence of their voters and are then reduced to an injective mapping by
    `greedy_injective`, so the ensemble never reuses a source or a target and
    does not depend on the input order.  Only the predictions of the single
    representations are read — no API request is made — so the ensemble can be
    recomputed from cached per-mode predictions.
    """
    votes: dict[tuple[str, str], list[float]] = {}
    for candidates in per_mode.values():
        for src, tgt, conf in candidates:
            votes.setdefault((src, tgt), []).append(conf)

    accepted: list[Candidate] = [
        (src, tgt, mean(confs))
        for (src, tgt), confs in votes.items()
        if len(confs) >= min_votes
    ]
    return greedy_injective(accepted)


def aggregate_mode_f1(
    per_pair_modes: dict[str, dict[str, float]],
    per_pair_ensemble: dict[str, float] | None = None,
) -> dict[str, float]:
    """Summarise per-pair, per-mode F1 into published aggregation rules.

    `mean_over_modes` averages the three representations, per pair and then
    over pairs.  `loo_mode_selection` picks, for every pair, the representation
    with the best mean over the *other* pairs and applies it to that pair, so
    the pair's own ground truth never selects its representation.
    `oracle_max` is the per-pair maximum over the representations; it needs the
    pair's own ground truth and is therefore an oracle, not a deployable rule.
    `ensemble` is added when the per-pair ensemble F1 values are given.
    """
    pairs = sorted(per_pair_modes)
    if not pairs:
        return {}
    modes = sorted({m for d in per_pair_modes.values() for m in d})

    mean_over_modes = mean(
        mean(per_pair_modes[p][m] for m in modes if m in per_pair_modes[p])
        for p in pairs
    )

    loo_values: list[float] = []
    for held_out in pairs:
        others = [p for p in pairs if p != held_out]
        best_mode = max(
            modes,
            key=lambda m: mean(per_pair_modes[p][m] for p in others if m in per_pair_modes[p]),
        )
        loo_values.append(per_pair_modes[held_out].get(best_mode, 0.0))

    aggregates = {
        "mean_over_modes": mean_over_modes,
        "loo_mode_selection": mean(loo_values),
        "oracle_max": mean(max(per_pair_modes[p].values()) for p in pairs),
        "n_pairs": len(pairs),
    }
    if per_pair_ensemble is not None:
        values = [per_pair_ensemble[p] for p in pairs if p in per_pair_ensemble]
        aggregates["ensemble"] = mean(values) if values else 0.0
    return aggregates


class LLMMatcher:
    """LLM-based ontology matching via LLMs4OM framework.

    1. Concept representation (C, CP, CC) with preprocessing.
    2. Embedding retrieval (top-k candidates, full descriptions).
    3. Binary classification per source node (one API request per node).
    4. Post-processing: confidence threshold + cardinality filtering (1:1).
    5. Optional ensemble over the three representations
       (`ensemble_candidates`), free of API calls once the modes are cached.

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

        # The MiniLM matcher and its model are created on first use, so a
        # BM25-only run never loads a model (see the `_embed_matcher` property).
        self._embed_matcher_instance = None
        self._embed_cache: dict[tuple[str, str], dict[str, list[tuple[str, float]]]] = {}

        self.last_timing: dict[str, float] = {}
        self._retry_count = 0  # Retries issued by the last classification sweep.

    @property
    def _embed_matcher(self):
        """MiniLM matcher for candidate retrieval, created on first access."""
        if self._embed_matcher_instance is None:
            from src.matchers.embedding import EmbeddingMatcher

            self._embed_matcher_instance = EmbeddingMatcher(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        return self._embed_matcher_instance

    # ── Concept representations (LLMs4OM §3, step 1) ─────────────────────

    @staticmethod
    def _describe_concept(
        tax: Taxonomy,
        node_id: str,
        mode: str = "concept",
    ) -> str:
        """Build a preprocessed text description for one LLMs4OM input type.

        Every mode starts from the node name plus all characteristics that are
        available — attributes, comment and `Disjoint with` targets — so C is
        not name-only.  `concept-parent` appends one shortest ancestor chain
        and `concept-children` the child names; neither replaces the base part.
        """
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
            path = tax.get_representative_chain(node_id)
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

    def retrieve_candidates(
        self, source: Taxonomy, target: Taxonomy
    ) -> dict[str, list[tuple[str, float]]]:
        """Pre-compute top-k target candidates via embedding cosine similarity.

        Both sides are encoded with `description_mode="full"`, so retrieval and
        prompting use different texts (the prompt uses C/CP/CC; the paper uses
        one representation for both stages).  The returned similarities are
        kept for the ceiling analysis; the LLMs4OM high-precision step
        (`S_ir > 0.9`) that would consume them is not implemented.
        """
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
                self._retry_count += 1
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
        self._retry_count = 0

        # Build task list: (source_id, candidate_ids).
        tasks: list[tuple[str, list[str]]] = []
        for src_id in src_ids_sorted:
            candidates = top_k_map.get(src_id, [])
            if not candidates:
                continue
            tasks.append((src_id, [cid for cid, _ in candidates]))

        # One request per source node with at least one candidate.
        self.last_timing["requests"] = len(tasks)

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

                    # Fallback: top embedding candidate; the confidence is
                    # above CONFIDENCE_THRESHOLD so the candidate survives
                    # `_postprocess` instead of becoming a silent empty cell.
                    if cand_ids:
                        best_id = cand_ids[0]
                        judgements = [PairJudgement(
                            source_id=src_id, candidate_id=best_id,
                            match=True, confidence=FALLBACK_CONFIDENCE,
                            fallback=True,
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
        self.last_timing["judgements"] = len(all_judgements)
        self.last_timing["retries"] = self._retry_count
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
        top_k_map = self.retrieve_candidates(source, target)
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
        """LLMs4OM post-processing: confidence threshold + cardinality 1:1.

        Step 1 keeps the "yes" judgements above `CONFIDENCE_THRESHOLD`; the
        paper's high-precision step `S_ir > 0.9` is not implemented (see the
        module docstring).  Step 3 reduces the survivors to a 1:1 mapping with
        `greedy_injective`, whose tie-break is the source and target id: the
        result therefore does not depend on the order in which the parallel
        classifications finished, unlike the previous arrival-order greedy.
        """
        candidates: list[Candidate] = [
            (j.source_id, j.candidate_id, j.confidence)
            for j in judgements
            if j.match and j.confidence > CONFIDENCE_THRESHOLD
        ]
        assigned = {src: (tgt, conf) for src, tgt, conf in greedy_injective(candidates)}

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
        mode_results: dict[str, list[LLMMatchResult]] | None = None,
    ) -> tuple[Alignment, list[LLMMatchResult]]:
        """Ensemble of the three representations via `ensemble_candidates`.

        Runs the three representations itself when `mode_results` is not given
        (three sweeps, the last one's timing), otherwise it combines the
        predictions passed in — the demo path and the cache recomputation use
        that form and make no API calls at all.
        """
        if mode_results is None:
            mode_results = {}
            for i, mode in enumerate(ENSEMBLE_MODES):
                if i > 0:
                    # Cooldown between modes to avoid rate limiting.
                    time.sleep(3.0)
                _, results = self.match(source, target, mode=mode)
                mode_results[mode] = results

        per_mode: dict[str, list[Candidate]] = {
            mode: [
                (r.source_id, r.target_id, r.confidence)
                for r in results if r.target_id
            ]
            for mode, results in mode_results.items()
        }
        selected = ensemble_candidates(per_mode)

        alignment = Alignment(
            source=source.name,
            target=target.name,
            matches=[
                TaxonMatch(source_id=src, target_id=tgt, confidence=conf)
                for src, tgt, conf in selected
            ],
        )
        results_by_source = {
            src: LLMMatchResult(source_id=src, target_id=tgt, confidence=conf)
            for src, tgt, conf in selected
        }
        aggregated = [
            results_by_source.get(
                src_id, LLMMatchResult(source_id=src_id, target_id="", confidence=0.0)
            )
            for src_id in sorted(source.nodes.keys())
        ]
        return alignment, aggregated
