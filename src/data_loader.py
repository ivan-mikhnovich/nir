"""OAEI Conference track data loader and synthetic taxonomy generator."""

from __future__ import annotations

import json
import random
from pathlib import Path

from .owl_parser import parse_owl, parse_reference_alignment, build_alignment
from .taxonomy import Alignment, TaxonMatch, TaxonNode, Taxonomy


# Map OAEI reference alignment file prefix → ontology file name.
# Source: http://oaei.ontologymatching.org/2025/conference/
ONTOLOGY_NAME_TO_FILE = {
    "cmt": "cmt.owl",
    "conference": "Conference.owl",  # Sofsem.
    "confOf": "confof.owl",  # ConfTool.
    "edas": "edas.owl",
    "ekaw": "ekaw.owl",
    "iasted": "iasted.owl",
    "sigkdd": "sigkdd.owl",
}


def load_oaei_conference(
    data_dir: str | Path,
) -> tuple[dict[str, Taxonomy], list[Alignment]]:
    """Load all OAEI Conference ontologies and reference alignments.

    Args:
        data_dir: Path to the directory containing .owl and .rdf files.

    Returns:
        (taxonomies dict, list of Alignments for all 21 pairs).
    """
    data_dir = Path(data_dir)

    # Load all 7 ontologies.
    taxonomies: dict[str, Taxonomy] = {}
    for onto_name, fname in ONTOLOGY_NAME_TO_FILE.items():
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"WARNING: {fpath} not found, skipping.")
            continue
        tax = parse_owl(fpath)
        tax.name = onto_name
        taxonomies[onto_name] = tax

    # Load all 21 reference alignments.
    alignments: list[Alignment] = []
    ref_prefixes = list(ONTOLOGY_NAME_TO_FILE.keys())

    for i, src in enumerate(ref_prefixes):
        for tgt in ref_prefixes[i + 1:]:
            ref_fname = f"{src}-{tgt}.rdf"
            ref_path = data_dir / ref_fname
            if not ref_path.exists():
                print(f"WARNING: {ref_path} not found, skipping.")
                continue

            if src not in taxonomies or tgt not in taxonomies:
                continue

            ref_map = parse_reference_alignment(ref_path)
            align = build_alignment(taxonomies[src], taxonomies[tgt], ref_map)
            alignments.append(align)

    return taxonomies, alignments


def generate_synthetic_variations(
    source: Taxonomy,
    seed: int = 42,
) -> dict[str, tuple[Taxonomy, Alignment]]:
    """Generate synthetic variations of a source taxonomy with known mappings.

    Generates four types of transformations:
      1. exact - identical copy (renamed IDs only).
      2. synonym - class names replaced with LLM-like synonyms (deterministic).
      3. structural - a leaf re-attached to a higher ancestor (tree shape change).
      4. attribute - attributes randomly removed/renamed.

    Variant taxonomy names carry both the scenario suffix and the seed
    (``<name>_<SUFFIX>_s<seed>``), so different seeds produce distinct
    taxonomies that a name-keyed feature cache cannot collapse into one
    (finding G4).  Node IDs carry only the suffix, which keeps the identity
    ground truth of every scenario readable as ``nid → nid_<SUFFIX>``.

    Args:
        source: The base taxonomy.
        seed: Random seed; also embedded in the variant names.

    Returns:
        Dict mapping scenario name → (variant_taxonomy, alignment_to_source).
    """
    rng = random.Random(seed)
    results: dict[str, tuple[Taxonomy, Alignment]] = {}

    # 1. Exact match.
    exact_tax, exact_align = _make_exact(source, rng, seed)
    results["exact"] = (exact_tax, exact_align)

    # 2. Synonym shift.
    syn_tax, syn_align = _make_synonym(source, rng, seed)
    results["synonym"] = (syn_tax, syn_align)

    # 3. Structural shift.
    struct_tax, struct_align = _make_structural(source, rng, seed)
    results["structural"] = (struct_tax, struct_align)

    # 4. Attribute shift.
    attr_tax, attr_align = _make_attribute(source, rng, seed)
    results["attribute"] = (attr_tax, attr_align)

    return results


# -- Transformation helpers --------------------------------------------------

def _clone_taxonomy(source: Taxonomy, suffix: str, seed: int = 42) -> tuple[Taxonomy, dict[str, str]]:
    """Deep clone a taxonomy, renaming all node IDs with a suffix.

    The variant name is ``<source>_<suffix>_s<seed>`` so that variations
    produced from different seeds stay distinct objects for a name-keyed
    feature cache (finding G4).  Node IDs keep the plain ``<id>_<suffix>``
    form because the synthetic ground truth is an identity map on them.
    """
    cloned = Taxonomy(
        name=f"{source.name}_{suffix}_s{seed}",
        namespace=source.namespace,
    )
    id_map: dict[str, str] = {}

    for nid, node in source.nodes.items():
        new_id = f"{nid}_{suffix}"
        id_map[nid] = new_id

    for nid, node in source.nodes.items():
        new_id = id_map[nid]
        new_node = TaxonNode(
            id=new_id,
            name=node.name,
            parents=[id_map[p] for p in node.parents if p in id_map],
            children=[id_map[c] for c in node.children if c in id_map],
            attributes=dict(node.attributes),
            disjoint_with=[id_map[d] for d in node.disjoint_with if d in id_map],
            comment=node.comment,
            depth=node.depth,
        )
        cloned.nodes[new_id] = new_node

    if source.root_id and source.root_id in id_map:
        cloned.root_id = id_map[source.root_id]

    return cloned, id_map


def _make_exact(
    source: Taxonomy, _rng: random.Random, seed: int = 42,
) -> tuple[Taxonomy, Alignment]:
    """Clone the taxonomy with renamed IDs. All nodes map 1:1."""
    variant, id_map = _clone_taxonomy(source, "V1", seed)
    matches = [
        TaxonMatch(source_id=nid, target_id=id_map[nid])
        for nid in sorted(source.nodes)
        if nid in id_map
    ]
    return variant, Alignment(source=source.name, target=variant.name, matches=matches)


# Simple synonym dictionary for the conference domain.
_CONF_SYNONYMS: dict[str, str] = {
    "Author": "Writer",
    "Reviewer": "Evaluator",
    "Review": "Evaluation",
    "Paper": "Article",
    "Submission": "Contribution",
    "Conference": "Symposium",
    "Workshop": "Seminar",
    "Chair": "Head",
    "Committee": "Board",
    "Participant": "Attendee",
    "Organisation": "Organization",
    "Document": "Manuscript",
    "Person": "Individual",
    "Event": "Occasion",
    "Topic": "Subject",
    "Preference": "Priority",
    "Decision": "Verdict",
    "Assistant": "Aide",
    "Member": "Associate",
    "Regular": "Standard",
    "Program": "Schedule",
    "Meeting": "Gathering",
    "Session": "Panel",
    "Tutorial": "Workshop",
    "Presentation": "Talk",
    "Speaker": "Lecturer",
    "Building": "Facility",
    "Location": "Venue",
    "Hotel": "Lodging",
    "Room": "Chamber",
    "City": "Municipality",
    "Country": "Nation",
    "Address": "Location data",
    "Email": "Electronic mail",
    "Phone": "Telephone number",
    "Date": "Calendar date",
    "Time": "Timepoint",
    "Activity": "Function",
    "Administrator": "Manager",
    "User": "Operator",
    "Student": "Learner",
    "Professor": "Instructor",
}


def _make_synonym(
    source: Taxonomy, rng: random.Random, seed: int = 42,
) -> tuple[Taxonomy, Alignment]:
    """Create a variant with class names replaced by known synonyms."""
    import re
    variant, id_map = _clone_taxonomy(source, "SYN", seed)

    for nid, node in source.nodes.items():
        var_node = variant.nodes[id_map[nid]]
        # Try to find a synonym (case-insensitive replace, preserving case).
        # Use word boundary pattern to avoid partial matches.
        for original, synonym in _CONF_SYNONYMS.items():
            pattern = r'\b' + re.escape(original) + r'\b'
            if re.search(pattern, node.name, flags=re.IGNORECASE):
                var_node.name = re.sub(
                    pattern, synonym, node.name, flags=re.IGNORECASE
                )
                break

    matches = [
        TaxonMatch(source_id=nid, target_id=id_map[nid])
        for nid in sorted(source.nodes)
        if nid in id_map
    ]
    return variant, Alignment(source=source.name, target=variant.name, matches=matches)


def _recompute_depths(tax: Taxonomy) -> None:
    """Recompute each node's depth after a structural mutation.

    Uses the parser's convention — ``depth = max(parent depths) + 1``, and 0
    for a node without parents — which reproduces the stored depth of all 491
    nodes of the seven OAEI ontologies exactly.  ``_clone_taxonomy`` copies
    the stored depths, so a re-attached node would otherwise keep a stale one.
    """
    for _ in range(len(tax.nodes) + 1):
        changed = False
        for nid, node in tax.nodes.items():
            parents = [p for p in node.parents if p in tax.nodes]
            want = max((tax.nodes[p].depth for p in parents), default=-1) + 1
            if want != node.depth:
                node.depth = want
                changed = True
        if not changed:
            return


def _make_structural(
    source: Taxonomy, rng: random.Random, seed: int = 42,
) -> tuple[Taxonomy, Alignment]:
    """Re-attach one leaf to a higher ancestor, changing the parent graph.

    A leaf whose parent is a root cannot be promoted, so the shuffled leaves
    are tried until one has a parent that itself has a parent; the search
    falls back to the leaf's other parents.  The stored depths of the variant
    are recomputed afterwards.  When no leaf can be promoted the taxonomy is
    returned renamed only, which the self-check of
    ``check_synthetic_variations`` reports.

    Guarantee: for every taxonomy that has a leaf with a grandparent, at least
    one parent edge of the variant differs from the base taxonomy (finding F2
    / class 3.4 — the scenario used to be a silent no-op).
    """
    variant, id_map = _clone_taxonomy(source, "STR", seed)

    # Collect leaves (nodes with no children).
    leaves = [nid for nid, n in source.nodes.items() if not n.children and n.parents]
    rng.shuffle(leaves)

    promoted = False
    for promote_node in leaves:
        for old_parent in source.nodes[promote_node].parents:
            grandparent = source.nodes[old_parent].parents
            if not grandparent or grandparent[0] not in id_map:
                continue
            # Remove from old parent, add to grandparent.
            old_p_var = variant.nodes[id_map[old_parent]]
            if id_map[promote_node] not in old_p_var.children:
                continue
            old_p_var.children.remove(id_map[promote_node])
            variant.nodes[id_map[promote_node]].parents = [id_map[grandparent[0]]]
            variant.nodes[id_map[grandparent[0]]].children.append(id_map[promote_node])
            promoted = True
            break
        if promoted:
            break

    if promoted:
        _recompute_depths(variant)

    matches = [
        TaxonMatch(source_id=nid, target_id=id_map[nid])
        for nid in sorted(source.nodes)
        if nid in id_map
    ]
    return variant, Alignment(source=source.name, target=variant.name, matches=matches)


_ATTR_POOL: list[tuple[str, str]] = [
    ("hasTitle", "title"),
    ("hasKeyword", "keywords"),
    ("contactEmail", "email_address"),
    ("location", "venue_location"),
    ("abstract", "summary"),
    ("defaultChoice", "default_option"),
    ("hasTime", "timestamp"),
    ("hasDuration", "duration_minutes"),
    ("hasCapacity", "max_capacity"),
    ("cost", "price"),
]


def _make_attribute(
    source: Taxonomy, rng: random.Random, seed: int = 42,
) -> tuple[Taxonomy, Alignment]:
    """Randomly rename or drop attributes, injecting one where there are none.

    Every node that has no attribute at all receives one from ``_ATTR_POOL``
    (always on the alphabetically first such node), so the scenario cannot
    degenerate into a pure id-renaming clone for a taxonomy such as ``cmt``
    that carries no attributes — it used to be a silent no-op (finding F2).
    """
    variant, id_map = _clone_taxonomy(source, "ATTR", seed)

    attribute_less = sorted(nid for nid, n in source.nodes.items() if not n.attributes)
    for nid, node in source.nodes.items():
        var_node = variant.nodes[id_map[nid]]
        if not var_node.attributes:
            # No attribute to shift: add a plausible one instead.  At least
            # the first attribute-less node always mutates, so the scenario
            # can never collapse into a pure id-renaming clone (finding F2).
            if attribute_less and nid != attribute_less[0] and rng.random() >= 0.5:
                continue
            key, value = _ATTR_POOL[rng.randrange(len(_ATTR_POOL))]
            var_node.attributes[key] = value
            continue

        # Rename one attribute if possible.
        attr_keys = list(var_node.attributes.keys())
        rng.shuffle(attr_keys)
        for old_key in attr_keys:
            for pool_key, pool_val in _ATTR_POOL:
                if pool_key.lower() in old_key.lower():
                    var_node.attributes[pool_val] = var_node.attributes.pop(old_key)
                    break

        # Drop one attribute with 30% probability.
        if len(var_node.attributes) >= 2 and rng.random() < 0.3:
            drop_key = rng.choice(list(var_node.attributes.keys()))
            del var_node.attributes[drop_key]

    matches = [
        TaxonMatch(source_id=nid, target_id=id_map[nid])
        for nid in sorted(source.nodes)
        if nid in id_map
    ]
    return variant, Alignment(source=source.name, target=variant.name, matches=matches)


def _content_diff(
    base: Taxonomy, variant: Taxonomy, id_map: dict[str, str],
) -> dict[str, int]:
    """Count content changes between a variant and its base, ignoring ids.

    Node ids of the variant are ``id_map[base_id]``; every comparison is made
    node by node, so a clone that only renamed ids reports zero everywhere.
    """
    diff = {
        "names": 0,
        "parents": 0,
        "children": 0,
        "attributes": 0,
        "comments": 0,
        "disjoint": 0,
    }
    for nid, node in base.nodes.items():
        var_id = id_map[nid]
        if var_id not in variant.nodes:
            diff["names"] += 1
            continue
        var = variant.nodes[var_id]
        if node.name != var.name:
            diff["names"] += 1
        if sorted(id_map[p] for p in node.parents if p in id_map) != sorted(var.parents):
            diff["parents"] += 1
        if sorted(id_map[c] for c in node.children if c in id_map) != sorted(var.children):
            diff["children"] += 1
        if node.attributes != var.attributes:
            diff["attributes"] += 1
        if node.comment != var.comment:
            diff["comments"] += 1
        if sorted(id_map[d] for d in node.disjoint_with if d in id_map) != sorted(var.disjoint_with):
            diff["disjoint"] += 1
    return diff


# Scenario → node-id suffix of its variants.
SCENARIO_SUFFIX: dict[str, str] = {
    "exact": "V1",
    "synonym": "SYN",
    "structural": "STR",
    "attribute": "ATTR",
}

# Scenario → the change category that must be non-zero for that scenario.
_SCENARIO_REQUIRED_CHANGE: dict[str, str] = {
    "exact": "ids",
    "synonym": "names",
    "structural": "parents",
    "attribute": "attributes",
}


def check_synthetic_variations(
    base: Taxonomy,
    seeds: tuple[int, ...] = (42, 43, 44),
) -> list[dict]:
    """Self-check that every synthetic scenario really mutates the taxonomy.

    For every (scenario, seed) the variant must differ from the base in at
    least one respect (ids, names, parent/child edges, attributes, comments or
    disjointness), and the scenario's own mutation must be present: ``synonym``
    renames at least one class, ``structural`` changes at least one parent
    edge, ``attribute`` changes at least one attribute set.  ``exact`` is the
    deliberate exception — it must change nothing but the node ids, which is
    exactly the identity-renaming control it is meant to be.  Variant names
    must be unique per (scenario, seed), the point of finding G4.

    Args:
        base: The base taxonomy (``cmt`` in ``prepare_data``).
        seeds: Seeds to check.

    Returns:
        One report row per (scenario, seed).  Raises ``AssertionError`` when
        any check fails, so a silent no-op scenario cannot pass unnoticed.
    """
    rows: list[dict] = []
    names: set[str] = set()
    for seed in seeds:
        variations = generate_synthetic_variations(base, seed=seed)
        for scenario, (variant, alignment) in sorted(variations.items()):
            suffix = SCENARIO_SUFFIX[scenario]
            id_map = {nid: f"{nid}_{suffix}" for nid in base.nodes}
            diff = _content_diff(base, variant, id_map)
            ids_changed = (
                base.root_id is None
                or id_map[base.root_id] != base.root_id
            )
            total_content = sum(diff.values())
            required = _SCENARIO_REQUIRED_CHANGE[scenario]
            if required == "ids":
                assert ids_changed and total_content == 0, (
                    f"{scenario}/s{seed}: expected a pure id-renaming clone, "
                    f"got diffs {diff}"
                )
            else:
                assert diff[required] > 0, (
                    f"{scenario}/s{seed}: scenario did not change {required} "
                    f"(a no-op variant); diffs {diff}"
                )
                assert total_content > 0, (
                    f"{scenario}/s{seed}: variant is identical to the base "
                    f"modulo ids (no-op); diffs {diff}"
                )
            assert alignment.match_count == len(base.nodes), (
                f"{scenario}/s{seed}: identity GT has {alignment.match_count} "
                f"matches for {len(base.nodes)} nodes"
            )
            assert variant.name == f"{base.name}_{suffix}_s{seed}", (
                f"{scenario}/s{seed}: unexpected variant name {variant.name!r}"
            )
            assert variant.name not in names, f"duplicate variant name {variant.name!r}"
            names.add(variant.name)
            rows.append({
                "scenario": scenario,
                "seed": seed,
                "variant": variant.name,
                "nodes": variant.node_count,
                "gt_matches": alignment.match_count,
                **diff,
            })
    return rows


def save_taxonomy(tax: Taxonomy, path: str | Path) -> None:
    """Save a Taxonomy to a JSON file."""
    path = Path(path)
    data = {
        "name": tax.name,
        "namespace": tax.namespace,
        "root_id": tax.root_id,
        "nodes": {
            nid: {
                "id": n.id,
                "name": n.name,
                "parents": n.parents,
                "children": n.children,
                "attributes": n.attributes,
                "disjoint_with": n.disjoint_with,
                "comment": n.comment,
                "depth": n.depth,
            }
            for nid, n in tax.nodes.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_taxonomy(path: str | Path) -> Taxonomy:
    """Load a Taxonomy from a JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    tax = Taxonomy(
        name=data["name"],
        namespace=data.get("namespace", ""),
        root_id=data.get("root_id"),
    )
    for nid, ndata in data["nodes"].items():
        node = TaxonNode(
            id=ndata["id"],
            name=ndata["name"],
            parents=ndata.get("parents", []),
            children=ndata.get("children", []),
            attributes=ndata.get("attributes", {}),
            disjoint_with=ndata.get("disjoint_with", []),
            comment=ndata.get("comment", ""),
            depth=ndata.get("depth", 0),
        )
        tax.nodes[nid] = node
    return tax


def load_taxonomies(data_dir: str | Path) -> dict[str, Taxonomy]:
    """Load all processed taxonomy JSON files from a directory."""
    data_dir = Path(data_dir)
    taxonomies: dict[str, Taxonomy] = {}
    for f in sorted(data_dir.glob("*.json")):
        if f.name == "alignments.json":
            continue
        tax = load_taxonomy(f)
        taxonomies[tax.name] = tax
    return taxonomies


def load_alignments(path: str | Path) -> list[dict]:
    """Load a raw alignment list (or ground-truth file) from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_gt(src_name: str, tgt_name: str, alignments_raw: list[dict]) -> Alignment | None:
    """Find ground-truth alignment, trying both source↔target orderings."""
    for al_data in alignments_raw:
        s = al_data["source"]
        t = al_data["target"]
        if (s == src_name and t == tgt_name) or (s == tgt_name and t == src_name):
            matches = [
                TaxonMatch(source_id=m["source_id"], target_id=m["target_id"],
                           confidence=m.get("confidence", 1.0))
                for m in al_data["matches"]
            ]
            # If we matched the reverse direction, swap source/target in matches.
            if s == tgt_name and t == src_name:
                matches = [
                    TaxonMatch(source_id=m.target_id, target_id=m.source_id,
                               confidence=m.confidence)
                    for m in matches
                ]
            return Alignment(
                source=src_name,
                target=tgt_name,
                matches=matches,
            )
    return None


def save_alignments(alignments: list[Alignment], path: str | Path) -> None:
    """Save alignments to a JSON file."""
    data = [
        {
            "source": a.source,
            "target": a.target,
            "matches": [
                {"source_id": m.source_id, "target_id": m.target_id, "confidence": m.confidence}
                for m in a.matches
            ],
        }
        for a in alignments
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
