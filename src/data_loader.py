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
      3. structural - some nodes merged/split to change tree shape.
      4. attribute - attributes randomly removed/renamed.

    Args:
        source: The base taxonomy.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping scenario name → (variant_taxonomy, alignment_to_source).
    """
    rng = random.Random(seed)
    results: dict[str, tuple[Taxonomy, Alignment]] = {}

    # 1. Exact match.
    exact_tax, exact_align = _make_exact(source, rng)
    results["exact"] = (exact_tax, exact_align)

    # 2. Synonym shift.
    syn_tax, syn_align = _make_synonym(source, rng)
    results["synonym"] = (syn_tax, syn_align)

    # 3. Structural shift.
    struct_tax, struct_align = _make_structural(source, rng)
    results["structural"] = (struct_tax, struct_align)

    # 4. Attribute shift.
    attr_tax, attr_align = _make_attribute(source, rng)
    results["attribute"] = (attr_tax, attr_align)

    return results


# -- Transformation helpers --------------------------------------------------

def _clone_taxonomy(source: Taxonomy, suffix: str) -> Taxonomy:
    """Deep clone a taxonomy, renaming all node IDs with a suffix."""
    cloned = Taxonomy(
        name=f"{source.name}_{suffix}",
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
    source: Taxonomy, _rng: random.Random,
) -> tuple[Taxonomy, Alignment]:
    """Clone the taxonomy with renamed IDs. All nodes map 1:1."""
    variant, id_map = _clone_taxonomy(source, "V1")
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
    source: Taxonomy, rng: random.Random,
) -> tuple[Taxonomy, Alignment]:
    """Create a variant with class names replaced by known synonyms."""
    import re
    variant, id_map = _clone_taxonomy(source, "SYN")

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


def _make_structural(
    source: Taxonomy, rng: random.Random,
) -> tuple[Taxonomy, Alignment]:
    """Introduce structural changes: merge some sibling nodes, split others.

    Strategy:
      - Pick 1-2 leaf nodes and "promote" them (make them children of a higher ancestor).
      - Merge 1-2 pairs of sibling leaves into a single node.
    """
    variant, id_map = _clone_taxonomy(source, "STR")

    # Collect leaves (nodes with no children).
    leaves = [nid for nid, n in source.nodes.items() if not n.children and n.parents]
    if len(leaves) < 3:
        # Not enough leaves for meaningful structural changes; return as-is.
        matches = [
            TaxonMatch(source_id=nid, target_id=id_map[nid])
            for nid in sorted(source.nodes)
            if nid in id_map
        ]
        return variant, Alignment(source=source.name, target=variant.name, matches=matches)

    rng.shuffle(leaves)

    # 1. Promote: move one leaf up the tree by 1-2 levels.
    promote_node = leaves[0]
    promote_src = source.nodes[promote_node]
    promote_var = variant.nodes[id_map[promote_node]]

    if promote_src.parents:
        old_parent = promote_src.parents[0]
        grandparent = source.nodes[old_parent].parents
        if grandparent and grandparent[0] in id_map:
            # Remove from old parent, add to grandparent.
            old_p_var = variant.nodes[id_map[old_parent]]
            old_p_var.children.remove(id_map[promote_node])
            promote_var.parents = [id_map[grandparent[0]]]
            variant.nodes[id_map[grandparent[0]]].children.append(id_map[promote_node])

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
    source: Taxonomy, rng: random.Random,
) -> tuple[Taxonomy, Alignment]:
    """Randomly drop or rename attributes."""
    variant, id_map = _clone_taxonomy(source, "ATTR")

    for nid, node in source.nodes.items():
        var_node = variant.nodes[id_map[nid]]
        if var_node.attributes:
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
