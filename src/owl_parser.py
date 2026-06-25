"""OWL ontology parser — extracts taxonomies from OWL/RDF files."""

from __future__ import annotations

import re
from pathlib import Path

from owlready2 import Restriction, ThingClass, AllDisjoint
from owlready2 import get_ontology, owl

from .taxonomy import Alignment, TaxonMatch, TaxonNode, Taxonomy


# OWL properties that represent is-a relations we care about.
# Only rdfs:subClassOf is the primary taxonomy edge.     
# We also store disjointWith for richer node descriptions.

def parse_owl(path: str | Path) -> Taxonomy:
    """Parse an OWL file and extract a Taxonomy.

    Extracts:
      - Class hierarchy via rdfs:subClassOf.
      - Disjointness axioms via owl:disjointWith.
      - rdfs:comment for each class.
      - Datatype and object property restrictions as textual attributes.

    Args:
        path: Path to the OWL file.

    Returns:
        A Taxonomy object with nodes and parent/child edges.
    """
    path = Path(path).resolve()

    # Use owlready2 to load and reason over the ontology.
    onto = get_ontology(f"file://{path}").load()

    # Determine a short namespace name from the base IRI.
    base_iri = onto.base_iri
    namespace = base_iri.rstrip("/#")
    short_name = Path(path).stem  # e.g., 'cmt', 'confof'.

    taxonomy = Taxonomy(name=short_name, namespace=namespace)

    # Collect all classes first.
    classes: list[ThingClass] = list(onto.classes())

    # Build mapping from class IRI to local id.
    iri_to_id: dict[str, str] = {}

    for cls in classes:
        # Skip owl:Thing itself — we track it but don't make a node for it.
        if cls == owl.Thing:
            continue
        cls_iri = cls.iri
        # Extract local name: 'http://cmt#Author' → 'cmt#Author'.
        local_id = cls_iri_to_local_id(cls_iri, namespace, short_name)
        iri_to_id[cls_iri] = local_id

    # Build nodes.
    for cls in classes:
        if cls == owl.Thing:
            continue
        cls_iri = cls.iri
        node_id = iri_to_id[cls_iri]

        # Name: use label or local part of IRI.
        label = cls.label.first() if hasattr(cls, 'label') and cls.label else ""
        name = label or cls.name.replace("_", " ")

        # Comment.
        comment = cls.comment.first() if hasattr(cls, 'comment') and cls.comment else ""

        node = TaxonNode(id=node_id, name=name, comment=comment)
        taxonomy.nodes[node_id] = node

    # Extract parent/child relations (subClassOf).
    # We collect parents for each class, but skip owl:Thing as a parent
    # (unless it's the only parent — then the node is a root of its taxonomy).
    has_non_thing_parent: set[str] = set()

    for cls in classes:
        if cls == owl.Thing:
            continue
        node_id = iri_to_id[cls.iri]
        node = taxonomy.nodes[node_id]

        for parent in cls.is_a:
            if isinstance(parent, ThingClass):
                parent_iri = parent.iri
                if parent_iri not in iri_to_id:
                    continue  # Skip Thing or unknown.
                parent_id = iri_to_id[parent_iri]
                node.parents.append(parent_id)
                has_non_thing_parent.add(node_id)

                # Add reverse child link.
                if parent_id in taxonomy.nodes:
                    taxonomy.nodes[parent_id].children.append(node_id)

    # Extract disjointness axioms.
    for cls in classes:
        if cls == owl.Thing:
            continue
        node_id = iri_to_id[cls.iri]
        node = taxonomy.nodes[node_id]

        for other_cls in cls.disjoints():
            if isinstance(other_cls, AllDisjoint):
                # AllDisjoint is a set of mutually disjoint classes.
                for member in other_cls.entities:
                    if hasattr(member, 'iri'):
                        member_iri = member.iri
                        if member_iri in iri_to_id and member_iri != cls.iri:
                            node.disjoint_with.append(iri_to_id[member_iri])
            elif hasattr(other_cls, 'iri'):
                other_iri = other_cls.iri
                if other_iri in iri_to_id:
                    node.disjoint_with.append(iri_to_id[other_iri])

    # Deduplicate disjoint_with lists (may have duplicates from
    # bidirectional axioms and AllDisjoint expansion).
    for node in taxonomy.nodes.values():
        node.disjoint_with = list(dict.fromkeys(node.disjoint_with))

    # Extract property restrictions as textual attributes.
    _extract_restrictions(taxonomy, classes, iri_to_id)

    # Compute depths (BFS from nodes without non-Thing parents).
    _compute_depths(taxonomy, has_non_thing_parent)

    # Identify root: the node that has a parent in OWL but it's Thing.
    # Use the node with depth=0.
    for node_id, node in taxonomy.nodes.items():
        if node.depth == 0:
            taxonomy.root_id = node_id
            break

    return taxonomy


def _extract_restrictions(
    taxonomy: Taxonomy,
    classes: list[ThingClass],
    iri_to_id: dict[str, str],
) -> None:
    """Extract OWL property restrictions as textual key=value attributes."""

    for cls in classes:
        if cls == owl.Thing:
            continue
        node_id = iri_to_id.get(cls.iri)
        if node_id is None:
            continue

        for equiv in cls.equivalent_to:
            if isinstance(equiv, Restriction):
                prop = equiv.property
                if prop is not None:
                    prop_name = prop.label.first() if prop.label else prop.name
                    value = equiv.value
                    if hasattr(value, 'name'):
                        value_str = value.name
                    else:
                        value_str = str(value)
                    taxonomy.nodes[node_id].attributes[prop_name] = value_str


def cls_iri_to_local_id(cls_iri: str, namespace: str, short_name: str) -> str:
    """Convert a full class IRI to a short local identifier.

    Examples:
        'http://cmt#Author' → 'cmt#Author'
        'http://confof.owl#Scholar' → 'confof#Scholar'
    """
    # Try to match namespace + separator.
    for sep in ("#", "/"):
        if cls_iri.startswith(f"{namespace}{sep}"):
            local = cls_iri[len(namespace) + len(sep):]
            return f"{short_name}#{local}"

    # Fallback: use the last part after # or /.
    match = re.search(r"[#/]([^#/]+)$", cls_iri)
    if match:
        return f"{short_name}#{match.group(1)}"

    # Last resort: just use the short name.
    return f"{short_name}#{cls_iri.rsplit('/', 1)[-1]}"


def _compute_depths(taxonomy: Taxonomy, has_non_thing_parent: set[str]) -> None:
    """BFS to assign depth to each node."""
    from collections import deque

    # Roots: nodes that have no non-Thing parent, or nodes w/o parents.
    queue: deque[str] = deque()
    for node_id, node in taxonomy.nodes.items():
        if node_id not in has_non_thing_parent or not node.parents:
            node.depth = 0
            queue.append(node_id)

    visited: set[str] = set()
    while queue:
        node_id = queue.popleft()
        if node_id in visited:
            continue
        visited.add(node_id)
        node = taxonomy.nodes.get(node_id)
        if node is None:
            continue
        for child_id in node.children:
            child = taxonomy.nodes.get(child_id)
            if child is not None and child.depth <= node.depth:
                child.depth = node.depth + 1
                queue.append(child_id)


def parse_reference_alignment(path: str | Path) -> dict[str, str]:
    """Parse an OAEI reference alignment RDF file.

    Returns a dict mapping source_entity → target_entity.

    Args:
        path: Path to a reference alignment RDF file
               (e.g., 'cmt-conference.rdf').

    Returns:
        Dict of {entity1_URI: entity2_URI}.
    """
    import xml.etree.ElementTree as ET

    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()

    # Namespace mapping.
    ns = {
        "align": "http://knowledgeweb.semanticweb.org/heterogeneity/alignment",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    }
    align_ns = ns["align"]
    rdf_ns = ns["rdf"]

    mappings: dict[str, str] = {}

    for cell_elem in root.iter(f"{{{align_ns}}}Cell"):
        entity1_elem = cell_elem.find(f"{{{align_ns}}}entity1")
        entity2_elem = cell_elem.find(f"{{{align_ns}}}entity2")
        if entity1_elem is not None and entity2_elem is not None:
            entity1 = entity1_elem.get(f"{{{rdf_ns}}}resource")
            entity2 = entity2_elem.get(f"{{{rdf_ns}}}resource")
            if entity1 and entity2:
                mappings[entity1] = entity2

    return mappings


def build_alignment(
    source_tax: Taxonomy,
    target_tax: Taxonomy,
    ref_map: dict[str, str],
) -> Alignment:
    """Build an Alignment object from a raw reference mapping.

    The reference mapping uses full URIs; we convert them to local node IDs.
    """
    matches: list[TaxonMatch] = []

    for uri1, uri2 in ref_map.items():
        # Convert URIs to local IDs matching the taxonomy's naming convention.
        local1 = _uri_to_local(uri1, source_tax)
        local2 = _uri_to_local(uri2, target_tax)

        # Verify the node exists.
        if local1 in source_tax.nodes and local2 in target_tax.nodes:
            matches.append(TaxonMatch(source_id=local1, target_id=local2, confidence=1.0))

    return Alignment(
        source=source_tax.name,
        target=target_tax.name,
        matches=matches,
    )


def _uri_to_local(uri: str, taxonomy: Taxonomy) -> str:
    """Convert a reference alignment URI to a local node ID.

    Reference alignment format: 'http://cmt#Conference'
    Taxonomy node IDs: 'cmt#Conference'
    """
    # Try exact match in namespace.
    for node_id in taxonomy.nodes:
        # Check if the URI ends with the node name part.
        local_part = node_id.split("#", 1)[-1] if "#" in node_id else node_id
        if uri.endswith(f"#{local_part}") or uri.endswith(f"/{local_part}"):
            return node_id

    # Fallback: just use the IRI's last fragment.
    fragment = uri.rsplit("#", 1)[-1] if "#" in uri else uri.rsplit("/", 1)[-1]
    for node_id in taxonomy.nodes:
        if node_id.endswith(f"#{fragment}"):
            return node_id

    return fragment
