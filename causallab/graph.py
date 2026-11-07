"""DAG separation by ancestor restriction, moralization and connectivity.

No distributions, independence tests, graph learning or active-path enumeration.
A returned witness is in the moral graph, not necessarily a path in the DAG.
"""
from collections import deque
import hashlib
import itertools
import json
from .core import fields, name, _model


def _graph(payload):
    fields(payload, ['version', 'nodes', 'edges'], 'graph')
    if type(payload['version']) is not int or payload['version'] != 1:
        raise ValueError('graph version must be integer 1')
    nodes, edges = payload['nodes'], payload['edges']
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 32:
        raise ValueError('graph requires 1..32 nodes')
    nodes = [name(v) for v in nodes]
    if len(set(nodes)) != len(nodes):
        raise ValueError('duplicate graph node')
    if not isinstance(edges, list) or len(edges) > 128:
        raise ValueError('graph requires at most 128 directed edges')
    found = set()
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError('edges must be [parent, child] lists')
        a, b = (name(v) for v in edge)
        if a not in nodes or b not in nodes or a == b:
            raise ValueError('edge endpoints must be distinct graph nodes')
        if (a, b) in found:
            raise ValueError('duplicate directed edge')
        found.add((a, b))
    parents = {v: set() for v in nodes}
    children = {v: set() for v in nodes}
    for a, b in found:
        parents[b].add(a); children[a].add(b)
    degree = {v: len(parents[v]) for v in nodes}
    queue = deque(sorted(v for v in nodes if not degree[v])); visited = 0
    while queue:
        v = queue.popleft(); visited += 1
        for c in sorted(children[v]):
            degree[c] -= 1
            if degree[c] == 0: queue.append(c)
    if visited != len(nodes):
        raise ValueError('graph must be acyclic')
    return {'version': 1, 'nodes': sorted(nodes), 'edges': [list(e) for e in sorted(found)]}


def graph_from_model(model):
    """Include every exogenous root, including shared latent common causes."""
    m = _model(model)
    return _graph({'version': 1, 'nodes': list(m.cards),
                   'edges': [[p, v.name] for v in m.nodes for p in v.inputs]})


def _names(values, nodes, label, required=False):
    if not isinstance(values, (list, tuple)) or (required and not values):
        raise ValueError(label + ' must be a ' + ('nonempty ' if required else '') + 'list/tuple of names')
    vals = [name(v) for v in values]
    if len(vals) != len(set(vals)) or any(v not in nodes for v in vals):
        raise ValueError(label + ' must contain unique known graph nodes')
    return sorted(vals)


def _hash(graph):
    return hashlib.sha256(json.dumps(graph, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def d_separation(graph, x, y, given=(), *, interventions=()):
    """Decide d-separation in a supplied DAG, optionally after incoming-edge cuts.

    x, y and given are pairwise disjoint. Interventions names mark graph surgery,
    not values or extra conditioning. The result concerns the supplied graph:
    d-connection does NOT assert statistical dependence or faithfulness.
    """
    g = _graph(graph); nodes = g['nodes']
    x = _names(x, nodes, 'x', True); y = _names(y, nodes, 'y', True)
    z = _names(given, nodes, 'given'); do = _names(interventions, nodes, 'interventions')
    if set(x) & set(y) or set(x) & set(z) or set(y) & set(z):
        raise ValueError('x, y and given must be pairwise disjoint')
    removed = [e for e in g['edges'] if e[1] in do]
    edges = [e for e in g['edges'] if e[1] not in do]
    effective = {'version': 1, 'nodes': nodes[:], 'edges': edges}
    parents = {v: set() for v in nodes}
    for a, b in edges: parents[b].add(a)
    ancestors = set(x + y + z); stack = list(ancestors)
    while stack:
        for p in parents[stack.pop()]:
            if p not in ancestors: ancestors.add(p); stack.append(p)
    reasons = {}
    def add(a, b, reason):
        reasons.setdefault(tuple(sorted((a, b))), []).append(reason)
    for a, b in edges:
        if a in ancestors and b in ancestors:
            add(a, b, {'kind': 'directed', 'parent': a, 'child': b})
    for c in sorted(ancestors):
        for a, b in itertools.combinations(sorted(parents[c]), 2):
            add(a, b, {'kind': 'coparents', 'child': c})
    moral = [{'nodes': list(e), 'origins': reasons[e]} for e in sorted(reasons)]
    remaining = [r for r in moral if not set(r['nodes']) & set(z)]
    neighbors = {v: set() for v in ancestors - set(z)}
    for r in remaining:
        a, b = r['nodes']; neighbors[a].add(b); neighbors[b].add(a)
    previous = {v: None for v in x}; queue = deque(x)
    while queue:
        v = queue.popleft()
        for n in sorted(neighbors[v]):
            if n not in previous: previous[n] = v; queue.append(n)
    target = next((v for v in y if v in previous), None)
    path = []
    while target is not None:
        path.append(target); target = previous[target]
    path.reverse()
    witness = [{'nodes': [a, b], 'origins': reasons[tuple(sorted((a, b)))]}
               for a, b in zip(path, path[1:])]
    return {'version': 1, 'graph': g, 'graph_sha256': _hash(g),
            'x': x, 'y': y, 'given': z, 'interventions': do,
            'effective_graph_sha256': _hash(effective), 'removed_incoming_edges': removed,
            'effective_edges': edges, 'ancestors': sorted(ancestors),
            'pruned_nodes': sorted(set(nodes) - ancestors), 'moral_edges': moral,
            'remaining_edges': remaining, 'reachable_from_x': sorted(previous),
            'd_separated': not path, 'moral_witness_path': path, 'moral_witness_edges': witness,
            'semantics': 'ancestral moral graph after incoming-edge surgery; witness is not an original-DAG active path',
            'limits': ['A supplied DAG assumption, not graph discovery or an independence test.',
                       'Separation implies conditional independence for distributions Markov to this DAG; connection does not imply dependence.',
                       'Explicitly include common causes; omitting latent nodes can invalidate conclusions.',
                       'Graph surgery encodes incoming-edge removal, not intervention values. Fixed values and deterministic cancellations can add independences not shown by d-separation.']}


def graph_demo():
    chain = {'version': 1, 'nodes': ['X', 'M', 'Y'], 'edges': [['X', 'M'], ['M', 'Y']]}
    fork = {'version': 1, 'nodes': ['U', 'X', 'Y'], 'edges': [['U', 'X'], ['U', 'Y']]}
    collider = {'version': 1, 'nodes': ['X', 'C', 'Y', 'D'],
                'edges': [['X', 'C'], ['Y', 'C'], ['C', 'D']]}
    specs = [('chain_open', chain, [], []), ('chain_given_M', chain, ['M'], []),
             ('fork_open', fork, [], []), ('fork_given_U', fork, ['U'], []),
             ('collider_closed', collider, [], []), ('collider_given_C', collider, ['C'], []),
             ('collider_given_D', collider, ['D'], []), ('fork_do_X', fork, [], ['X'])]
    results = {key: d_separation(g, ['X'], ['Y'], z, interventions=do) for key, g, z, do in specs}
    expected = [False, True, False, True, True, False, False, True]
    assert [r['d_separated'] for r in results.values()] == expected
    return {'version': 1, 'selection': 'fixed chain, fork and collider motifs; no graph or parameter search',
            'cases': results, 'expected_separated': dict(zip(results, expected)),
            'checks': {'all_expected': True},
            'limits': 'Graphical implications under supplied DAG assumptions, not empirical dependence or causal discovery.'}
