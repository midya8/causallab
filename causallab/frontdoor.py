"""Bounded sufficient frontdoor certificates for supplied causal DAGs.

Established Pearl criterion, not general identification. Availability, causal
assumptions and observed positivity are not inferred from graph node names.
"""
from collections import deque
import itertools
from .graph import _graph, _hash, _names, d_separation

LIMITS = [
    'A sufficient frontdoor criterion for one treatment and one outcome, not general identification or a necessary criterion for a particular frontdoor functional.',
    'The supplied causal DAG must include relevant latent common causes. Candidates are explicitly caller-declared measurable names; no data availability is inferred.',
    'A passing graph licenses the stated functional only under the causal DAG assumptions and appropriate positive observed P(x,z) support. No probabilities or estimates are computed here.',
    'The second and third conditions concern proper backdoor paths. Removing outgoing edges is a criterion test, not do(), which removes incoming edges.',
    'No valid set among the supplied candidates does not prove non-identifiability. Inclusion-minimal and minimum-cardinality sets are distinct.'
]


def _request(graph, treatment, outcome, mediators, label):
    g = _graph(graph)
    x = _names([treatment], g['nodes'], 'treatment', True)[0]
    y = _names([outcome], g['nodes'], 'outcome', True)[0]
    if x == y:
        raise ValueError('treatment and outcome must be distinct')
    z = _names(mediators, g['nodes'], label)
    if x in z or y in z:
        raise ValueError(label + ' must exclude treatment and outcome')
    return g, x, y, z


def _cut(g, sources):
    removed = [e[:] for e in g['edges'] if e[0] in sources]
    graph = {'version': 1, 'nodes': g['nodes'][:],
             'edges': [e[:] for e in g['edges'] if e[0] not in sources]}
    return graph, removed


def _directed(g, x, y, z):
    # Reachability stops at every mediator; a surviving X/Y path refutes (i).
    children = {v: [] for v in g['nodes']}
    for a, b in g['edges']:
        children[a].append(b)
    previous = {x: None}
    queue = deque([x])
    while queue:
        a = queue.popleft()
        if a in z:
            continue
        for b in children[a]:
            if b not in previous:
                previous[b] = a
                queue.append(b)
    path = []
    if y in previous:
        v = y
        while v is not None:
            path.append(v)
            v = previous[v]
        path.reverse()
    return {'all_directed_paths_intercepted': y not in previous,
            'stopped_at': z[:], 'reachable_with_stops': sorted(previous),
            'unintercepted_directed_witness': path}


def _check(g, x, y, z, x_cut, x_removed):
    directed = _directed(g, x, y, z)
    z_cut, z_removed = _cut(g, z)
    # Empty Z has no backdoor endpoints. Do not call d_separation on empty sets.
    xz = d_separation(x_cut, [x], z) if z else None
    zy = d_separation(z_cut, z, [y], [x]) if z else None
    second = xz is None or xz['d_separated']
    third = zy is None or zy['d_separated']
    conditions = {'all_directed_paths_intercepted': directed['all_directed_paths_intercepted'],
                  'no_treatment_mediator_backdoor': second,
                  'treatment_blocks_mediator_outcome_backdoors': third}
    valid = all(conditions.values())
    return {'version': 1, 'graph': g, 'graph_sha256': _hash(g),
            'treatment': x, 'outcome': y, 'mediators': z,
            'conditions': conditions, 'criterion_met': valid,
            'directed_interception': directed,
            'removed_treatment_outgoing_edges': x_removed,
            'removed_mediator_outgoing_edges': z_removed,
            'treatment_mediator_separation': xz,
            'mediator_outcome_separation': zy,
            'empty_mediator_backdoor_conditions_vacuous': not z,
            'formula': 'P(y | do(x)) = sum_z P(z | x) sum_xprime P(y | xprime,z) P(xprime)' if valid else None,
            'semantics': 'intercept directed X/Y paths; test X/Z in G with X outgoing cut; test Z/Y given X in G with all Z outgoing cut; proper backdoor criterion, not intervention graphs',
            'limits': LIMITS[:]}


def frontdoor_criterion(graph, treatment, outcome, mediators=()):
    """Keep all three conditions and full moral certificates, even on failure.

    Empty mediators may pass only when no directed treatment/outcome path exists.
    This encodes a graph-implied absence of a causal effect, not observed absence.
    """
    g, x, y, z = _request(graph, treatment, outcome, mediators, 'mediators')
    cut, removed = _cut(g, [x])
    return _check(g, x, y, z, cut, removed)


def frontdoor_sets(graph, treatment, outcome, available):
    """Exhaust every subset of at most ten explicitly available node names."""
    if not isinstance(available, (list, tuple)) or len(available) > 10:
        raise ValueError('available requires at most 10 candidate names (1024 subsets)')
    g, x, y, candidates = _request(graph, treatment, outcome, available, 'available')
    cut, removed = _cut(g, [x])
    audits, valid = [], []
    for k in range(len(candidates) + 1):
        for subset in itertools.combinations(candidates, k):
            z = list(subset)
            result = _check(g, x, y, z, cut, removed)
            a, b = result['treatment_mediator_separation'], result['mediator_outcome_separation']
            audits.append({'mediators': z, 'criterion_met': result['criterion_met'],
                           'conditions': result['conditions'],
                           'unintercepted_directed_witness': result['directed_interception']['unintercepted_directed_witness'],
                           'treatment_mediator_moral_witness': a['moral_witness_path'] if a else [],
                           'mediator_outcome_moral_witness': b['moral_witness_path'] if b else []})
            if result['criterion_met']:
                valid.append(z)
    minimal = [z for z in valid if not any(set(m) < set(z) for m in valid)]
    return {'version': 1, 'graph': g, 'graph_sha256': _hash(g),
            'treatment': x, 'outcome': y, 'available': candidates,
            'subsets_checked': len(audits), 'valid_sets': valid,
            'inclusion_minimal_sets': minimal,
            'minimum_size': min(map(len, valid)) if valid else None,
            'subset_audits': audits, 'limits': LIMITS[:]}


def frontdoor_demo():
    base = {'version': 1, 'nodes': ['U', 'X', 'M', 'Y'],
            'edges': [['U', 'X'], ['U', 'Y'], ['X', 'M'], ['M', 'Y']]}
    specs = [('latent_confounding', []), ('direct_bypass', [['X', 'Y']]),
             ('treatment_mediator_confounding', [['U', 'M']]),
             ('mediator_outcome_confounding', [['X', 'V'], ['V', 'M'], ['V', 'Y']])]
    cases = {}
    for key, extra in specs:
        g = {'version': 1, 'nodes': base['nodes'][:] + (['V'] if key == 'mediator_outcome_confounding' else []),
             'edges': [e[:] for e in base['edges']] + extra}
        cases[key] = frontdoor_criterion(g, 'X', 'Y', ['M'])
    # In the last motif V is caused by X; conditioning on X does not block M<-V->Y.
    expected = {'latent_confounding': [True, True, True],
                'direct_bypass': [False, True, True],
                'treatment_mediator_confounding': [True, False, False],
                'mediator_outcome_confounding': [False, True, False]}
    checks = {k: list(r['conditions'].values()) == expected[k] for k, r in cases.items()}
    assert all(checks.values())
    return {'version': 1, 'selection': 'fixed latent frontdoor motif and explicit edge-addition failures, not discovered graphs',
            'cases': cases, 'expected_conditions': expected, 'checks': checks,
            'limits': LIMITS[:]}
