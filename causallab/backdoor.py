"""Pearl's sufficient backdoor criterion, not a complete adjustment criterion.

Candidate variables are declared available by the caller. No observed-data
availability, positivity, graph validity or estimation is inferred from names.
"""
import itertools
from .graph import _graph, _hash, _names, d_separation

LIMITS = [
    'A sufficient single-treatment backdoor criterion, not complete general adjustment or do-calculus identification.',
    'The supplied causal DAG must include relevant latent common causes. Available names are caller-declared measurable candidates, not verified data columns.',
    'A criterion-valid set licenses the adjustment formula only under the causal graph assumptions and appropriate observed support/positivity. No data, probabilities or estimates are produced.',
    'No valid set within the supplied candidates does not establish non-identifiability. Adding a variable can open a collider and invalidate a valid set.'
]


def _targets(g, treatment, outcome):
    x = _names([treatment], g['nodes'], 'treatment', True)[0]
    y = _names([outcome], g['nodes'], 'outcome', True)[0]
    if x == y: raise ValueError('treatment and outcome must be distinct')
    children = {v: [] for v in g['nodes']}
    for a, b in g['edges']: children[a].append(b)
    descendants = set(); stack = children[x][:]
    while stack:
        v = stack.pop()
        if v not in descendants: descendants.add(v); stack.extend(children[v])
    removed = [e for e in g['edges'] if e[0] == x]
    backdoor_graph = {'version':1, 'nodes':g['nodes'][:], 'edges':[e[:] for e in g['edges'] if e[0] != x]}
    return x, y, sorted(descendants), removed, backdoor_graph


def _adjust(values, g, x, y, label):
    z = _names(values, g['nodes'], label)
    if x in z or y in z: raise ValueError(label + ' must exclude treatment and outcome')
    return z


def _check(g, x, y, descendants, removed, cut_graph, z):
    # Always report both conditions, even if a descendant already disqualifies Z.
    separation = d_separation(cut_graph, [x], [y], z)
    excluded = sorted(set(z) & set(descendants))
    valid = not excluded and separation['d_separated']
    return {'version':1, 'graph':g, 'graph_sha256':_hash(g), 'treatment':x, 'outcome':y,
            'adjustment':z, 'treatment_descendants':descendants, 'forbidden_descendants':excluded,
            'removed_outgoing_edges':removed, 'backdoor_separation':separation,
            'criterion_met':valid, 'formula':'P(y | do(x)) = sum_z P(y | x,z) P(z)' if valid else None,
            'semantics':'exclude original treatment descendants, then d-separate in graph with treatment outgoing edges removed; this removal is NOT an intervention graph',
            'limits':LIMITS[:]}


def backdoor_criterion(graph, treatment, outcome, adjustment=()):
    """Return both conditions and a full separation audit for one supplied set."""
    g = _graph(graph); x,y,desc,removed,cut = _targets(g,treatment,outcome)
    z = _adjust(adjustment,g,x,y,'adjustment')
    return _check(g,x,y,desc,removed,cut,z)


def backdoor_sets(graph, treatment, outcome, available):
    """Exhaust every subset of at most 10 explicitly available candidate names.

    Return every criterion-valid set and inclusion-minimal set. Empty adjustment
    is included. Per-subset compact certificates avoid repeating the full graph.
    Call backdoor_criterion for full moral-edge provenance of a selected set.
    """
    g = _graph(graph); x,y,desc,removed,cut = _targets(g,treatment,outcome)
    if not isinstance(available,(list,tuple)) or len(available)>10:
        raise ValueError('available requires at most 10 candidate names (1024 subsets)')
    candidates = _adjust(available,g,x,y,'available')
    audits = []; valid = []
    for k in range(len(candidates)+1):
        for subset in itertools.combinations(candidates,k):
            z = list(subset); result = _check(g,x,y,desc,removed,cut,z)
            sep = result['backdoor_separation']
            row = {'adjustment':z, 'criterion_met':result['criterion_met'],
                   'forbidden_descendants':result['forbidden_descendants'],
                   'backdoor_d_separated':sep['d_separated'], 'ancestors':sep['ancestors'],
                   'reachable_from_treatment':sep['reachable_from_x'],
                   'moral_witness_path':sep['moral_witness_path']}
            audits.append(row)
            if result['criterion_met']: valid.append(z)
    # Cardinality-first order means all strict subsets occur before a superset.
    # Compare against prior minimal sets; do not assume validity is upward closed.
    minimal = []
    for z in valid:
        if not any(set(m) <= set(z) for m in minimal): minimal.append(z)
    return {'version':1, 'graph':g, 'graph_sha256':_hash(g), 'treatment':x, 'outcome':y,
            'available':candidates, 'treatment_descendants':desc, 'removed_outgoing_edges':removed,
            'backdoor_graph':cut, 'backdoor_graph_sha256':_hash(cut),
            'subsets_checked':len(audits), 'valid_sets':valid, 'inclusion_minimal_sets':minimal,
            'minimum_size':min(map(len,valid)) if valid else None, 'subset_audits':audits,
            'limits':LIMITS[:]}


def backdoor_demo():
    confounded = {'version':1,'nodes':['Z','X','Y'],
                  'edges':[['Z','X'],['Z','Y'],['X','Y']]}
    collider = {'version':1,'nodes':['X','Y','A','B','C'],
                'edges':[['A','X'],['A','C'],['B','C'],['B','Y'],['X','Y']]}
    mediator = {'version':1,'nodes':['X','M','Y'],'edges':[['X','M'],['M','Y']]}
    hidden = {'version':1,'nodes':['U','X','M','Y'],
              'edges':[['U','X'],['U','Y'],['X','M'],['M','Y']]}
    specs=[('confounded',confounded,['Z']),('collider',collider,['A','B','C']),
           ('mediator',mediator,['M']),('hidden_common_cause',hidden,['M'])]
    cases={key:backdoor_sets(g,'X','Y',available) for key,g,available in specs}
    checks={'confounder_required':cases['confounded']['valid_sets']==[['Z']],
            'empty_collider_valid':[] in cases['collider']['valid_sets'],
            'adding_collider_invalid':['C'] not in cases['collider']['valid_sets'],
            'blocking_parent_repairs':['A','C'] in cases['collider']['valid_sets'],
            'mediator_forbidden':cases['mediator']['valid_sets']==[[]],
            'hidden_no_backdoor_set':not cases['hidden_common_cause']['valid_sets']}
    assert all(checks.values())
    return {'version':1,'selection':'fixed confounding, collider, mediator and latent-common-cause motifs; no graph search',
            'cases':cases,'checks':checks,
            'limits':'Sufficient backdoor search within explicit candidates, not complete identification or empirical effect estimation.'}
