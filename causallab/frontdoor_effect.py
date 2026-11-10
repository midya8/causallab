"""Exact rational frontdoor functional, conditional on supplied DAG assumptions.

A positive observed joint does not validate the causal graph or compatibility
with its latent-variable assumptions. This computes a functional, not an estimate
learned from samples, confidence interval or general identification procedure.
"""
from fractions import Fraction
import hashlib
import itertools
import json
import math
from .core import fields, name
from .frontdoor import frontdoor_criterion


def _fraction(v):
    if type(v) is int:
        result = Fraction(v)
    elif isinstance(v, list) and len(v) == 2 and all(type(x) is int for x in v):
        if not 1 <= v[1] <= 1000000 or abs(v[0]) > 1000000:
            raise ValueError('rational components require bounded integers and positive denominator')
        result = Fraction(*v)
    else:
        raise ValueError('probabilities must be integers or [integer,positive denominator]')
    if not 0 <= result <= 1:
        raise ValueError('probabilities must lie in [0,1]')
    return result


def _encode(v):
    if isinstance(v, Fraction):
        return [v.numerator, v.denominator]
    if isinstance(v, dict):
        return {k: _encode(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_encode(x) for x in v]
    return v


def frontdoor_effect(graph, treatment, outcome, mediators, observed_joint):
    """Evaluate every P(Y=y | do(X=x)) row with all exact inner/outer terms.

    The joint's variables must be exactly X, the supplied mediator set, and Y.
    Require P(X=x,Z=z)>0 for every declared x/z cell, even if one requested
    treatment would permit a weaker support condition. Never fill 0/0 as zero.
    """
    certificate = frontdoor_criterion(graph, treatment, outcome, mediators)
    if not certificate['criterion_met']:
        raise ValueError('supplied mediator set fails the sufficient frontdoor criterion')
    x, y, z = certificate['treatment'], certificate['outcome'], certificate['mediators']
    fields(observed_joint, ['version', 'variables', 'probabilities'], 'observed_joint')
    if type(observed_joint['version']) is not int or observed_joint['version'] != 1:
        raise ValueError('observed joint version must be integer 1')
    variables = observed_joint['variables']
    if not isinstance(variables, list) or not 2 <= len(variables) <= 8:
        raise ValueError('observed joint requires 2..8 variables')
    cards, order, cells = {}, [], 1
    for v in variables:
        fields(v, ['name', 'states'], 'observed variable')
        n = name(v['name']); c = v['states']
        if n in cards or type(c) is not int or not 2 <= c <= 4:
            raise ValueError('observed variables must be unique with 2..4 integer states')
        cards[n] = c; order.append(n); cells *= c
    if set(cards) != {x, y, *z}:
        raise ValueError('observed variables must be exactly treatment, mediators and outcome')
    if cells > 4096:
        raise ValueError('observed joint exceeds 4096 cells')
    raw = observed_joint['probabilities']
    if not isinstance(raw, list) or len(raw) != cells:
        raise ValueError('observed probability count must match the full Cartesian table')
    probs, lcm = [], 1
    for v in raw:
        p = _fraction(v); lcm = math.lcm(lcm, p.denominator)
        if lcm > 1000000:
            raise ValueError('observed joint denominator LCM exceeds 1000000')
        probs.append(p)
    if sum(probs) != 1:
        raise ValueError('observed probabilities must sum exactly to 1')
    canonical = [x, *z, y]
    data = {}
    for values, mass in zip(itertools.product(*(range(cards[n]) for n in order)), probs):
        assignment = dict(zip(order, values))
        data[tuple(assignment[n] for n in canonical)] = mass
    zs = list(itertools.product(*(range(cards[n]) for n in z)))
    pxz = {(a, m): sum(data[(a, *m, b)] for b in range(cards[y]))
           for a in range(cards[x]) for m in zs}
    if any(p == 0 for p in pxz.values()):
        raise ValueError('frontdoor functional requires positive P(x,z) in every declared cell; undefined conditional not replaced by zero')
    px = {a: sum(pxz[(a, m)] for m in zs) for a in range(cards[x])}
    rows = []
    for a in range(cards[x]):
        for b in range(cards[y]):
            terms = []
            for m in zs:
                inner = []
                for aprime in range(cards[x]):
                    conditional = data[(aprime, *m, b)] / pxz[(aprime, m)]
                    inner.append({'treatment_prime': aprime, 'p_y_given_xprime_z': conditional,
                                  'p_xprime': px[aprime], 'weighted': conditional * px[aprime]})
                inner_sum = sum(t['weighted'] for t in inner)
                mediator = pxz[(a, m)] / px[a]
                terms.append({'mediator_values': list(m), 'p_z_given_x': mediator,
                              'inner_terms': inner, 'inner_sum': inner_sum,
                              'contribution': mediator * inner_sum})
            rows.append({'treatment_value': a, 'outcome_value': b,
                         'probability': sum(t['contribution'] for t in terms),
                         'mediator_terms': terms})
    normalized = _encode({'version': 1, 'variables': [{'name': n, 'states': cards[n]} for n in canonical],
                          'probabilities': [data[v] for v in itertools.product(*(range(cards[n]) for n in canonical))]})
    digest = hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return _encode({'version': 1, 'graph_certificate': certificate, 'observed_joint': normalized,
                    'observed_joint_sha256': digest, 'cell_count': cells,
                    'positive_xz_cells': len(pxz), 'denominator_lcm': lcm,
                    'support': [{'treatment_value': a, 'mediator_values': list(m),
                                 'p_xz': pxz[(a, m)], 'p_x': px[a]} for a in range(cards[x]) for m in zs],
                    'rows': rows, 'arithmetic': 'exact rational functional; final input index changes fastest',
                    'limits': ['The causal graph and latent-variable assumptions are supplied, not learned or validated by this observed joint.',
                               'The result is a functional of a supplied exact positive joint, not a statistical estimate, confidence interval or medical recommendation.',
                               'Strict positivity in every declared treatment/mediator cell is deliberately stronger than query-specific extensions.',
                               'Integer/fraction inputs only; 2..8 variables with2..4 states,4096 cells and normalized denominator LCM at most1000000. Intermediate fraction sizes are not a speed or RAM guarantee.']})


def frontdoor_effect_demo():
    graph = {'version': 1, 'nodes': ['U', 'X', 'M', 'Y'], 'edges': [['U', 'X'], ['U', 'Y'], ['X', 'M'], ['M', 'Y']]}
    joint = {'version': 1, 'variables': [{'name': 'X', 'states': 2}, {'name': 'M', 'states': 2}, {'name': 'Y', 'states': 2}], 'probabilities': [[43, 128], [13, 128], [13, 384], [43, 384], [17, 384], [23, 384], [23, 128], [17, 128]]}
    effect = frontdoor_effect(graph, 'X', 'Y', ['M'], joint)
    values = [Fraction(*p) for p in joint['probabilities']]
    observed = []
    for x in range(2):
        numerator = sum(values[4*x+2*m+1] for m in range(2))
        denominator = sum(values[4*x+i] for i in range(4))
        observed.append({'treatment_value': x, 'p_y1_given_x': _encode(numerator/denominator)})
    return {'version': 1,
            'selection': 'fixed synthetic SCM: independent U~Bernoulli(1/3), E1/E2~Bernoulli(1/4), E3~Bernoulli(1/8); X=U xor E1, M=X xor E2, Y=M xor U xor E3; no parameter search',
            'observational_rows': observed, 'effect': effect,
            'limits': 'The graph is supplied. Exact rational calculation does not validate a real causal model or estimate effects from samples.'}
