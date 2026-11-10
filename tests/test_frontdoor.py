import copy
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from causallab.frontdoor import frontdoor_criterion, frontdoor_sets, frontdoor_demo

ROOT = Path(__file__).resolve().parents[1]


def original_path_conditions(g, x, y, z):
    """Original DAG simple paths only. No surgery, moralization or production call."""
    arrows = set(map(tuple, g['edges']))
    neighbors = {n: set() for n in g['nodes']}
    for a, b in arrows:
        neighbors[a].add(b); neighbors[b].add(a)
    descendants = {n: set() for n in g['nodes']}
    for n in g['nodes']:
        for _ in g['nodes']:
            descendants[n] |= {b for a, b in arrows if a == n or a in descendants[n]}

    def paths(a, b, prefix=None):
        prefix = [a] if prefix is None else prefix
        if prefix[-1] == b:
            yield prefix
        else:
            for c in sorted(neighbors[prefix[-1]] - set(prefix)):
                yield from paths(a, b, prefix + [c])

    def active(p, given):
        for a, b, c in zip(p, p[1:], p[2:]):
            collider = (a, b) in arrows and (c, b) in arrows
            if collider and not ({b} | descendants[b]) & set(given):
                return False
            if not collider and b in given:
                return False
        return True

    directed = all(set(p) & set(z) for p in paths(x, y)
                   if all((a, b) in arrows for a, b in zip(p, p[1:])))
    second = not any((p[1], x) in arrows and active(p, [])
                     for m in z for p in paths(x, m))
    third = not any((p[1], m) in arrows and active(p, [x])
                    for m in z for p in paths(m, y) if not set(p[1:]) & set(z))
    return [directed, second, third]


class FrontdoorTests(unittest.TestCase):
    def setUp(self):
        self.g = {'version': 1, 'nodes': ['U', 'X', 'M', 'Y'],
                  'edges': [['U', 'X'], ['U', 'Y'], ['X', 'M'], ['M', 'Y']]}

    def test_fixed_controls_three_separate_conditions(self):
        r = frontdoor_demo()
        self.assertEqual(len(r['cases']), 4)
        self.assertTrue(all(r['checks'].values()))
        self.assertTrue(r['cases']['latent_confounding']['criterion_met'])
        for name, a in r['cases'].items():
            self.assertEqual(list(a['conditions'].values()), original_path_conditions(a['graph'], 'X', 'Y', ['M']), name)
            self.assertEqual(bool(a['formula']), a['criterion_met'])

    def test_cuts_are_outgoing_not_interventions(self):
        a = frontdoor_criterion(self.g, 'X', 'Y', ['M'])
        self.assertEqual(a['removed_treatment_outgoing_edges'], [['X', 'M']])
        self.assertEqual(a['removed_mediator_outgoing_edges'], [['M', 'Y']])
        for key in ['treatment_mediator_separation', 'mediator_outcome_separation']:
            self.assertEqual(a[key]['interventions'], [])
            self.assertEqual(a[key]['removed_incoming_edges'], [])
        self.assertEqual(a['mediator_outcome_separation']['given'], ['X'])
        self.assertEqual(a['treatment_mediator_separation']['given'], [])

    def test_all_fixed_order_four_node_dags_and_all_disjoint_queries(self):
        nodes = ['A', 'B', 'C', 'D']; possible = list(itertools.combinations(nodes, 2))
        checks = searches = 0
        for mask in range(64):
            g = {'version': 1, 'nodes': nodes, 'edges': [list(e) for i, e in enumerate(possible) if mask & (1 << i)]}
            for x, y in itertools.permutations(nodes, 2):
                available = [n for n in nodes if n not in [x, y]]
                search = frontdoor_sets(g, x, y, available)
                valid = []
                for row in search['subset_audits']:
                    z = row['mediators']; expected = original_path_conditions(g, x, y, z)
                    a = frontdoor_criterion(g, x, y, z)
                    self.assertEqual(list(a['conditions'].values()), expected, (g, x, y, z))
                    self.assertEqual(row['conditions'], a['conditions'])
                    self.assertEqual(row['criterion_met'], all(expected))
                    self.assertEqual(row['unintercepted_directed_witness'], a['directed_interception']['unintercepted_directed_witness'])
                    if all(expected): valid.append(z)
                    checks += 1
                self.assertEqual(search['valid_sets'], valid)
                self.assertEqual(search['inclusion_minimal_sets'], [z for z in valid if not any(set(m) < set(z) for m in valid)])
                self.assertEqual(search['minimum_size'], min(map(len, valid)) if valid else None)
                self.assertEqual(search['subsets_checked'], 4); searches += 1
        self.assertEqual(checks, 3072); self.assertEqual(searches, 768)

    def test_seeded_five_six_node_dags_full_candidate_sets(self):
        rng = random.Random(220926); checks = 0
        for n in [5, 6]:
            names = ['N' + str(i) for i in range(n)]
            possible = list(itertools.combinations(names, 2))
            for _ in range(30):
                rng.shuffle(names)
                # Relabel a random topological order, retaining full original paths.
                ordered = names[:]
                edges = [[a, b] for a, b in itertools.combinations(ordered, 2) if rng.randrange(3) == 0]
                g = {'version': 1, 'nodes': names[:], 'edges': edges}
                x, y = rng.sample(names, 2)
                a = frontdoor_sets(g, x, y, [m for m in names if m not in [x, y]])
                for row in a['subset_audits']:
                    self.assertEqual(list(row['conditions'].values()), original_path_conditions(g, x, y, row['mediators']), (g, x, y, row))
                    checks += 1
        self.assertEqual(checks, 720)

    def test_directed_bypass_witness_uses_original_edges_and_no_mediator(self):
        g = copy.deepcopy(self.g); g['edges'] += [['X', 'Y']]
        a = frontdoor_criterion(g, 'X', 'Y', ['M'])
        self.assertEqual(a['directed_interception']['unintercepted_directed_witness'], ['X', 'Y'])
        self.assertFalse(a['criterion_met'])
        self.assertTrue(a['conditions']['no_treatment_mediator_backdoor'])
        self.assertTrue(a['conditions']['treatment_blocks_mediator_outcome_backdoors'])

    def test_empty_mediator_set_vacuity_and_no_directed_causal_path(self):
        a = frontdoor_criterion(self.g, 'X', 'Y', [])
        self.assertFalse(a['criterion_met']); self.assertTrue(a['empty_mediator_backdoor_conditions_vacuous'])
        self.assertIsNone(a['treatment_mediator_separation']); self.assertIsNone(a['mediator_outcome_separation'])
        g = {'version': 1, 'nodes': ['U', 'X', 'Y'], 'edges': [['U', 'X'], ['U', 'Y']]}
        b = frontdoor_criterion(g, 'X', 'Y', [])
        self.assertTrue(b['criterion_met']); self.assertIn('sum_z', b['formula'])
        self.assertEqual(frontdoor_sets(g, 'X', 'Y', [])['valid_sets'], [[]])

    def test_joint_mediators_proper_backdoor_paths(self):
        g = {'version': 1, 'nodes': ['U', 'X', 'A', 'B', 'Y'],
             'edges': [['U', 'X'], ['U', 'Y'], ['X', 'A'], ['A', 'B'], ['B', 'Y']]}
        for z in [['A'], ['B'], ['A', 'B']]:
            a = frontdoor_criterion(g, 'X', 'Y', z)
            self.assertTrue(a['criterion_met'])
            self.assertEqual(list(a['conditions'].values()), original_path_conditions(g, 'X', 'Y', z))
        a = frontdoor_sets(g, 'X', 'Y', ['A', 'B'])
        self.assertEqual(a['valid_sets'], [['A'], ['B'], ['A', 'B']])
        self.assertEqual(a['inclusion_minimal_sets'], [['A'], ['B']])

    def test_availability_is_explicit(self):
        self.assertEqual(frontdoor_sets(self.g, 'X', 'Y', [])['valid_sets'], [])
        self.assertEqual(frontdoor_sets(self.g, 'X', 'Y', ['M'])['valid_sets'], [['M']])
        self.assertTrue(any('does not prove non-identifiability' in t for t in frontdoor_sets(self.g, 'X', 'Y', [])['limits']))

    def test_full_1024_subset_boundary_and_preflight(self):
        names = ['M' + str(i) for i in range(10)]
        g = {'version': 1, 'nodes': ['X', 'Y'] + names, 'edges': []}
        a = frontdoor_sets(g, 'X', 'Y', names)
        self.assertEqual(a['subsets_checked'], 1024); self.assertEqual(len(a['valid_sets']), 1024)
        self.assertEqual(a['inclusion_minimal_sets'], [[]]); self.assertEqual(a['minimum_size'], 0)
        with patch('causallab.frontdoor.d_separation', side_effect=AssertionError('must reject first')):
            with self.assertRaisesRegex(ValueError, 'at most 10'):
                frontdoor_sets(g, 'X', 'Y', names + ['extra'])

    def test_normalization_hash_and_input_output_isolation(self):
        g = copy.deepcopy(self.g); before = copy.deepcopy(g)
        a = frontdoor_criterion(g, 'X', 'Y', ['M']); b = copy.deepcopy(a)
        g['nodes'].reverse(); g['edges'].reverse()
        self.assertEqual(a, frontdoor_criterion(g, 'X', 'Y', ['M']))
        self.assertEqual(before, self.g)
        a['graph']['edges'].clear(); a['limits'].clear()
        self.assertEqual(b, frontdoor_criterion(self.g, 'X', 'Y', ['M']))

    def test_strict_schema_unknown_duplicate_overlap_names(self):
        for args in [('X', 'X', []), ('unknown', 'Y', []), ('X', 'Y', ['M', 'M']),
                     ('X', 'Y', ['X']), ('X', 'Y', ['Y']), ('X', 'Y', 'M'), ('X', 'Y', [False])]:
            with self.assertRaises((ValueError, TypeError)): frontdoor_criterion(self.g, *args)
        for update in [{'unknown': 0}, {'version': True}, {'edges': self.g['edges'] + [['Y', 'X']]}]:
            with self.assertRaises((ValueError, TypeError)):
                frontdoor_criterion(dict(self.g, **update), 'X', 'Y', ['M'])
        for candidate in ['M', None, ['M', 'M'], ['U', 'X']]:
            with self.assertRaises((ValueError, TypeError)): frontdoor_sets(self.g, 'X', 'Y', candidate)

    def test_cli_demo_certificate_search_and_strict_failures(self):
        def cli(*args):
            return subprocess.run([sys.executable, '-m', 'causallab', *args], cwd=ROOT, text=True, capture_output=True)
        a = cli('frontdoor-demo'); self.assertEqual(a.returncode, 0, a.stderr)
        self.assertEqual(json.loads(a.stdout), frontdoor_demo())
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'g.json'; p.write_text(json.dumps(self.g))
            for cmd, flag, expected in [('frontdoor', '--mediators', frontdoor_criterion(self.g, 'X', 'Y', ['M'])),
                                        ('frontdoor-sets', '--available', frontdoor_sets(self.g, 'X', 'Y', ['M']))]:
                a = cli(cmd, str(p), '--treatment', 'X', '--outcome', 'Y', flag, 'M')
                self.assertEqual(a.returncode, 0, a.stderr); self.assertEqual(json.loads(a.stdout), expected)
            for bad in ['{"version":1,"version":1,"nodes":[],"edges":[]}', '{}', ' ' * 262145]:
                p.write_text(bad); a = cli('frontdoor', str(p), '--treatment', 'X', '--outcome', 'Y', '--mediators', 'M')
                self.assertEqual(a.returncode, 2); self.assertEqual(a.stdout, ''); self.assertIn('causallab:', a.stderr)
                self.assertNotIn('Traceback', a.stderr)


if __name__ == '__main__':
    unittest.main()
