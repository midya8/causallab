import copy
from fractions import Fraction as F
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from causallab.frontdoor_effect import frontdoor_effect

ROOT = Path(__file__).resolve().parents[1]
G = {'version': 1, 'nodes': ['U', 'X', 'M', 'Y'],
     'edges': [['U', 'X'], ['U', 'Y'], ['X', 'M'], ['M', 'Y']]}


def encoded(v):
    return [v.numerator, v.denominator]


def noise_reference(pu, pe, pm, py):
    """Literal Boolean SCM world sums, with intervention evaluated separately."""
    observed = [F(0) for _ in range(8)]; action = [F(0), F(0)]
    for u, e, m, n in itertools.product(range(2), repeat=4):
        mass = (pu if u else 1-pu) * (pe if e else 1-pe) * (pm if m else 1-pm) * (py if n else 1-py)
        x = u ^ e; mediator = x ^ m; y = mediator ^ u ^ n
        observed[4*x+2*mediator+y] += mass
        for forced_x in range(2):
            if (forced_x ^ m ^ u ^ n) == 1: action[forced_x] += mass
    return {'version': 1, 'variables': [{'name': n, 'states': 2} for n in ['X', 'M', 'Y']],
            'probabilities': [encoded(p) for p in observed]}, action


class FrontdoorEffectTests(unittest.TestCase):
    def test_81_exact_scm_world_references_and_all_contributions(self):
        cases = 0
        for probs in itertools.product([F(1, 4), F(1, 3), F(3, 4)], repeat=4):
            joint, expected = noise_reference(*probs)
            a = frontdoor_effect(G, 'X', 'Y', ['M'], joint)
            self.assertEqual(a['cell_count'], 8); self.assertEqual(a['positive_xz_cells'], 4)
            for row in a['rows']:
                x, y = row['treatment_value'], row['outcome_value']
                self.assertEqual(F(*row['probability']), expected[x] if y else 1-expected[x])
                contributions = []
                for term in row['mediator_terms']:
                    inner = sum(F(*q['p_y_given_xprime_z']) * F(*q['p_xprime']) for q in term['inner_terms'])
                    self.assertEqual(F(*term['inner_sum']), inner)
                    for q in term['inner_terms']:
                        self.assertEqual(F(*q['weighted']), F(*q['p_y_given_xprime_z']) * F(*q['p_xprime']))
                    value = F(*term['p_z_given_x']) * inner
                    self.assertEqual(F(*term['contribution']), value); contributions.append(value)
                self.assertEqual(sum(contributions), F(*row['probability']))
            cases += 1
        self.assertEqual(cases, 81)

    def test_analytic_fixture_and_normalized_permuted_joint(self):
        joint, _ = noise_reference(F(1, 3), F(1, 4), F(1, 4), F(1, 8))
        a = frontdoor_effect(G, 'X', 'Y', ['M'], joint)
        self.assertEqual([F(*r['probability']) for r in a['rows'] if r['outcome_value'] == 1], [F(7,16), F(9,16)])
        permuted = {'version': 1, 'variables': [{'name': n, 'states': 2} for n in ['Y', 'X', 'M']],
                    'probabilities': [joint['probabilities'][4*x+2*m+y] for y, x, m in itertools.product(range(2), repeat=3)]}
        self.assertEqual(a, frontdoor_effect(G, 'X', 'Y', ['M'], permuted))

    def test_nonbinary_joint_against_independent_factorized_interventions(self):
        px = [[F(1,6), F(2,6), F(3,6)], [F(3,6), F(2,6), F(1,6)]]
        pm = [[F(1,6),F(2,6),F(3,6)], [F(3,6),F(2,6),F(1,6)], [F(2,6),F(3,6),F(1,6)]]
        py = [[F(1,4),F(1,2),F(3,4)], [F(3,4),F(1,4),F(1,2)]]
        pu = [F(2,3), F(1,3)]
        probs = [sum(pu[u]*px[u][x]*pm[x][m]*(py[u][m] if y else 1-py[u][m]) for u in range(2))
                 for x, m, y in itertools.product(range(3), range(3), range(2))]
        joint = {'version': 1, 'variables': [{'name': 'X','states':3},{'name':'M','states':3},{'name':'Y','states':2}],
                 'probabilities': list(map(encoded, probs))}
        a = frontdoor_effect(G, 'X', 'Y', ['M'], joint)
        for row in a['rows']:
            x, y = row['treatment_value'], row['outcome_value']
            expected = sum(pu[u]*pm[x][m]*(py[u][m] if y else 1-py[u][m]) for u in range(2) for m in range(3))
            self.assertEqual(F(*row['probability']), expected)
        self.assertEqual(a['cell_count'], 18); self.assertEqual(a['positive_xz_cells'], 9)

    def test_joint_mediators_and_empty_mediator_functionals(self):
        g = {'version':1,'nodes':['U','X','A','B','Y'],
             'edges':[['U','X'],['U','Y'],['X','A'],['A','B'],['B','Y']]}
        table = [F(0) for _ in range(16)]; expected = [F(0), F(0)]
        ps = [F(1,3),F(1,4),F(1,4),F(1,3),F(1,8)]
        for u,e1,e2,e3,e4 in itertools.product(range(2), repeat=5):
            mass = F(1)
            for v,p in zip([u,e1,e2,e3,e4],ps):mass *= p if v else 1-p
            x=u^e1;a=x^e2;b=a^e3;y=b^u^e4;table[8*x+4*a+2*b+y]+=mass
            for forced in range(2):
                if forced^e2^e3^u^e4:expected[forced]+=mass
        joint={'version':1,'variables':[{'name':n,'states':2}for n in ['X','A','B','Y']], 'probabilities':list(map(encoded,table))}
        a=frontdoor_effect(g,'X','Y',['B','A'],joint)
        self.assertEqual(a['positive_xz_cells'],8)
        self.assertEqual([F(*r['probability'])for r in a['rows']if r['outcome_value']==1],expected)
        g={'version':1,'nodes':['U','X','Y'],'edges':[['U','X'],['U','Y']]}
        joint={'version':1,'variables':[{'name':'X','states':2},{'name':'Y','states':2}], 'probabilities':[[1,10],[2,10],[3,10],[4,10]]}
        a=frontdoor_effect(g,'X','Y',[],joint)
        self.assertEqual([F(*r['probability'])for r in a['rows']if r['outcome_value']==1],[F(3,5),F(3,5)])

    def test_refuses_invalid_graph_or_zero_support_without_partial_result(self):
        joint,_=noise_reference(F(1,3),F(1,4),F(1,4),F(1,8))
        g=copy.deepcopy(G);g['edges'].append(['X','Y'])
        with self.assertRaisesRegex(ValueError,'fails'):frontdoor_effect(g,'X','Y',['M'],joint)
        joint['probabilities']=[1,0,0,0,0,0,0,0]
        with self.assertRaisesRegex(ValueError,'positive P'):frontdoor_effect(G,'X','Y',['M'],joint)

    def test_strict_rational_schema_lcm_cell_and_name_guards(self):
        good,_=noise_reference(F(1,3),F(1,4),F(1,4),F(1,8))
        for raw in [True,0.1,'0.1',[1,0],[1,-1],[1,True],[-1,8],[9,8],[1,1000001],[1000001,1000001],{},[1,2,3]]:
            bad=copy.deepcopy(good);bad['probabilities'][0]=raw
            with self.assertRaises((ValueError,TypeError)):frontdoor_effect(G,'X','Y',['M'],bad)
        for edit in [{'extra':0},{'version':True},{'probabilities':[0]*8}, {'probabilities':[0]*7},
                     {'variables':[{'name':'X','states':2},{'name':'M','states':2},{'name':'M','states':2}]},
                     {'variables':[{'name':'X','states':True},{'name':'M','states':2},{'name':'Y','states':2}]}]:
            with self.assertRaises((ValueError,TypeError)):frontdoor_effect(G,'X','Y',['M'],dict(good,**edit))
        bad=copy.deepcopy(good);bad['probabilities']=[[1,997],[1,991]]+[0]*6
        with self.assertRaises(ValueError):frontdoor_effect(G,'X','Y',['M'],bad)

    def test_input_output_isolation(self):
        joint,_=noise_reference(F(1,3),F(1,4),F(1,4),F(1,8));before=copy.deepcopy(joint)
        a=frontdoor_effect(G,'X','Y',['M'],joint);snapshot=copy.deepcopy(a)
        self.assertEqual(joint,before);a['observed_joint']['probabilities'].clear();a['rows'].clear()
        self.assertEqual(frontdoor_effect(G,'X','Y',['M'],joint),snapshot)

    def test_demo_literal_joint_and_observed_fraction_values(self):
        from causallab.frontdoor_effect import frontdoor_effect_demo
        joint,_=noise_reference(F(1,3),F(1,4),F(1,4),F(1,8));d=frontdoor_effect_demo()
        self.assertEqual(d['effect'],frontdoor_effect(G,'X','Y',['M'],joint))
        self.assertEqual([r['p_y1_given_x']for r in d['observational_rows']],[[41,112],[37,80]])
        self.assertEqual(json.loads((ROOT/'examples/frontdoor_joint.json').read_text()),joint)
        self.assertEqual(json.loads((ROOT/'examples/frontdoor_graph.json').read_text()),G)

    def test_full_cell_boundary_and_lcm_preflight(self):
        names=['X','A','B','C','D','Y'];g={'version':1,'nodes':names,'edges':[]}
        joint={'version':1,'variables':[{'name':n,'states':4}for n in names],'probabilities':[[1,4096]for _ in range(4096)]}
        a=frontdoor_effect(g,'X','Y',['A','B','C','D'],joint)
        self.assertEqual(a['cell_count'],4096);self.assertEqual(a['positive_xz_cells'],1024)
        self.assertEqual(len(a['rows']),16);self.assertTrue(all(r['probability']==[1,4]for r in a['rows']))
        bad=copy.deepcopy(joint);bad['variables'].append({'name':'E','states':4});g['nodes'].append('E')
        with self.assertRaisesRegex(ValueError,'4096'):frontdoor_effect(g,'X','Y',['A','B','C','D','E'],bad)
        good,_=noise_reference(F(1,3),F(1,4),F(1,4),F(1,8));good['probabilities']=[[1,997],[1,991],[1,983]]+[0]*5
        with self.assertRaisesRegex(ValueError,'LCM'):frontdoor_effect(G,'X','Y',['M'],good)

    def test_cli_exact_effect_and_joint_file_errors(self):
        joint,_=noise_reference(F(1,3),F(1,4),F(1,4),F(1,8))
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'g.json';q=Path(tmp)/'p.json';p.write_text(json.dumps(G));q.write_text(json.dumps(joint))
            args=[sys.executable,'-m','causallab','frontdoor-effect',str(p),'--treatment','X','--outcome','Y','--mediators','M','--joint',str(q)]
            a=subprocess.run(args,cwd=ROOT,text=True,capture_output=True)
            self.assertEqual(a.returncode,0,a.stderr);self.assertEqual(json.loads(a.stdout),frontdoor_effect(G,'X','Y',['M'],joint))
            for text in ['{}','{"version":1,"version":1}',' '*262145]:
                q.write_text(text);a=subprocess.run(args,cwd=ROOT,text=True,capture_output=True)
                self.assertEqual(a.returncode,2);self.assertEqual(a.stdout,'');self.assertNotIn('Traceback',a.stderr)


if __name__=='__main__':unittest.main()
