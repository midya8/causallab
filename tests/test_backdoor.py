import copy
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest.mock import patch
from causallab import backdoor_criterion, backdoor_sets, graph_from_model, infer
from causallab.backdoor import backdoor_demo

ROOT=Path(__file__).resolve().parents[1]


def direct_backdoor_reference(g,x,y,z):
    """Original-DAG path test, first arrow INTO X, no outgoing-edge deletion.

    Independent descendant traversal and local collider checks. Returns the
    standard sufficient criterion, not the more general adjustment criterion.
    """
    arrows={tuple(e) for e in g['edges']};neighbors={n:set() for n in g['nodes']}
    for a,b in arrows:neighbors[a].add(b);neighbors[b].add(a)
    desc={n:set() for n in g['nodes']}
    for n in g['nodes']:
        stack=[n]
        while stack:
            a=stack.pop()
            for p,c in arrows:
                if p==a and c not in desc[n]:desc[n].add(c);stack.append(c)
    if desc[x]&set(z):return False
    def visit(path):
        if path[-1]==y:return True
        for n in neighbors[path[-1]]-set(path):
            if len(path)==1 and (n,x) not in arrows:continue
            if len(path)>1:
                a,b=path[-2:];collider=(a,b) in arrows and (n,b) in arrows
                if collider and not (desc[b]|{b})&set(z):continue
                if not collider and b in z:continue
            if visit(path+[n]):return True
        return False
    return not visit([x])


class BackdoorTests(unittest.TestCase):
    def setUp(self):
        self.g={'version':1,'nodes':['Z','X','Y'],'edges':[['Z','X'],['Z','Y'],['X','Y']]}

    def test_fixed_demo(self):
        d=backdoor_demo();self.assertEqual(len(d['cases']),4);self.assertEqual(len(d['checks']),6)
        self.assertTrue(all(d['checks'].values()))

    def test_confounder_and_outgoing_not_incoming_removal(self):
        a=backdoor_criterion(self.g,'X','Y');b=backdoor_criterion(self.g,'X','Y',['Z'])
        self.assertFalse(a['criterion_met']);self.assertTrue(b['criterion_met'])
        self.assertEqual(a['removed_outgoing_edges'],[['X','Y']])
        self.assertEqual(a['backdoor_separation']['effective_edges'],[['Z','X'],['Z','Y']])
        self.assertEqual(a['backdoor_separation']['interventions'],[])
        self.assertEqual(a['backdoor_separation']['moral_witness_path'],['X','Z','Y'])
        self.assertIsNone(a['formula']);self.assertIn('sum_z',b['formula'])

    def test_exhaustive_original_path_reference_all_fixed_order_dags(self):
        nodes=['A','B','C','D'];edges=list(itertools.combinations(nodes,2));checks=0;searches=0
        for mask in range(64):
            g={'version':1,'nodes':nodes,'edges':[list(e) for i,e in enumerate(edges) if mask&(1<<i)]}
            for x,y in itertools.permutations(nodes,2):
                available=[n for n in nodes if n not in [x,y]]
                search=backdoor_sets(g,x,y,available);valid=[];minimal=[]
                for row in search['subset_audits']:
                    z=row['adjustment'];expected=direct_backdoor_reference(g,x,y,z)
                    self.assertEqual(row['criterion_met'],expected,(g,x,y,z))
                    single=backdoor_criterion(g,x,y,z)
                    self.assertEqual(single['criterion_met'],expected)
                    self.assertEqual(single['forbidden_descendants'],row['forbidden_descendants'])
                    self.assertEqual(single['backdoor_separation']['moral_witness_path'],row['moral_witness_path'])
                    if expected:valid.append(z)
                    checks+=1
                for z in valid:
                    if not any(set(other)<set(z) for other in valid):minimal.append(z)
                self.assertEqual(search['valid_sets'],valid)
                self.assertEqual(search['inclusion_minimal_sets'],minimal)
                self.assertEqual(search['subsets_checked'],4);searches+=1
        self.assertEqual(checks,3072);self.assertEqual(searches,768)

    def test_collider_validity_is_not_upward_closed(self):
        c=backdoor_demo()['cases']['collider'];rows={tuple(r['adjustment']):r for r in c['subset_audits']}
        self.assertTrue(rows[()]['criterion_met']);self.assertFalse(rows[('C',)]['criterion_met'])
        self.assertEqual(rows[('C',)]['moral_witness_path'],['X','A','B','Y'])
        self.assertTrue(rows[('A','C')]['criterion_met']);self.assertTrue(rows[('B','C')]['criterion_met'])
        self.assertEqual(c['inclusion_minimal_sets'],[[]]);self.assertEqual(c['minimum_size'],0)
        self.assertEqual(len(c['valid_sets']),7)

    def test_descendant_exclusion_even_when_separated_after_cut(self):
        c=backdoor_demo()['cases']['mediator'];row=c['subset_audits'][1]
        self.assertEqual(row['adjustment'],['M']);self.assertEqual(row['forbidden_descendants'],['M'])
        self.assertTrue(row['backdoor_d_separated']);self.assertFalse(row['criterion_met'])
        self.assertEqual(c['valid_sets'],[[]])

    def test_non_mediator_descendant_still_excluded_by_this_sufficient_rule(self):
        g={'version':1,'nodes':['X','D','Y'],'edges':[['X','D'],['X','Y']]}
        r=backdoor_criterion(g,'X','Y',['D'])
        self.assertFalse(r['criterion_met']);self.assertEqual(r['forbidden_descendants'],['D'])
        self.assertTrue(r['backdoor_separation']['d_separated'])

    def test_transitive_descendants_original_graph(self):
        g={'version':1,'nodes':['X','A','B','C','Y'],
           'edges':[['X','A'],['A','B'],['B','C'],['C','Y']]}
        r=backdoor_criterion(g,'X','Y',['C'])
        self.assertEqual(r['treatment_descendants'],['A','B','C','Y'])
        self.assertEqual(r['forbidden_descendants'],['C'])

    def test_inclusion_minimal_is_not_only_minimum_cardinality(self):
        g={'version':1,'nodes':['X','Y','A','B','C'],
           'edges':[['A','X'],['B','A'],['C','A'],['B','Y'],['C','Y']]}
        r=backdoor_sets(g,'X','Y',['A','B','C'])
        self.assertEqual(r['inclusion_minimal_sets'],[['A'],['B','C']]);self.assertEqual(r['minimum_size'],1)
        self.assertEqual(len(r['valid_sets']),5)

    def test_candidates_never_silently_include_unavailable_common_cause(self):
        c=backdoor_demo()['cases']['hidden_common_cause']
        self.assertEqual(c['available'],['M']);self.assertEqual(c['valid_sets'],[])
        self.assertEqual(c['inclusion_minimal_sets'],[]);self.assertIsNone(c['minimum_size'])
        r=backdoor_sets(c['graph'],'X','Y',['U','M'])
        self.assertEqual(r['valid_sets'],[['U']])
        self.assertTrue(any('does not establish non-identifiability' in s for s in c['limits']))

    def test_empty_candidates_still_test_empty_adjustment(self):
        a=backdoor_sets(self.g,'X','Y',[]);self.assertEqual(a['subsets_checked'],1);self.assertEqual(a['valid_sets'],[])
        g={'version':1,'nodes':['X','Y'],'edges':[['X','Y']]}
        b=backdoor_sets(g,'X','Y',());self.assertEqual(b['valid_sets'],[[]]);self.assertEqual(b['inclusion_minimal_sets'],[[]])

    def test_full_1024_subset_boundary_and_no_early_exit(self):
        names=['Z'+str(i) for i in range(10)]
        g={'version':1,'nodes':['X','Y']+names,'edges':[['X','Y']]}
        r=backdoor_sets(g,'X','Y',names)
        self.assertEqual(r['subsets_checked'],1024);self.assertEqual(len(r['subset_audits']),1024)
        self.assertEqual(len(r['valid_sets']),1024);self.assertEqual(r['inclusion_minimal_sets'],[[]])
        self.assertEqual(r['subset_audits'][-1]['adjustment'],names)
        self.assertLess(len(json.dumps(r)),400000)  # this sparse fixture only, not a universal output-size promise

    def test_eleven_candidates_rejected_before_any_separation_call(self):
        names=['Z'+str(i) for i in range(11)]
        g={'version':1,'nodes':['X','Y']+names,'edges':[]}
        with patch('causallab.backdoor.d_separation',side_effect=AssertionError('unexpected evaluation')):
            with self.assertRaisesRegex(ValueError,'at most 10'):backdoor_sets(g,'X','Y',names)

    def test_single_set_checker_not_limited_to_ten(self):
        names=['Z'+str(i) for i in range(30)]
        g={'version':1,'nodes':['X','Y']+names,'edges':[]}
        self.assertTrue(backdoor_criterion(g,'X','Y',names)['criterion_met'])

    def test_original_full_scm_confounder_and_adjustment_formula(self):
        model=json.loads((ROOT/'examples/confounded.json').read_text())
        r=backdoor_sets(graph_from_model(model),'X','Y',['Z'])
        self.assertEqual(r['valid_sets'],[['Z']])
        joint=infer(model,['X','Y','Z'])['distribution']
        for x,expected in [(0,.6),(1,.5)]:
            total=0.
            for z in [0,1]:
                weight=sum(row['probability'] for row in joint if row['values']['Z']==z)
                denom=sum(row['probability'] for row in joint if row['values']['X']==x and row['values']['Z']==z)
                numer=sum(row['probability'] for row in joint if row['values']=={'X':x,'Y':1,'Z':z})
                self.assertGreater(denom,0);total+=weight*numer/denom
            direct=infer(model,['Y'],interventions={'X':x})['distribution'][1]['probability']
            self.assertAlmostEqual(total,expected);self.assertAlmostEqual(total,direct)

    def test_order_hash_and_no_mutation_or_rng_effects(self):
        g=backdoor_demo()['cases']['collider']['graph'];before=copy.deepcopy(g);state=random.getstate()
        a=backdoor_sets(g,'X','Y',['C','A','B']);g['nodes'].reverse();g['edges'].reverse()
        self.assertEqual(a,backdoor_sets(g,'X','Y',['B','C','A']))
        self.assertEqual(random.getstate(),state)
        a['graph']['nodes'].append('bad');a['limits'].append('bad')
        fresh=backdoor_sets(before,'X','Y',['A','B','C']);self.assertNotIn('bad',fresh['graph']['nodes'])
        self.assertNotIn('bad',fresh['limits'])

    def test_invalid_graph_target_and_adjustment_inputs(self):
        for graph,x,y,z in [(None,'X','Y',[]),(self.g,'X','X',[]),(self.g,'missing','Y',[]),
                            (self.g,['X'],'Y',[]),(self.g,'X','Y',['X']),
                            (self.g,'X','Y',['Y']),(self.g,'X','Y',['Z','Z']),
                            (self.g,'X','Y',['missing']),(self.g,'X','Y','Z'),
                            (self.g,'X','Y',None),(self.g,'X','Y',[True])]:
            with self.subTest(args=(graph,x,y,z)),self.assertRaises(ValueError):backdoor_criterion(graph,x,y,z)
        for available in [None,'Z',{'Z'},['Z','Z'],['X'],['Y'],['unknown']]:
            with self.subTest(available=available),self.assertRaises(ValueError):backdoor_sets(self.g,'X','Y',available)

    def test_both_failed_conditions_retained(self):
        g={'version':1,'nodes':['Z','X','Y','M'],
           'edges':[['Z','X'],['Z','Y'],['X','M'],['X','Y']]}
        r=backdoor_criterion(g,'X','Y',['M'])
        self.assertEqual(r['forbidden_descendants'],['M']);self.assertFalse(r['backdoor_separation']['d_separated'])
        self.assertFalse(r['criterion_met'])

    def cli(self,*args):
        return subprocess.run([sys.executable,'-m','causallab',*args],cwd=ROOT,capture_output=True,text=True)

    def test_cli_demo_criterion_and_search(self):
        p=self.cli('backdoor-demo');self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout),backdoor_demo())
        for command,flag,fn in [('backdoor','--adjust',backdoor_criterion),('backdoor-sets','--available',backdoor_sets)]:
            p=self.cli(command,'examples/backdoor_graph.json','--treatment','X','--outcome','Y',flag,'Z')
            self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout),fn(self.g,'X','Y',['Z']))

    def test_cli_empty_candidates_and_invalid_inputs(self):
        p=self.cli('backdoor-sets','examples/backdoor_graph.json','--treatment','X','--outcome','Y','--available','')
        self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout)['subsets_checked'],1)
        for command,tail in [('backdoor',['--adjust','X']),('backdoor',['--adjust','unknown']),
                             ('backdoor-sets',[]),('backdoor-sets',['--available','Z,Z'])]:
            p=self.cli(command,'examples/backdoor_graph.json','--treatment','X','--outcome','Y',*tail)
            self.assertEqual(p.returncode,2);self.assertEqual(p.stdout,'');self.assertNotIn('Traceback',p.stderr)
