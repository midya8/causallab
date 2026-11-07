import copy
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest
from causallab import d_separation, graph_from_model, Model, infer
from causallab.graph import graph_demo

ROOT = Path(__file__).resolve().parents[1]


def active_path_reference(g, x, y, z, cuts=()):
    """Independent simple-path search, with descendant-open collider rule.

    Does not build ancestors of the query, moralize, or use production helpers.
    """
    arrows = {tuple(e) for e in g['edges'] if e[1] not in cuts}
    skeleton = {v: set() for v in g['nodes']}
    descendants = {v: {v} for v in g['nodes']}
    for a, b in arrows:
        skeleton[a].add(b); skeleton[b].add(a)
    for start in g['nodes']:
        stack = [start]
        while stack:
            a = stack.pop()
            for p, c in arrows:
                if p == a and c not in descendants[start]:
                    descendants[start].add(c); stack.append(c)
    def search(path):
        if path[-1] in y: return True
        for v in skeleton[path[-1]] - set(path):
            if len(path) >= 2:
                a, b = path[-2:]
                collider = (a, b) in arrows and (v, b) in arrows
                if collider and not descendants[b] & set(z): continue
                if not collider and b in z: continue
            if search(path + [v]): return True
        return False
    return not any(search([v]) for v in x)


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.g = {'version': 1, 'nodes': ['X','C','Y','D'],
                  'edges': [['X','C'],['Y','C'],['C','D']]}

    def test_fixed_demo(self):
        d = graph_demo()
        self.assertTrue(d['checks']['all_expected'])
        self.assertEqual(len(d['cases']), 8)
        self.assertEqual({k:r['d_separated'] for k,r in d['cases'].items()}, d['expected_separated'])

    def test_exhaustive_four_node_dags_singleton_queries_and_all_surgeries(self):
        nodes = ['A','B','C','D']; possible = list(itertools.combinations(nodes, 2)); count = 0
        for mask in range(64):
            g = {'version':1,'nodes':nodes,'edges':[list(e) for i,e in enumerate(possible) if mask & (1 << i)]}
            for x, y in itertools.permutations(nodes, 2):
                rest = [n for n in nodes if n not in [x,y]]
                for zmask in range(4):
                    z = [n for i,n in enumerate(rest) if zmask & (1 << i)]
                    for cmask in range(16):
                        cuts = [n for i,n in enumerate(nodes) if cmask & (1 << i)]
                        result = d_separation(g, [x], [y], z, interventions=cuts)
                        self.assertEqual(result['d_separated'], active_path_reference(g,{x},{y},z,cuts),
                                         (g,x,y,z,cuts))
                        count += 1
        self.assertEqual(count, 49152)

    def test_exhaustive_set_queries_four_node_dags(self):
        nodes = ['A','B','C','D']; possible = list(itertools.combinations(nodes, 2)); count = 0
        for mask in range(64):
            g = {'version':1,'nodes':nodes,'edges':[list(e) for i,e in enumerate(possible) if mask & (1 << i)]}
            for labels in itertools.product(range(4), repeat=4):
                groups = [[v for v,l in zip(nodes,labels) if l==i] for i in range(1,4)]
                x,y,z = groups
                if not x or not y: continue
                self.assertEqual(d_separation(g,x,y,z)['d_separated'], active_path_reference(g,set(x),set(y),z))
                count += 1
        self.assertEqual(count, 7040)

    def test_ancestor_restriction_before_moralization(self):
        r = d_separation(self.g,['X'],['Y'])
        self.assertTrue(r['d_separated']); self.assertEqual(r['ancestors'],['X','Y'])
        self.assertEqual(r['pruned_nodes'],['C','D']); self.assertEqual(r['moral_edges'],[])

    def test_conditioned_collider_and_descendant_open(self):
        for z in [['C'],['D']]:
            r = d_separation(self.g,['X'],['Y'],z)
            self.assertFalse(r['d_separated']); self.assertEqual(r['moral_witness_path'],['X','Y'])
            self.assertEqual(r['moral_witness_edges'][0]['origins'],[{'kind':'coparents','child':'C'}])
            self.assertIn('C', r['ancestors'])

    def test_surgery_is_not_conditioning(self):
        conditioned=d_separation(self.g,['X'],['Y'],['C'])
        cut=d_separation(self.g,['X'],['Y'],interventions=['C'])
        self.assertFalse(conditioned['d_separated']); self.assertTrue(cut['d_separated'])
        self.assertEqual(cut['removed_incoming_edges'],[['X','C'],['Y','C']])
        self.assertEqual(cut['effective_edges'],[['C','D']])
        self.assertEqual(cut['given'],[])

    def test_every_demo_witness_and_separation_certificate(self):
        for r in graph_demo()['cases'].values():
            edges={tuple(e['nodes']) for e in r['remaining_edges']}
            reached=set(r['reachable_from_x']); z=set(r['given'])
            self.assertTrue(set(r['x'])<=reached); self.assertFalse(z & reached)
            for a,b in edges: self.assertEqual(a in reached,b in reached)
            path=r['moral_witness_path']
            self.assertEqual(r['d_separated'],not bool(path))
            if path:
                self.assertIn(path[0],r['x']);self.assertIn(path[-1],r['y'])
                self.assertEqual(len(path),len(set(path)));self.assertFalse(set(path)&z)
                for a,b in zip(path,path[1:]):self.assertIn(tuple(sorted((a,b))),edges)
            else:self.assertFalse(reached & set(r['y']))
            effective={tuple(e) for e in r['effective_edges']}
            for e in r['moral_edges']:
                a,b=e['nodes']
                for origin in e['origins']:
                    if origin['kind']=='directed':
                        self.assertIn((origin['parent'],origin['child']),effective)
                        self.assertEqual({a,b},{origin['parent'],origin['child']})
                    else:
                        c=origin['child'];self.assertIn(c,r['ancestors'])
                        self.assertIn((a,c),effective);self.assertIn((b,c),effective)

    def test_multiple_sources_targets_and_all_reachable(self):
        g={'version':1,'nodes':['A','B','C','D','E'],'edges':[['A','C'],['C','D']]}
        r=d_separation(g,['B','A'],['E','D'])
        self.assertEqual(r['moral_witness_path'],['A','C','D'])
        self.assertEqual(r['reachable_from_x'],['A','B','C','D'])
        self.assertTrue(d_separation(g,['A','B'],['D','E'],['C'])['d_separated'])

    def test_order_invariance_and_hashes(self):
        a=d_separation(self.g,['Y'],['X'],['D'],interventions=['C','X'])
        b=copy.deepcopy(self.g);b['nodes'].reverse();b['edges'].reverse()
        self.assertEqual(a,d_separation(b,['Y'],['X'],('D',),interventions=['X','C']))
        self.assertNotEqual(a['graph_sha256'],a['effective_graph_sha256'])
        r=d_separation(self.g,['X'],['Y']);self.assertEqual(r['graph_sha256'],r['effective_graph_sha256'])

    def test_non_topological_input_and_disconnected_nodes(self):
        g={'version':1,'nodes':['Z','B','A'],'edges':[['B','A'],['Z','B']]}
        self.assertFalse(d_separation(g,['Z'],['A'])['d_separated'])
        self.assertTrue(d_separation(g,['Z'],['A'],['B'])['d_separated'])

    def test_overlapping_moral_edge_origins_retained(self):
        g={'version':1,'nodes':['A','B','C','D'],
           'edges':[['A','B'],['A','C'],['B','C'],['A','D'],['B','D']]}
        r=d_separation(g,['A'],['B'],['C','D'])
        edge=next(e for e in r['moral_edges'] if e['nodes']==['A','B'])
        self.assertEqual(edge['origins'],[{'kind':'directed','parent':'A','child':'B'},
                                        {'kind':'coparents','child':'C'},{'kind':'coparents','child':'D'}])

    def test_model_conversion_keeps_shared_noise(self):
        m=json.loads((ROOT/'examples/common.json').read_text())
        g=graph_from_model(m)
        self.assertEqual(g,graph_from_model(Model.from_dict(m)))
        self.assertIn(['U','X'],g['edges']);self.assertIn(['U','Y'],g['edges'])
        self.assertFalse(d_separation(g,['X'],['Y'])['d_separated'])
        self.assertTrue(d_separation(g,['X'],['Y'],['U'])['d_separated'])
        self.assertTrue(d_separation(g,['X'],['Y'],interventions=['X'])['d_separated'])

    def test_graph_connection_does_not_imply_dependence(self):
        m={'version':1,'exogenous':[{'name':n,'probabilities':[.5,.5]} for n in ['U','V']],
           'nodes':[{'name':'X','states':2,'inputs':['U'],'table':[0,1]},
                    {'name':'Y','states':2,'inputs':['X','V'],'table':[0,1,1,0]}]}
        self.assertFalse(d_separation(graph_from_model(m),['X'],['Y'])['d_separated'])
        joint=infer(m,['X','Y'])
        self.assertEqual([r['probability'] for r in joint['distribution']],[.25]*4)

    def test_input_and_output_mutation_isolation_rng(self):
        state=random.getstate();before=copy.deepcopy(self.g)
        r=d_separation(self.g,['X'],['Y'],['D']);self.assertEqual(before,self.g)
        r['graph']['nodes'].append('bad');r['effective_edges'][0].append('bad')
        self.assertEqual(before,self.g);self.assertEqual(random.getstate(),state)
        self.assertFalse(d_separation(self.g,['X'],['Y'],['D'])['d_separated'])

    def test_node_and_edge_capacity_boundaries(self):
        nodes=['N'+str(i) for i in range(32)]
        edges=[list(e) for e in itertools.combinations(nodes,2)][:128]
        g={'version':1,'nodes':nodes,'edges':edges}
        r=d_separation(g,[nodes[0]],[nodes[-1]])
        self.assertFalse(r['d_separated']);self.assertEqual(len(r['effective_edges']),128)
        g['edges'].append([nodes[5],nodes[31]])
        with self.assertRaises(ValueError):d_separation(g,[nodes[0]],[nodes[-1]])
        g['edges']=[];g['nodes'].append('extra')
        with self.assertRaises(ValueError):d_separation(g,[nodes[0]],[nodes[-1]])

    def test_graph_schema_errors(self):
        bad=[None,[],{},dict(self.g,extra=1),dict(self.g,version=True),dict(self.g,version=2),
             dict(self.g,nodes=[]),dict(self.g,nodes=['X','X']),dict(self.g,nodes=['bad name','Y']),
             dict(self.g,edges='X'),dict(self.g,edges=[['X','X']]),dict(self.g,edges=[['X','missing']]),
             dict(self.g,edges=[['X','C'],['X','C']]),dict(self.g,edges=[['X']]),
             dict(self.g,edges=[('X','C')]),dict(self.g,edges=[['X','C'],['C','X']])]
        for g in bad:
            with self.subTest(g=g),self.assertRaises(ValueError):d_separation(g,['X'],['Y'])

    def test_query_schema_errors(self):
        for x,y,z,cuts in [([],['Y'],[],[]),('X',['Y'],[],[]),(['X'],[],[],[]),
                            (['X','X'],['Y'],[],[]),(['X'],['X'],[],[]),(['X'],['Y'],['X'],[]),
                            (['X'],['Y'],['Y'],[]),(['X'],['Y'],['C','C'],[]),
                            (['missing'],['Y'],[],[]),(['X'],['Y'],[],['missing']),
                            (['X'],['Y'],[],['C','C']),(['X'],['Y'],set(),[]),
                            (['X'],['Y'],[],None),([True],['Y'],[],[])]:
            with self.subTest(args=(x,y,z,cuts)),self.assertRaises(ValueError):
                d_separation(self.g,x,y,z,interventions=cuts)

    def test_model_conversion_revalidates(self):
        with self.assertRaises(ValueError):graph_from_model(self.g)
        with self.assertRaises(ValueError):graph_from_model(Model((),()))

    def cli(self,*args):
        return subprocess.run([sys.executable,'-m','causallab',*args],cwd=ROOT,capture_output=True,text=True)

    def test_cli_graph_demo_and_conditioning(self):
        p=self.cli('graph-demo');self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(json.loads(p.stdout),graph_demo())
        p=self.cli('dsep','examples/collider_graph.json','--x','X','--y','Y','--given','D')
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(json.loads(p.stdout),d_separation(self.g,['X'],['Y'],['D']))

    def test_cli_cuts(self):
        p=self.cli('dsep','examples/collider_graph.json','--x','X','--y','Y','--cut','C')
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(json.loads(p.stdout),d_separation(self.g,['X'],['Y'],interventions=['C']))

    def test_cli_failures(self):
        for args in [('dsep','examples/collider_graph.json','--x','X','--y','X'),
                     ('dsep','examples/collider_graph.json','--x','X','--y','Y','--given','Y'),
                     ('dsep','examples/collider_graph.json','--x','X','--y','Y','--cut','unknown'),
                     ('dsep','examples/confounded.json','--x','X','--y','Y'),
                     ('dsep','examples/collider_graph.json','--x','X')]:
            p=self.cli(*args);self.assertEqual(p.returncode,2);self.assertEqual(p.stdout,'')
            self.assertNotIn('Traceback',p.stderr)
