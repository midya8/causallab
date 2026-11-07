import copy
from fractions import Fraction as F
import itertools
import json
import math
import random
import unittest
from unittest.mock import patch
from causallab import Model,infer
from causallab.demo import demo,probability
from causallab.fixtures import confounded,observational_twins


class InferenceTests(unittest.TestCase):
    def test_truth_table_order_rightmost_fastest(self):
        m={'version':1,'exogenous':[{'name':'A','probabilities':[.2,.8]},{'name':'B','probabilities':[.1,.3,.6]}],
            'nodes':[{'name':'Y','states':4,'inputs':['A','B'],'table':[0,1,2,3,0,1]}]}
        p=infer(m,['Y']);expected=[.2*.1+.8*.3,.2*.3+.8*.6,.2*.6,.8*.1]
        for row,e in zip(p['distribution'],expected):self.assertAlmostEqual(row['probability'],e,14)
    def test_constant_node_and_unused_noise_integrate_out(self):
        m={'version':1,'exogenous':[{'name':'U','probabilities':[.3,.7]}],'nodes':[{'name':'Y','states':2,'inputs':[],'table':[1]}]}
        p=infer(m,['Y']);self.assertEqual([r['probability'] for r in p['distribution']],[0.,1.]);self.assertIsNone(p['distribution'][0]['log_probability'])
        self.assertEqual(p['positive_prior_worlds'],2)
    def test_full_joint_independent_fraction_enumeration(self):
        p=infer(confounded(),['Z','X','Y']);expected={key:F(0) for key in itertools.product(range(2),repeat=3)}
        # No production truth tables, evaluation or probability normalization.
        for u,v,w in itertools.product(range(2),range(5),range(10)):
            z=u;x=int(v<(4 if z==0 else 1));y=int(w<[[9,8],[3,2]][z][x]);expected[z,x,y]+=F(1,100)
        for row in p['distribution']:self.assertAlmostEqual(row['probability'],float(expected[tuple(row['values'][n] for n in ['Z','X','Y'])]),14)
    def test_confounded_observations_and_interventions_hand_values(self):
        m=confounded()
        for x,observed,acted in [(0,F(42,100),F(6,10)),(1,F(68,100),F(5,10))]:
            self.assertAlmostEqual(probability(infer(m,['Y'],evidence={'X':x}),Y=1),float(observed),14)
            self.assertAlmostEqual(probability(infer(m,['Y'],interventions={'X':x}),Y=1),float(acted),14)
    def test_stratified_backdoor_adjustment_matches_action(self):
        m=confounded()
        for x in [0,1]:
            adjusted=sum(probability(infer(m,['Y'],{'X':x,'Z':z}),Y=1)*probability(infer(m,['Z']),Z=z) for z in [0,1])
            self.assertAlmostEqual(adjusted,probability(infer(m,['Y'],interventions={'X':x}),Y=1),14)
    def test_observational_equivalence_not_interventional_equivalence(self):
        a,b=observational_twins().values()
        self.assertEqual(infer(a,['X','Y'])['distribution'],infer(b,['X','Y'])['distribution'])
        self.assertEqual(probability(infer(a,['Y'],interventions={'X':1}),Y=1),1.)
        self.assertEqual(probability(infer(b,['Y'],interventions={'X':1}),Y=1),.5)
    def test_counterfactual_shared_noise_and_factual_abduction(self):
        for n,m in observational_twins().items():
            r=infer(m,['Y'],{'X':0,'Y':0},{'X':1},counterfactual=True,trace=True)
            self.assertEqual(probability(r,Y=1),1. if n=='direct' else 0.)
            t=r['world_trace'];self.assertEqual(len(t),1);self.assertEqual(t[0]['exogenous'],{'U':0})
            self.assertEqual(t[0]['factual'],{'X':0,'Y':0});self.assertEqual(t[0]['prediction']['X'],1)
            self.assertEqual(t[0]['posterior_exogenous_probability'],1.);self.assertEqual(r['evidence_probability'],.5)
    def test_query_post_action_evidence_is_not_factual(self):
        m=observational_twins()['direct']
        with self.assertRaisesRegex(ValueError,'intervened model'):infer(m,['Y'],{'X':0},{'X':1})
        self.assertEqual(probability(infer(m,['Y'],{'X':0},{'X':1},counterfactual=True),Y=1),1.)
    def test_counterfactual_matches_independent_fraction_unit_enumeration(self):
        m=confounded();numerator=F(0);denominator=F(0)
        for z,v,w in itertools.product(range(2),range(5),range(10)):
            x=int(v<(4 if z==0 else 1));y=int(w<[[9,8],[3,2]][z][x])
            if x==0 and y==1:
                denominator+=F(1,100)
                if w<[8,2][z]:numerator+=F(1,100)
        result=infer(m,['Y'],{'X':0,'Y':1},{'X':1},counterfactual=True)
        self.assertAlmostEqual(probability(result,Y=1),float(numerator/denominator),14)
    def test_counterfactual_empty_evidence_equals_intervention(self):
        for m in [confounded(),*observational_twins().values()]:
            a=infer(m,['X','Y'],interventions={'X':1});b=infer(m,['X','Y'],interventions={'X':1},counterfactual=True)
            self.assertEqual(a['distribution'],b['distribution'])
    def test_counterfactual_no_action_equals_observational_conditioning(self):
        m=confounded();self.assertEqual(infer(m,['Y'],{'X':1})['distribution'],infer(m,['Y'],{'X':1},counterfactual=True)['distribution'])
    def test_consistency_when_factual_action_already_holds(self):
        m=confounded();self.assertEqual(infer(m,['Y'],{'X':1})['distribution'],infer(m,['Y'],{'X':1},{'X':1},counterfactual=True)['distribution'])
    def test_intervention_replaces_whole_equation_and_multiple_nodes(self):
        m=confounded();r=infer(m,['Z','X','Y'],interventions={'X':0,'Y':1},trace=True)
        for row in r['world_trace']:self.assertEqual(row['prediction']['X'],0);self.assertEqual(row['prediction']['Y'],1)
        self.assertEqual(probability(r,Z=0),.5);self.assertEqual(probability(r,Y=1),1.)
    def test_joint_marginalization_and_query_order(self):
        m=confounded();a=infer(m,['Y','Z']);b=infer(m,['Z','Y']);c=infer(m,['Y'])
        for y in [0,1]:self.assertAlmostEqual(probability(a,Y=y),probability(c,Y=y),14)
        for z,y in itertools.product(range(2),repeat=2):self.assertAlmostEqual(probability(a,Y=y,Z=z),probability(b,Y=y,Z=z),14)
    def test_trace_counts_and_posterior_reconstruct_distribution(self):
        r=infer(confounded(),['Y'],{'X':0,'Y':1},{'X':1},counterfactual=True,trace=True)
        self.assertEqual(r['world_slots'],100);self.assertEqual(r['positive_prior_worlds'],100)
        self.assertEqual(r['evaluations'],{'factual':100,'action':r['matching_worlds']})
        p=sum(t['posterior_exogenous_probability'] for t in r['world_trace'] if t['prediction']['Y']==1)
        self.assertAlmostEqual(p,probability(r,Y=1),14)
        self.assertAlmostEqual(sum(t['posterior_exogenous_probability'] for t in r['world_trace']),1.,14)
    def test_zero_noise_bins_and_impossible_evidence(self):
        m=observational_twins()['direct'];m['exogenous'][0]['probabilities']=[1.,0.]
        r=infer(m,['Y']);self.assertEqual(r['world_slots'],2);self.assertEqual(r['positive_prior_worlds'],1)
        with self.assertRaisesRegex(ValueError,'zero probability'):infer(m,['Y'],{'X':1})
    def test_rare_evidence_does_not_become_impossible_by_underflow(self):
        m={'version':1,'exogenous':[{'name':n,'probabilities':[1e-300,1.]} for n in ['A','B','C','D']],
            'nodes':[{'name':'E','states':2,'inputs':['A','B','C','D'],'table':[1]+[0]*15},{'name':'Y','states':2,'inputs':[],'table':[1]}]}
        r=infer(m,['Y'],{'E':1},trace=True)
        self.assertEqual(r['evidence_probability'],0.);self.assertAlmostEqual(r['log_evidence_probability'],4*math.log(1e-300),12)
        self.assertEqual(probability(r,Y=1),1.);self.assertEqual(r['matching_worlds'],1)
    def test_rare_output_keeps_finite_log_probability(self):
        m={'version':1,'exogenous':[{'name':n,'probabilities':[1e-300,1.]} for n in ['A','B']],
            'nodes':[{'name':'Y','states':2,'inputs':['A','B'],'table':[1,0,0,0]}]}
        r=infer(m,['Y']);row=r['distribution'][1]
        self.assertEqual(row['probability'],0.);self.assertTrue(math.isfinite(row['log_probability']));self.assertAlmostEqual(row['log_probability'],2*math.log(1e-300),12)
    def test_immutability_fingerprint_copy_and_rng(self):
        payload=confounded();original=copy.deepcopy(payload);model=Model.from_dict(payload);before=random.getstate()
        a=infer(model,['Y'],trace=True);self.assertEqual(before,random.getstate());self.assertEqual(original,payload)
        self.assertEqual(a,infer(model,['Y'],trace=True));self.assertEqual(model.fingerprint,Model.from_dict(original).fingerprint)
        d=model.as_dict();d['nodes'][0]['table'][0]=1;self.assertEqual(model.as_dict(),original)
        a['world_trace'][0]['prediction']['Y']=99;self.assertNotEqual(a['world_trace'][0]['post_action']['Y'],99)
        self.assertEqual(infer(model,['Y'])['model_sha256'],model.fingerprint)
    def test_demo_both_failures_and_adjustment(self):
        v=demo();self.assertTrue(all(v['checks'].values()));json.dumps(v,allow_nan=False)
        self.assertAlmostEqual(v['confounding']['observational_difference'],.26,14);self.assertAlmostEqual(v['confounding']['interventional_difference'],-.1,14)


class ValidationTests(unittest.TestCase):
    def test_reject_model_fields_names_and_probability_errors(self):
        base=observational_twins()['direct']
        invalid=[None,dict(base,extra=1),dict(base,version=True),dict(base,version=2),dict(base,exogenous=[]),dict(base,nodes=[])]
        for key,value in [('name','bad-name'),('name','U'*25),('probabilities',[True,0.]),('probabilities',[float('nan'),.5]),('probabilities',[.2,.2]),('probabilities',[-.1,1.1]),('probabilities',[1.])]:
            m=copy.deepcopy(base);m['exogenous'][0][key]=value;invalid.append(m)
        for m in invalid:
            with self.assertRaises(ValueError):Model.from_dict(m)
    def test_reject_cycles_forward_refs_duplicate_names_and_truth_tables(self):
        base=observational_twins()['direct'];invalid=[]
        for changes in [dict(name='U'),dict(name='X'),dict(states=True),dict(states=5),dict(inputs=['Y']),dict(inputs=['X','X']),dict(inputs=['missing']),dict(table=[True,1]),dict(table=[0]),dict(table=[0,2]),dict(extra=1)]:
            m=copy.deepcopy(base);m['nodes'][1].update(changes);invalid.append(m)
        m=copy.deepcopy(base);m['nodes'][0]['inputs']=['Y'];invalid.append(m)
        for m in invalid:
            with self.assertRaises(ValueError):Model.from_dict(m)
    def test_reject_unknown_exogenous_duplicate_out_of_range_query(self):
        m=confounded()
        for q,e,d in [([],None,None),(['U'],None,None),(['Y','Y'],None,None),('Y',None,None),(['Y'],{'U':0},None),(['Y'],{'Y':True},None),(['Y'],{'Y':2},None),(['Y'],None,{'U':0}),(['Y'],None,{'Y':-1}),(['Y'],[],None)]:
            with self.assertRaises(ValueError):infer(m,q,e,d)
        for kwargs in [dict(counterfactual=1),dict(trace=0)]:
            with self.assertRaises(ValueError):infer(m,['Y'],**kwargs)
    def test_world_table_query_trace_caps_before_evaluation(self):
        def model(n):return {'version':1,'exogenous':[{'name':'U'+str(i),'probabilities':[1/16]*16} for i in range(n)],'nodes':[{'name':'Y','states':2,'inputs':[],'table':[1]}]}
        cases=[(model(5),['Y'],False)]
        m=model(4);m['nodes'][0]['inputs']=['U0','U1','U2','U3'];m['nodes'][0]['table']=[];cases.append((m,['Y'],False))
        m=model(3);m['nodes']=[{'name':'Y'+str(i),'states':4,'inputs':['U0','U1','U2'],'table':[0]*4096} for i in range(5)];cases.append((m,['Y0'],False))
        m=model(1);m['nodes']=[{'name':'Y'+str(i),'states':4,'inputs':[],'table':[0]} for i in range(7)];cases.append((m,['Y'+str(i) for i in range(7)],False))
        cases.append((model(4),['Y'],True))
        for m,q,trace in cases:
            with patch('causallab.core._evaluate') as evaluate:
                with self.assertRaises(ValueError):infer(m,q,trace=trace)
                evaluate.assert_not_called()
    def test_probability_sum_tolerance_is_normalized_not_silently_arbitrary(self):
        m=observational_twins()['direct'];m['exogenous'][0]['probabilities']=[.5,.5000000000001]
        compiled=Model.from_dict(m);self.assertEqual(math.fsum(compiled.noises[0].probabilities),1.)
        m['exogenous'][0]['probabilities']=[.5,.50001]
        with self.assertRaises(ValueError):Model.from_dict(m)


class ExhaustiveTests(unittest.TestCase):
    def test_all_64_tiny_boolean_mechanisms_against_fraction_units(self):
        for tx in itertools.product(range(2),repeat=2):
            for ty in itertools.product(range(2),repeat=4):
                m={'version':1,'exogenous':[{'name':'U','probabilities':[.3,.7]},{'name':'V','probabilities':[.6,.4]}],
                    'nodes':[{'name':'X','states':2,'inputs':['U'],'table':list(tx)},{'name':'Y','states':2,'inputs':['X','V'],'table':list(ty)}]}
                for action in [0,1]:
                    expected=sum(([F(3,10),F(7,10)][u]*[F(6,10),F(4,10)][v]) for u,v in itertools.product(range(2),repeat=2) if ty[action*2+v]==1)
                    self.assertAlmostEqual(probability(infer(m,['Y'],interventions={'X':action}),Y=1),float(expected),14)
                    denominator=sum(([F(3,10),F(7,10)][u]*[F(6,10),F(4,10)][v]) for u,v in itertools.product(range(2),repeat=2) if ty[tx[u]*2+v]==0)
                    if denominator:
                        numerator=sum(([F(3,10),F(7,10)][u]*[F(6,10),F(4,10)][v]) for u,v in itertools.product(range(2),repeat=2) if ty[tx[u]*2+v]==0 and ty[action*2+v]==1)
                        actual=probability(infer(m,['Y'],{'Y':0},{'X':action},counterfactual=True),Y=1)
                        self.assertAlmostEqual(actual,float(numerator/denominator),14)
                    else:
                        with self.assertRaises(ValueError):infer(m,['Y'],{'Y':0},{'X':action},counterfactual=True)
    def test_world_cap_boundary_and_no_trace_output(self):
        m={'version':1,'exogenous':[{'name':'U'+str(i),'probabilities':[1/16]*16} for i in range(4)],'nodes':[{'name':'Y','states':2,'inputs':[],'table':[1]}]}
        r=infer(m,['Y']);self.assertEqual(r['world_slots'],65536);self.assertEqual(r['positive_prior_worlds'],65536)
        self.assertEqual(probability(r,Y=1),1.);self.assertNotIn('world_trace',r)
    def test_directly_constructed_invalid_model_is_rejected(self):
        for m in [Model(('invalid',),()),Model((),()),Model((),('invalid',))]:
            with self.assertRaises(ValueError):infer(m,['Y'])
