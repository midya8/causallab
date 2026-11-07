import copy
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from causallab import joint_counterfactual, infer, Model
from causallab.multiworld import response_models, multiworld_demo

S=[{'name':'off','query':['Y'],'do':{'X':0}}, {'name':'on','query':['Y'],'do':{'X':1}}]


def probabilities(result):return [r['probability'] for r in result['distribution']]


class MultiWorldTests(unittest.TestCase):
    def setUp(self):self.models=response_models()

    def test_same_observation_and_all_action_marginals_different_joint(self):
        for model in self.models.values():
            self.assertEqual(probabilities(infer(model,['X','Y'])),[.25]*4)
            for x in [0,1]:self.assertEqual(probabilities(infer(model,['Y'],interventions={'X':x})),[.5,.5])
        self.assertEqual(probabilities(joint_counterfactual(self.models['stable'],S)),[.5,0,0,.5])
        self.assertEqual(probabilities(joint_counterfactual(self.models['flip'],S)),[0,.5,.5,0])

    def test_factual_abduction_not_fresh_noise(self):
        for label,expected in [('stable',[1,0,0,0]),('flip',[0,1,0,0])]:
            result=joint_counterfactual(self.models[label],S,{'X':0,'Y':0},trace=True)
            self.assertEqual(probabilities(result),expected);self.assertEqual(result['evidence_probability'],.25)
            self.assertEqual(result['world_trace'][0]['exogenous'],{'U':0,'V':0})
            self.assertEqual(result['evaluations'],{'factual':4,'action':2})

    def test_single_scenario_equals_original_counterfactual(self):
        for model in self.models.values():
            for evidence in [{},{'X':0},{'Y':1},{'X':1,'Y':0}]:
                r=joint_counterfactual(model,[S[1]],evidence)
                old=infer(model,['Y'],evidence,{'X':1},counterfactual=True)
                self.assertEqual(probabilities(r),probabilities(old));self.assertEqual(r['log_evidence_probability'],old['log_evidence_probability'])

    def test_marginals_equal_each_original_query(self):
        for model in self.models.values():
            for evidence in [{},{'Y':0}]:
                result=joint_counterfactual(model,S,evidence)
                for spec in S:
                    marginal=[sum(r['probability'] for r in result['distribution'] if r['values'][spec['name']]['Y']==y) for y in [0,1]]
                    self.assertEqual(marginal,probabilities(infer(model,['Y'],evidence,spec['do'],counterfactual=True)))

    def test_repeated_worlds_perfectly_coupled_not_independent(self):
        specs=copy.deepcopy(S);specs[1]['do']={'X':0}
        self.assertEqual(probabilities(joint_counterfactual(self.models['flip'],specs)),[.5,0,0,.5])

    def test_empty_actions_reproduce_factual_joint(self):
        specs=[{'name':'a','query':['X','Y'],'do':{}},{'name':'b','query':['Y'],'do':{}}]
        result=joint_counterfactual(self.models['flip'],specs,{'X':0})
        for row in result['distribution']:
            if row['probability']:self.assertEqual(row['values']['a']['Y'],row['values']['b']['Y']);self.assertEqual(row['values']['a']['X'],0)

    def test_scenario_permutation_and_nested_query_order(self):
        specs=[{'name':'a','query':['Y','X'],'do':{'X':0}},{'name':'b','query':['X','Y'],'do':{'X':1}}]
        left=joint_counterfactual(self.models['flip'],specs);right=joint_counterfactual(self.models['flip'],list(reversed(specs)))
        keyed=lambda r:{json.dumps(x['values'],sort_keys=True):x['probability'] for x in r['distribution']}
        self.assertEqual(keyed(left),keyed(right));self.assertEqual(list(left['distribution'][0]['values']['a']),['Y','X'])

    def test_multi_node_intervention_and_no_sequential_contamination(self):
        specs=[{'name':'a','query':['X','Y'],'do':{'X':1,'Y':1}}, {'name':'b','query':['X','Y'],'do':{'X':0}}]
        r=joint_counterfactual(self.models['stable'],specs,{'Y':0})
        row=next(x for x in r['distribution'] if x['probability'])
        self.assertEqual(row['values'],{'a':{'X':1,'Y':1},'b':{'X':0,'Y':0}})

    def test_trace_reconstructs_joint_and_is_fresh(self):
        specs=copy.deepcopy(S);r=joint_counterfactual(self.models['flip'],specs,trace=True)
        self.assertAlmostEqual(sum(t['posterior_exogenous_probability'] for t in r['world_trace']),1)
        for row in r['distribution']:
            mass=sum(t['posterior_exogenous_probability'] for t in r['world_trace'] if all(t['predictions'][s['name']]['Y']==row['values'][s['name']]['Y'] for s in specs))
            self.assertAlmostEqual(mass,row['probability'])
        r['world_trace'][0]['predictions']['off']['Y']=999
        self.assertEqual(r['world_trace'][0]['predictions']['on']['Y'],1)
        specs[0]['query'][0]='X';self.assertEqual(r['scenarios'][0]['query'],['Y'])

    def test_all_64_boolean_mechanisms_independent_rational_reference(self):
        # Reference uses only explicit tuple indexing and Fraction, not engine internals.
        for xt in itertools.product(range(2),repeat=2):
            for yt in itertools.product(range(2),repeat=4):
                model={'version':1,'exogenous':[{'name':'U','probabilities':[.25,.75]}, {'name':'V','probabilities':[.5,.5]}],
                       'nodes':[{'name':'X','states':2,'inputs':['U'],'table':list(xt)}, {'name':'Y','states':2,'inputs':['X','V'],'table':list(yt)}]}
                for evidence in [{},{'Y':0},{'Y':1}]:
                    counts={(a,b):Fraction(0) for a,b in itertools.product(range(2),repeat=2)};denom=Fraction(0)
                    for u,v in itertools.product(range(2),repeat=2):
                        weight=[Fraction(1,4),Fraction(3,4)][u]*Fraction(1,2)
                        fy=yt[xt[u]*2+v]
                        if evidence and fy!=evidence['Y']:continue
                        counts[yt[v],yt[2+v]]+=weight;denom+=weight
                    if not denom:
                        with self.assertRaisesRegex(ValueError,'zero probability'):joint_counterfactual(model,S,evidence)
                    else:
                        r=joint_counterfactual(model,S,evidence)
                        for row,(key,mass) in zip(r['distribution'],counts.items()):self.assertAlmostEqual(row['probability'],float(mass/denom))

    def test_nonbinary_and_four_scenarios(self):
        model={'version':1,'exogenous':[{'name':'U','probabilities':[.2,.3,.5]}],
               'nodes':[{'name':'X','states':3,'inputs':['U'],'table':[0,1,2]}, {'name':'Y','states':3,'inputs':['X'],'table':[2,0,1]}]}
        specs=[{'name':'a'+str(x),'query':['Y'],'do':{'X':x}} for x in [0,1,2]]+[{'name':'factual','query':['X'],'do':{}}]
        r=joint_counterfactual(model,specs);self.assertEqual(len(r['distribution']),81)
        mass=[row for row in r['distribution'] if row['probability']]
        self.assertEqual([row['values']['a0']['Y'] for row in mass],[2]*3);self.assertEqual([row['values']['a1']['Y'] for row in mass],[0]*3)
        for row in mass:self.assertAlmostEqual(row['probability'],[.2,.3,.5][row['values']['factual']['X']])

    def test_rare_evidence_and_log_support(self):
        model={'version':1,'exogenous':[{'name':'U'+str(i),'probabilities':[1.,1e-300]} for i in range(4)],
               'nodes':[{'name':'A'+str(i),'states':2,'inputs':['U'+str(i)],'table':[0,1]} for i in range(4)]}
        specs=[{'name':'a','query':['A0'],'do':{'A0':0}}, {'name':'b','query':['A0'],'do':{'A0':1}}]
        r=joint_counterfactual(model,specs,{f'A{i}':1 for i in range(4)},trace=True)
        self.assertEqual(r['evidence_probability'],0);self.assertTrue(math.isfinite(r['log_evidence_probability']));self.assertEqual(probabilities(r),[0,1,0,0])
        self.assertEqual([x['log_probability'] for x in r['distribution']],[None,0.,None,None])
        rare=joint_counterfactual(model,[{'name':'a','query':['A0','A1'],'do':{}}])
        row=rare['distribution'][-1];self.assertEqual(row['probability'],0);self.assertTrue(math.isfinite(row['log_probability']))

    def test_zero_noise_bins_and_impossible_factual_evidence(self):
        model=self.models['stable'];model['exogenous'][0]['probabilities']=[1.,0.]
        r=joint_counterfactual(model,S);self.assertEqual(r['world_slots'],4);self.assertEqual(r['positive_prior_worlds'],2)
        with self.assertRaisesRegex(ValueError,'factual model'):joint_counterfactual(model,S,{'Y':1})

    def test_scenario_validation_before_evaluation(self):
        bad=[None,{},[],S*3,[{}],[{'name':'a','query':['Y'],'do':None}],
             [{'name':'a','query':['Y'],'do':{},'extra':1}],[S[0],S[0]],
             [{'name':'bad-name','query':['Y'],'do':{}}],
             [{'name':'a','query':['Y','Y'],'do':{}}],
             [{'name':'a','query':['U'],'do':{}}],
             [{'name':'a','query':['Y'],'do':{'U':0}}],
             [{'name':'a','query':['Y'],'do':{'X':True}}]]
        with patch('causallab.multiworld._evaluate',side_effect=AssertionError('evaluated too early')):
            for spec in bad:
                with self.subTest(spec=spec),self.assertRaises(ValueError):joint_counterfactual(self.models['flip'],spec)
            with self.assertRaises(ValueError):joint_counterfactual(self.models['flip'],S,trace=1)
            with self.assertRaises(ValueError):joint_counterfactual(self.models['flip'],S,{'U':0})

    def test_combined_joint_and_trace_caps_preflight(self):
        model={'version':1,'exogenous':[{'name':'U','probabilities':[.5,.5]}],
               'nodes':[{'name':'A'+str(i),'states':4,'inputs':[],'table':[0]} for i in range(8)]}
        specs=[{'name':'a','query':['A0','A1','A2','A3'],'do':{}}, {'name':'b','query':['A0','A1','A2'],'do':{}}]
        with patch('causallab.multiworld._evaluate',side_effect=AssertionError('late cap')):
            with self.assertRaisesRegex(ValueError,'joint scenario'):joint_counterfactual(model,specs)
        large=copy.deepcopy(self.models['flip']);large['exogenous']=[{'name':'U','probabilities':[.5,.5]}, {'name':'V','probabilities':[1/16]*16}]+[{'name':'N'+str(i),'probabilities':[1/16]*16} for i in range(2)]
        large['nodes'][0]['table']=[0,1]*8
        with patch('causallab.multiworld._evaluate',side_effect=AssertionError('late cap')):
            with self.assertRaisesRegex(ValueError,'trace work/output'):joint_counterfactual(large,S,trace=True)

    def test_inputs_fingerprint_rng_and_compiled_model(self):
        model=self.models['flip'];original=copy.deepcopy(model);specs=copy.deepcopy(S);saved=copy.deepcopy(specs);state=random.getstate()
        result=joint_counterfactual(Model.from_dict(model),specs,trace=True)
        self.assertEqual(model,original);self.assertEqual(specs,saved);self.assertEqual(state,random.getstate());self.assertEqual(result['model_sha256'],Model.from_dict(model).fingerprint)
        result['scenarios'][0]['do']['X']=1;self.assertEqual(specs,saved)

    def test_trace_boundary_32768_and_next_scenario_refused(self):
        model=self.models['flip']
        model['exogenous'] += [{'name':'N0','probabilities':[1/16]*16}, {'name':'N1','probabilities':[1/16]*16}, {'name':'N2','probabilities':[.25]*4}]
        specs=copy.deepcopy(S)+[{'name':'third','query':['Y'],'do':{}}]
        r=joint_counterfactual(model,specs,trace=True)
        self.assertEqual(len(r['world_trace']),4096);self.assertEqual(r['evaluations'],{'factual':4096,'action':12288})
        with self.assertRaisesRegex(ValueError,'trace work/output'):joint_counterfactual(model,specs+[{'name':'fourth','query':['Y'],'do':{}}],trace=True)

    def test_joint_row_boundary_4096(self):
        model={'version':1,'exogenous':[{'name':'U','probabilities':[.5,.5]}],
               'nodes':[{'name':'X','states':4,'inputs':[],'table':[2]}]}
        specs=[{'name':'s'+str(i),'query':['X'],'do':{'X':i}} for i in range(4)]
        # Each scenario may ask multiple variables; exactly 4^6 joint rows.
        model['nodes']+=[{'name':'Y','states':4,'inputs':[],'table':[1]}]
        specs[0]['query']=['X','Y'];specs[1]['query']=['Y','X']
        r=joint_counterfactual(model,specs);self.assertEqual(len(r['distribution']),4096);self.assertEqual(sum(probabilities(r)),1)

    def test_demo_response_types_and_naive_product_fail(self):
        d=multiworld_demo()
        for group in d['models'].values():self.assertNotEqual(probabilities(group['joint']),group['independent_marginal_product'])
        self.assertEqual(d['models']['stable']['response_types']['helped'],0)
        self.assertEqual(d['models']['flip']['response_types']['helped'],.5)


class MultiWorldCLI(unittest.TestCase):
    def run_cli(self,*args):return subprocess.run([sys.executable,'-m','causallab',*args],text=True,capture_output=True)

    def test_demo_and_joint_file(self):
        demo=self.run_cli('multiworld-demo');self.assertEqual(demo.returncode,0,demo.stderr);self.assertEqual(json.loads(demo.stdout),multiworld_demo())
        r=self.run_cli('joint','examples/response_flip.json','--scenarios','examples/scenarios.json','--evidence','X=0,Y=0','--trace')
        self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(probabilities(json.loads(r.stdout)),[0,1,0,0])

    def test_scenario_file_duplicate_malformed_oversized_and_missing(self):
        for content in ['[{"name":"a","name":"b","query":["Y"],"do":{}}]','oops',' '*262145]:
            with tempfile.NamedTemporaryFile(mode='w',suffix='.json') as f:
                f.write(content);f.flush();r=self.run_cli('joint','examples/response_flip.json','--scenarios',f.name)
            self.assertEqual(r.returncode,2);self.assertEqual(r.stdout,'');self.assertNotIn('Traceback',r.stderr)
        r=self.run_cli('joint','examples/response_flip.json','--scenarios','missing-scenarios.json')
        self.assertEqual(r.returncode,2);self.assertEqual(r.stdout,'')

    def test_query_rejects_joint_only_arguments_and_joint_has_no_postaction_flag(self):
        r=self.run_cli('joint','examples/response_flip.json','--scenarios','examples/scenarios.json','--do','X=1')
        self.assertEqual(r.returncode,2);self.assertEqual(r.stdout,'')
