import copy
from fractions import Fraction as F
import itertools
import json
import math
import random
import subprocess
import sys
import unittest
from causallab import response_bounds
from causallab.bounds import bounds_demo
from causallab.multiworld import multiworld_demo


def exact(number):return F(number['exact'])


def metrics(q):
    # Independent reference from the four response masses, not parameter formulas.
    a,b,c,d=q;p0=c+d;p1=b+d;mean=b-c;second=b+c
    return {'always_zero':a,'helped':b,'harmed':c,'always_one':d,'changed':b+c,'unchanged':a+d,
            'mean_effect':mean,'effect_variance':sum(mass*(effect-mean)**2 for mass,effect in zip(q,[0,1,-1,0])),
            'covariance':sum(mass*(y0-p0)*(y1-p1) for mass,y0,y1 in zip(q,[0,0,1,1],[0,1,0,1])),
            'sufficiency':None if a+b==0 else b/(a+b),'necessity':None if b+d==0 else b/(b+d)}


def witness_vector(w):return [exact(w['cells'][key]) for key in ['always_zero','helped','harmed','always_one']]


class ResponseBoundsTests(unittest.TestCase):
    def test_equal_marginal_ranges_and_witnesses(self):
        r=response_bounds(.5,.5)
        for k,lo,hi in [('helped',0,.5),('harmed',0,.5),('changed',0,1),('unchanged',0,1),('mean_effect',0,0),('effect_variance',0,1),('covariance',-.25,.25),('sufficiency',0,1),('necessity',0,1)]:
            self.assertEqual(exact(r['ranges'][k]['lower']),F(lo));self.assertEqual(exact(r['ranges'][k]['upper']),F(hi))
        self.assertEqual(witness_vector(r['witnesses']['low_joint11']),[0,F(1,2),F(1,2),0])
        self.assertEqual(witness_vector(r['witnesses']['high_joint11']),[F(1,2),0,0,F(1,2)])

    def test_positive_mean_does_not_rule_out_harm(self):
        r=response_bounds(.25,.75)
        self.assertEqual(exact(r['ranges']['mean_effect']['lower']),F(1,2))
        self.assertEqual(exact(r['ranges']['harmed']['upper']),F(1,4))
        self.assertEqual(exact(r['ranges']['helped']['lower']),F(1,2));self.assertEqual(exact(r['ranges']['helped']['upper']),F(3,4))
        self.assertEqual(exact(r['ranges']['effect_variance']['lower']),F(1,4));self.assertEqual(exact(r['ranges']['effect_variance']['upper']),F(3,4))

    def test_all_denominator8_couplings_exhaustive_sharp_reference(self):
        groups={}
        for a,b,c in itertools.product(range(9),repeat=3):
            d=8-a-b-c
            if d<0:continue
            q=tuple(F(n,8) for n in [a,b,c,d]);key=(c+d,b+d);groups.setdefault(key,[]).append(q)
        self.assertEqual(len(groups),81)
        for (m0,m1),joints in groups.items():
            for monotone in [False,True]:
                candidates=[q for q in joints if not monotone or q[2]==0]
                if not candidates:
                    with self.assertRaises(ValueError):response_bounds(m0/8,m1/8,monotone=True)
                    continue
                result=response_bounds(m0/8,m1/8,monotone=monotone)
                for key,bound in result['ranges'].items():
                    vals=[metrics(q)[key] for q in candidates]
                    if vals[0] is None:self.assertFalse(bound['defined']);continue
                    self.assertTrue(bound['defined']);self.assertEqual(exact(bound['lower']),min(vals));self.assertEqual(exact(bound['upper']),max(vals))

    def test_every_endpoint_witness_attains_bound(self):
        for p0,p1 in itertools.product([0.,.125,.5,.75,1.],repeat=2):
            for monotone in [False,True]:
                if monotone and p1<p0:continue
                r=response_bounds(p0,p1,monotone=monotone)
                for k,v in r['ranges'].items():
                    if not v['defined']:continue
                    for side in ['lower','upper']:
                        q=witness_vector(r['witnesses'][v[side+'_witness']]);self.assertEqual(metrics(q)[k],exact(v[side]))

    def test_all_witnesses_feasible_and_preserve_marginals(self):
        for p0,p1 in itertools.product([0.,.1,.3,.5,.9,1.],repeat=2):
            r=response_bounds(p0,p1)
            for w in r['witnesses'].values():
                a,b,c,d=witness_vector(w);self.assertEqual(a+b+c+d,1);self.assertTrue(all(x>=0 for x in [a,b,c,d]))
                self.assertEqual(c+d,F(p0));self.assertEqual(b+d,F(p1));self.assertEqual(d,exact(w['joint_11']))

    def test_convex_mixtures_span_feasible_interval(self):
        r=response_bounds(.125,.625);lo=witness_vector(r['witnesses']['low_joint11']);hi=witness_vector(r['witnesses']['high_joint11'])
        for weight in [F(0),F(1,7),F(1,2),F(1)]:
            q=[(1-weight)*a+weight*b for a,b in zip(lo,hi)]
            self.assertEqual(q[2]+q[3],F(1,8));self.assertEqual(q[1]+q[3],F(5,8))
            for key,v in metrics(q).items():
                if v is None:continue
                self.assertLessEqual(exact(r['ranges'][key]['lower']),v);self.assertLessEqual(v,exact(r['ranges'][key]['upper']))

    def test_monotone_positive_marginals_collapses_and_has_no_harm(self):
        r=response_bounds(.25,.75,monotone=True)
        self.assertEqual(witness_vector(r['witnesses']['low_joint11']),[F(1,4),F(1,2),0,F(1,4)])
        self.assertEqual(r['witnesses']['low_joint11'],r['witnesses']['high_joint11'])
        for key,v in r['ranges'].items():
            if v['defined']:self.assertEqual(v['lower'],v['upper'])
        self.assertEqual(exact(r['ranges']['harmed']['upper']),0)

    def test_monotone_equal_means_no_changed_units(self):
        for p in [0.,.1,.5,1.]:
            r=response_bounds(p,p,monotone=True)
            self.assertEqual(exact(r['ranges']['changed']['upper']),0)
            self.assertEqual(exact(r['ranges']['effect_variance']['upper']),0)

    def test_monotonicity_rejects_negative_mean_without_tolerance(self):
        for p0,p1 in [(1.,0.),(.75,.25),(.5,math.nextafter(.5,0.))]:
            with self.assertRaisesRegex(ValueError,'incompatible'):response_bounds(p0,p1,monotone=True)
        r=response_bounds(.75,.25);self.assertEqual(exact(r['ranges']['mean_effect']['upper']),F(-1,2))

    def test_zero_conditioning_events_are_undefined_not_zero(self):
        for p0 in [0.,.5,1.]:self.assertFalse(response_bounds(p0,0)['ranges']['necessity']['defined'])
        for p1 in [0.,.5,1.]:self.assertFalse(response_bounds(1,p1)['ranges']['sufficiency']['defined'])
        for key in ['necessity','sufficiency']:self.assertNotIn('lower',response_bounds(1,0)['ranges'][key])

    def test_degenerate_marginals_have_single_joint(self):
        for p0,p1 in itertools.product([0,1],repeat=2):
            r=response_bounds(p0,p1)
            self.assertEqual(r['witnesses']['low_joint11'],r['witnesses']['high_joint11'])
            self.assertEqual(sum(x!=0 for x in witness_vector(r['witnesses']['low_joint11'])),1)
            self.assertEqual(exact(r['ranges']['effect_variance']['lower']),0)

    def test_rational_exactness_for_binary_float_not_decimal_text(self):
        r=response_bounds(.1,.3)
        self.assertEqual(exact(r['marginals']['p0']),F(.1));self.assertNotEqual(exact(r['marginals']['p0']),F(1,10))
        self.assertEqual(exact(r['ranges']['mean_effect']['lower']),F(.3)-F(.1))
        for w in r['witnesses'].values():self.assertEqual(sum(witness_vector(w)),1)

    def test_tiny_positive_denominators_do_not_become_undefined(self):
        tiny=math.ulp(0.);r=response_bounds(0.,tiny)
        self.assertTrue(r['ranges']['necessity']['defined']);self.assertEqual(exact(r['ranges']['necessity']['lower']),1)
        self.assertEqual(exact(r['ranges']['helped']['lower']),F(tiny));self.assertTrue(r['ranges']['sufficiency']['defined'])
        near=math.nextafter(1.,0.);r=response_bounds(near,1.)
        self.assertEqual(exact(r['ranges']['sufficiency']['lower']),1)

    def test_variance_underflow_does_not_lose_authoritative_fraction(self):
        tiny=math.ulp(0.);r=response_bounds(0,tiny)
        exactvar=F(tiny)*(1-F(tiny))
        self.assertEqual(exact(r['ranges']['effect_variance']['lower']),exactvar)
        self.assertGreater(exactvar,0)
        # Covariance at p0=p1=tiny can be negative with magnitude tiny^2.
        cov=response_bounds(tiny,tiny)['ranges']['covariance']['lower']
        self.assertEqual(cov['value'],-0.);self.assertEqual(exact(cov),-F(tiny)**2)

    def test_relabel_actions_swaps_help_harm_and_negates_mean(self):
        for p0,p1 in [(.125,.75),(.5,.5),(.9,.1)]:
            r=response_bounds(p0,p1);s=response_bounds(p1,p0)
            self.assertEqual(r['ranges']['helped']['lower'],s['ranges']['harmed']['lower']);self.assertEqual(r['ranges']['helped']['upper'],s['ranges']['harmed']['upper'])
            self.assertEqual(exact(r['ranges']['mean_effect']['lower']),-exact(s['ranges']['mean_effect']['lower']))
            self.assertEqual(r['ranges']['effect_variance']['lower'],s['ranges']['effect_variance']['lower'])

    def test_invalid_numbers_and_assumption_flag(self):
        for bad in [True,False,None,'0.5',F(1,2),math.nan,math.inf,-math.inf,-1.,1.0001,10**10000]:
            for args in [(bad,.5),(.5,bad)]:
                with self.subTest(value=type(bad).__name__),self.assertRaises(ValueError):response_bounds(*args)
        for bad in [0,1,None,'yes']:
            with self.assertRaises(ValueError):response_bounds(.5,.5,monotone=bad)

    def test_original_mechanisms_match_endpoint_witnesses(self):
        d=multiworld_demo();r=response_bounds(.5,.5)
        for label,w in [('stable','high_joint11'),('flip','low_joint11')]:
            actual=d['models'][label]['response_types']
            for key,mass in r['witnesses'][w]['cells'].items():self.assertEqual(F(actual[key]),exact(mass))

    def test_rng_output_mutation_and_demo_repeatability(self):
        state=random.getstate();a=bounds_demo();saved=copy.deepcopy(a);self.assertEqual(a,bounds_demo());self.assertEqual(state,random.getstate())
        a['cases']['equal']['unrestricted']['witnesses']['low_joint11']['cells']['helped']['exact']='999'
        self.assertEqual(bounds_demo(),saved)


class BoundsCLI(unittest.TestCase):
    def run_cli(self,*args):return subprocess.run([sys.executable,'-m','causallab',*args],capture_output=True,text=True)
    def test_api_cli_and_demo_exact(self):
        r=self.run_cli('bounds','--p0','.25','--p1','.75');self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout),response_bounds(.25,.75))
        r=self.run_cli('bounds-demo');self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout),bounds_demo())
    def test_monotone_cli_and_incompatibility(self):
        r=self.run_cli('bounds','--p0','.25','--p1','.75','--monotone');self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout),response_bounds(.25,.75,monotone=True))
        r=self.run_cli('bounds','--p0','.75','--p1','.25','--monotone');self.assertEqual(r.returncode,2);self.assertEqual(r.stdout,'');self.assertIn('incompatible',r.stderr)
    def test_bad_nonfinite_range_and_missing_arguments_no_partial_json(self):
        for tail in [['--p0','nan','--p1','.5'],['--p0','inf','--p1','.5'],['--p0','1.1','--p1','.5'],['--p0','.5'],['--p0','oops','--p1','.5']]:
            r=self.run_cli('bounds',*tail);self.assertEqual(r.returncode,2);self.assertEqual(r.stdout,'');self.assertNotIn('Traceback',r.stderr)
    def test_tiny_scientific_notation_and_help(self):
        r=self.run_cli('bounds','--p0','0','--p1','5e-324');self.assertEqual(r.returncode,0,r.stderr);self.assertTrue(json.loads(r.stdout)['ranges']['necessity']['defined'])
        r=self.run_cli('bounds','--help');self.assertEqual(r.returncode,0);self.assertIn('confidence intervals',' '.join(r.stdout.split()))
