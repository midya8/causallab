"""Sharp binary-response bounds with exact rational attaining witnesses."""
from fractions import Fraction
import math


def _marginal(value,label):
    if type(value) not in (int,float) or not 0<=value<=1 or not math.isfinite(value):
        raise ValueError(label+' must be a finite number in [0,1], not a boolean')
    return Fraction(value)


def _number(value):return {'value':float(value),'exact':str(value)}


def _cells(p0,p1,t):return {'always_zero':1-p0-p1+t,'helped':p1-t,'harmed':p0-t,'always_one':t}


def _quantities(p0,p1,t):
    c=_cells(p0,p1,t);delta=p1-p0
    return {**c,'changed':c['helped']+c['harmed'],'unchanged':c['always_zero']+c['always_one'],
            'mean_effect':delta,'effect_variance':c['helped']+c['harmed']-delta*delta,
            'covariance':t-p0*p1,
            'sufficiency':None if p0==1 else c['helped']/(1-p0),
            'necessity':None if p1==0 else c['helped']/p1}


def response_bounds(p0,p1,*,monotone=False):
    """Sharp ranges over all binary (Y0,Y1) joints with these action marginals.

    p0=P(Y0=1), p1=P(Y1=1); optional monotonicity means Y1>=Y0.
    Internal arithmetic is rational, exact for the supplied int/binary float.
    Float display values are approximations; exact strings retain tiny masses.
    No observed conditional is automatically accepted as an action marginal.
    """
    if type(monotone) is not bool:raise ValueError('monotone requires a boolean')
    a=_marginal(p0,'p0');b=_marginal(p1,'p1')
    lo=max(Fraction(0),a+b-1);hi=min(a,b)
    if monotone:
        if b<a:raise ValueError('monotonicity Y1>=Y0 is incompatible with p1<p0')
        lo=hi=a
    ends={'low_joint11':lo,'high_joint11':hi};witnesses={};values={}
    for label,t in ends.items():
        cells=_cells(a,b,t)
        witnesses[label]={'joint_11':_number(t),'cells':{key:_number(v) for key,v in cells.items()},
                          'marginals':{'p0':_number(cells['harmed']+cells['always_one']),
                                       'p1':_number(cells['helped']+cells['always_one'])}}
        values[label]=_quantities(a,b,t)
    ranges={}
    for key,left in values['low_joint11'].items():
        right=values['high_joint11'][key]
        if left is None:
            ranges[key]={'defined':False,'reason':'conditioning event Y0=0 has zero probability' if key=='sufficiency' else 'conditioning event Y1=1 has zero probability'}
            continue
        lower_label='low_joint11' if left<=right else 'high_joint11'
        upper_label='high_joint11' if right>=left else 'low_joint11'
        ranges[key]={'defined':True,'lower':_number(min(left,right)),'upper':_number(max(left,right)),
                     'lower_witness':lower_label,'upper_witness':upper_label}
    return {'version':1,'marginals':{'p0':_number(a),'p1':_number(b)},'monotone':monotone,
            'joint_11_interval':{'lower':_number(lo),'upper':_number(hi)},
            'ranges':ranges,'witnesses':witnesses,
            'definitions':{'Y0':'binary outcome under do(X=0)','Y1':'binary outcome under do(X=1)',
                           'always_zero':'P(Y0=0,Y1=0)','helped':'P(Y0=0,Y1=1)',
                           'harmed':'P(Y0=1,Y1=0)','always_one':'P(Y0=1,Y1=1)',
                           'changed':'P(Y0!=Y1)','unchanged':'P(Y0=Y1)',
                           'mean_effect':'E[Y1-Y0]','effect_variance':'Var(Y1-Y0)',
                           'covariance':'Cov(Y0,Y1)',
                           'sufficiency':'P(Y1=1 | Y0=0)','necessity':'P(Y0=0 | Y1=1)'},
            'semantics':'sharp over all binary response joints with the supplied point action marginals and, only when requested, Y1>=Y0; witnesses attain every reported endpoint individually, not necessarily all endpoints simultaneously; any intermediate joint is a convex mixture of the two witnesses; rational arithmetic is exact for supplied binary-float/int values, not intended decimal text; value is rounded display and exact is the authoritative fraction, including tiny underflowed display values; point action marginals are assumptions, not observational conditionals, fitted estimates, confidence intervals or empirical benefit/harm claims; monotonicity is an extra structural restriction, not inferred from a nonnegative average effect'}


def bounds_demo():
    from .multiworld import multiworld_demo
    models=multiworld_demo()['models']
    cases={}
    for label,p0,p1 in [('equal',.5,.5),('positive',.25,.75)]:
        cases[label]={'unrestricted':response_bounds(p0,p1),'monotone':response_bounds(p0,p1,monotone=True)}
    stable=models['stable']['response_types'];flip=models['flip']['response_types']
    equal=cases['equal']['unrestricted'];mono=cases['equal']['monotone']
    assert equal['ranges']['helped']['lower']['value']==0
    assert equal['ranges']['helped']['upper']['value']==.5
    assert mono['ranges']['changed']['upper']['value']==0
    assert cases['positive']['unrestricted']['ranges']['helped']['lower']['value']==.5
    assert cases['positive']['unrestricted']['ranges']['helped']['upper']['value']==.75
    return {'cases':cases,'existing_mechanisms':{'stable':stable,'flip':flip},
            'checks':{'same_marginals_both_mechanisms_feasible':stable['helped']==0 and flip['helped']==.5,
                      'flip_violates_monotonicity':flip['harmed']>0,
                      'stable_satisfies_monotonicity':stable['harmed']==0},
            'selection':'predeclared exact quarters: equal .5/.5 connects to the existing stable/flip counterexample; positive .25/.75 shows positive mean effect with possible negative unit effects; no optimization, data, seed or search',
            'limit':'point action marginals assumed known; monotonicity is added, not learned; exact rational feasible-joint bounds, not uncertainty intervals for unknown population marginals'}
