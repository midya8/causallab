"""Exact hand-specified failures of causal conclusions from association alone."""
from .core import infer
from .fixtures import confounded,observational_twins


def probability(result,**values):
    return sum(r['probability'] for r in result['distribution'] if all(r['values'][n]==v for n,v in values.items()))


def demo():
    m=confounded();observed=[probability(infer(m,['Y'],evidence={'X':x}),Y=1) for x in [0,1]]
    acted=[probability(infer(m,['Y'],interventions={'X':x}),Y=1) for x in [0,1]]
    strata=[[probability(infer(m,['Y'],evidence={'X':x,'Z':z}),Y=1) for x in [0,1]] for z in [0,1]]
    adjustment=[sum(probability(infer(m,['Y'],evidence={'X':x,'Z':z}),Y=1)*probability(infer(m,['Z']),Z=z) for z in [0,1]) for x in [0,1]]
    twins={}
    for label,model in observational_twins().items():
        twins[label]={'model':model,'observational_joint':infer(model,['X','Y']),
            'do_X1':infer(model,['Y'],interventions={'X':1}),
            'counterfactual_X1_given_X0Y0':infer(model,['Y'],evidence={'X':0,'Y':0},interventions={'X':1},counterfactual=True,trace=True)}
    a,b=twins['direct'],twins['common']
    checks={'association_positive':observed[1]>observed[0],
        'effect_negative':acted[1]<acted[0],
        'negative_in_both_strata':all(p[1]<p[0] for p in strata),
        'adjustment_matches_intervention':all(abs(x-y)<1e-12 for x,y in zip(adjustment,acted)),
        'same_observational_joint':a['observational_joint']['distribution']==b['observational_joint']['distribution'],
        'different_interventions':probability(a['do_X1'],Y=1)!=probability(b['do_X1'],Y=1),
        'different_counterfactuals':probability(a['counterfactual_X1_given_X0Y0'],Y=1)!=probability(b['counterfactual_X1_given_X0Y0'],Y=1)}
    if not all(checks.values()):raise ValueError('causal fixture invariant failed')
    return {'confounding':{'model':m,'observational_Y1_given_X':observed,'interventional_Y1':acted,'stratum_Y1_given_X':strata,
        'backdoor_adjusted_Y1':adjustment,'observational_difference':observed[1]-observed[0],'interventional_difference':acted[1]-acted[0]},
        'twins':twins,'checks':checks,'selection':'first hand-specified illustrative mechanisms; exhaustive100-world confounding example and two2-world models; no learned parameters, randomized trials, real data, empirical uncertainty or held-out performance; examples demonstrate model-conditional semantics and observational non-identifiability, not an automatic identification algorithm'}
