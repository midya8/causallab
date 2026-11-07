"""Joint potential outcomes in one supplied SCM, never products of marginals."""
import itertools
import math
from .core import (_model, _assignment, _query, _worlds, _evaluate,
                   _logsum, fields, name)


def joint_counterfactual(payload, scenarios, evidence=None, *, trace=False):
    """Return a joint across 1..4 named action worlds sharing factual noise.

    Each scenario has exactly name, query (endogenous names), and do (assignment).
    Evidence is always factual. Scenario order and within-query order are kept.
    No scenario's output becomes another scenario's evidence or input.
    """
    if type(trace) is not bool:raise ValueError('trace requires a boolean')
    model=_model(payload);cards={n.name:n.states for n in model.nodes}
    evidence=_assignment(evidence,cards,'factual evidence')
    if not isinstance(scenarios,list) or not 1<=len(scenarios)<=4:raise ValueError('requires 1..4 scenarios')
    specs=[];seen=set();axes=[]
    for spec in scenarios:
        fields(spec,['name','query','do'],'scenario');label=name(spec['name'])
        if label in seen:raise ValueError('duplicate scenario name')
        seen.add(label);query=_query(spec['query'],cards)
        if not isinstance(spec['do'],dict):raise ValueError('scenario do must be an assignment object')
        action=_assignment(spec['do'],cards,'scenario intervention')
        specs.append({'name':label,'query':query,'do':action})
        axes.extend(cards[n] for n in query)
    if math.prod(axes)>4096:raise ValueError('joint scenario Cartesian product must not exceed4096 rows')
    if trace and model.world_slots*len(model.nodes)*(len(specs)+1)>32768:raise ValueError('multi-world trace work/output cap32768 node-world slots exceeded')
    keys=list(itertools.product(*(range(k) for k in axes)));groups={key:[] for key in keys}
    logs=[];traces=[];positive=0
    for noise,lp in _worlds(model):
        positive+=1;factual=_evaluate(model,noise,{})
        if not all(factual[n]==v for n,v in evidence.items()):continue
        predictions={s['name']:_evaluate(model,noise,s['do']) for s in specs}
        key=tuple(predictions[s['name']][n] for s in specs for n in s['query'])
        logs.append(lp);groups[key].append(lp)
        if trace:traces.append({'exogenous':noise,'log_prior':lp,'factual':factual,'predictions':predictions})
    denominator=_logsum(logs)
    if denominator is None:raise ValueError('evidence has zero probability under the factual model')
    peak=max(logs);total=math.fsum(math.exp(v-peak) for v in logs)
    rows=[]
    for key in keys:
        offset=0;values={}
        for s in specs:
            size=len(s['query']);values[s['name']]=dict(zip(s['query'],key[offset:offset+size]));offset+=size
        lp=_logsum(groups[key])
        rows.append({'values':values,'probability':math.fsum(math.exp(v-peak) for v in groups[key])/total,
                     'log_probability':None if lp is None else lp-denominator})
    result={'model_sha256':model.fingerprint,'mode':'joint_counterfactual','scenarios':specs,
            'evidence':evidence,'distribution':rows,'evidence_probability':math.exp(denominator),
            'log_evidence_probability':denominator,'world_slots':model.world_slots,
            'positive_prior_worlds':positive,'matching_worlds':len(logs),
            'evaluations':{'factual':positive,'action':len(logs)*len(specs)},
            'semantics':'factual evidence selects one exogenous posterior, reused jointly across all named action worlds; scenarios replace equations independently and never condition one another; not independent draws or a product of action marginals; cross-world dependence is an assumption of the supplied SCM, not identified by single-world observations or experiments in general; exhaustive finite enumeration in floating point; impossible support has null log, underflowed possible mass has finite log'}
    if trace:
        for row in traces:row['posterior_exogenous_probability']=math.exp(row['log_prior']-peak)/total
        result['world_trace']=traces
    return result


def response_models():
    """Two hand-specified mechanisms, no fitted data or model/seed search."""
    base={'version':1,'exogenous':[{'name':'U','probabilities':[.5,.5]},
                                    {'name':'V','probabilities':[.5,.5]}],
          'nodes':[{'name':'X','states':2,'inputs':['V'],'table':[0,1]}]}
    from copy import deepcopy
    out={}
    for label,inputs,table in [('stable',['U'],[0,1]),('flip',['X','U'],[0,1,1,0])]:
        model=deepcopy(base);model['nodes'].append({'name':'Y','states':2,'inputs':inputs,'table':table});out[label]=model
    return out


def multiworld_demo():
    from .core import infer
    scenarios=[{'name':'off','query':['Y'],'do':{'X':0}},
               {'name':'on','query':['Y'],'do':{'X':1}}]
    output={}
    for label,model in response_models().items():
        joint=joint_counterfactual(model,scenarios,trace=True)
        factual=joint_counterfactual(model,scenarios,{'X':0,'Y':0},trace=True)
        masses={(r['values']['off']['Y'],r['values']['on']['Y']):r['probability'] for r in joint['distribution']}
        output[label]={'model':model,'observed':infer(model,['X','Y']),
                       'interventions':{str(x):infer(model,['Y'],interventions={'X':x}) for x in [0,1]},
                       'joint':joint,'factual_X0_Y0':factual,
                       'response_types':{'always_zero':masses[0,0],'helped':masses[0,1],
                                         'harmed':masses[1,0],'always_one':masses[1,1]},
                       'independent_marginal_product':[.25]*4}
    stable=output['stable'];flip=output['flip']
    assert [r['probability'] for r in stable['observed']['distribution']]==[.25]*4
    assert [r['probability'] for r in flip['observed']['distribution']]==[.25]*4
    assert stable['response_types']=={'always_zero':.5,'helped':0.,'harmed':0.,'always_one':.5}
    assert flip['response_types']=={'always_zero':0.,'helped':.5,'harmed':.5,'always_one':0.}
    for group in output.values():
        for r in group['interventions'].values():assert [v['probability'] for v in r['distribution']]==[.5,.5]
    return {'models':output,'scenario_spec':scenarios,
            'selection':'first hand-specified stable Y=U and flip Y=X xor U; independent equiprobable U,V; X=V; no data, fitting, seed or mechanism search',
            'limit':'identical observational joint and single-action marginals do not determine cross-world response dependence; no monotonicity assumed; helped/harmed describe synthetic binary outcomes only, not real-world benefit or injury'}
