"""Bounded exact enumeration for fully specified finite acyclic SCMs."""
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import re

NAME=re.compile(r'^[A-Za-z][A-Za-z0-9_]{0,23}$')


def integer(v,lo,hi,label):
    if type(v) is not int or not lo<=v<=hi:raise ValueError(f'{label} must be an integer in [{lo},{hi}]')
    return v


def name(v):
    if not isinstance(v,str) or not NAME.fullmatch(v):raise ValueError('names must be 1..24 ASCII letters/digits/underscores, starting with a letter')
    return v


def fields(v,keys,label):
    if not isinstance(v,dict) or set(v)!=set(keys):raise ValueError(f'{label} requires exactly: '+', '.join(keys))


@dataclass(frozen=True)
class Noise:
    name:str
    probabilities:tuple


@dataclass(frozen=True)
class Node:
    name:str
    states:int
    inputs:tuple
    table:tuple


@dataclass(frozen=True)
class Model:
    noises:tuple
    nodes:tuple

    @classmethod
    def from_dict(cls,payload):
        fields(payload,['version','exogenous','nodes'],'model')
        if type(payload['version']) is not int or payload['version']!=1:raise ValueError('model version must be integer1')
        exo=payload['exogenous'];rows=payload['nodes']
        if not isinstance(exo,list) or not 1<=len(exo)<=8:raise ValueError('requires 1..8 independent exogenous variables')
        if not isinstance(rows,list) or not 1<=len(rows)<=12:raise ValueError('requires 1..12 topologically ordered endogenous nodes')
        cards={};noises=[];nodes=[];worlds=1;total_table=0
        for row in exo:
            fields(row,['name','probabilities'],'exogenous variable');n=name(row['name']);p=row['probabilities']
            if n in cards:raise ValueError('duplicate variable name')
            if not isinstance(p,list) or not 2<=len(p)<=16:raise ValueError('exogenous cardinality must be 2..16')
            if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not 0<=x<=1 for x in p):raise ValueError('probabilities must be finite numbers in [0,1]')
            total=math.fsum(p)
            if abs(total-1)>1e-12:raise ValueError('exogenous probabilities must sum to one within 1e-12')
            cards[n]=len(p);noises.append(Noise(n,tuple(x/total for x in p)));worlds*=len(p)
            if worlds>65536:raise ValueError('exogenous Cartesian product must not exceed 65536 worlds')
        for row in rows:
            fields(row,['name','states','inputs','table'],'node');n=name(row['name'])
            if n in cards:raise ValueError('duplicate variable name')
            states=integer(row['states'],2,4,'node cardinality');inputs=row['inputs'];table=row['table']
            if not isinstance(inputs,list) or len(inputs)>8 or any(not isinstance(x,str) or x not in cards for x in inputs):raise ValueError('inputs must name exogenous or earlier endogenous variables, at most8')
            if len(set(inputs))!=len(inputs):raise ValueError('duplicate mechanism input')
            size=math.prod(cards[x] for x in inputs)
            if size>4096:raise ValueError('individual truth table must not exceed 4096 entries')
            if not isinstance(table,list) or len(table)!=size:raise ValueError('truth table length must match Cartesian input cardinalities')
            for output in table:integer(output,0,states-1,'truth table output')
            total_table+=size
            if total_table>16384:raise ValueError('total truth tables must not exceed 16384 entries')
            nodes.append(Node(n,states,tuple(inputs),tuple(table)));cards[n]=states
        return cls(tuple(noises),tuple(nodes))

    def as_dict(self):
        return {'version':1,'exogenous':[{'name':n.name,'probabilities':list(n.probabilities)} for n in self.noises],
                'nodes':[{'name':n.name,'states':n.states,'inputs':list(n.inputs),'table':list(n.table)} for n in self.nodes]}

    @property
    def cards(self):return {**{n.name:len(n.probabilities) for n in self.noises},**{n.name:n.states for n in self.nodes}}

    @property
    def world_slots(self):return math.prod(len(n.probabilities) for n in self.noises)

    @property
    def fingerprint(self):return hashlib.sha256(json.dumps(self.as_dict(),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _model(payload):
    # Revalidate even directly constructed dataclasses rather than trusting a tag.
    try:source=payload.as_dict() if isinstance(payload,Model) else payload
    except (AttributeError,TypeError):raise ValueError('invalid Model object; use Model.from_dict') from None
    return Model.from_dict(source)


def _assignment(values,cards,label):
    if values is None:return {}
    if not isinstance(values,dict):raise ValueError(label+' must be a variable-to-integer object')
    out={}
    for n,value in values.items():
        if not isinstance(n,str) or n not in cards:raise ValueError(label+' names an unknown or exogenous variable')
        out[n]=integer(value,0,cards[n]-1,label+' value')
    return out


def _query(variables,cards):
    if not isinstance(variables,list) or not 1<=len(variables)<=8 or any(not isinstance(n,str) or n not in cards for n in variables):raise ValueError('query must be a list of1..8 endogenous names')
    if len(set(variables))!=len(variables):raise ValueError('duplicate query variable')
    if math.prod(cards[n] for n in variables)>4096:raise ValueError('query Cartesian product must not exceed4096 rows')
    return variables[:]


def _worlds(model):
    for states in itertools.product(*(range(len(n.probabilities)) for n in model.noises)):
        probabilities=[n.probabilities[s] for n,s in zip(model.noises,states)]
        if any(p==0 for p in probabilities):continue
        yield dict(zip((n.name for n in model.noises),states)),math.fsum(math.log(p) for p in probabilities)


def _evaluate(model,noise,interventions):
    values=dict(noise);cards=model.cards
    for node in model.nodes:
        if node.name in interventions:values[node.name]=interventions[node.name]
        else:
            index=0
            for n in node.inputs:index=index*cards[n]+values[n]
            values[node.name]=node.table[index]
    return {node.name:values[node.name] for node in model.nodes}


def _logsum(values):
    if not values:return None
    peak=max(values);return peak+math.log(math.fsum(math.exp(x-peak) for x in values))


def infer(payload,variables,evidence=None,interventions=None,*,counterfactual=False,trace=False):
    """Query under do with optional post-action evidence, or factual abduction.

    For counterfactual=True, evidence is strictly factual (unmodified model),
    and the action prediction reuses each factual world's exogenous assignment.
    """
    if type(counterfactual) is not bool or type(trace) is not bool:raise ValueError('counterfactual and trace require booleans')
    model=_model(payload);cards={n.name:n.states for n in model.nodes};variables=_query(variables,cards)
    evidence=_assignment(evidence,cards,'evidence');interventions=_assignment(interventions,cards,'intervention')
    if trace and model.world_slots*len(model.nodes)*(2 if counterfactual else 1)>32768:raise ValueError('trace work/output cap32768 node-world slots exceeded')
    keys=list(itertools.product(*(range(cards[n]) for n in variables)))
    groups={key:[] for key in keys};matched_logs=[];traces=[];factual_evaluations=0;action_evaluations=0;world_count=0
    for noise,log_prior in _worlds(model):
        world_count+=1
        base=_evaluate(model,noise,{} if counterfactual else interventions)
        if counterfactual:factual_evaluations+=1
        else:action_evaluations+=1
        if not all(base[n]==v for n,v in evidence.items()):continue
        prediction=_evaluate(model,noise,interventions) if counterfactual else base
        if counterfactual:action_evaluations+=1
        key=tuple(prediction[n] for n in variables);groups[key].append(log_prior)
        matched_logs.append(log_prior)
        if trace:traces.append({'exogenous':noise,'log_prior':log_prior,'factual' if counterfactual else 'post_action':base.copy(),'prediction':prediction.copy()})
    denominator=_logsum(matched_logs)
    if denominator is None:raise ValueError('evidence has zero probability under '+('the factual model' if counterfactual else 'the intervened model'))
    # Relative terms preserve rare evidence even if the reported mass underflows.
    peak=max(matched_logs);scaled_total=math.fsum(math.exp(v-peak) for v in matched_logs)
    rows=[]
    for key in keys:
        lp=_logsum(groups[key]);prob=math.fsum(math.exp(x-peak) for x in groups[key])/scaled_total
        rows.append({'values':dict(zip(variables,key)),'probability':prob,'log_probability':None if lp is None else lp-denominator})
    result={'model_sha256':model.fingerprint,'mode':'counterfactual' if counterfactual else 'interventional_query','variables':variables,
            'evidence':evidence,'interventions':interventions,'distribution':rows,'evidence_probability':math.exp(denominator),
            'log_evidence_probability':denominator,'world_slots':model.world_slots,'positive_prior_worlds':world_count,'matching_worlds':len(matched_logs),
            'evaluations':{'factual':factual_evaluations,'action':action_evaluations},
            'semantics':'fully specified finite acyclic structural model; independent named exogenous distributions may be shared by equations; observational conditioning when do is empty; intervention replaces an equation, not a selection event; counterfactual evidence is factual and action reuses the same noise posterior; ordinary query evidence is post-action; exhaustive discrete enumeration in floating point, not symbolic arithmetic, fitted causes, effect identification from data, a real-world effect or policy recommendation; negative-infinite log probability is represented by null only for impossible support, finite rare log masses can coexist with underflowed probability0'}
    if trace:
        for r in traces:r['posterior_exogenous_probability']=math.exp(r['log_prior']-peak)/scaled_total
        result['world_trace']=traces
    return result
