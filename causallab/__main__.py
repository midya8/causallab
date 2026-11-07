import argparse
import json
from .core import infer
from .demo import demo
from .multiworld import joint_counterfactual, multiworld_demo


def unique_object(pairs):
    out={}
    for key,value in pairs:
        if key in out:raise ValueError('duplicate JSON key: '+key)
        out[key]=value
    return out


def assignments(text):
    if not text:return {}
    out={}
    for term in text.split(','):
        parts=term.split('=')
        if len(parts)!=2 or parts[0] in out:raise ValueError('assignments require unique NAME=INTEGER pairs separated by commas')
        try:out[parts[0]]=int(parts[1])
        except ValueError:raise ValueError('assignment values must be integers') from None
    return out


def main():
    parser=argparse.ArgumentParser(prog='causallab',description='Exact queries for supplied finite structural causal models, not causes learned from data')
    commands=parser.add_subparsers(dest='command',required=True);commands.add_parser('demo',help='run exact confounding and observational-equivalence counterexamples')
    commands.add_parser('multiworld-demo',help='same observed/action marginals, different cross-world dependence')
    joint=commands.add_parser('joint',help='joint shared-noise counterfactuals, factual evidence only')
    joint.add_argument('model');joint.add_argument('--scenarios',required=True,help='JSON file:1..4 objects with name/query/do')
    joint.add_argument('--evidence',default='');joint.add_argument('--trace',action='store_true')
    for command in ['query','counterfactual']:
        p=commands.add_parser(command,help='post-action evidence' if command=='query' else 'factual abduction followed by action with shared noise')
        p.add_argument('model');p.add_argument('--query',required=True,help='comma-separated endogenous names,1..8')
        p.add_argument('--evidence',default='',help='NAME=INTEGER pairs; factual for counterfactual, post-action for query')
        p.add_argument('--do',dest='interventions',default='',help='NAME=INTEGER pairs; replaces those mechanisms')
        p.add_argument('--trace',action='store_true',help='bounded matching noise-world posterior and factual/action values')
    args=parser.parse_args()
    try:
        if args.command=='demo':result=demo()
        elif args.command=='multiworld-demo':result=multiworld_demo()
        else:
            with open(args.model,encoding='utf8') as f:text=f.read(262145)
            if len(text.encode('utf8'))>262144:raise ValueError('model JSON exceeds262144 bytes')
            model=json.loads(text,object_pairs_hook=unique_object)
            if args.command=='joint':
                with open(args.scenarios,encoding='utf8') as f:scenario_text=f.read(262145)
                if len(scenario_text.encode('utf8'))>262144:raise ValueError('scenario JSON exceeds262144 bytes')
                result=joint_counterfactual(model,json.loads(scenario_text,object_pairs_hook=unique_object),assignments(args.evidence),trace=args.trace)
            else:result=infer(model,args.query.split(','),assignments(args.evidence),assignments(args.interventions),counterfactual=args.command=='counterfactual',trace=args.trace)
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError,TypeError,OSError,OverflowError) as error:parser.exit(2,f'causallab: {error}\n')


if __name__=='__main__':main()
