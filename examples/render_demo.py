"""Fixed causal-model counterexamples with all eight plotted probabilities."""
import argparse
import html
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causallab.demo import demo


def manifest(v):
    if v!=demo():raise ValueError('requires unchanged complete causal demo')
    c=v['confounding'];t=v['twins']
    observed=c['observational_Y1_given_X'];acted=c['interventional_Y1']
    value=lambda result:next(row['probability'] for row in result['distribution'] if row['values']=={'Y':1})
    series={'observed':observed,'do':acted,'twins_do':[value(t[n]['do_X1']) for n in ['direct','common']],
            'twins_counterfactual':[value(t[n]['counterfactual_X1_given_X0Y0']) for n in ['direct','common']]}
    expected={'observed':[.42,.68],'do':[.6,.5],'twins_do':[1.,.5],'twins_counterfactual':[1.,0.]}
    for name,values in series.items():
        if len(values)!=2 or any(not math.isfinite(x) or not 0<=x<=1 or abs(x-y)>1e-12 for x,y in zip(values,expected[name])):raise ValueError('requires documented fixed probabilities')
    if t['direct']['observational_joint']['distribution']!=t['common']['observational_joint']['distribution']:raise ValueError('requires observationally equivalent models')
    if t['direct']['do_X1']['world_slots']!=2 or t['common']['do_X1']['world_slots']!=2:raise ValueError('requires two-world twins')
    if c['model']['nodes'][1]['table']!=[1,1,1,1,0,1,0,0,0,0]:raise ValueError('requires documented confounding mechanism')
    return {'series':series,'domain':[0,1],'models':{'confounded':c['model'],**{n:t[n]['model'] for n in t}},
            'strata':c['stratum_Y1_given_X'],'adjustment':c['backdoor_adjusted_Y1'],
            'observational_joints':{n:t[n]['observational_joint']['distribution'] for n in t},
            'counterfactual_traces':{n:t[n]['counterfactual_X1_given_X0Y0']['world_trace'] for n in t},
            'checks':v['checks'],'selection':v['selection']}


def render(v):
    m=manifest(v);out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="580" viewBox="0 0 1120 580" role="img" aria-labelledby="title desc">',
        '<title id="title">CausalLab: same observations do not fix an intervention</title>',
        '<desc id="desc">Supplied finite structural models, exhaustive enumeration. Association is positive .26 while the interventional difference is negative .10. Two observationally identical models give intervention probabilities 1 and .5; shared-noise counterfactual probabilities 1 and 0 given factual X0,Y0. These are synthetic mechanism assumptions, not learned real-world causes.</desc>',
        '<metadata>'+html.escape(json.dumps(m,sort_keys=True,allow_nan=False))+'</metadata>','<rect width="1120" height="580" fill="white"/>']
    def text(x,y,s,size=14,weight=400,anchor='start'):
        out.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="#22283a" text-anchor="{anchor}">{html.escape(str(s))}</text>')
    text(28,36,'CausalLab | same observations, different interventions',25,600)
    text(28,64,'Exhaustive finite worlds under supplied mechanisms. Probability scale 0..1 in every panel.',14)
    text(28,88,'Synthetic model counterexamples, not effects learned from observational data or policy advice.',14)
    panels=[(65,'A. Confounding reverses the sign',[(35,'observed',0,'obs X=0'),(100,'observed',1,'obs X=1'),(185,'do',0,'do X=0'),(250,'do',1,'do X=1')]),
            (425,'B. Same observational joint',[(85,'twins_do',0,'direct'),(225,'twins_do',1,'common')]),
            (785,'C. Shared-noise counterfactual',[(85,'twins_counterfactual',0,'direct'),(225,'twins_counterfactual',1,'common')])]
    for panel,(x,title,bars) in enumerate(panels):
        text(x,128,title,16,600)
        text(x,153,['P(Y=1 | X) versus P(Y=1 | do(X))','P(Y=1 | do(X=1))','P(Y under X=1 is 1 | factual X=0,Y=0)'][panel],12)
        for tick in [0,.5,1.]:
            yy=385-tick*195;out.append(f'<path d="M{x} {yy:.3f}H{x+285}" stroke="#e0e5ea"/>');text(x-9,yy+4,f'{tick:g}',12,anchor='end')
        for dx,name,index,label in bars:
            value=m['series'][name][index];xx=x+dx;yy=385-value*195;color='#8264bb' if name=='observed' else '#13877d'
            out.append(f'<rect data-series="{name}" data-index="{index}" data-value="{value:.17g}" x="{xx-21}" y="{yy:.3f}" width="42" height="{value*195:.3f}" fill="{color}"/>')
            text(xx,yy-9,f'{value:.2f}',13,600,anchor='middle');text(xx,407,label,12,anchor='middle')
    text(65,439,'Association difference: +.26',14,600)
    text(65,462,'Interventional difference: -.10',14,600)
    text(425,439,'Both: P(X=0,Y=0)=P(X=1,Y=1)=.5',12)
    text(425,462,'Direct: X=U,Y=X. Common: X=U,Y=U.',12)
    text(785,439,'Factual evidence fixes U=0 in both.',12)
    text(785,462,'Reusing U is not drawing fresh noise.',12)
    text(28,512,'One observational distribution does not identify the causal mechanism. The equations are assumptions.',16,600)
    text(28,542,'Observation selects worlds; action replaces an equation; counterfactual prediction reuses factual noise.',14)
    text(28,565,'Full models, probability values, factual/action traces and fixed domains are embedded in metadata.',13)
    out.append('</svg>');return '\n'.join(out)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--check',action='store_true');args=p.parse_args()
    svg=render(json.loads(Path(args.input).read_text()));target=Path(__file__).resolve().parents[1]/'assets'/'causal-audit.svg'
    if args.check:
        if target.read_text()!=svg:raise SystemExit('causal SVG differs')
    else:target.write_text(svg)
