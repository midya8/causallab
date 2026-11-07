"""Fixed response-type audit, not an arbitrary-data plotting interface."""
import argparse
import html
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causallab.multiworld import multiworld_demo


def manifest(value):
    if value!=multiworld_demo():raise ValueError('requires unchanged complete multi-world demo')
    return {'domain':[0,1],'naive_product':.25,'response_order':['always_zero','helped','harmed','always_one'],
            'complete_demo':value}


def render(value):
    meta=manifest(value)
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="620" viewBox="0 0 1120 620" role="img" aria-labelledby="title desc">',
         '<title id="title">CausalLab: action marginals do not determine a cross-world joint</title>',
         '<desc id="desc">Both models have uniform observed X,Y and outcome probability one half under both actions. Stable Y=U has half always zero and half always one. Flip Y=X xor U has half helped and half harmed. Multiplying marginals incorrectly assigns one quarter to each response type. These are supplied synthetic mechanisms, not identified effects.</desc>',
         '<metadata>'+html.escape(json.dumps(meta,sort_keys=True,allow_nan=False))+'</metadata>',
         '<rect width="1120" height="620" fill="white"/>']
    def text(x,y,t,size=14,weight=400,anchor='start'):
        out.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="#22283a" text-anchor="{anchor}">{html.escape(str(t))}</text>')
    text(28,37,'CausalLab | action marginals miss cross-world dependence',25,600)
    text(28,66,'Same observed joint: P(X=x,Y=y)=.25 for every pair. Same action marginals: P(Y=1 | do(X=x))=.5.',15)
    text(28,94,'Independent U,V ~ Bernoulli(.5), X=V. Only the supplied Y mechanism differs. No data or model search.',14)
    labels=['always zero','helped','harmed','always one'];pairs=['(Y0,Y1)=(0,0)','(Y0,Y1)=(0,1)','(Y0,Y1)=(1,0)','(Y0,Y1)=(1,1)']
    for panel,(label,title) in enumerate([('stable','Stable: Y=U'),('flip','Flip: Y=X xor U')]):
        x=65+panel*550;text(x,144,title,21,600);text(x,169,'Shared-noise joint over outcomes Y0 and Y1',14)
        for tick in [0,.5,1]:
            yy=390-180*tick;out.append(f'<path d="M{x} {yy}H{x+420}" stroke="#e0e5ea"/>');text(x-10,yy+4,f'{tick:g}',12,anchor='end')
        # The same .25 independent-product error applies to every category.
        out.append(f'<path data-product="{label}" d="M{x} 345H{x+420}" stroke="#8264bb" stroke-dasharray="5 4"/>')
        for i,key in enumerate(meta['response_order']):
            v=value['models'][label]['response_types'][key];xx=x+45+i*108
            out.append(f'<rect data-model="{label}" data-response="{key}" data-value="{v:.17g}" x="{xx-22}" y="{390-v*180:.3f}" width="44" height="{v*180:.3f}" fill="#13877d"/>')
            text(xx,390-v*180-10,f'{v:.2f}',15,600,'middle');text(xx,414,labels[i],13,anchor='middle');text(xx,436,pairs[i],11,anchor='middle')
        text(x,468,'Dashed .25: naive product of identical action marginals (wrong here).',12)
    text(28,513,'Given factual X=0,Y=0, the same noise gives (Y0,Y1)=(0,0) for stable and (0,1) for flip.',16,600)
    text(28,544,'Y0 means Y under do(X=0); Y1 means Y under do(X=1). Both are evaluated on each same noise world.',14)
    text(28,573,'Cross-world dependence requires structural assumptions beyond single-world observations and experiments.',14)
    text(28,600,'Synthetic binary responses only. No monotonicity, learned causes, empirical benefit or real-world harm claim.',13)
    return '\n'.join(out+['</svg>'])+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--check',action='store_true');a=p.parse_args()
    svg=render(json.loads(Path(a.input).read_text()));target=Path(__file__).resolve().parents[1]/'assets'/'multiworld-audit.svg'
    if a.check:
        if target.read_text()!=svg:raise SystemExit('multi-world SVG differs')
    else:target.write_text(svg)
