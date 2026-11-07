"""Fixed binary response-bound witness audit with all 24 segment masses."""
import argparse
import html
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causallab.bounds import bounds_demo


ORDER=['always_zero','helped','harmed','always_one']
COLORS=['#22283a','#13877d','#8264bb','#cbd1dc']


def manifest(value):
    if value!=bounds_demo():raise ValueError('requires unchanged complete response-bounds demo')
    bars=[]
    for case in ['equal','positive']:
        for assumption,witness,label in [('unrestricted','low_joint11','min P(1,1)'),('unrestricted','high_joint11','max P(1,1)'),('monotone','low_joint11','monotone')]:
            cells=value['cases'][case][assumption]['witnesses'][witness]['cells']
            bars.append({'case':case,'assumption':assumption,'witness':witness,'label':label,'cells':cells})
    return {'domain':[0,1],'response_order':ORDER,'bars':bars,'complete_demo':value}


def render(value):
    m=manifest(value);out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="650" viewBox="0 0 1120 650" role="img" aria-labelledby="title desc">',
        '<title id="title">CausalLab: sharp response bounds expose the monotonicity assumption</title>',
        '<desc id="desc">Two supplied action-marginal pairs, with endpoint joint witnesses and a monotone witness. Equal marginals permit helped fractions zero to one half, but monotonicity forces zero. Marginals one quarter and three quarters permit helped fractions one half to three quarters and harmed fractions up to one quarter, despite positive mean effect. These are feasible-joint ranges, not confidence intervals.</desc>',
        '<metadata>'+html.escape(json.dumps(m,sort_keys=True,allow_nan=False))+'</metadata>',
        '<rect width="1120" height="650" fill="white"/>']
    def text(x,y,s,size=14,weight=400,anchor='start'):
        out.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="#22283a" text-anchor="{anchor}">{html.escape(str(s))}</text>')
    text(28,36,'CausalLab | sharp bounds, explicit assumptions',25,600)
    text(28,65,'Supplied point action marginals constrain a joint. Extra monotonicity Y1 >= Y0 rules out harmed (1,0).',15)
    for i,(label,color) in enumerate(zip(['always zero (0,0)','helped (0,1)','harmed (1,0)','always one (1,1)'],COLORS)):
        x=30+i*268;out.append(f'<rect x="{x}" y="88" width="15" height="15" fill="{color}"/>');text(x+23,100,label,13)
    for panel,case in enumerate(['equal','positive']):
        x=70+panel*550;text(x,146,['A. p0=.50, p1=.50','B. p0=.25, p1=.75'][panel],20,600)
        text(x,173,['Mean effect = 0; helped range [0,.50]','Mean effect = +.50; helped range [.50,.75]'][panel],14)
        for tick in [0,.5,1]:
            y=405-190*tick;out.append(f'<path d="M{x} {y}H{x+405}" stroke="#e0e5ea"/>');text(x-10,y+4,f'{tick:g}',12,anchor='end')
        for i,b in enumerate([v for v in m['bars'] if v['case']==case]):
            xx=x+60+i*143;cumulative=0.
            for key,color in zip(ORDER,COLORS):
                v=b['cells'][key]['value'];y=405-(cumulative+v)*190
                out.append(f'<rect data-case="{case}" data-bar="{i}" data-response="{key}" data-value="{v:.17g}" x="{xx-30}" y="{y:.3f}" width="60" height="{v*190:.3f}" fill="{color}"/>')
                if v>0:
                    ink='white' if key!='always_one' else '#22283a'
                    out.append(f'<text x="{xx}" y="{y+v*95+5:.3f}" fill="{ink}" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" font-weight="600">{v:.2f}</text>')
                cumulative+=v
            text(xx,430,b['label'],13,anchor='middle')
        text(x,462,['Unrestricted endpoints: flip and stable.','Positive mean still permits .25 harmed.'][panel],14,600)
        text(x,485,['Monotonicity keeps stable, excludes flip.','Monotonicity forces harmed=0, helped=.50.'][panel],13)
    text(28,535,'Each stacked column is an attaining joint witness; two endpoint witnesses span every feasible joint.',16,600)
    text(28,565,'p0=P(Y0=1), p1=P(Y1=1). Ranges are sharp for supplied point marginals, not sampling confidence intervals.',14)
    text(28,593,'All calculations use exact rational arithmetic for the supplied binary-float inputs; displayed values are rounded.',14)
    text(28,622,'No monotonicity is learned from a positive mean. Synthetic response names do not imply empirical benefit or harm.',13)
    return '\n'.join(out+['</svg>'])+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--check',action='store_true');a=p.parse_args()
    svg=render(json.loads(Path(a.input).read_text()));target=Path(__file__).resolve().parents[1]/'assets'/'bounds-audit.svg'
    if a.check:
        if target.read_text()!=svg:raise SystemExit('response-bounds SVG differs')
    else:target.write_text(svg)
