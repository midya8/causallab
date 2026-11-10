"""Fixed frontdoor conditions and exact observed/intervention functional."""
import argparse
from fractions import Fraction
import html
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from causallab.frontdoor import frontdoor_demo
from causallab.frontdoor_effect import frontdoor_effect_demo

POSITIONS = {'U': (225, 185), 'X': (100, 320), 'M': (245, 320), 'Y': (390, 320)}


def demo():
    return {'graph_cases': frontdoor_demo(), 'functional_case': frontdoor_effect_demo()}


def render(value):
    if value != demo():
        raise ValueError('requires unchanged complete frontdoor demos')
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="900" viewBox="0 0 1120 900" role="img" aria-labelledby="title desc">',
           '<title id="title">CausalLab: a mediator can identify an effect despite hidden confounding</title>',
           '<desc id="desc">Fixed graph U to X and Y, X to M to Y, with U unobserved. All three sufficient frontdoor conditions pass for M. Added-edge controls fail separate conditions. Under a supplied positive exact synthetic joint, observational P(Y=1|X=0)=41/112 and P(Y=1|X=1)=37/80; the frontdoor functional gives 7/16 and 9/16. Causal interpretation requires the supplied DAG and support assumptions. No graph discovery or sample estimate.</desc>',
           '<metadata>' + html.escape(json.dumps({'positions': POSITIONS, 'bar_baseline': 760, 'bar_scale': 180, 'complete_demo': value}, sort_keys=True, allow_nan=False)) + '</metadata>',
           '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#555f75"/></marker></defs>',
           '<rect width="1120" height="900" fill="white"/>']
    def text(x, y, s, size=14, weight=400, fill='#22283a', anchor='start'):
        out.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{html.escape(str(s))}</text>')
    text(28, 38, 'CausalLab | a mediator can identify an effect', 27, 600)
    text(28, 69, 'Three sufficient graphical conditions, then an exact functional of a supplied positive observed joint.', 15)
    out.append('<rect x="28" y="94" width="460" height="356" rx="7" fill="#f7f9fc" stroke="#dce2ea"/>')
    text(48, 125, 'Supplied causal DAG; U is unavailable', 18, 600)
    g = value['graph_cases']['cases']['latent_confounding']['graph']
    for a, b in g['edges']:
        x1,y1 = POSITIONS[a]; x2,y2 = POSITIONS[b]; dx=x2-x1;dy=y2-y1;length=(dx*dx+dy*dy)**.5
        out.append(f'<line data-parent="{a}" data-child="{b}" x1="{x1+23*dx/length:.3f}" y1="{y1+23*dy/length:.3f}" x2="{x2-27*dx/length:.3f}" y2="{y2-27*dy/length:.3f}" stroke="#555f75" stroke-width="2" marker-end="url(#arrow)"/>')
    for n,(x,y) in POSITIONS.items():
        out.append(f'<circle data-node="{n}" cx="{x}" cy="{y}" r="22" fill="white" stroke="#555f75" stroke-width="2" stroke-dasharray="{"5 4" if n=="U" else "none"}"/>')
        text(x,y+5,n,16,600,anchor='middle')
    text(48, 380, 'M intercepts every directed X-to-Y path.', 14)
    text(48, 408, 'U stays in the graph even when not observed.', 14, 600)
    text(518, 124, 'Every condition remains visible on failure', 18, 600)
    for x,t in [(525,'Supplied graph'),(800,'i'),(890,'ii'),(990,'iii')]:text(x,159,t,14,600)
    labels = {'latent_confounding':'Base: latent U, mediator M', 'direct_bypass':'Add direct X -> Y',
              'treatment_mediator_confounding':'Add hidden U -> M', 'mediator_outcome_confounding':'Add X -> V -> {M,Y}'}
    for i,(key,r) in enumerate(value['graph_cases']['cases'].items()):
        y = 201 + 45*i
        out.append(f'<g data-case="{key}"><rect x="518" y="{y-24}" width="574" height="39" fill="#f7f9fc"/>')
        text(527,y,labels[key],14)
        for j,(condition,yes) in enumerate(r['conditions'].items()):
            out.append(f'<g data-condition="{condition}" data-pass="{str(yes).lower()}">')
            text(800+95*j,y,'PASS' if yes else 'FAIL',14,600,'#13877d' if yes else '#8264bb');out.append('</g>')
        out.append('</g>')
    text(518, 390, 'i: M intercepts directed X/Y paths.', 13)
    text(518, 414, 'ii: no X/M backdoor. iii: X blocks proper M/Y backdoors.', 13)
    text(28, 492, 'Same positive observed joint, different operations', 21, 600)
    text(28, 520, 'P(Y = 1 | X = x) is not P(Y = 1 | do(X = x)). All displayed values are exact fractions.', 14)
    text(680, 552, 'Observed conditional', 13, 600, '#8264bb')
    text(890, 552, 'Frontdoor functional', 13, 600, '#13877d')
    for prob in [0, .5, 1]:
        y=760-180*prob;out.append(f'<line x1="75" y1="{y}" x2="1065" y2="{y}" stroke="#dce2ea"/>');text(60,y+4,prob,12,anchor='end')
    fixture = value['functional_case']
    for x in range(2):
        observed=Fraction(*fixture['observational_rows'][x]['p_y1_given_x'])
        effect=next(Fraction(*r['probability']) for r in fixture['effect']['rows'] if r['treatment_value']==x and r['outcome_value']==1)
        for j,(kind,p,color) in enumerate([('observed',observed,'#8264bb'),('frontdoor',effect,'#13877d')]):
            bx=200+460*x+100*j;by=760-180*float(p);height=180*float(p)
            out.append(f'<rect data-action="{x}" data-kind="{kind}" data-probability="{p.numerator}/{p.denominator}" x="{bx}" y="{by:.3f}" width="65" height="{height:.3f}" fill="{color}"/>')
            text(bx+32.5,by-12,str(p),14,600,color,anchor='middle')
        text(282+460*x,790,'X = '+str(x),15,600,anchor='middle')
    text(28, 831, 'The graph and latent assumptions are supplied, not validated by the joint. Strict positive P(x,m) is required.', 13, 600)
    text(28, 856, 'Original bounded code and synthetic fixture. Established frontdoor semantics are credited in FRONTDOOR.md.', 13)
    text(28, 881, 'No general identification, causal discovery, medical claim, empirical estimator or confidence interval.', 13)
    return '\n'.join(out + ['</svg>']) + '\n'


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--check',action='store_true');a=p.parse_args()
    svg=render(json.loads(Path(a.input).read_text()));target=Path(__file__).resolve().parents[1]/'assets'/'frontdoor-audit.svg'
    if a.check:
        if target.read_text()!=svg:raise SystemExit('frontdoor SVG differs')
    else:target.write_text(svg)
