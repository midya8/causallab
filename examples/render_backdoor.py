"""Fixed backdoor-set audit: more conditioning need not preserve validity."""
import argparse
import html
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causallab.backdoor import backdoor_demo

POSITIONS={'A':(130,172),'B':(400,172),'C':(265,252),'X':(130,340),'Y':(400,340)}


def manifest(value):
    if value!=backdoor_demo():raise ValueError('requires unchanged complete backdoor demo')
    return {'positions':POSITIONS,'row_height':34,'complete_demo':value}


def render(value):
    m=manifest(value);out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="790" viewBox="0 0 1120 790" role="img" aria-labelledby="title desc">',
        '<title id="title">CausalLab: adding a collider can invalidate a backdoor adjustment set</title>',
        '<desc id="desc">A fixed DAG and all eight subsets of available A,B,C. The empty set passes the sufficient backdoor criterion; C alone fails by opening a collider; adding A or B repairs it. Separate fixtures require confounder Z, forbid descendant mediator M, and find no set among an available mediator when common cause U is unavailable. No set is not a proof of non-identifiability.</desc>',
        '<metadata>'+html.escape(json.dumps(m,sort_keys=True,allow_nan=False))+'</metadata>',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#555f75"/></marker></defs>',
        '<rect width="1120" height="790" fill="white"/>']
    def text(x,y,s,size=14,weight=400,fill='#22283a',anchor='start'):
        out.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{html.escape(str(s))}</text>')
    text(28,36,'CausalLab | more controls can create bias',26,600)
    text(28,65,'A supplied causal DAG, a sufficient backdoor criterion, and every subset of explicitly available candidates.',14)
    out.append('<rect x="28" y="88" width="510" height="405" rx="7" fill="#f7f9fc" stroke="#dce2ea"/>')
    text(48,120,'Original supplied graph',18,600)
    for a,b in value['cases']['collider']['graph']['edges']:
        x1,y1=POSITIONS[a];x2,y2=POSITIONS[b];dx=x2-x1;dy=y2-y1;length=(dx*dx+dy*dy)**.5
        out.append(f'<line data-parent="{a}" data-child="{b}" x1="{x1+20*dx/length:.3f}" y1="{y1+20*dy/length:.3f}" x2="{x2-24*dx/length:.3f}" y2="{y2-24*dy/length:.3f}" stroke="#555f75" stroke-width="2" marker-end="url(#arrow)"/>')
    for name,(x,y) in POSITIONS.items():
        out.append(f'<circle data-node="{name}" cx="{x}" cy="{y}" r="19" fill="white" stroke="#555f75" stroke-width="2"/>')
        text(x,y+5,name,16,600,anchor='middle')
    text(48,395,'Conditioning on C opens X <- A -> C <- B -> Y.',13)
    text(48,423,'For the criterion, remove outgoing X -> Y first.',13)
    text(48,450,'That graph transformation is not do(X).',13,600)
    text(48,477,'Empty adjustment works here; C alone does not.',13)
    text(568,119,'All available subsets: A, B, C',18,600)
    for x,label in [(575,'Adjustment'),(765,'Backdoor criterion'),(990,'Minimal?')]:text(x,150,label,13,600)
    for i,row in enumerate(value['cases']['collider']['subset_audits']):
        y=181+i*34;valid=row['criterion_met'];z=row['adjustment'];minimal=z in value['cases']['collider']['inclusion_minimal_sets']
        out.append(f'<g data-adjustment="{",".join(z)}" data-valid="{str(valid).lower()}" data-minimal="{str(minimal).lower()}">')
        out.append(f'<rect x="568" y="{y-19}" width="520" height="30" fill="{"#eef7f5" if valid else "#f3eef9"}"/>')
        text(580,y,'{' + ','.join(z) + '}' if z else '{} (empty)',14)
        text(765,y,'PASS' if valid else 'FAIL: collider opens',13,600,'#13877d' if valid else '#8264bb')
        text(1000,y,'yes' if minimal else 'no',13);out.append('</g>')
    text(568,474,'Inclusion-minimal = no strict valid subset, not fewest controls only.',12)
    specs=[('confounded','Confounding','Available: Z','Only {Z} passes.'),
           ('mediator','Post-treatment mediator','Available: M','Only {} passes; M is a descendant.'),
           ('hidden_common_cause','Unavailable common cause','Available: M, not U','No candidate subset passes.')]
    for i,(key,title,available,summary) in enumerate(specs):
        x=28+i*366;out.append(f'<g data-summary="{key}"><rect x="{x}" y="518" width="352" height="122" rx="6" fill="#f7f9fc" stroke="#dce2ea"/>')
        text(x+15,547,title,16,600);text(x+15,575,available,13);text(x+15,604,summary,13);out.append('</g>')
    text(28,677,'Valid sets license sum_z P(y | x,z) P(z) only under the causal graph and support assumptions.',15,600)
    text(28,706,'The caller declares measurable candidates. Latent common causes must stay in the graph even when unavailable.',13)
    text(28,735,'No valid backdoor set does not prove non-identifiability. This is not complete adjustment or do-calculus search.',13)
    text(28,764,'Fixed motifs; full subset audits and graph hashes embedded. No data, empirical effect estimate or graph discovery.',13)
    return '\n'.join(out+['</svg>'])+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--check',action='store_true');a=p.parse_args()
    svg=render(json.loads(Path(a.input).read_text()));target=Path(__file__).resolve().parents[1]/'assets'/'backdoor-audit.svg'
    if a.check:
        if target.read_text()!=svg:raise SystemExit('backdoor SVG differs')
    else:target.write_text(svg)
