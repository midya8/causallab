"""Fixed topology audit: eight graphical queries, not empirical associations."""
import argparse
import html
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from causallab.graph import graph_demo

PANELS = [
    ('Chain', ['chain_open','chain_given_M'], {'X':(70,78),'M':(240,78),'Y':(410,78)}),
    ('Fork / common cause', ['fork_open','fork_given_U'], {'U':(240,55),'X':(100,125),'Y':(380,125)}),
    ('Collider / descendant', ['collider_closed','collider_given_C','collider_given_D'],
     {'X':(70,55),'C':(240,90),'Y':(410,55),'D':(240,155)}),
    ('Incoming-edge surgery', ['fork_do_X'], {'U':(240,55),'X':(100,125),'Y':(380,125)})]


def manifest(value):
    if value != graph_demo(): raise ValueError('requires unchanged complete graph demo')
    return {'panels':[{'title':t,'cases':keys,'positions':p} for t,keys,p in PANELS], 'complete_demo':value}


def render(value):
    m=manifest(value)
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="860" viewBox="0 0 1120 860" role="img" aria-labelledby="title desc">',
         '<title id="title">CausalLab: conditioning can block a chain or open a collider</title>',
         '<desc id="desc">Eight d-separation queries on four fixed DAG panels. A chain and common-cause fork connect X and Y, but conditioning on the middle node separates them. A collider is closed until it or its descendant is conditioned on. Removing incoming edges into X separates a common-cause fork. Connected means a graph does not imply independence, not that dependence has been measured.</desc>',
         '<metadata>'+html.escape(json.dumps(m,sort_keys=True,allow_nan=False))+'</metadata>',
         '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#555f75"/></marker></defs>',
         '<rect width="1120" height="860" fill="white"/>']
    def text(x,y,s,size=14,weight=400,fill='#22283a',anchor='start'):
        out.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{html.escape(str(s))}</text>')
    text(28,35,'CausalLab | d-separation makes graph assumptions inspectable',24,600)
    text(28,63,'Ancestor restriction -> moralize -> remove conditioned nodes -> connectivity. No probabilities or data fitting.',14)
    for i,(title,keys,positions) in enumerate(PANELS):
        ox=28+(i%2)*550;oy=86+(i//2)*325
        out.append(f'<rect x="{ox}" y="{oy}" width="528" height="310" rx="7" fill="#f7f9fc" stroke="#dce2ea"/>')
        text(ox+18,oy+29,title,19,600)
        r=value['cases'][keys[0]];removed={tuple(e) for e in r['removed_incoming_edges']}
        for a,b in r['graph']['edges']:
            x1,y1=positions[a];x2,y2=positions[b];dx=x2-x1;dy=y2-y1;length=(dx*dx+dy*dy)**.5
            x1+=ox+dx/length*19;y1+=oy+dy/length*19
            x2+=ox-dx/length*23;y2+=oy-dy/length*23
            cut=(a,b) in removed;dash=' stroke-dasharray="5 4"' if cut else ''
            out.append(f'<line data-panel="{i}" data-parent="{a}" data-child="{b}" data-cut="{str(cut).lower()}" x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" stroke="{"#acb3c0" if cut else "#555f75"}" stroke-width="2"{dash} marker-end="url(#arrow)"/>')
        for node,(x,y) in positions.items():
            out.append(f'<circle data-panel="{i}" data-node="{node}" cx="{ox+x}" cy="{oy+y}" r="18" fill="white" stroke="#555f75" stroke-width="2"/>')
            text(ox+x,oy+y+5,node,15,600,anchor='middle')
        if i==3:
            text(ox+30,oy+176,'do(X): dashed U -> X is removed, not conditioned on.',12)
        start=oy+203
        for j,key in enumerate(keys):
            result=value['cases'][key];sep=result['d_separated'];status='SEPARATED' if sep else 'CONNECTED'
            label='given '+','.join(result['given']) if result['given'] else 'no conditioning'
            if result['interventions']:label+='; cut '+','.join(result['interventions'])
            yy=start+j*32
            out.append(f'<g data-case="{key}" data-separated="{str(sep).lower()}">')
            text(ox+18,yy,label,13)
            text(ox+250,yy,status,13,600,'#13877d' if sep else '#8264bb')
            path=result['moral_witness_path']
            text(ox+372,yy,' / '.join(path) if path else 'no route',12)
            out.append('</g>')
        if i==2:text(ox+18,oy+299,'X / Y is a moral-graph witness via coparent bridge at C.',11)
        elif i==3:text(ox+18,oy+254,'Surgery happens before finding ancestors or moralizing.',13)
        else:text(ox+18,oy+282,'Right column: moral witness or no remaining route.',12)
    text(28,770,'Separated => conditional independence for distributions Markov to the supplied DAG.',16,600)
    text(28,797,'Connected does not assert dependence. Include latent common causes; deterministic effects can add independences.',13)
    text(28,824,'Witnesses belong to the ancestral moral graph, not necessarily to original-DAG active paths. No graph learning.',13)
    text(28,847,'Fixed motifs, eight full query traces and graph hashes are embedded in this SVG. No graph or parameter search.',12)
    return '\n'.join(out+['</svg>'])+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--check',action='store_true');a=p.parse_args()
    svg=render(json.loads(Path(a.input).read_text()));target=Path(__file__).resolve().parents[1]/'assets'/'graph-audit.svg'
    if a.check:
        if target.read_text()!=svg:raise SystemExit('graph SVG differs')
    else:target.write_text(svg)
