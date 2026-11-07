import copy
import importlib.util
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
from causallab.graph import graph_demo

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('graph_renderer',ROOT/'examples/render_graph.py')
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
NS={'s':'http://www.w3.org/2000/svg'}


class GraphFigureTests(unittest.TestCase):
    def test_all_eight_statuses_and_complete_metadata(self):
        demo=graph_demo();root=ET.fromstring(renderer.render(demo))
        meta=json.loads(root.find('s:metadata',NS).text)
        self.assertEqual(meta['complete_demo'],demo)
        rows=root.findall('.//s:g[@data-case]',NS)
        self.assertEqual(len(rows),8)
        for row in rows:
            result=demo['cases'][row.attrib['data-case']]
            self.assertEqual(row.attrib['data-separated'],str(result['d_separated']).lower())
            texts=[t.text for t in row.findall('s:text',NS)]
            self.assertIn('SEPARATED' if result['d_separated'] else 'CONNECTED',texts)
            self.assertIn(' / '.join(result['moral_witness_path']) if result['moral_witness_path'] else 'no route',texts)

    def test_topology_coordinates_arrow_endpoints_and_surgery(self):
        root=ET.fromstring(renderer.render(graph_demo()));count=0;node_count=0
        for i,(_,keys,positions) in enumerate(renderer.PANELS):
            ox=28+i%2*550;oy=86+i//2*325;r=graph_demo()['cases'][keys[0]]
            lines=root.findall(f'.//s:line[@data-panel="{i}"]',NS)
            self.assertEqual(len(lines),len(r['graph']['edges']))
            for line in lines:
                a,b=line.attrib['data-parent'],line.attrib['data-child'];self.assertIn([a,b],r['graph']['edges'])
                x1,y1=positions[a];x2,y2=positions[b];dx=x2-x1;dy=y2-y1;d=(dx*dx+dy*dy)**.5
                for attr,value in [('x1',ox+x1+19*dx/d),('y1',oy+y1+19*dy/d),
                                   ('x2',ox+x2-23*dx/d),('y2',oy+y2-23*dy/d)]:
                    self.assertAlmostEqual(float(line.attrib[attr]),value,delta=.00051)
                cut=[a,b] in r['removed_incoming_edges']
                self.assertEqual(line.attrib['data-cut'],str(cut).lower())
                self.assertEqual('stroke-dasharray' in line.attrib,cut);count+=1
            for node in root.findall(f'.//s:circle[@data-panel="{i}"]',NS):
                x,y=positions[node.attrib['data-node']]
                self.assertEqual(float(node.attrib['cx']),ox+x);self.assertEqual(float(node.attrib['cy']),oy+y);node_count+=1
        self.assertEqual(count,9);self.assertEqual(node_count,13)

    def test_byte_reproduction_and_no_mutation(self):
        d=graph_demo();before=copy.deepcopy(d);svg=renderer.render(d)
        self.assertEqual(d,before);self.assertEqual(svg,(ROOT/'assets/graph-audit.svg').read_text())
        self.assertEqual(svg,renderer.render(d))

    def test_changed_results_provenance_graph_witness_and_selection_refused(self):
        for change in [lambda d:d['cases']['chain_open'].__setitem__('d_separated',True),
                       lambda d:d['cases']['fork_do_X'].__setitem__('removed_incoming_edges',[]),
                       lambda d:d['cases']['collider_given_D'].__setitem__('moral_witness_path',[]),
                       lambda d:d['cases']['chain_open']['graph']['edges'].pop(),
                       lambda d:d.__setitem__('selection','changed')]:
            d=graph_demo();change(d)
            with self.assertRaises(ValueError):renderer.render(d)
