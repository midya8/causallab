import copy
import importlib.util
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
from causallab.backdoor import backdoor_demo
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('backdoor_renderer',ROOT/'examples/render_backdoor.py')
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
NS={'s':'http://www.w3.org/2000/svg'}


class BackdoorFigureTests(unittest.TestCase):
    def test_eight_subsets_statuses_minimality_and_complete_metadata(self):
        d=backdoor_demo();root=ET.fromstring(renderer.render(d));m=json.loads(root.find('s:metadata',NS).text)
        self.assertEqual(m['complete_demo'],d)
        rows=root.findall('.//s:g[@data-adjustment]',NS);self.assertEqual(len(rows),8)
        for i,(row,actual) in enumerate(zip(rows,d['cases']['collider']['subset_audits'])):
            self.assertEqual(row.attrib['data-adjustment'],','.join(actual['adjustment']))
            self.assertEqual(row.attrib['data-valid'],str(actual['criterion_met']).lower())
            self.assertEqual(row.attrib['data-minimal'],str(actual['adjustment'] in d['cases']['collider']['inclusion_minimal_sets']).lower())
            self.assertEqual(float(row.find('s:rect',NS).attrib['y']),181+i*34-19)
        self.assertEqual(len(root.findall('.//s:g[@data-summary]',NS)),3)

    def test_original_graph_nodes_and_all_five_arrow_coordinates(self):
        d=backdoor_demo();root=ET.fromstring(renderer.render(d));lines=root.findall('.//s:line',NS)
        self.assertEqual(len(lines),5)
        for line in lines:
            a,b=line.attrib['data-parent'],line.attrib['data-child'];self.assertIn([a,b],d['cases']['collider']['graph']['edges'])
            x1,y1=renderer.POSITIONS[a];x2,y2=renderer.POSITIONS[b];dx=x2-x1;dy=y2-y1;dist=(dx*dx+dy*dy)**.5
            for key,value in [('x1',x1+20*dx/dist),('y1',y1+20*dy/dist),('x2',x2-24*dx/dist),('y2',y2-24*dy/dist)]:
                self.assertAlmostEqual(float(line.attrib[key]),value,delta=.00051)
        circles=root.findall('.//s:circle',NS);self.assertEqual(len(circles),5)
        for c in circles:
            x,y=renderer.POSITIONS[c.attrib['data-node']]
            self.assertEqual(float(c.attrib['cx']),x);self.assertEqual(float(c.attrib['cy']),y)

    def test_byte_reproduction_and_no_mutation(self):
        d=backdoor_demo();before=copy.deepcopy(d);svg=renderer.render(d)
        self.assertEqual(before,d);self.assertEqual(svg,(ROOT/'assets/backdoor-audit.svg').read_text())
        self.assertEqual(svg,renderer.render(d))

    def test_changed_validity_candidates_provenance_and_selection_refused(self):
        for change in [lambda d:d['cases']['collider']['subset_audits'][0].__setitem__('criterion_met',False),
                       lambda d:d['cases']['confounded']['valid_sets'].append([]),
                       lambda d:d['cases']['hidden_common_cause']['available'].append('U'),
                       lambda d:d['cases']['collider']['graph']['edges'].pop(),
                       lambda d:d.__setitem__('selection','changed')]:
            d=backdoor_demo();change(d)
            with self.assertRaises(ValueError):renderer.render(d)
