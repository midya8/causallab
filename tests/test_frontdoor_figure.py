import copy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('frontdoor_renderer',ROOT/'examples/render_frontdoor.py')
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
NS={'s':'http://www.w3.org/2000/svg'}


class FrontdoorFigureTests(unittest.TestCase):
    def test_every_condition_cell_and_full_metadata(self):
        d=renderer.demo();root=ET.fromstring(renderer.render(d));m=json.loads(root.find('s:metadata',NS).text)
        self.assertEqual(m['complete_demo'],d)
        rows=root.findall('.//s:g[@data-case]',NS);self.assertEqual(len(rows),4)
        for row in rows:
            expected=d['graph_cases']['cases'][row.attrib['data-case']]['conditions']
            cells=row.findall('s:g',NS);self.assertEqual(len(cells),3)
            for cell in cells:
                v=expected[cell.attrib['data-condition']]
                self.assertEqual(cell.attrib['data-pass'],str(v).lower())
                self.assertEqual(cell.find('s:text',NS).text,'PASS'if v else 'FAIL')

    def test_all_four_exact_bars_on_one_scale(self):
        d=renderer.demo();root=ET.fromstring(renderer.render(d));bars=root.findall('.//s:rect[@data-probability]',NS)
        self.assertEqual(len(bars),4)
        for bar in bars:
            x=int(bar.attrib['data-action']);kind=bar.attrib['data-kind']
            if kind=='observed':p=Fraction(*d['functional_case']['observational_rows'][x]['p_y1_given_x'])
            else:p=next(Fraction(*r['probability'])for r in d['functional_case']['effect']['rows']if r['treatment_value']==x and r['outcome_value']==1)
            self.assertEqual(bar.attrib['data-probability'],str(p))
            self.assertAlmostEqual(float(bar.attrib['height']),180*float(p),delta=.00051)
            self.assertAlmostEqual(float(bar.attrib['y']),760-180*float(p),delta=.00051)
            self.assertEqual(float(bar.attrib['x']),200+460*x+(100 if kind=='frontdoor'else 0))

    def test_every_original_arrow_and_node(self):
        root=ET.fromstring(renderer.render(renderer.demo()));lines=root.findall('.//s:line[@data-parent]',NS)
        self.assertEqual(len(lines),4)
        self.assertEqual({(l.attrib['data-parent'],l.attrib['data-child'])for l in lines},{('U','X'),('U','Y'),('X','M'),('M','Y')})
        for l in lines:
            x1,y1=renderer.POSITIONS[l.attrib['data-parent']];x2,y2=renderer.POSITIONS[l.attrib['data-child']];dx=x2-x1;dy=y2-y1;length=(dx*dx+dy*dy)**.5
            for key,v in [('x1',x1+23*dx/length),('y1',y1+23*dy/length),('x2',x2-27*dx/length),('y2',y2-27*dy/length)]:self.assertAlmostEqual(float(l.attrib[key]),v,delta=.00051)
        circles=root.findall('.//s:circle',NS);self.assertEqual(len(circles),4)
        for c in circles:
            self.assertEqual((float(c.attrib['cx']),float(c.attrib['cy'])),renderer.POSITIONS[c.attrib['data-node']])
            self.assertEqual(c.attrib['stroke-dasharray'],'5 4'if c.attrib['data-node']=='U'else'none')

    def test_byte_reproduction_isolation_and_undisplayed_mutation_refusal(self):
        d=renderer.demo();before=copy.deepcopy(d);svg=renderer.render(d)
        self.assertEqual(d,before);self.assertEqual(svg,(ROOT/'assets/frontdoor-audit.svg').read_text());self.assertEqual(svg,renderer.render(d))
        for change in [lambda a:a['graph_cases']['cases']['direct_bypass']['conditions'].__setitem__('all_directed_paths_intercepted',True),
                       lambda a:a['functional_case']['effect']['rows'][0]['mediator_terms'][0]['inner_terms'][0].__setitem__('weighted',[0,1]),
                       lambda a:a['functional_case'].__setitem__('selection','changed')]:
            d=renderer.demo();change(d)
            with self.assertRaises(ValueError):renderer.render(d)
