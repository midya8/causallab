import copy
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
from causallab.bounds import bounds_demo
from examples.render_bounds import manifest,render,ORDER


class BoundsFigure(unittest.TestCase):
    def setUp(self):self.demo=bounds_demo();self.svg=render(self.demo)
    def test_all_24_segment_values_coordinates_and_stack_totals(self):
        root=ET.fromstring(self.svg);bars=[r for r in root.findall('{http://www.w3.org/2000/svg}rect') if 'data-response' in r.attrib]
        self.assertEqual(len(bars),24);m=manifest(self.demo);totals={}
        for case in ['equal','positive']:
            for i in range(3):
                spec=[b for b in m['bars'] if b['case']==case][i];cum=0
                for key in ORDER:
                    r=next(b for b in bars if b.attrib['data-case']==case and int(b.attrib['data-bar'])==i and b.attrib['data-response']==key)
                    v=spec['cells'][key]['value'];self.assertEqual(float(r.attrib['data-value']),v)
                    self.assertEqual(float(r.attrib['x']),70+(550 if case=='positive' else 0)+60+i*143-30)
                    self.assertEqual(float(r.attrib['y']),405-(cum+v)*190);self.assertEqual(float(r.attrib['height']),v*190);cum+=v
                self.assertEqual(cum,1)
        self.assertEqual(m['domain'],[0,1])
    def test_complete_metadata_all_ranges_exact_witnesses_and_assumptions(self):
        m=json.loads(ET.fromstring(self.svg).find('{http://www.w3.org/2000/svg}metadata').text)
        self.assertEqual(m,manifest(self.demo));self.assertEqual(m['complete_demo'],self.demo)
        self.assertEqual(len(m['bars']),6);self.assertFalse(m['complete_demo']['cases']['equal']['unrestricted']['monotone']);self.assertTrue(m['complete_demo']['cases']['equal']['monotone']['monotone'])
    def test_exact_regeneration_and_no_mutation(self):
        saved=copy.deepcopy(self.demo);self.assertEqual(self.svg,Path('assets/bounds-audit.svg').read_text());self.assertEqual(self.svg,render(bounds_demo()));self.assertEqual(saved,self.demo)
    def test_changed_exact_mass_range_assumption_or_selection_rejected(self):
        for mode in ['mass','range','assumption','selection','mechanism']:
            d=copy.deepcopy(self.demo)
            if mode=='mass':d['cases']['equal']['unrestricted']['witnesses']['low_joint11']['cells']['helped']['exact']='0'
            elif mode=='range':d['cases']['positive']['unrestricted']['ranges']['helped']['upper']['value']=.6
            elif mode=='assumption':d['cases']['equal']['monotone']['monotone']=False
            elif mode=='selection':d['selection']='searched for results'
            else:d['existing_mechanisms']['flip']['harmed']=0
            with self.subTest(mode=mode),self.assertRaises(ValueError):render(d)
