import copy
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
from causallab.multiworld import multiworld_demo
from examples.render_multiworld import render,manifest


class MultiWorldFigure(unittest.TestCase):
    def setUp(self):self.demo=multiworld_demo();self.svg=render(self.demo)
    def test_eight_bars_joint_values_coordinates_and_product_reference(self):
        root=ET.fromstring(self.svg);bars=[r for r in root.findall('{http://www.w3.org/2000/svg}rect') if 'data-response' in r.attrib]
        self.assertEqual(len(bars),8)
        keys=['always_zero','helped','harmed','always_one']
        for b in bars:
            label=b.attrib['data-model'];key=b.attrib['data-response'];v=self.demo['models'][label]['response_types'][key]
            self.assertEqual(float(b.attrib['data-value']),v);self.assertEqual(float(b.attrib['x']),65+(550 if label=='flip' else 0)+45+keys.index(key)*108-22)
            self.assertEqual(float(b.attrib['y']),390-180*v);self.assertEqual(float(b.attrib['height']),180*v)
        dashed=[p for p in root.findall('{http://www.w3.org/2000/svg}path') if 'data-product' in p.attrib]
        self.assertEqual(len(dashed),2)
        for p in dashed:self.assertIn(' 345H',p.attrib['d'])
        self.assertEqual(manifest(self.demo)['domain'],[0,1])
    def test_metadata_includes_all_models_observed_and_action_joints_traces(self):
        m=json.loads(ET.fromstring(self.svg).find('{http://www.w3.org/2000/svg}metadata').text)
        self.assertEqual(m,manifest(self.demo));self.assertEqual(m['complete_demo'],self.demo)
        self.assertEqual(m['naive_product'],.25)
    def test_regeneration_exact_and_no_input_mutation(self):
        old=copy.deepcopy(self.demo);self.assertEqual(render(self.demo),Path('assets/multiworld-audit.svg').read_text())
        self.assertEqual(old,self.demo);self.assertEqual(render(multiworld_demo()),self.svg)
    def test_changed_model_mass_factual_trace_selection_or_naive_product_rejected(self):
        for change in ['model','mass','trace','selection','product']:
            d=copy.deepcopy(self.demo)
            if change=='model':d['models']['flip']['model']['nodes'][1]['table'][0]=1
            elif change=='mass':d['models']['stable']['response_types']['helped']=.1
            elif change=='trace':d['models']['flip']['factual_X0_Y0']['world_trace'][0]['predictions']['on']['Y']=0
            elif change=='selection':d['selection']='selected best-performing models'
            else:d['models']['stable']['independent_marginal_product'][0]=.3
            with self.subTest(change=change),self.assertRaises(ValueError):render(d)
