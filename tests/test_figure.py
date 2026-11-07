import copy
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
from causallab.demo import demo
from examples.render_demo import manifest,render


class FigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.demo=demo();cls.svg=render(cls.demo)
    def test_all_eight_bars_values_coordinates_and_shared_domain(self):
        root=ET.fromstring(self.svg);bars=[r for r in root.findall('{http://www.w3.org/2000/svg}rect') if 'data-series' in r.attrib]
        self.assertEqual(len(bars),8);m=manifest(self.demo)
        positions={'observed':[100,165],'do':[250,315],'twins_do':[510,650],'twins_counterfactual':[870,1010]}
        for r in bars:
            n=r.attrib['data-series'];i=int(r.attrib['data-index']);v=m['series'][n][i]
            self.assertEqual(float(r.attrib['data-value']),v);self.assertEqual(float(r.attrib['x']),positions[n][i]-21)
            self.assertEqual(float(r.attrib['y']),round(385-v*195,3));self.assertEqual(float(r.attrib['height']),round(v*195,3))
        self.assertEqual(m['domain'],[0,1])
    def test_metadata_complete_models_traces_and_reproducibility(self):
        root=ET.fromstring(self.svg);m=json.loads(root.find('{http://www.w3.org/2000/svg}metadata').text)
        self.assertEqual(m,manifest(self.demo));self.assertEqual(set(m['models']),{'confounded','direct','common'})
        self.assertEqual(self.svg,render(demo()));self.assertEqual(self.svg,Path('assets/causal-audit.svg').read_text())
    def test_changed_values_model_trace_and_missing_checks_are_rejected(self):
        for change in ['value','model','trace','checks']:
            v=copy.deepcopy(self.demo)
            if change=='value':v['confounding']['interventional_Y1'][0]=1.01
            elif change=='model':v['confounding']['model']['exogenous'][0]['probabilities']=[.4,.6]
            elif change=='trace':v['twins']['common']['counterfactual_X1_given_X0Y0']['world_trace'][0]['exogenous']['U']=1
            else:v['checks']['different_interventions']=False
            with self.assertRaises(ValueError):render(v)
    def test_render_does_not_mutate_input(self):
        v=copy.deepcopy(self.demo);old=copy.deepcopy(v);render(v);self.assertEqual(v,old)
