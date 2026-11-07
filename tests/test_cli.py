import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from causallab.__main__ import assignments


class CliTests(unittest.TestCase):
    def run_cli(self,*args):return subprocess.run([sys.executable,'-m','causallab',*args],capture_output=True,text=True)
    def test_demo_is_deterministic_and_both_counterexamples_pass(self):
        a=self.run_cli('demo');b=self.run_cli('demo');self.assertEqual(a.returncode,0,a.stderr);self.assertEqual(a.stdout,b.stdout)
        self.assertTrue(all(json.loads(a.stdout)['checks'].values()))
    def test_query_and_counterfactual_commands_have_distinct_evidence_semantics(self):
        a=self.run_cli('query','examples/direct.json','--query','Y','--do','X=1');self.assertEqual(a.returncode,0,a.stderr)
        self.assertEqual(json.loads(a.stdout)['distribution'][1]['probability'],1.)
        b=self.run_cli('counterfactual','examples/common.json','--query','Y','--evidence','X=0,Y=0','--do','X=1','--trace');self.assertEqual(b.returncode,0,b.stderr)
        r=json.loads(b.stdout);self.assertEqual(r['distribution'][1]['probability'],0.);self.assertIn('world_trace',r)
    def test_assignment_parser_duplicate_malformed_and_integer_errors(self):
        self.assertEqual(assignments('X=0,Y=1'),{'X':0,'Y':1})
        for text in ['X=0,X=1','X:0','X=1=2','X=.5','X=true',',']:
            with self.assertRaises(ValueError):assignments(text)
    def test_strict_payload_malformed_duplicate_key_and_large_file(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'model.json'
            for content in ['{','[]','{"version":1,"version":2}', ' '*262145]:
                p.write_text(content);r=self.run_cli('query',str(p),'--query','Y')
                self.assertEqual(r.returncode,2);self.assertEqual(r.stdout,'');self.assertNotIn('Traceback',r.stderr);self.assertIn('causallab:',r.stderr)
    def test_impossible_evidence_and_unknown_variables_are_clean_errors(self):
        for args in [('query','examples/direct.json','--query','Y','--evidence','X=0','--do','X=1'),('query','examples/direct.json','--query','U'),('query','missing-file','--query','Y')]:
            r=self.run_cli(*args);self.assertEqual(r.returncode,2);self.assertEqual(r.stdout,'');self.assertNotIn('Traceback',r.stderr)
    def test_help_missing_command_and_unknown_option(self):
        for command in ['query','counterfactual']:
            p=self.run_cli(command,'--help');self.assertEqual(p.returncode,0);self.assertIn('--query',p.stdout);self.assertIn('--evidence',p.stdout);self.assertIn('--do',p.stdout)
        for args in [(),('other',),('query','examples/direct.json','--query','Y','--unknown')]:self.assertEqual(self.run_cli(*args).returncode,2)
