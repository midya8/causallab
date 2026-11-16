"""Decoder resource failures must follow the same CLI error contract as bad JSON."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class JsonNestingTests(unittest.TestCase):
    def check_files(self, arguments):
        # Each document is below the byte cap, but exceeds the JSON decoder's
        # recursion capacity. Exercise arrays and objects, complete and cut off.
        documents = ['['*1500 + '0' + ']'*1500,
                     '{"nested":'*1500 + '0' + '}'*1500,
                     '['*1500, '{"nested":'*1500]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested.json'
            for document in documents:
                with self.subTest(prefix=document[:12], size=len(document)):
                    path.write_text(document)
                    result = subprocess.run([sys.executable, '-m', 'causallab',
                                             *arguments(str(path))], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertEqual(result.stdout, '')
                    self.assertTrue(result.stderr.startswith('causallab: '), result.stderr)
                    self.assertNotIn('Traceback', result.stderr)
                    self.assertLess(len(result.stderr), 1024)

    def test_primary_model_file(self):
        self.check_files(lambda path: ['query', path, '--query', 'Y'])

    def test_auxiliary_scenario_file(self):
        self.check_files(lambda path: ['joint', 'examples/common.json', '--scenarios', path])

    def test_auxiliary_observed_joint_file(self):
        self.check_files(lambda path: ['frontdoor-effect', 'examples/frontdoor_graph.json',
                                      '--treatment', 'X', '--outcome', 'Y',
                                      '--mediators', 'M', '--joint', path])
