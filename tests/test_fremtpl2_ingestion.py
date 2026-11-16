import tempfile
import unittest
from pathlib import Path

from experiments.data.fetch_fremtpl2 import inspect_arff


class Fremtpl2IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.path = Path(self.temporary_directory.name) / 'sample.arff'

    def test_inspects_arff_schema_and_counts_rows(self):
        self.path.write_text(
            '% synthetic fixture\n'
            '@relation sample\n'
            '@attribute IDpol numeric\n'
            '@attribute "ClaimNb" numeric\n'
            '@data\n'
            '1,0\n'
            '2,1\n',
            encoding='utf-8',
        )

        result = inspect_arff(self.path, ['IDpol', 'ClaimNb'])

        self.assertEqual(result['records'], 2)
        self.assertEqual(result['missing_values'], 0)
        self.assertEqual(result['attributes'], ['IDpol', 'ClaimNb'])

    def test_rejects_schema_mismatch(self):
        self.path.write_text(
            '@relation sample\n'
            '@attribute IDpol numeric\n'
            '@data\n'
            '1\n',
            encoding='utf-8',
        )

        with self.assertRaisesRegex(ValueError, 'schema mismatch'):
            inspect_arff(self.path, ['PolicyID'])

    def test_rejects_malformed_rows_and_missing_values(self):
        self.path.write_text(
            '@relation sample\n'
            '@attribute IDpol numeric\n'
            '@attribute ClaimNb numeric\n'
            '@data\n'
            '1,?\n',
            encoding='utf-8',
        )

        with self.assertRaisesRegex(ValueError, 'missing values'):
            inspect_arff(self.path, ['IDpol', 'ClaimNb'])

        self.path.write_text(
            '@relation sample\n'
            '@attribute IDpol numeric\n'
            '@attribute ClaimNb numeric\n'
            '@data\n'
            '1\n',
            encoding='utf-8',
        )

        with self.assertRaisesRegex(ValueError, 'expected 2'):
            inspect_arff(self.path, ['IDpol', 'ClaimNb'])


if __name__ == '__main__':
    unittest.main()
