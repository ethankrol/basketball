import json
from pathlib import Path
import tempfile
import unittest

from etl.backfill_history import run_history
from etl.features import content_hash
from test_backfill import fixture


class HistoryTests(unittest.TestCase):
    def test_continuous_elo_chain_and_poll_free_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for season in (2001,2002,2003):
                payload=fixture()
                payload['season']=season
                for row in payload['tables']['games']:
                    row['season']=season%100
                    row['date']=f'{season}-04-01'
                # No poll data in warmup seasons is normal, not an Elo reset.
                payload['tables']['polls']=[]
                (root/str(season)).mkdir()
                (root/str(season)/'source.json').write_text(json.dumps(payload))
            manifest=run_history(root,2001,2003,{})
            states=[json.loads(Path(e['elo_state']).read_text()) for e in manifest['seasons']]
            self.assertTrue(states[0]['bootstrap'])
            self.assertFalse(states[1]['bootstrap'])
            self.assertEqual(states[2]['initial_season'],2001)
            self.assertEqual(states[1]['prior_state_hash'],content_hash(states[0]))
            self.assertEqual(states[2]['prior_state_hash'],content_hash(states[1]))
            self.assertNotEqual(states[0]['ratings'],states[1]['ratings'])
            self.assertEqual(manifest,run_history(root,2001,2003,{}))

    def test_missing_season_does_not_silently_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                run_history(Path(directory),2001,2003,{})
