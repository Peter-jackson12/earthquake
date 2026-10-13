import sys,unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from scripts.finalize_personal import choose,evaluate,validate_holdout_tables
from test_personal_sampling import fixture
from preprocess import ID,TARGET

class PersonalFinalTests(unittest.TestCase):
    def test_selects_cv_only(self):
        rows=[dict(experiment='baseline_v1',model='lr',cv_mean=.68,converged=True),
              dict(experiment='new',model='lr',cv_mean=.67,converged=True)]
        self.assertEqual(choose(rows)['experiment'],'baseline_v1')
    def test_nonconverged_excluded(self):
        rows=[dict(experiment='base',model='lr',cv_mean=.6,converged=True),
              dict(experiment='bad',model='lr',cv_mean=.9,converged=False)]
        self.assertEqual(choose(rows)['experiment'],'base')
    def test_tie_prespecified_simplicity(self):
        rows=[dict(experiment='baseline_v1',model='lr',cv_mean=.68,converged=True),
              dict(experiment='synth',model='lr',cv_mean=.6800005,converged=True)]
        self.assertEqual(choose(rows)['experiment'],'baseline_v1')
    def test_invalid_digest_stops_before_open(self):
        with patch('scripts.finalize_personal.development_frame',side_effect=AssertionError('read too soon')):
            with self.assertRaises(ValueError):evaluate('not-a-selection-sha')
    def test_holdout_join_uses_ids(self):
        X,y=fixture();X.insert(0,ID,range(1,len(X)+1));labels=pd.DataFrame({ID:X[ID],TARGET:y})
        got=validate_holdout_tables(X,labels.iloc[::-1],X[ID],range(500,600))
        self.assertEqual(got.tolist(),y.tolist())
    def test_duplicate_or_overlap_holdout_rejected(self):
        X,y=fixture();X.insert(0,ID,range(1,len(X)+1));labels=pd.DataFrame({ID:X[ID],TARGET:y})
        with self.assertRaises(ValueError):validate_holdout_tables(X,labels,X[ID],[1])
        labels.loc[1,ID]=1
        with self.assertRaises(ValueError):validate_holdout_tables(X,labels,X[ID],[])
    def test_bad_label_rejected(self):
        X,y=fixture();X.insert(0,ID,range(1,len(X)+1));labels=pd.DataFrame({ID:X[ID],TARGET:y});labels.loc[0,TARGET]=4
        with self.assertRaises(ValueError):validate_holdout_tables(X,labels,X[ID],[])
    def test_unseen_id_rejected(self):
        X,y=fixture();X.insert(0,ID,range(1,len(X)+1));labels=pd.DataFrame({ID:X[ID],TARGET:y})
        with self.assertRaises(ValueError):validate_holdout_tables(X,labels,range(90),[])

if __name__=='__main__':unittest.main()
