"""Optional private-artifact adversarial checks. No fitting or holdout access."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.model_families.verify_aggregates import verify

KEYS=('MODEL_FAMILIES_DATA','MODEL_FAMILIES_RUN','MODEL_FAMILIES_LR','MODEL_FAMILIES_AGGREGATES')


@unittest.skipUnless(all(os.environ.get(k) for k in KEYS),'Private model-family audit artifacts not configured.')
class AggregateTamperingTests(unittest.TestCase):
    def reject_tamper(self,filename,column):
        data,run,lr,original=[Path(os.environ[k]) for k in KEYS]
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'audit';shutil.copytree(original,out)
            p=out/filename;table=pd.read_csv(p);table.loc[0,column]+=1
            table.to_csv(p,index=False)
            # Repair the checksum too: arithmetic, not hashes alone, must reject it.
            receipt=json.loads((out/'verification.json').read_text())
            receipt['aggregate_sha256'][filename]=hashlib.sha256(p.read_bytes()).hexdigest()
            (out/'verification.json').write_text(json.dumps(receipt))
            with self.assertRaises((ValueError,AssertionError)):
                verify(data,run,lr,out)
    def test_model_cv_mean(self):self.reject_tamper('model_summary.csv','cv_mean')
    def test_group_grade3_denominator(self):self.reject_tamper('group_summary.csv','grade3_count')
    def test_profile_macro_f1(self):self.reject_tamper('error_profiles.csv','macro_f1')
    def test_new_severe_errors(self):self.reject_tamper('paired_transitions.csv','severe_new')


if __name__=='__main__':unittest.main()
