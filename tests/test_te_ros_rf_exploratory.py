"""Aggregate-only test. No private CSV, OOF, joblib, or model fitting."""
import json
import math
import statistics
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
J = ROOT / "reports" / "te_ros_rf_exploratory_summary.json"

def check(cm):
    n=sum(map(sum,cm))
    true=[sum(r) for r in cm]
    pred=[sum(row[j] for row in cm) for j in range(3)]
    f1=[2*cm[j][j]/(true[j]+pred[j]) for j in range(3)]
    return n, sum(f1)/3, sum(cm[j][j] for j in range(3))/n

class TestTEROSRFExploratorySummary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=json.loads(J.read_text(encoding="utf-8"))
        cls.rf=cls.d["random_forest_5fold"]

    def test_folds_complete_and_disjoint_counts(self):
        self.assertEqual([f["fold"] for f in self.rf["folds"]],[0,1,2,3,4])
        for f in self.rf["folds"]:
            n, macro, micro=check(f["confusion"])
            self.assertEqual(n,41696)
            self.assertAlmostEqual(macro,f["macro_f1"],places=12)
            self.assertAlmostEqual(micro,f["micro_f1"],places=12)

    def test_mean_sd(self):
        values=[f["macro_f1"] for f in self.rf["folds"]]
        self.assertAlmostEqual(statistics.mean(values),self.rf["cv_macro_f1_mean"],places=12)
        self.assertAlmostEqual(statistics.stdev(values),self.rf["cv_macro_f1_std_ddof1"],places=12)

    def test_pooled_confusion(self):
        cm=[[sum(f["confusion"][i][j] for f in self.rf["folds"]) for j in range(3)] for i in range(3)]
        self.assertEqual(cm,self.rf["confusion"])
        n,macro,micro=check(cm)
        self.assertEqual(n,208480)
        self.assertAlmostEqual(macro,self.rf["pooled_macro_f1"],places=12)
        self.assertAlmostEqual(micro,self.rf["pooled_micro_f1"],places=12)
        self.assertEqual(cm[2][0],self.rf["error31"])
        self.assertEqual(cm[2][1],self.rf["error32"])

    def test_grade3_recall(self):
        cm=self.rf["confusion"]
        self.assertEqual(self.rf["support"],[20099,118607,69774])
        self.assertAlmostEqual(cm[2][2]/sum(cm[2]),self.rf["grade_recall"][2],places=12)

    def test_scope_not_overstated(self):
        self.assertEqual(len(self.d["fixed_fold0"]["models"]),5)
        self.assertGreater(self.rf["cv_macro_f1_mean"],0.7)
        self.assertLess(self.rf["folds"][4]["macro_f1"],0.7)
        self.assertTrue(self.d["audit_scope"]["row_level_oof_not_saved"])
        self.assertFalse(self.d["audit_scope"]["training_or_model_reaudit"])
        self.assertFalse(self.d["data"]["final_holdout_accessed"])

if __name__=="__main__":
    unittest.main()
