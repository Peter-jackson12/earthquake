"""Public aggregate integrity and Streamlit AppTest. No private data required."""
import copy
import builtins
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dashboard_data import load_bundle,validate_bundle,model_table,error_stats,region_table,load_stratified

class AggregateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.bundle=load_bundle()
    def test_public_bundle_and_metrics(self):
        table=model_table(self.bundle)
        self.assertEqual(len(table),len(self.bundle['models']))
        self.assertTrue(table.cv_macro_f1.between(0,1).all())
        if self.bundle['schema_version']==1:
            self.assertIsNone(self.bundle['holdout_result'])
        else:
            self.assertEqual(self.bundle['holdout_result']['scope'],'personal_final_not_team')
    def test_cv_and_pooled_metrics_are_distinct(self):
        lr=next(x for x in self.bundle['models'] if x['experiment']=='baseline_v1' and x['model']=='lr')
        self.assertNotEqual(lr['cv_mean'],error_stats(lr)['oof_macro_f1'])
        self.assertAlmostEqual(error_stats(lr)['rate_3_to_1'],467/69774)
    def test_mean_corruption_rejected(self):
        bad=copy.deepcopy(self.bundle);bad['models'][0]['cv_mean']+=.01
        with self.assertRaises(ValueError): validate_bundle(bad)
    def test_confusion_corruption_rejected(self):
        bad=copy.deepcopy(self.bundle);bad['models'][0]['matrix'][0][0]+=1
        with self.assertRaises(ValueError): validate_bundle(bad)
    def test_duplicate_identity_rejected(self):
        bad=copy.deepcopy(self.bundle);bad['models'].append(bad['models'][0])
        with self.assertRaises(ValueError): validate_bundle(bad)
    def test_no_legacy_regions_fabricated(self):
        self.assertTrue(region_table(self.bundle,'baseline_v1','lr').empty)
    def test_region_denominators_reconcile(self):
        t=region_table(self.bundle,'class_weights_v1','rf')
        self.assertEqual(len(t),31);self.assertEqual(t.grade3_count.sum(),69774)
        r=t.loc[t.region.eq('26')].iloc[0]
        self.assertAlmostEqual(r.rate_3_to_1,591/1532)
    def test_zero_severe_support_is_missing_not_zero(self):
        b=copy.deepcopy(self.bundle)
        r=next(x for x in b['regions'] if x['experiment']=='class_weights_v1' and x['model']=='rf')
        r.update(grade3_count=0,error_3_to_1=0,error_3_to_2=0)
        row=region_table(b,'class_weights_v1','rf').set_index('region').loc[r['region']]
        self.assertTrue(np.isnan(row.rate_3_to_1));self.assertTrue(np.isnan(row.grade3_recall))
    def test_eda_rate_corruption_rejected(self):
        b=copy.deepcopy(self.bundle);b['eda'][0]['grade3_rate']+=.1
        with self.assertRaises(ValueError):validate_bundle(b)
    def test_loader_only_opens_public_bundle(self):
        opened=[]; original=Path.open; original_csv=__import__('pandas').read_csv
        def traced(path,*args,**kwargs):
            opened.append(Path(path).name);return original(path,*args,**kwargs)
        def traced_csv(path,*args,**kwargs):
            opened.append(Path(path).name);return original_csv(path,*args,**kwargs)
        with patch.object(Path,'open',traced), patch('dashboard_data.pd.read_csv',traced_csv): load_bundle()
        expected={'dashboard_context.json','dashboard_models.json','dashboard_regions.json','eda_summary.csv'}
        if self.bundle['schema_version']==2:expected|={'personal_selection.json','personal_final_holdout.json'}
        self.assertEqual(set(opened),expected)

    def test_final_results_cannot_silently_appear(self):
        b=copy.deepcopy(self.bundle);b['holdout_result']={'f1':.7}
        with self.assertRaises(ValueError):validate_bundle(b)

class StratifiedRankingTests(unittest.TestCase):
    def test_audited_values(self):
        t=load_stratified().set_index('설정')
        self.assertEqual(len(t),11)
        self.assertEqual(t.index[0],'EY 파이프라인 구판 재현 (T33 E1)')
        self.assertAlmostEqual(t.loc['RF+TE+ROS (T14)','층화 CV Macro F1'],0.700427)
        self.assertAlmostEqual(t.loc['GBC 튜닝 (T38, 600단계·깊이 7)','층화 CV Macro F1'],0.682123)
        self.assertEqual(t.loc['DT 튜닝 (T38, 깊이 제한 없음·잎 20)','3→1 오류'],552)
        self.assertEqual(t.loc['LR C=1 · 공식 개인 최종 모델 (T0)','3→1 오류'],467)
    def test_corrupted_mean_rejected(self):
        import json,shutil,tempfile
        with tempfile.TemporaryDirectory() as d:
            for name in ['stratified_ranking_20261016.json','wide_tuning_20261017.json']:
                shutil.copy(ROOT/'reports'/name,Path(d)/name)
            path=Path(d)/'stratified_ranking_20261016.json'
            data=json.loads(path.read_text(encoding='utf-8'));data['settings']['T14']['cv_mean']+=.01
            path.write_text(json.dumps(data),encoding='utf-8')
            with self.assertRaises(ValueError): load_stratified(Path(d))

class PersonalFinalAggregateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.bundle=load_bundle()
    def test_actual_final_metrics_and_scope(self):
        f=self.bundle['holdout_result']
        self.assertEqual(f['scope'],'personal_final_not_team')
        self.assertEqual(f['holdout_rows'],52121)
        self.assertAlmostEqual(f['metrics']['macro_f1'],0.6867713036658166,places=12)
    def test_final_count_metric_and_scope_corruption_rejected(self):
        for key,value in [('scope','team_final'),('evaluation_count',2),('post_holdout_tuning',True)]:
            with self.subTest(key=key):
                b=copy.deepcopy(self.bundle);b['holdout_result'][key]=value
                with self.assertRaises(ValueError):validate_bundle(b)
        b=copy.deepcopy(self.bundle);b['holdout_result']['metrics']['macro_f1']+=.01
        with self.assertRaises(ValueError):validate_bundle(b)
    def test_post_holdout_selection_rejected(self):
        b=copy.deepcopy(self.bundle)
        b['holdout_result']['frozen_at_utc']='2026-10-14T00:00:00+00:00'
        with self.assertRaises(ValueError):validate_bundle(b)
    def test_selected_model_and_hash_mismatch_rejected(self):
        b=copy.deepcopy(self.bundle);b['holdout_result']['mode']='context_smote'
        with self.assertRaises(ValueError):validate_bundle(b)
        b=copy.deepcopy(self.bundle);b['personal_selection_sha256']='bad'
        with self.assertRaises(ValueError):validate_bundle(b)

@unittest.skipUnless(importlib.util.find_spec('streamlit'),'Install requirements-dashboard.txt for UI tests.')
class AppTests(unittest.TestCase):
    def app(self):
        from streamlit.testing.v1 import AppTest
        at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
        self.assertEqual(len(at.exception),0)
        return at
    def test_all_views_render(self):
        at=self.app()
        for view in ['실험 비교','층화 순위 (팀 검토용)','개발 데이터 EDA','오류 분석','개인 최종 평가','검증·진행 상태']:
            at.sidebar.radio[0].set_value(view).run()
            self.assertEqual(len(at.exception),0,view)
    def test_empty_model_filter(self):
        at=self.app();at.multiselect(key='families').set_value([]).run()
        self.assertEqual(len(at.exception),0)
        self.assertIn('선택',at.info[0].value)
    def test_age_boundaries_display(self):
        at=self.app();at.sidebar.radio[0].set_value('개발 데이터 EDA').run()
        at.selectbox(key='eda_feature').set_value('age_group').run()
        self.assertEqual(len(at.exception),0)
        self.assertEqual(len(at.dataframe[0].value),8)
        self.assertEqual(at.dataframe[0].value.n.sum(),208480)
    def test_legacy_error_caveat(self):
        at=self.app();at.sidebar.radio[0].set_value('오류 분석').run()
        at.selectbox(key='error_model').set_value('LR 기본 · C=1').run()
        self.assertIn('과거',at.warning[0].value)
        at.checkbox(key='normalize_cm').check().run()
        self.assertEqual(len(at.exception),0)
    def test_verified_regions(self):
        at=self.app();at.sidebar.radio[0].set_value('오류 분석').run()
        at.selectbox(key='error_model').set_value('RF balanced · 0.1 / leaf=2').run()
        self.assertEqual(len(at.exception),0)
        self.assertEqual(len(at.dataframe[-1].value),31)
        self.assertEqual(at.metric[0].value,'1,862건')
    def test_final_metrics_are_actual_and_personal(self):
        at=self.app();at.sidebar.radio[0].set_value('개인 최종 평가').run()
        self.assertEqual(len(at.exception),0)
        self.assertEqual(at.metric[0].value,'0.686771')
        self.assertEqual(at.metric[2].value,'52,121')
    def test_holdout_status_is_explicit(self):
        at=self.app();at.sidebar.radio[0].set_value('검증·진행 상태').run()
        expected='완료' if load_bundle().get('holdout_result') else '미실행'
        self.assertEqual(at.metric[0].value,expected)
        self.assertEqual(len(at.exception),0)

if __name__=='__main__': unittest.main()
