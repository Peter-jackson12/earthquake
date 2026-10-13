import sys,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from preprocess import FEATURES,GEO,BINARY,NUMERIC,ALLOWED_TEXT,normalize_features
from personal_sampling import make_plan,PersonalSamplingLR,PROTECTED,CONTINUOUS
from baseline import build_pipeline


def fixture():
    n=90
    d={c:np.zeros(n,dtype=int) for c in GEO+BINARY}
    d.update({c:np.repeat(v[0],n) for c,v in ALLOWED_TEXT.items()})
    d[GEO[2]]=np.arange(n)//30
    d.update(count_floors_pre_eq=np.repeat(2,n),age=np.tile([0,995,10,20,30,40],15),
             area_percentage=np.arange(n)%20+1,height_percentage=np.arange(n)%10+1,
             count_families=np.repeat(1,n))
    return normalize_features(pd.DataFrame(d)[FEATURES]),np.repeat([1,2,3],[18,42,30])


class PersonalSamplingTests(unittest.TestCase):
    def test_original_unchanged(self):
        X,y=fixture();before=X.copy(deep=True)
        make_plan(X,y,[1,1,1]);pd.testing.assert_frame_equal(X,before)
    def test_balanced_and_same_control_donors(self):
        X,y=fixture();a,b,w,d=make_plan(X,y,[1,1,1]);a2,_,_,d2=make_plan(X,y,[1,1,1],interpolate=False)
        np.testing.assert_array_equal(a,a2)
        self.assertEqual(d['after'],{'1':42,'2':42,'3':42})
        self.assertEqual(d['after'],d2['after'])
    def test_seed_reproducibility(self):
        X,y=fixture();v=make_plan(X,y,[1,2,3]);v2=make_plan(X,y,[1,2,3])
        for a,b in zip(v[:3],v2[:3]):np.testing.assert_array_equal(a,b)
    def test_protected_context_and_labels(self):
        X,y=fixture();a,b,w,d=make_plan(X,y,[1,1,1])
        pd.testing.assert_frame_equal(X.iloc[a][PROTECTED].reset_index(drop=True),X.iloc[b][PROTECTED].reset_index(drop=True))
        np.testing.assert_array_equal(y[a],y[b]);self.assertGreater(d['actually_changed_rows'],0)
    def test_age_special_states_never_cross(self):
        X,y=fixture();a,b,w,d=make_plan(X,y,[1,1,1]);syn=X.iloc[a].age.to_numpy()+w*(X.iloc[b].age.to_numpy()-X.iloc[a].age.to_numpy())
        for sentinel in [0,995]:
            m=X.iloc[a].age.to_numpy()==sentinel
            self.assertTrue((syn[m]==sentinel).all())
        normal=~np.isin(X.iloc[a].age,[0,995]);self.assertTrue(((syn[normal]>=10)&(syn[normal]<=40)).all())
    def test_all_singletons_fallback(self):
        X,y=fixture();X[GEO[2]]=np.arange(len(X)).astype(str)
        a,b,w,d=make_plan(X,y,[1,1,1]);np.testing.assert_array_equal(a,b)
        self.assertEqual(d['eligible_interpolation_rows'],0);self.assertEqual(d['exact_copy_rows'],len(a))
    def test_invalid_parameters(self):
        X,y=fixture()
        for scale,k in [([0,1,1],5),([1,1],5),([1,1,1],0),([1,1,1],.5)]:
            with self.assertRaises(ValueError):make_plan(X,y,scale,k=k)
        with self.assertRaises(ValueError):make_plan(X,y[:-1],[1,1,1])
    def test_noninteger_raw_remains_rejected(self):
        X,y=fixture();X['age']=X.age.astype(float);X.loc[0,'age']=10.5
        with self.assertRaises(ValueError):PersonalSamplingLR().fit(X,y)
    def test_three_modes_fit_sparse_and_predict(self):
        X,y=fixture()
        with threadpool_limits(limits=1):
            for mode in ['none','random_over','context_smote']:
                m=PersonalSamplingLR(mode=mode).fit(X,y)
                self.assertTrue(sparse.isspmatrix_csr(m.preprocess_.transform(X)))
                np.testing.assert_allclose(m.predict_proba(X).sum(1),1)
                self.assertEqual(m.fit_rows_,90 if mode=='none' else 126)
                self.assertFalse(hasattr(clone(m),'model_'))
    def test_no_sampling_in_predict(self):
        X,y=fixture()
        with threadpool_limits(limits=1):m=PersonalSamplingLR().fit(X,y)
        with patch('personal_sampling.make_plan',side_effect=AssertionError('Sampler called during predict')):
            self.assertEqual(len(m.predict(X.iloc[:7])),7)
    def test_original_lr_parity(self):
        X,y=fixture();cfg=json.loads((ROOT/'configs/baseline.json').read_text())
        with threadpool_limits(limits=1):
            original=build_pipeline('lr',cfg,jobs=1).fit(X,y)
            current=PersonalSamplingLR(mode='none').fit(X,y)
        np.testing.assert_array_equal(original.predict(X),current.predict(X))
        np.testing.assert_allclose(original.predict_proba(X),current.predict_proba(X),atol=0,rtol=0)
    def test_train_statistics_not_resampled(self):
        X,y=fixture()
        with threadpool_limits(limits=1):m=PersonalSamplingLR().fit(X,y)
        sc=m.preprocess_['columns'].named_transformers_['numeric']['scale']
        np.testing.assert_allclose(sc.mean_,X[NUMERIC].mean())
        before=sc.mean_.copy();valid=X.iloc[:3].copy();valid[GEO[2]]='999999';valid.age=995
        m.predict(valid);np.testing.assert_array_equal(before,sc.mean_)
    def test_reload(self):
        import joblib
        X,y=fixture()
        with threadpool_limits(limits=1):m=PersonalSamplingLR().fit(X,y)
        with tempfile.TemporaryDirectory() as p:
            joblib.dump(m,Path(p)/'m.joblib');r=joblib.load(Path(p)/'m.joblib')
            np.testing.assert_array_equal(m.predict(X),r.predict(X))
    def test_unknown_mode(self):
        X,y=fixture()
        with self.assertRaises(ValueError):PersonalSamplingLR(mode='bad').fit(X,y)

if __name__=='__main__':unittest.main()
