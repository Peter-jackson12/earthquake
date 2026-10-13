"""Predict from the hash-verified, trusted personal model. Never read labels."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from preprocess import ID,FEATURES,CATEGORICAL,normalize_features,sha256_file


def predict(input_path,output_path,model_path=ROOT/'outputs/personal_final_v1/model.joblib'):
    output=Path(output_path);model_path=Path(model_path)
    if output.exists():raise FileExistsError('Do not overwrite input or previous predictions')
    record=json.loads((ROOT/'reports/personal_final_holdout.json').read_text())
    if sha256_file(model_path)!=record['model_sha256']:
        raise ValueError('Model hash mismatch. Never load an untrusted joblib file.')
    frame=pd.read_csv(input_path,dtype={c:str for c in CATEGORICAL})
    if list(frame.columns)!=[ID]+FEATURES or not len(frame) or frame[ID].isna().any() or frame[ID].duplicated().any():
        raise ValueError('Require unique building_id + the original 38 features, without damage_grade')
    X=normalize_features(frame[FEATURES]);model=joblib.load(model_path)
    with threadpool_limits(limits=1):p=model.predict_proba(X)
    if model.classes_.tolist()!=[1,2,3] or not np.isfinite(p).all():raise ValueError('Invalid model output')
    np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
    result=pd.DataFrame({ID:frame[ID],'predicted_damage_grade':p.argmax(1)+1,
                          'p_grade_1':p[:,0],'p_grade_2':p[:,1],'p_grade_3':p[:,2]})
    output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(output,index=False)
    return len(result)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,default=ROOT/'outputs/personal_final_v1/model.joblib');a=p.parse_args()
    print('Predicted rows:',predict(a.input,a.output,a.model))
