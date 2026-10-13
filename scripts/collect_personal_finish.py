"""Recompute available OOF and publish aggregate-only personal completion bundle."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from preprocess import ID,TARGET,write_json,sha256_file
from scripts.audit_eda_oof import development_frame,validate_oof,verify_metrics,confusion_stats
from dashboard_data import load_bundle,model_table


def collect():
    frame,contract=development_frame(ROOT/'data');rows=[];folds=[];checks=[];packed={}
    old=json.loads((ROOT/'reports/personal_reference_snapshot.json').read_text())
    regions=json.loads((ROOT/'reports/dashboard_regions.json').read_text())
    for path in sorted((ROOT/'outputs').glob('**/oof_predictions.csv.gz')):
        saved_path=path.parent/'metrics.json'
        if not saved_path.exists():continue
        saved=json.loads(saved_path.read_text())
        if saved.get('rows')!=len(frame) or saved.get('status')!='complete':continue
        joined=validate_oof(frame,pd.read_csv(path));stats=verify_metrics(joined,saved)
        if saved.get('oof_sha256') and sha256_file(path)!=saved['oof_sha256']:raise ValueError('OOF digest mismatch')
        checks.append({'path':str(path.relative_to(ROOT)),'rows':len(joined),
                       'oof_sha256':sha256_file(path),'metrics_sha256':sha256_file(saved_path),'status':'PASS'})
        run=path.parent.parent
        if not run.name.startswith('personal_'):continue
        mode=run.name[len('personal_'):-len('_v1')]
        label={'none':'LR C=1 재현 점검','random_over':'LR RandomOverSampler','context_smote':'LR 제약형 SMOTE + 복제'}[mode]
        rows.append(dict(mode=mode,label=label,cv_mean=saved['cv_macro_f1_mean'],cv_std=saved['cv_macro_f1_std'],
                    **{k:v for k,v in stats.items() if k not in ['matrix','macro_f1']},oof_macro_f1=stats['macro_f1'],
                    fit_seconds=saved['total_fit_seconds'],new_configuration=mode!='none'))
        for f in saved['fold_results']:
            treatment=f.get('sampling') or {}
            folds.append(dict(mode=mode,fold=f['fold'],macro_f1=f['validation']['macro_f1'],
                    train_rows=f['train_rows'],validation_rows=f['validation_rows'],
                    fit_rows=f['fit_rows'],fit_seconds=f['fit_seconds'],
                    convergence_ok=f['convergence_ok'],warning_count=len(f['warnings']),
                    **{k:treatment.get(k,0) for k in ['synthetic_or_copied_rows','actually_changed_rows','exact_copy_rows',
                     'singleton_fallback_rows','protected_context_violations','age_state_violations','fractional_continuous_rows']}))
        regional=[]
        for region,part in joined.groupby('geo_level_1_id'):
            m=confusion_stats(part.y_true,part.y_pred)
            regional.append([str(region),len(part),m['grade3_count'],m['error_3_to_1'],m['error_3_to_2']])
        packed[run.name+'|lr']=regional
        if mode=='none':continue  # evidence restoration, not a 20th unique configuration
        record=dict(experiment=run.name,model='lr',label=label,stage='개인 보수적 샘플링',
                    cv_mean=saved['cv_macro_f1_mean'],cv_std=saved['cv_macro_f1_std'],oof_macro_f1=stats['macro_f1'],
                    fold_scores=[x['validation']['macro_f1'] for x in saved['fold_results']],
                    train_mean=float(np.mean([x['train_macro_f1'] for x in saved['fold_results']])),
                    fit_seconds=saved['total_fit_seconds'],converged=saved['all_folds_converged'],
                    matrix=stats['matrix'],evidence_level='row_oof_checked')
        old.append(record);regions[run.name+'|lr']=regional
    if len(rows)!=3 or len(old)!=19 or len(checks)!=12:raise ValueError('Incomplete evidence coverage')
    final_path=ROOT/'reports/personal_final_holdout.json'
    final=json.loads(final_path.read_text());selection=json.loads((ROOT/'reports/personal_selection.json').read_text())
    # A second arithmetic verifier reads saved predictions, not another model score.
    predictions=ROOT/'outputs/personal_final_v1/holdout_predictions.csv.gz'
    if sha256_file(predictions)!=final['predictions_sha256']:raise ValueError('Final prediction hash')
    saved_pred=pd.read_csv(predictions)
    final_stats=confusion_stats(saved_pred.y_true,saved_pred.y_pred)
    if final_stats['matrix']!=final['metrics']['confusion_matrix']:raise ValueError('Final CM mismatch')
    np.testing.assert_allclose(final_stats['macro_f1'],final['metrics']['macro_f1'],atol=1e-12,rtol=0)
    context=json.loads((ROOT/'reports/dashboard_context.json').read_text())
    context['schema_version']=2
    context['holdout_result']=None  # populated from its checksum-bound public file by load_bundle
    context['personal_selection_sha256']=sha256_file(ROOT/'reports/personal_selection.json')
    context['personal_final_sha256']=sha256_file(final_path)
    context['personal_sampling']=[{k:r[k] for k in ['label','cv_mean','cv_std','error_3_to_1','error_3_to_2','new_configuration']} for r in rows]
    context['status']=[{'항목':a,'상태':b} for a,b in [
       ('개인 처리 정책','사전 동결 · 제약형 SMOTE/복제 비교 완료'),('기준선·개선·가중치·RUS·튜닝','19개 고유 설정 · CV 기록 95개'),
       ('기준선 근거 복구','별도 재현 5 Fold · 새 OOF 검증, 원래 파일 복구 아님'),
       ('개인 최종 모델','무가중치 LR C=1 · 합성 미사용'),('개인 홀드아웃','선택 동결 후 1회 완료 · 재튜닝 없음'),
       ('기존 원래 OOF','8개 원래 파일 미확보 · 개인 재현과 구분'),('팀 상태','main 통합·팀 선정 별도, 개인 수행을 팀원 수행으로 대체하지 않음'),
       ('공통 홀드아웃','개인 평가 후 이미 노출됨 · 팀 반복 선택에 재사용 금지')]]
    context['provenance']['personal_policy_commit']=final['policy_commit']
    context['provenance']['personal_selection_sha256']=context['personal_selection_sha256']
    write_json(ROOT/'reports/dashboard_context.json',context)
    (ROOT/'reports/dashboard_models.json').write_text('[\n'+',\n'.join(json.dumps(r,ensure_ascii=False,separators=(',',':')) for r in old)+'\n]\n')
    (ROOT/'reports/dashboard_regions.json').write_text('{\n'+',\n'.join(json.dumps(k)+':'+json.dumps(v,separators=(',',':')) for k,v in regions.items())+'\n}\n')
    pd.DataFrame(rows).to_csv(ROOT/'reports/personal_sampling_summary.csv',index=False)
    pd.DataFrame(folds).to_csv(ROOT/'reports/personal_sampling_fold_summary.csv',index=False)
    write_json(ROOT/'reports/personal_reference_regions.json',{'note':'New LR C=1 reproduction, NOT recovery of historical raw OOF',
                'rows':packed['personal_none_v1|lr']})
    receipt={'status':'PASS','data_contract':contract,'available_oof_runs_checked':len(checks),
             'available_oof_rows_checked':sum(x['rows'] for x in checks),'sources':checks,
             'final_prediction_arithmetic_recheck':'PASS','final_model_scored_once':True,
             'reference_reproduction_distinct_from_original_oof':True,
             'original_eight_raw_oof_files_available':False}
    write_json(ROOT/'reports/personal_validation.json',receipt)
    model_table(load_bundle()).to_csv(ROOT/'reports/personal_comparison_summary.csv',index=False)
    print('PERSONAL BUNDLE PASS',len(old),len(checks),final['metrics']['macro_f1'])

if __name__=='__main__':collect()
