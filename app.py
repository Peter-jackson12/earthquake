"""Streamlit results dashboard: aggregate files only, no training or holdout access."""
from __future__ import annotations
import pandas as pd
import altair as alt
import streamlit as st
from dashboard_data import load_bundle, model_table, error_stats, region_table

st.set_page_config(page_title='Earthquake · 실험 검증',layout='wide')
try:
    bundle=load_bundle()
except (OSError,ValueError,KeyError) as exc:
    st.error(f'검증된 집계 파일을 읽을 수 없습니다: {exc}')
    st.stop()
models=model_table(bundle)
st.title('네팔 지진 피해 등급 · 분석과 검증')
st.caption('Peter 개인 실험 | 개발 데이터 208,480건 | 잠정 80/20 · 고정 층화 5-Fold | 최종 모델 미확정')
page=st.sidebar.radio('보기',['실험 비교','개발 데이터 EDA','오류 분석','검증·진행 상태'],key='view')
st.sidebar.caption('집계 전용 화면입니다. 원본·행별 예측·모델·홀드아웃 파일을 읽지 않습니다.')

if page=='실험 비교':
    chosen=st.multiselect('모델 계열',options=['dummy','lr','dt','rf','gbc'],
                          default=['lr','dt','rf','gbc'],key='families')
    selected=models.loc[models.model.isin(chosen)].copy()
    if selected.empty:
        st.info('비교할 모델 계열을 선택하세요.')
    else:
        eligible=selected.loc[selected.converged]
        a,b,c=st.columns(3)
        if len(eligible):
            best=eligible.iloc[0]
            a.metric('선택 범위의 최고 CV Macro F1',f'{best.cv_macro_f1:.6f}')
            a.caption(best.label+' · 최종 선정 아님')
        else: a.metric('수렴 완료 후보','없음')
        b.metric('비교 설정',len(selected))
        c.metric('설정별 검증 건물',f'{bundle["development_rows"]:,}')
        st.subheader('같은 Fold에서 비교한 성능')
        selected['lower']=(selected.cv_macro_f1-selected.fold_std).clip(lower=0)
        selected['upper']=(selected.cv_macro_f1+selected.fold_std).clip(upper=1)
        chart=alt.Chart(selected).encode(y=alt.Y('label:N',sort='-x',title=None))
        bars=chart.mark_bar().encode(x=alt.X('cv_macro_f1:Q',title='CV Macro F1',scale=alt.Scale(domain=[0,1])),
             tooltip=['label',alt.Tooltip('cv_macro_f1:Q',format='.6f'),alt.Tooltip('fold_std:Q',format='.6f'),'evidence'])
        whiskers=chart.mark_rule().encode(x='lower:Q',x2='upper:Q')
        st.altair_chart((bars+whiskers).properties(height=max(280,len(selected)*30)),width='stretch')
        st.caption('막대: 5개 Fold 평균. 선: 평균 ± Fold 표본 표준편차(ddof=1), 신뢰구간 아님. 탐색에 사용한 CV는 최종 독립 평가가 아닙니다.')
        st.dataframe(selected.drop(columns=['lower','upper']),hide_index=True,width='stretch')
        st.download_button('현재 비교표 CSV',selected.drop(columns=['lower','upper']).to_csv(index=False).encode('utf-8-sig'),
                            'earthquake_comparison_summary.csv','text/csv')
        st.info('과거 8개 설정은 집계 재계산 수준입니다. 후속·튜닝 설정의 행별 OOF 검증과 구분해 표시했습니다.')

elif page=='개발 데이터 EDA':
    st.subheader('피해 분포와 관측 연관성')
    support=pd.DataFrame({'피해 등급':['1','2','3'],'건물 수':bundle['class_support']})
    st.bar_chart(support,x='피해 등급',y='건물 수')
    labels={'foundation_type':'기초 코드','roof_type':'지붕 코드','geo_level_1_id':'상위 지역 코드',
            'age_group':'연령 구간','has_superstructure_mud_mortar_stone':'흙 모르타르·석재 구조 플래그',
            'has_superstructure_rc_engineered':'설계된 철근콘크리트 구조 플래그','region_foundation':'지역 × 기초 대조'}
    feature=st.selectbox('분석 변수',list(labels),format_func=labels.get,key='eda_feature')
    eda=pd.DataFrame([x for x in bundle['eda'] if x['feature']==feature])
    eda['3등급 비율 (%)']=eda.grade3_count/eda.n*100
    order=['0','1-10','11-20','21-30','31-50','51-100','101-994','995'] if feature=='age_group' else '-x'
    chart=alt.Chart(eda).mark_bar().encode(
        y=alt.Y('category:N',sort=order,title=None),
        x=alt.X('3등급 비율 (%):Q',scale=alt.Scale(domain=[0,100])),
        tooltip=['category',alt.Tooltip('n:Q',format=',d'),alt.Tooltip('grade3_count:Q',format=',d'),
                 alt.Tooltip('3등급 비율 (%):Q',format='.2f')]).properties(height=max(230,len(eda)*23))
    st.altair_chart(chart,width='stretch')
    st.dataframe(eda[['category','n','grade3_count','3등급 비율 (%)']],hide_index=True,width='stretch')
    st.caption('분모는 각 그룹의 전체 건물 수입니다. 익명 코드를 재료명·좌표로 해석하지 않으며 지역·구조·연령 차이는 인과효과가 아닙니다. 0·995는 별도 연령 구간입니다.')

elif page=='오류 분석':
    record_by_label={m['label']:m for m in bundle['models']}
    label=st.selectbox('설정',models.label.tolist(),key='error_model')
    record=record_by_label[label]; stats=error_stats(record)
    a,b,c=st.columns(3)
    a.metric('3 → 1 오류',f'{stats["error_3_to_1"]:,}건',help='실제 심한 피해를 경미한 피해로 예측')
    a.caption(f'실제 3등급의 {stats["rate_3_to_1"]:.2%}')
    b.metric('3 → 2 오류',f'{stats["error_3_to_2"]:,}건')
    b.caption(f'실제 3등급의 {stats["rate_3_to_2"]:.2%}')
    c.metric('3등급 Recall',f'{stats["recall"][2]:.2%}')
    st.caption(f'위 세 값의 분모: 실제 3등급 {stats["grade3_count"]:,}건. 다른 오류와 함께 해석해야 합니다.')
    normalized=st.checkbox('혼동행렬: 실제 등급별 비율로 표시',key='normalize_cm')
    cells=[]
    for i in range(3):
        for j in range(3):
            count=int(stats['matrix'][i,j]); val=count/stats['support'][i] if normalized else count
            cells.append({'actual':str(i+1),'predicted':str(j+1),'count':count,'value':val,
                          'label':f'{val:.1%}' if normalized else f'{count:,}'})
    frame=pd.DataFrame(cells)
    base=alt.Chart(frame).encode(x=alt.X('predicted:O',title='예측 등급'),y=alt.Y('actual:O',title='실제 등급'))
    heat=base.mark_rect(opacity=0.65).encode(color=alt.Color('value:Q',legend=None),tooltip=['actual','predicted','count','label'])
    text=base.mark_text(fontSize=16).encode(text='label:N')
    st.altair_chart((heat+text).properties(height=280),width='stretch')
    per_class=pd.DataFrame({'등급':['1','2','3'],'Precision':stats['precision'],'Recall':stats['recall'],
                             'F1':stats['f1'],'표본 수':stats['support']})
    st.dataframe(per_class,hide_index=True,width='stretch')
    st.subheader('지역별 심각한 과소예측')
    regional=region_table(bundle,record['experiment'],record['model'])
    if regional.empty:
        st.warning('이 과거 설정의 행별 OOF가 없어 지역별 결합을 독립 검증하지 못했습니다. 지역 오류를 0으로 표시하지 않습니다.')
    else:
        st.dataframe(regional[['region','n','grade3_count','error_3_to_1','rate_3_to_1',
                                'error_3_to_2','rate_3_to_2','grade3_recall']],hide_index=True,width='stretch')
    st.caption('지역 코드는 좌표가 아닙니다. 같은 지진의 무작위 Fold 결과이며 미관측 지역의 일반화 성능이 아닙니다.')

else:
    st.subheader('검증 범위와 다음 승인 지점')
    st.dataframe(pd.DataFrame(bundle['status']),hide_index=True,width='stretch')
    st.metric('최종 홀드아웃 평가','미실행')
    st.info('팀 공통 분할 확정 · 최종 후보 선정 · 홀드아웃 평가 · main 통합은 승인 전입니다.')
    pilot=bundle['smotenc_pilot']
    st.subheader('SMOTENC 안전성 파일럿')
    st.write(f'Fold 0 학습 구간 {pilot["pilot_rows"]:,}건 → 합성 {pilot["synthetic_rows"]:,}건. 모델 학습·성능 평가는 하지 않았습니다.')
    st.warning('전체 실험 확대 보류: '+ ' / '.join(pilot['blocking_reasons']))
    st.dataframe(pd.DataFrame([{'검사':k,'값':v} for k,v in pilot['counts'].items()]),hide_index=True,width='stretch')
    with st.expander('출처·계산 정의'):
        st.json(bundle['provenance'])
        st.write('CV: Fold Macro F1의 산술 평균. OOF: 전체 혼동행렬에서 계산한 Macro F1. 표준편차는 ddof=1입니다.')
