"""Presentation view for the Streamlit dashboard: audited aggregates only, team-review tone (not a final decision)."""
from __future__ import annotations
import pandas as pd
import altair as alt
import streamlit as st
from dashboard_data import load_stratified, load_tuning

EY_PREFIX='EY 파이프라인'


def _cover(ctx):
    st.caption('네팔 지진 피해 등급 분류 · 팀 프로젝트')
    st.title('층화 5-Fold 비교와 튜닝 결과')
    st.warning('팀 검토용 자료입니다. 팀 최종 모델과 최종 평가 방법은 아직 정하지 않았습니다.')
    st.write('2015년 네팔 지진의 건물별 피해 등급(1·2·3)을 건물의 위치·구조·재료 특성으로 분류합니다. '
             '이 발표는 같은 층화 5-Fold에서 비교한 모델 순위, 튜닝 과정, 심한 피해의 과소예측(3→1 오류), 해석의 한계를 다룹니다.')
    st.caption('모든 수치는 독립 감사를 마친 reports/ 집계 파일에서 읽습니다. 공식 개인 최종 모델은 T0 LR C=1입니다.')


def _data(ctx):
    st.header('데이터와 클래스 불균형')
    bundle=ctx['bundle']
    support=pd.DataFrame({'피해 등급':['1','2','3'],'건물 수':bundle['class_support']})
    left,right=st.columns([3,2])
    with left:
        st.altair_chart(alt.Chart(support).mark_bar().encode(
            x=alt.X('피해 등급:N',axis=alt.Axis(labelAngle=0)),y=alt.Y('건물 수:Q'),
            tooltip=['피해 등급',alt.Tooltip('건물 수:Q',format=',d')]).properties(height=280),width='stretch')
    with right:
        st.metric('개발 데이터 건물',f'{bundle["development_rows"]:,}')
        st.write(f'1등급 {support["건물 수"][0]:,} · 2등급 {support["건물 수"][1]:,} · 3등급 {support["건물 수"][2]:,}')
        st.write('1등급이 가장 적어 Accuracy 대신 **Macro F1**을 주지표로 씁니다. '
                 '심한 피해(3등급)를 경미한 피해(1등급)로 예측하는 **3→1 오류**도 함께 봅니다.')
    st.caption('홀드아웃 52,121건은 이 비교에 쓰지 않았습니다.')


def _criteria(ctx):
    st.header('평가 기준')
    a,b,c=st.columns(3)
    a.metric('분할','Stratified 5-Fold'); a.caption('seed 42, split_manifest.csv 고정')
    b.metric('주지표','Macro F1'); b.caption('labels 1·2·3, Fold 평균')
    c.metric('함께 보는 오류','3→1'); c.caption('실제 3등급을 1등급으로 예측한 건수')
    st.write('과제 가이드의 "층화 분할, 모든 모델에 같은 분할" 요구에 따라 모든 모델을 같은 5개 Fold로 비교했습니다. '
             '지역 단위 분할(geo2 Group)은 심화 분석이라 이 순위에 섞지 않았습니다.')


def _ranking(ctx):
    st.header('모델 순위')
    data=ctx['ranking'].copy()
    data['구분']=data['설정'].map(lambda s:'은영님 파이프라인' if s.startswith(EY_PREFIX) else '그 밖의 설정')
    color=alt.Color('구분:N',scale=alt.Scale(domain=['은영님 파이프라인','그 밖의 설정'],range=['#2c5b88','#9aa7b4']),title=None)
    chart=alt.Chart(data).mark_bar().encode(
        y=alt.Y('설정:N',sort='-x',title=None),
        x=alt.X('층화 CV Macro F1:Q',scale=alt.Scale(domain=[0.64,0.71],clamp=True),title='층화 CV Macro F1 (축은 0.64부터)'),
        color=color,tooltip=['설정',alt.Tooltip('층화 CV Macro F1:Q',format='.6f'),alt.Tooltip('3→1 오류:Q',format=',d')])
    st.altair_chart(chart.properties(height=380),width='stretch')
    st.write('은영님 파이프라인 4종이 0.7068~0.7073으로 가장 높았고, RF+TE+ROS(T14) 0.700427, LightGBM native(T8) 0.699342가 뒤를 이었습니다.')


def _ranking_table(ctx):
    st.header('순위 상세')
    table=ctx['ranking'].drop(columns=['근거 보고서'])
    st.dataframe(table,hide_index=True,width='stretch',column_config={
        '층화 CV Macro F1':st.column_config.NumberColumn(format='%.6f'),'Fold SD':st.column_config.NumberColumn(format='%.6f'),
        '3등급 Recall':st.column_config.NumberColumn(format='%.4f'),'3→1 비율':st.column_config.NumberColumn(format='%.4f')})
    st.caption('A3 비교에서 묶음이 다른 모든 쌍은 위 묶음이 5개 Fold 모두 높았습니다(검정 아님). '
               '은영님 파이프라인 4종 안의 차이(0.0005 이내)는 구별하지 않습니다. T38 튜닝 3개는 A3 이후 결과입니다.')


def _method(ctx):
    st.header('튜닝 방법: 그리드 서치 + 중첩 CV')
    left,right=st.columns(2)
    with left:
        st.markdown('- 바깥 **층화 5-Fold**: 검증 Fold는 성능 측정에만 사용\n'
                    '- 각 바깥 학습 Fold 안의 **층화 3-Fold**로만 후보 선택\n'
                    '- 고른 후보로 바깥 학습 Fold 전체를 다시 학습해 검증\n'
                    '- 클래스 가중치·재표집은 끔')
        st.graphviz_chart('''digraph{rankdir=LR;node[shape=box,fontsize=11];
            a[label="바깥 학습 Fold\\n(4/5)"];b[label="안쪽 층화 3-Fold\\n후보별 Macro F1"];c[label="최고 후보 재학습"];d[label="바깥 검증 Fold\\n(1/5) 평가"];
            a->b->c->d}''')
    with right:
        with st.container(border=True):
            st.markdown('**T38 후보 범위**\n\n'
                        '- DT: max_depth 16·20·24·32·제한 없음 × min_samples_leaf 5·20·50·100 (20개)\n'
                        '- RF: max_features 0.3·0.45·0.6·0.8 × max_depth 20·30 (8개), 트리 100개\n'
                        '- GBC: n_estimators 300·600 × max_depth 5·7 (4개), 학습률 0.1')
            st.caption('학습 횟수: T36 93회, T37 436회, T38 465회 (안쪽 선택 + 바깥 재학습)')


def _progress(ctx):
    st.header('튜닝 경과')
    tuning=ctx['tuning']
    st.dataframe(tuning,hide_index=True,width='stretch',column_config={
        c:st.column_config.NumberColumn(format='%.6f') for c in ['기본값','1차 튜닝','범위 확장 (T38)']})
    st.write('DT·RF·GBC는 두 차례 튜닝으로 각각 +0.056, +0.040, +0.070 올랐습니다. LR(C=1이 5개 Fold 중 4번 선택)과 '
             '은영님 LightGBM(T36, Δ +0.000047)은 차이가 없어 기존 설정을 유지합니다.')
    st.info('GBC(600단계·깊이 7)와 RF(깊이 30), DT(깊이 제한 없음)는 T38에서도 후보 범위 끝이 선택됐습니다. 범위를 더 넓히면 값이 달라질 수 있습니다.')


def _errors(ctx):
    st.header('Macro F1이 오를수록 3→1 오류도 늘었습니다')
    data=ctx['ranking'].copy()
    data['구분']=data['설정'].map(lambda s:'은영님 파이프라인' if s.startswith(EY_PREFIX) else '그 밖의 설정')
    left,right=st.columns([3,2])
    with left:
        st.altair_chart(alt.Chart(data).mark_circle(size=90).encode(
            x=alt.X('층화 CV Macro F1:Q',scale=alt.Scale(domain=[0.64,0.71])),
            y=alt.Y('3→1 오류:Q',title='3→1 오류 (건)'),color=alt.Color('구분:N',scale=alt.Scale(domain=['은영님 파이프라인','그 밖의 설정'],range=['#b2432e','#2c5b88']),title=None),
            tooltip=['설정',alt.Tooltip('층화 CV Macro F1:Q',format='.6f'),alt.Tooltip('3→1 오류:Q',format=',d')]
        ).properties(height=320),width='stretch')
    with right:
        t=ctx['tuning'].set_index('모델')
        st.markdown(f'실제 3등급 {ctx["bundle"]["class_support"][2]:,}건 중 1등급으로 예측된 건수입니다.\n\n'
                    f'- 은영님 파이프라인 **779~1,002건**, LR-C 452건\n'
                    + ''.join(f'- {m} 튜닝: {int(t.loc[m,"3→1 기본값"])} → {int(t.loc[m,"3→1 1차"])} → {int(t.loc[m,"3→1 확장"])}건\n' for m in ['DT','GBC','RF'])
                    + f'- 은영님 LightGBM 튜닝(T36): {int(t.loc["은영님 LightGBM (T36)","3→1 기본값"])} → {int(t.loc["은영님 LightGBM (T36)","3→1 1차"])}건')
        st.caption('3→1 허용 상한은 정하지 않았습니다. 팀 최종 모델을 고를 때 함께 정해야 합니다.')


def _limits(ctx):
    st.header('해석할 때 지켜야 할 점')
    a,b=st.columns(2)
    with a.container(border=True):
        st.markdown('**선택 통계입니다**\n\n같은 5개 Fold를 여러 실험에서 반복해 썼습니다. 순위는 독립 성능이 아니며 최종 성능은 별도 평가 설계(T29 초안)가 필요합니다.')
    with b.container(border=True):
        st.markdown('**노트북 0.7131과 비교할 수 없습니다**\n\n노트북 값은 전체 260,601행의 단일 무작위 분할 점수입니다. 여기 값은 개발 208,480행의 5-Fold 평균이라 데이터 범위와 분할이 다릅니다.')
    c,d=st.columns(2)
    with c.container(border=True):
        st.markdown('**지역 일반화는 평가하지 않았습니다**\n\n층화 분할에서는 같은 지역이 학습·검증에 함께 들어갑니다. T14는 geo2 Group 분할에서 LR-C보다 낮았습니다(A2). 처음 보는 지역에서는 순위가 다를 수 있습니다.')
    with d.container(border=True):
        st.markdown('**튜닝 범위 경계**\n\nGBC·RF·DT는 T38에서도 후보 범위 끝이 선택됐고, 범위를 더 넓힌 실험은 하지 않았습니다.')


def _next(ctx):
    st.header('최종 결과와 남은 결정')
    final=ctx['bundle'].get('holdout_result')
    if final:
        st.metric('개인 최종 평가 (T0 LR C=1) 홀드아웃 Macro F1',f'{final["metrics"]["macro_f1"]:.6f}')
        st.caption('개인 분석에서 후보를 먼저 고정한 뒤 한 번 평가한 결과입니다. 이 홀드아웃은 이미 사용돼 팀 모델 선택에 다시 쓰지 않습니다.')
    a,b,c=st.columns(3)
    with a.container(border=True):
        st.markdown('**최종 평가 방법**\n\n새 확인 평가 방식(T29 초안)을 검토합니다.')
    with b.container(border=True):
        st.markdown('**팀 최종 모델**\n\n후보: 은영님 파이프라인, T14, T8, LR. Macro F1과 3→1 허용 상한을 함께 봅니다.')
    with c.container(border=True):
        st.markdown('**제출 정리**\n\nREADME·대시보드·Notion을 마감(10-19 23:59) 전에 맞춥니다.')
    st.caption('근거 보고서: STRATIFIED_RANKING_20261016 · EY_PIPELINE_V2_BENCHMARK_20261016 · EY_L_TUNING_20261016 · '
               'REQUIRED_MODELS_TUNING_20261017 · WIDE_TUNING_20261017 · STRATIFIED_SUMMARY_20261016 (reports/)')


SLIDES=[('표지',_cover),('데이터',_data),('평가 기준',_criteria),('모델 순위',_ranking),('순위 상세',_ranking_table),
        ('튜닝 방법',_method),('튜닝 경과',_progress),('3→1 오류',_errors),('한계',_limits),('남은 결정',_next)]


def _move(step:int):
    st.session_state.slide=min(len(SLIDES)-1,max(0,st.session_state.slide+step))


def render(bundle:dict):
    try:
        ctx={'bundle':bundle,'ranking':load_stratified(),'tuning':load_tuning()}
    except (OSError,ValueError,KeyError) as exc:
        st.error(f'발표용 집계 파일을 읽을 수 없습니다: {exc}')
        st.stop()
    st.session_state.setdefault('slide',0)
    prev,pick,nxt=st.columns([1,4,1],vertical_alignment='center')
    prev.button('← 이전',key='slide_prev',on_click=_move,args=(-1,),disabled=st.session_state.slide==0,width='stretch')
    pick.selectbox('슬라이드',range(len(SLIDES)),key='slide',label_visibility='collapsed',
                   format_func=lambda i:f'{i+1} / {len(SLIDES)} · {SLIDES[i][0]}')
    nxt.button('다음 →',key='slide_next',on_click=_move,args=(1,),disabled=st.session_state.slide==len(SLIDES)-1,width='stretch')
    st.progress((st.session_state.slide+1)/len(SLIDES))
    with st.container(border=True):
        SLIDES[st.session_state.slide][1](ctx)
