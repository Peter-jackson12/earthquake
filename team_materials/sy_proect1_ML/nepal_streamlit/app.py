"""
네팔 지진 건물 피해 등급 예측 — Streamlit 앱

실행:  streamlit run app.py
필요한 파일 (app.py와 같은 폴더):
  - nepal_building_damage.csv
  - nepal_building_damage_dictionary.xlsx (컬럼 정의서, 없어도 됨)
  - models/final_model.joblib (없으면 '피해 등급 예측' 페이지에서 학습 버튼으로 만들 수 있음)
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import results as R

BASE = Path(__file__).parent
DATA_PATH = BASE / "nepal_building_damage.csv"
DICT_PATH = BASE / "nepal_building_damage_dictionary.xlsx"
MODEL_PATH = BASE / "models" / "final_model.joblib"
METRIC_PATH = BASE / "models" / "test_metrics.json"
ASSETS = BASE / "assets"

GRADES = R.GRADE_LABELS
GCOLOR = R.GRADE_COLORS
INK, MUTED, BRICK = "#2A2521", "#7A7067", "#B4432F"

st.set_page_config(page_title="네팔 지진 건물 피해 예측", page_icon="🧱", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@600;800&family=IBM+Plex+Sans+KR:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, p, li, label, .stMetric {
  font-family: 'IBM Plex Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
}
h1, h2, h3 { font-family: 'Noto Serif KR', 'Malgun Gothic', serif !important; letter-spacing: -0.01em; }
h1 { font-weight: 800 !important; }
.block-container { padding-top: 2.2rem; max-width: 1200px; }
.hero-title { font-family: 'Noto Serif KR', serif; font-weight: 800; font-size: 2.6rem;
  line-height: 1.25; color: #2A2521; margin: 0 0 0.6rem 0; }
.hero-sub { color: #5E554D; font-size: 1.05rem; max-width: 760px; line-height: 1.7; }
.note { color: #6E655D; font-size: 0.92rem; line-height: 1.65; }
.rule { border-top: 2px solid #2A2521; margin: 1.6rem 0 1rem 0; }
.grade-chip { display:inline-block; padding: 2px 10px; border-radius: 3px; color: white;
  font-weight: 600; font-size: 0.85rem; margin-right: 6px; }
[data-testid="stMetricValue"] { font-family: 'Noto Serif KR', serif; }
</style>
""", unsafe_allow_html=True)


def chart_style(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="IBM Plex Sans KR, Malgun Gothic, sans-serif", color=INK),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(gridcolor="#E2DBD0", zeroline=False)
    fig.update_yaxes(gridcolor="#E2DBD0", zeroline=False)
    return fig


# ------------------------------------------------------------------ 데이터 불러오기
@st.cache_data(show_spinner="데이터를 불러오는 중…")
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["피해 등급"] = df["damage_grade"].map({1: "1등급", 2: "2등급", 3: "3등급"})
    return df


@st.cache_data(show_spinner=False)
def load_dictionary():
    if not DICT_PATH.exists():
        return None
    raw = pd.read_excel(DICT_PATH, sheet_name="컬럼정의서", header=None)
    header_row = raw.index[raw.iloc[:, 0].astype(str).str.strip() == "컬럼명"][0]
    table = raw.iloc[header_row + 1:].copy()
    table.columns = raw.iloc[header_row].astype(str).str.strip()
    return table.dropna(subset=["컬럼명"]).reset_index(drop=True)


@st.cache_data(show_spinner="테스트 데이터를 준비하는 중…")
def load_test_split():
    from train_model import split_data
    df = pd.read_csv(DATA_PATH)
    _, _, X_test, y_test = split_data(df)
    return X_test, y_test


@st.cache_resource(show_spinner="모델을 불러오는 중…")
def load_model():
    return joblib.load(MODEL_PATH)


if not DATA_PATH.exists():
    st.error(f"`{DATA_PATH.name}` 파일이 없습니다. app.py와 같은 폴더에 두고 다시 실행하세요.")
    st.stop()

df = load_data()
material_cols = [c for c in df.columns if c.startswith("has_superstructure_")]
secondary_cols = [c for c in df.columns if c.startswith("has_secondary_use_")]
num_cols = ["count_floors_pre_eq", "age", "area_percentage", "height_percentage", "count_families"]

# ------------------------------------------------------------------ 사이드바
with st.sidebar:
    st.markdown("### 🧱 네팔 지진 피해 예측")
    page = st.radio("페이지", ["개요", "데이터 탐색", "모델링 과정", "최종 모델 평가·해석", "피해 등급 예측"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div class="note">2015년 네팔 고르카 지진 건물 260,601채의 구조·재료·지역 정보로 '
                '피해 등급(1~3)을 예측합니다.<br><br>평가 지표: Macro F1<br>'
                '분할: 학습 70% · 검증 15% · 테스트 15%</div>', unsafe_allow_html=True)


# ================================================================== 1. 개요
if page == "개요":
    st.markdown('<div class="hero-title">같은 지진, 다른 피해.<br>26만 채 건물의 피해 등급을 예측합니다</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">2015년 4월 네팔 고르카 지진(규모 7.8) 뒤 조사된 건물 260,601채의 '
                '층수·나이·구조 재료·지역 정보로, 건물이 입은 피해를 세 등급으로 나눠 예측합니다. '
                '1등급은 경미한 피해, 2등급은 중간 피해, 3등급은 거의 완파입니다.</div>',
                unsafe_allow_html=True)

    counts = df["피해 등급"].value_counts().reindex(GRADES)
    fig = go.Figure()
    for g in GRADES:
        fig.add_bar(y=["건물"], x=[counts[g]], orientation="h", name=g, marker_color=GCOLOR[g],
                    text=f"{g}  {counts[g]:,}채 ({counts[g] / len(df):.1%})", textposition="inside",
                    insidetextanchor="middle", textfont=dict(color="white", size=14))
    fig.update_layout(barmode="stack", showlegend=False, height=110, margin=dict(l=0, r=0, t=10, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown('<div class="note">1등급이 10%밖에 되지 않는 불균형 데이터라서, 등급별 성능을 똑같이 '
                '평균하는 <b>Macro F1</b>을 주 지표로 썼습니다.</div>', unsafe_allow_html=True)

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.subheader("최종 결과")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("테스트 Macro F1", "0.7158", "+0.0487 (베이스라인 검증 0.6671 대비)")
    c2.metric("테스트 Accuracy", "0.7473")
    c3.metric("1등급 Recall", "0.6949", "+0.25 (베이스라인 XGBoost 0.4429 대비)")
    c4.metric("최종 모델", "HGB + LightGBM", "soft voting", delta_color="off")

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.subheader("성능이 오른 과정")
    j = R.JOURNEY.copy()
    j["단계명"] = j["단계"].str.replace("\n", " ")
    fig = px.line(j, x="단계명", y="검증 Macro F1", markers=True, text="검증 Macro F1", hover_data=["설명"])
    fig.update_traces(line_color=BRICK, marker=dict(size=10), texttemplate="%{y:.4f}", textposition="top center")
    fig.update_yaxes(range=[0.65, 0.735])
    fig.update_xaxes(title=None)
    st.plotly_chart(chart_style(fig, 360), width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**가장 큰 개선: 지역 정보를 직접 쓰기**")
        st.markdown('<div class="note">베이스라인은 지역 ID를 숫자로만 넣었습니다. 지역 ID를 LightGBM 범주형으로 '
                    '바꾸고 지역별 통계를 더하자 검증 F1이 0.667에서 0.718로 크게 올랐습니다. '
                    '이후의 파생변수 추가·튜닝·앙상블은 각각 0.002 안팎의 작은 개선이었습니다.</div>',
                    unsafe_allow_html=True)
    with right:
        st.markdown("**모델이 가장 많이 본 정보**")
        st.markdown('<div class="note">변수 중요도·permutation importance·SHAP 세 방법 모두 '
                    '"같은 지역에서 같은 재료로 지은 건물들이 얼마나 피해를 입었나"(지역×재료 피해 비율)를 '
                    '가장 중요한 변수로 꼽았습니다.</div>', unsafe_allow_html=True)


# ================================================================== 2. 데이터 탐색
elif page == "데이터 탐색":
    st.title("데이터 탐색")
    regions = sorted(df["geo_level_1_id"].unique())
    picked = st.multiselect("대지역(geo_level_1_id)으로 좁혀 보기 — 비워 두면 전체", regions)
    d = df[df["geo_level_1_id"].isin(picked)] if picked else df
    st.caption(f"표시 중: {len(d):,}채")

    tabs = st.tabs(["피해 등급", "건물 나이", "수치형 변수", "구조 재료", "부가 용도", "범주형 변수", "상관관계", "컬럼 정의서"])

    with tabs[0]:
        ratio = d["피해 등급"].value_counts(normalize=True).reindex(GRADES).mul(100).reset_index()
        ratio.columns = ["피해 등급", "비율(%)"]
        fig = px.bar(ratio, x="피해 등급", y="비율(%)", color="피해 등급", color_discrete_map=GCOLOR, text="비율(%)")
        fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(chart_style(fig), width="stretch")
        region_rate = (df.groupby("geo_level_1_id")["damage_grade"].value_counts(normalize=True)
                       .unstack().rename(columns={1: "1등급", 2: "2등급", 3: "3등급"}).mul(100))
        region_rate = region_rate.sort_values("3등급", ascending=False).reset_index()
        long = region_rate.melt(id_vars="geo_level_1_id", var_name="피해 등급", value_name="비율(%)")
        long["대지역"] = long["geo_level_1_id"].astype(str)
        fig = px.bar(long, x="대지역", y="비율(%)", color="피해 등급", color_discrete_map=GCOLOR,
                     title="대지역별 피해 등급 구성 (3등급 비율 높은 순)")
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=region_rate["geo_level_1_id"].astype(str))
        st.plotly_chart(chart_style(fig, 400), width="stretch")
        st.markdown('<div class="note">대지역에 따라 3등급 비율이 크게 달라집니다. 지역 정보가 모델의 핵심 변수가 된 이유입니다.</div>',
                    unsafe_allow_html=True)

    with tabs[1]:
        n995 = (d["age"] == 995).sum()
        st.markdown(f'<div class="note">age = 995인 건물이 {n995:,}채 있습니다. 실제 나이가 아닌 특수값일 가능성이 있어 '
                    '분석에서는 따로 표시(is_995)했습니다. 아래 그래프는 0~200년만 보여 줍니다.</div>', unsafe_allow_html=True)
        dd = d[d["age"] <= 200]
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(dd, x="age", nbins=40, title="건물 나이 분포", color_discrete_sequence=["#8C7B6B"])
            st.plotly_chart(chart_style(fig), width="stretch")
        with c2:
            fig = px.histogram(dd, x="age", color="피해 등급", nbins=40, histnorm="percent", barmode="overlay",
                               color_discrete_map=GCOLOR, opacity=0.55, title="피해 등급별 나이 분포 (등급 안에서 비율)")
            st.plotly_chart(chart_style(fig), width="stretch")
        age_bin = pd.cut(dd["age"], [-1, 5, 10, 20, 30, 50, 80, 200], labels=["0~5", "6~10", "11~20", "21~30", "31~50", "51~80", "81~200"])
        rate = dd.groupby(age_bin, observed=True)["damage_grade"].apply(lambda s: (s == 3).mean() * 100).reset_index()
        rate.columns = ["나이 구간", "3등급 비율(%)"]
        fig = px.bar(rate, x="나이 구간", y="3등급 비율(%)", text="3등급 비율(%)", color_discrete_sequence=[GCOLOR["3등급"]],
                     title="나이 구간별 3등급(거의 완파) 비율")
        fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
        st.plotly_chart(chart_style(fig), width="stretch")
        rho = d[["age", "damage_grade"]].corr(method="spearman").iloc[0, 1]
        st.markdown(f'<div class="note">나이와 피해 등급의 스피어만 상관계수는 {rho:.3f}로 약한 양의 관계입니다. '
                    '오래된 건물일수록 3등급 비율이 높아지지만, 나이만으로 피해를 설명하기는 어렵습니다.</div>',
                    unsafe_allow_html=True)

    with tabs[2]:
        col = st.selectbox("변수", num_cols, format_func=lambda c: {
            "count_floors_pre_eq": "count_floors_pre_eq (지진 전 층수)", "age": "age (건물 나이)",
            "area_percentage": "area_percentage (정규화 면적)", "height_percentage": "height_percentage (정규화 높이)",
            "count_families": "count_families (거주 가구 수)"}[c])
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(d, x="피해 등급", y=col, color="피해 등급", color_discrete_map=GCOLOR,
                         category_orders={"피해 등급": GRADES}, title=f"피해 등급별 {col}")
            fig.update_layout(showlegend=False)
            st.plotly_chart(chart_style(fig), width="stretch")
        with c2:
            s = d[col]
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            out = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean()
            st.markdown(f"**{col} 요약**")
            st.dataframe(s.describe().to_frame("값").T.round(2), width="stretch")
            st.markdown(f'<div class="note">IQR 기준 이상치 비율: <b>{out:.1%}</b><br>'
                        '층수·나이·면적은 오른쪽으로 긴 꼬리를 가집니다. 트리 모델은 값의 순서만 쓰므로 '
                        '이상치를 지우지 않고 그대로 사용했습니다.</div>', unsafe_allow_html=True)

    with tabs[3]:
        rows = []
        for c in material_cols:
            used = d[d[c] == 1]
            rows.append({"재료": c.replace("has_superstructure_", ""), "사용 건물 수": len(used),
                         "3등급 비율(%)": (used["damage_grade"] == 3).mean() * 100 if len(used) else np.nan,
                         "1등급 비율(%)": (used["damage_grade"] == 1).mean() * 100 if len(used) else np.nan})
        m = pd.DataFrame(rows).sort_values("3등급 비율(%)", ascending=False)
        avg3 = (d["damage_grade"] == 3).mean() * 100
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(m.sort_values("사용 건물 수"), x="사용 건물 수", y="재료", orientation="h",
                         title="상부 구조 재료별 사용 건물 수", color_discrete_sequence=["#8C7B6B"])
            st.plotly_chart(chart_style(fig, 430), width="stretch")
        with c2:
            fig = px.bar(m.sort_values("3등급 비율(%)"), x="3등급 비율(%)", y="재료", orientation="h",
                         title="재료별 3등급 비율", color_discrete_sequence=[GCOLOR["3등급"]])
            fig.add_vline(x=avg3, line_dash="dash", line_color=INK, annotation_text=f"전체 평균 {avg3:.1f}%")
            st.plotly_chart(chart_style(fig, 430), width="stretch")
        top = m.iloc[0]
        st.markdown(f'<div class="note">{top["재료"]}를 쓴 건물의 3등급 비율은 {top["3등급 비율(%)"]:.1f}%로 전체 평균 '
                    f'{avg3:.1f}%보다 {top["3등급 비율(%)"] - avg3:.1f}%p 높습니다. 흙·돌 계열은 피해가 크고, '
                    'RC(철근콘크리트)·시멘트 계열은 피해가 작습니다. 한 건물이 여러 재료를 함께 쓸 수 있습니다.</div>',
                    unsafe_allow_html=True)

    with tabs[4]:
        rows = []
        for c in secondary_cols:
            used = d[d[c] == 1]
            rows.append({"용도": c.replace("has_secondary_use_", ""), "건물 수": len(used),
                         "3등급 비율(%)": (used["damage_grade"] == 3).mean() * 100 if len(used) else np.nan})
        s = pd.DataFrame(rows).sort_values("건물 수")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(s, x="건물 수", y="용도", orientation="h", title="부가 용도별 건물 수",
                         color_discrete_sequence=["#8C7B6B"])
            st.plotly_chart(chart_style(fig, 400), width="stretch")
        with c2:
            fig = px.bar(s.sort_values("3등급 비율(%)"), x="3등급 비율(%)", y="용도", orientation="h",
                         title="부가 용도별 3등급 비율", color_discrete_sequence=[GCOLOR["3등급"]])
            fig.add_vline(x=(d["damage_grade"] == 3).mean() * 100, line_dash="dash", line_color=INK)
            st.plotly_chart(chart_style(fig, 400), width="stretch")
        st.markdown(f'<div class="note">부가 용도가 있는 건물은 {d["has_secondary_use"].mean():.1%}입니다. '
                    '건물 수가 적은 용도는 비율이 크게 흔들리므로 해석에 주의가 필요합니다.</div>', unsafe_allow_html=True)

    with tabs[5]:
        cat_cols = ["foundation_type", "roof_type", "ground_floor_type", "other_floor_type", "position",
                    "plan_configuration", "land_surface_condition", "legal_ownership_status"]
        col = st.selectbox("범주형 변수", cat_cols)
        t = d.groupby(col)["damage_grade"].value_counts(normalize=True).unstack().mul(100)
        t = t.rename(columns={1: "1등급", 2: "2등급", 3: "3등급"})
        t["건물 수"] = d[col].value_counts()
        t = t.sort_values("3등급", ascending=False).reset_index()
        long = t.melt(id_vars=[col, "건물 수"], value_vars=GRADES, var_name="피해 등급", value_name="비율(%)")
        fig = px.bar(long, x=col, y="비율(%)", color="피해 등급", color_discrete_map=GCOLOR, hover_data=["건물 수"],
                     title=f"{col} 값별 피해 등급 구성")
        st.plotly_chart(chart_style(fig, 400), width="stretch")
        st.markdown('<div class="note">문자 코드(h, i, r 등)의 실제 의미는 익명화되어 있습니다. '
                    '서로 다른 변수의 같은 문자는 같은 뜻이 아닙니다.</div>', unsafe_allow_html=True)

    with tabs[6]:
        corr = d[num_cols + ["damage_grade"]].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=["#4F6D7A", "#F6F3EE", BRICK],
                        zmin=-1, zmax=1, title="수치형 변수 상관계수")
        st.plotly_chart(chart_style(fig, 480), width="stretch")
        st.markdown('<div class="note">층수와 높이의 상관이 가장 높고(층이 많을수록 높음), '
                    '피해 등급과 강하게 연결된 수치형 변수는 없습니다.</div>', unsafe_allow_html=True)

    with tabs[7]:
        table = load_dictionary()
        if table is None:
            st.info(f"`{DICT_PATH.name}` 파일을 app.py와 같은 폴더에 두면 컬럼 정의서가 표시됩니다.")
        else:
            keep = [c for c in ["컬럼명", "한글명", "역할", "저장형 / 의미형", "정의", "단위·값 의미",
                                "실제 관측값·범위", "고유값 수", "해석·전처리 주의점"] if c in table.columns]
            q = st.text_input("검색 (컬럼명 또는 한글명)")
            show = table[keep]
            if q:
                show = show[show["컬럼명"].astype(str).str.contains(q, case=False)
                            | show["한글명"].astype(str).str.contains(q, case=False)]
            st.dataframe(show, width="stretch", hide_index=True, height=520)


# ================================================================== 3. 모델링 과정
elif page == "모델링 과정":
    st.title("모델링 과정")
    tabs = st.tabs(["① 베이스라인", "② 파생변수 실험", "③ 하이퍼파라미터 튜닝", "④ Soft voting"])

    with tabs[0]:
        st.markdown('<div class="note">원본 38개 변수를 원핫 인코딩해 4개 모델 × 5가지 불균형 처리 방법을 비교했습니다 '
                    '(5-Fold 교차검증 + 검증 데이터). CatBoost + SMOTENC가 검증 Macro F1 0.6671로 가장 좋았습니다.</div>',
                    unsafe_allow_html=True)
        b = R.BASELINE.copy()
        fig = px.bar(b, x="모델", y="검증 Macro F1", color="불균형 처리", barmode="group",
                     color_discrete_sequence=["#2A2521", BRICK, "#D4A03A", "#7E9C8F", "#B8ADA1"])
        fig.update_yaxes(range=[0.58, 0.68])
        st.plotly_chart(chart_style(fig, 400), width="stretch")
        st.dataframe(b.style.format({c: "{:.4f}" for c in b.columns[2:]})
                     .highlight_max(subset=["검증 Macro F1"], color="#F2D7CF"),
                     width="stretch", hide_index=True)
        st.markdown('<div class="note">Class Weight·Under Sampling은 Recall을 크게 올리지만 Accuracy가 떨어져 '
                    'F1은 오히려 낮아졌습니다. SMOTE 계열이 F1 기준으로 가장 균형이 좋았습니다.</div>', unsafe_allow_html=True)

    with tabs[1]:
        st.markdown('<div class="note">LightGBM 설정을 고정하고 파생변수 묶음을 차례로 쌓았습니다. 지역 ID는 범주형으로 그대로 넣고, '
                    '학습 후 1·3등급 예측 확률에 곱할 <b>배수</b>를 검증 데이터에서 찾아 판정 기준을 조정했습니다.</div>',
                    unsafe_allow_html=True)
        e = R.EXPERIMENTS
        fig = go.Figure()
        fig.add_bar(x=e["실험"], y=e["배수 적용 전 검증 F1"], name="배수 적용 전", marker_color="#B8ADA1")
        fig.add_bar(x=e["실험"], y=e["배수 적용 후 검증 F1"], name="배수 적용 후", marker_color=BRICK,
                    text=e["배수 적용 후 검증 F1"], texttemplate="%{y:.4f}", textposition="outside")
        fig.update_yaxes(range=[0.70, 0.725], title="검증 Macro F1")
        st.plotly_chart(chart_style(fig, 360), width="stretch")
        st.dataframe(e, width="stretch", hide_index=True)
        st.markdown('<div class="note">세 실험의 배수 적용 후 F1 차이는 0.0006 이내로, 파생변수를 더한 효과는 사실상 없었습니다. '
                    '배수 조정은 세 실험 모두 F1을 약 0.01 올렸습니다. 실험 ③은 트리 386개로 가장 빨리 수렴해 다음 단계에 썼습니다.</div>',
                    unsafe_allow_html=True)
        st.markdown("#### 실험별로 더한 변수")
        cols = st.columns(3)
        for col, (name, items) in zip(cols, R.EXPERIMENT_FEATURES.items()):
            with col:
                st.markdown(f"**{name}**")
                st.dataframe(pd.DataFrame(items, columns=["변수", "설명"]), hide_index=True, width="stretch")

    with tabs[2]:
        st.markdown('<div class="note">실험 ③에 대해 하이퍼파라미터 조합 30개를 무작위로 뽑아 비교했습니다. '
                    '시도마다 등급 배수도 새로 찾았습니다(범위 1등급 0.8~2.4, 3등급 0.7~2.0).</div>', unsafe_allow_html=True)
        t = R.TUNING
        fig = px.bar(t, x="설정", y="배수 적용 후 F1", text="배수 적용 후 F1",
                     color=t["설정"].eq("시도 5").map({True: "선택", False: "기타"}),
                     color_discrete_map={"선택": BRICK, "기타": "#B8ADA1"})
        fig.update_traces(texttemplate="%{y:.4f}", textposition="outside")
        fig.update_yaxes(range=[0.716, 0.722])
        fig.update_layout(showlegend=False)
        st.plotly_chart(chart_style(fig, 340), width="stretch")
        st.dataframe(t, width="stretch", hide_index=True)
        st.markdown('<div class="note">시도 5(num_leaves 127, learning_rate 0.03, cat_smooth 60, reg_lambda 10)가 선택되어 '
                    '검증 F1이 0.7184 → 0.7207(+0.0023), 테스트 F1이 0.7137 → 0.7155로 올랐습니다. 상위 시도들이 0.002 안에 '
                    '모여 있어 차이는 작습니다. 상위 10개 시도가 모두 cat_smooth 60이었습니다.</div>', unsafe_allow_html=True)

    with tabs[3]:
        st.markdown('<div class="note">배수 대신 <b>class_weight 1.9 / 1.0 / 1.3</b>을 학습 단계에 주고, '
                    'LightGBM·XGBoost·HGB·ExtraTrees 4개 모델 중 2~3개를 골라 예측 확률을 평균했습니다.</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1.4])
        with c1:
            st.markdown("**모델별 단독 성능 (검증)**")
            st.dataframe(R.SINGLE.style.format({c: "{:.4f}" for c in R.SINGLE.columns[1:]}),
                         width="stretch", hide_index=True)
        with c2:
            metric = st.radio("지표", ["Macro F1", "Accuracy", "Precision", "Recall"], horizontal=True)
            cb = R.COMBOS.sort_values(metric)
            fig = px.bar(cb, x=metric, y="조합", orientation="h", text=metric,
                         color=cb["조합"].eq("HGB + LightGBM").map({True: "선택", False: "기타"}),
                         color_discrete_map={"선택": BRICK, "기타": "#B8ADA1"})
            fig.update_traces(texttemplate="%{x:.4f}", textposition="outside")
            fig.update_xaxes(range=[cb[metric].min() - 0.003, cb[metric].max() + 0.002])
            fig.update_layout(showlegend=False)
            st.plotly_chart(chart_style(fig, 420), width="stretch")
        st.dataframe(R.COMBOS.style.format({c: "{:.4f}" for c in ["Macro F1", "Accuracy", "Precision", "Recall"]}),
                     width="stretch", hide_index=True)
        st.markdown('<div class="note">HGB + LightGBM이 검증 Macro F1 0.7195로 1위였지만, 단독 LightGBM(0.7187)보다 0.0008 높을 뿐입니다. '
                    '네 모델이 같은 변수와 같은 가중치를 써서 예측이 비슷했기 때문에, 섞어도 서로 보완하는 효과가 작았습니다.</div>',
                    unsafe_allow_html=True)


# ================================================================== 4. 최종 모델 평가·해석
elif page == "최종 모델 평가·해석":
    st.title("최종 모델 평가·해석")
    st.markdown('<div class="note">최종 모델: HGB + LightGBM soft voting · class_weight 1.9 / 1.0 / 1.3 · 변수 93개 · '
                '학습+검증(85%)으로 재학습 후 테스트(15%) 1회 평가</div>', unsafe_allow_html=True)
    tabs = st.tabs(["테스트 성능", "학습곡선", "변수 중요도", "SHAP"])

    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Macro F1", "0.7158")
        c2.metric("Accuracy", "0.7473")
        c3.metric("Macro Precision", "0.7080")
        c4.metric("Macro Recall", "0.7259")
        left, right = st.columns([1, 1.2])
        with left:
            view = st.radio("혼동행렬", ["건수", "실제 등급별 비율 (= Recall)"], horizontal=True)
            cm = np.array(R.FINAL_CM)
            z = cm if view == "건수" else cm / cm.sum(axis=1, keepdims=True)
            fig = px.imshow(z, x=GRADES, y=GRADES, text_auto=",d" if view == "건수" else ".2f",
                            color_continuous_scale=["#F6F3EE", "#D9B8A6", BRICK],
                            labels=dict(x="예측 등급", y="실제 등급", color=""))
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(chart_style(fig, 400), width="stretch")
        with right:
            st.markdown("**이전 방식과 비교 (테스트)**")
            tc = R.TEST_COMPARE
            st.dataframe(tc.style.format({c: "{:.4f}" for c in tc.columns[1:]}), width="stretch", hide_index=True)
            long = tc.melt(id_vars="방식", value_vars=["1등급 Recall", "2등급 Recall", "3등급 Recall"],
                           var_name="등급", value_name="등급별 Recall")
            fig = px.bar(long, x="등급", y="등급별 Recall", color="방식", barmode="group",
                         color_discrete_sequence=["#B8ADA1", "#8C7B6B", BRICK])
            fig.update_yaxes(range=[0.6, 0.82])
            st.plotly_chart(chart_style(fig, 300), width="stretch")
        st.markdown('<div class="note">오분류는 대부분 인접 등급 사이에서 생깁니다. 2·3등급을 서로 헷갈린 경우가 7,045건으로 가장 많고, '
                    '1↔3등급 직접 오분류는 225건(0.6%)뿐입니다. Soft voting은 튜닝한 LightGBM + 배수(0.7155)와 F1이 사실상 같고, '
                    '1등급 Recall만 조금 더 높습니다.</div>', unsafe_allow_html=True)
        if METRIC_PATH.exists():
            m = json.loads(METRIC_PATH.read_text())
            st.caption(f"이 컴퓨터에서 train_model.py로 다시 학습한 모델의 테스트 Macro F1: {m['macro_f1']:.4f} "
                       f"(Accuracy {m['accuracy']:.4f})")

    with tabs[1]:
        st.markdown("#### ① 트리 수에 따른 손실")
        st.image(str(ASSETS / "learning_curve_iter.png"), width="stretch")
        st.markdown('<div class="note">두 모델 모두 트리 약 150개부터 검증 손실이 거의 줄지 않고(LightGBM 약 0.577, HGB 약 0.580), '
                    '학습 손실만 계속 내려갑니다. 검증 손실이 다시 오르지는 않아 심한 과적합은 아닙니다.</div>', unsafe_allow_html=True)
        st.markdown("#### ② 학습 데이터 양에 따른 성능")
        sc = R.SIZE_CURVE
        fig = go.Figure()
        fig.add_scatter(x=sc["학습 건수"], y=sc["학습 F1"], name="학습", mode="lines+markers", line_color="#8C7B6B")
        fig.add_scatter(x=sc["학습 건수"], y=sc["검증 F1"], name="검증", mode="lines+markers+text", line_color=BRICK,
                        text=[f"{v:.4f}" for v in sc["검증 F1"]], textposition="bottom center")
        fig.update_xaxes(title="학습 데이터 건수")
        fig.update_yaxes(title="Macro F1")
        st.plotly_chart(chart_style(fig, 380), width="stretch")
        st.markdown('<div class="note">데이터가 늘수록 검증 F1이 0.6949 → 0.7195로 오르고 학습–검증 차이는 0.26 → 0.05로 줄어듭니다. '
                    '75% → 100% 구간의 상승은 +0.0021로 작아, 지금 변수로는 0.72 근처에서 정체되는 모습입니다.</div>',
                    unsafe_allow_html=True)

    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            g = R.GAIN.sort_values()
            fig = px.bar(x=g.values, y=g.index, orientation="h", title="LightGBM gain 중요도 (%)",
                         color_discrete_sequence=["#8C7B6B"], labels={"x": "중요도 (%)", "y": ""})
            st.plotly_chart(chart_style(fig, 560), width="stretch")
        with c2:
            p = R.PERMUTATION.sort_values("F1 감소")
            fig = px.bar(p, x="F1 감소", y="변수", orientation="h", error_x="표준편차",
                         title="Soft voting permutation importance (Macro F1 감소)", color_discrete_sequence=[BRICK])
            fig.update_yaxes(title=None)
            st.plotly_chart(chart_style(fig, 560), width="stretch")
        st.markdown('<div class="note">두 방법 모두 지역×재료 그룹별 피해 비율(g3/g2_material_low/high_rate)이 1~4위입니다. '
                    'g3_material_low_rate 하나를 섞으면 Macro F1이 0.048 떨어집니다. permutation importance가 0 이하인 변수는 '
                    '93개 중 31개로, 변수를 줄일 때 우선 검토할 후보입니다.</div>', unsafe_allow_html=True)

    with tabs[3]:
        sm = R.SHAP_MEAN.copy()
        sm["합계"] = sm[GRADES].sum(axis=1)
        sm = sm.sort_values("합계")
        fig = go.Figure()
        for g in GRADES:
            fig.add_bar(y=sm["변수"], x=sm[g], name=g, orientation="h", marker_color=GCOLOR[g])
        fig.update_layout(barmode="stack", title="SHAP 전체 중요도 (LightGBM, 등급별 평균 |SHAP|, log-odds)")
        st.plotly_chart(chart_style(fig, 560), width="stretch")
        grade = st.radio("등급별 summary plot", GRADES, horizontal=True)
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.image(str(ASSETS / f"shap_grade{GRADES.index(grade) + 1}.png"), width="stretch")
        with c2:
            notes = {
                "1등급": "같은 지역·재료 건물의 1등급 비율(g2/g3_material_low_rate)이 높을수록 1등급 쪽으로 강하게 밉니다. "
                         "지역 평균보다 오래된 건물(g3_age_gap이 큼)은 1등급 가능성을 낮추고, count_families가 0이거나 "
                         "신축(age가 낮음)이면 1등급 쪽으로 기여합니다.",
                "2등급": "2등급은 특정 변수 하나가 크게 좌우하지 않습니다. g3_material_high_rate와 "
                         "g3_material_high_gap의 기여가 상대적으로 큽니다.",
                "3등급": "같은 지역·재료 건물의 3등급 비율(g3_material_high_rate)의 기여가 0.38로 가장 크고, "
                         "g2_material_high_rate, geo_level_3_id_high_rate가 뒤따릅니다.",
            }
            st.markdown(f'<div class="note">{notes[grade]}<br><br>점 하나가 건물 하나입니다. 오른쪽일수록 그 등급일 가능성을 높이고, '
                        '색은 변수 값(빨강=큼, 파랑=작음)입니다. 범주형 변수는 숫자 코드로 색을 칠해서 색의 크고 작음에 의미가 없습니다.</div>',
                        unsafe_allow_html=True)


# ================================================================== 5. 피해 등급 예측
elif page == "피해 등급 예측":
    st.title("피해 등급 예측")
    st.markdown('<div class="note">건물 정보를 넣으면 최종 모델(HGB + LightGBM soft voting)이 피해 등급별 확률을 계산합니다. '
                '테스트 데이터의 실제 건물을 불러와 정답과 비교해 볼 수도 있습니다.</div>', unsafe_allow_html=True)

    if not MODEL_PATH.exists():
        st.warning("학습된 모델이 없습니다. 아래 버튼으로 최종 모델을 학습하세요 (2~5분). "
                   "터미널에서 `python train_model.py`로 학습해도 됩니다.")
        if st.button("최종 모델 학습하기", type="primary"):
            from train_model import main as train_main
            with st.status("학습 중…", expanded=True) as status:
                train_main(log=st.write)
                status.update(label="학습 완료", state="complete")
            st.rerun()
        st.stop()

    bundle = load_model()
    lgbm, hgb, builder = bundle["lgbm"], bundle["hgb"], bundle["builder"]
    X_test, y_test = load_test_split()
    raw_cols = list(X_test.columns)

    if "building" not in st.session_state:
        st.session_state.building = X_test.iloc[0].to_dict()
        st.session_state.true_grade = int(y_test.iloc[0]) + 1

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("🎲 테스트 건물 무작위로 불러오기", width="stretch"):
            i = np.random.randint(len(X_test))
            st.session_state.building = X_test.iloc[i].to_dict()
            st.session_state.true_grade = int(y_test.iloc[i]) + 1
            st.session_state.form_version = st.session_state.get("form_version", 0) + 1
    with c2:
        grade_pick = st.selectbox("등급 골라 불러오기", ["—", "1등급", "2등급", "3등급"], label_visibility="collapsed")
        if grade_pick != "—" and st.button(f"{grade_pick} 건물 불러오기", width="stretch"):
            idx = np.flatnonzero(y_test.to_numpy() == GRADES.index(grade_pick))
            i = int(np.random.choice(idx))
            st.session_state.building = X_test.iloc[i].to_dict()
            st.session_state.true_grade = GRADES.index(grade_pick) + 1
            st.session_state.form_version = st.session_state.get("form_version", 0) + 1

    b = st.session_state.building
    v = st.session_state.get("form_version", 0)

    def options(col):
        return sorted(df[col].unique().tolist())

    with st.form("building_form"):
        st.markdown("##### 위치")
        g1, g2, g3 = st.columns(3)
        geo1 = g1.number_input("geo_level_1_id (대지역)", 0, 30, int(b["geo_level_1_id"]), key=f"g1{v}")
        geo2 = g2.number_input("geo_level_2_id (중지역)", 0, 1427, int(b["geo_level_2_id"]), key=f"g2{v}")
        geo3 = g3.number_input("geo_level_3_id (소지역)", 0, 12567, int(b["geo_level_3_id"]), key=f"g3{v}")

        st.markdown("##### 건물 규모")
        s1, s2, s3, s4, s5 = st.columns(5)
        floors = s1.number_input("층수", 1, 9, int(b["count_floors_pre_eq"]), key=f"fl{v}")
        age = s2.number_input("나이 (년)", 0, 995, int(b["age"]), key=f"ag{v}")
        area = s3.number_input("정규화 면적", 1, 100, int(b["area_percentage"]), key=f"ar{v}")
        height = s4.number_input("정규화 높이", 1, 32, int(b["height_percentage"]), key=f"he{v}")
        families = s5.number_input("거주 가구 수", 0, 9, int(b["count_families"]), key=f"fa{v}")

        st.markdown("##### 구조 (문자 코드는 익명화된 값)")
        cat_fields = ["foundation_type", "roof_type", "ground_floor_type", "other_floor_type",
                      "position", "plan_configuration", "land_surface_condition", "legal_ownership_status"]
        cat_values = {}
        cc = st.columns(4)
        for k, col in enumerate(cat_fields):
            opts = options(col)
            cat_values[col] = cc[k % 4].selectbox(col, opts, index=opts.index(b[col]), key=f"{col}{v}")

        st.markdown("##### 상부 구조 재료 (여러 개 선택 가능)")
        mat_values = {}
        mc = st.columns(4)
        for k, col in enumerate(material_cols):
            mat_values[col] = int(mc[k % 4].checkbox(col.replace("has_superstructure_", ""), bool(b[col]), key=f"{col}{v}"))

        st.markdown("##### 부가 용도")
        sec_values = {}
        sc = st.columns(4)
        for k, col in enumerate(secondary_cols):
            sec_values[col] = int(sc[k % 4].checkbox(col.replace("has_secondary_use_", ""), bool(b[col]), key=f"{col}{v}"))

        submitted = st.form_submit_button("예측하기", type="primary", width="stretch")

    row = dict(b)
    row.update({"geo_level_1_id": geo1, "geo_level_2_id": geo2, "geo_level_3_id": geo3,
                "count_floors_pre_eq": floors, "age": age, "area_percentage": area,
                "height_percentage": height, "count_families": families, **cat_values, **mat_values, **sec_values})
    row["has_secondary_use"] = int(any(sec_values.values()))
    X_one = pd.DataFrame([row])[raw_cols]
    for col in raw_cols:
        X_one[col] = X_one[col].astype(X_test[col].dtype)

    unchanged = all(row[c] == b[c] for c in raw_cols if c != "has_secondary_use")
    feat = builder.transform(X_one)
    from features import to_codes
    p_lgbm = lgbm.predict_proba(feat)[0]
    p_hgb = hgb.predict_proba(to_codes(feat, missing_as_nan=True))[0]
    proba = (p_lgbm + p_hgb) / 2
    pred = int(proba.argmax())

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    r1, r2 = st.columns([1, 1.6])
    with r1:
        st.markdown(f'<span class="grade-chip" style="background:{GCOLOR[GRADES[pred]]}">예측</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="hero-title" style="font-size:3rem">{GRADES[pred]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="note">확률 {proba[pred]:.1%} · {["경미한 피해", "중간 피해", "거의 완파"][pred]}</div>',
                    unsafe_allow_html=True)
        if unchanged and st.session_state.get("true_grade"):
            tg = st.session_state.true_grade
            ok = "맞혔습니다 ✅" if tg == pred + 1 else "틀렸습니다 ❌"
            st.markdown(f'<div class="note" style="margin-top:0.8rem">테스트 데이터의 실제 등급: <b>{tg}등급</b> — {ok}</div>',
                        unsafe_allow_html=True)
        elif not unchanged:
            st.markdown('<div class="note" style="margin-top:0.8rem">값을 바꿔서 실제 등급과는 비교하지 않습니다.</div>',
                        unsafe_allow_html=True)
        if pd.isna(feat["geo_level_3_id"].iloc[0]):
            st.warning("학습 데이터에 없는 소지역(geo_level_3_id)입니다. 지역 정보 없이 평균값으로 예측합니다.")
    with r2:
        pdf = pd.DataFrame({"등급": GRADES, "Soft voting": proba, "LightGBM": p_lgbm, "HGB": p_hgb})
        fig = go.Figure()
        fig.add_bar(x=pdf["Soft voting"], y=pdf["등급"], orientation="h", marker_color=[GCOLOR[g] for g in GRADES],
                    text=[f"{x:.1%}" for x in pdf["Soft voting"]], textposition="outside", name="Soft voting")
        fig.update_layout(title="등급별 확률 (LightGBM과 HGB의 평균)", showlegend=False)
        fig.update_xaxes(range=[0, 1.1], tickformat=".0%")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(chart_style(fig, 240), width="stretch")
        st.caption(f"LightGBM: {' / '.join(f'{x:.1%}' for x in p_lgbm)}   ·   HGB: {' / '.join(f'{x:.1%}' for x in p_hgb)}")

    st.markdown(f"#### 이 건물이 {GRADES[pred]}로 예측된 이유 (LightGBM SHAP)")
    contrib = np.asarray(lgbm.predict(feat, pred_contrib=True)).reshape(3, feat.shape[1] + 1)[pred, :-1]
    shap_df = pd.DataFrame({"변수": feat.columns, "기여": contrib,
                            "값": [str(feat[c].iloc[0]) if not isinstance(feat[c].iloc[0], float)
                                  else f"{feat[c].iloc[0]:.3f}" for c in feat.columns]})
    shap_df = shap_df.reindex(shap_df["기여"].abs().sort_values(ascending=False).index).head(12).iloc[::-1]
    shap_df["라벨"] = shap_df["변수"] + " = " + shap_df["값"]
    fig = go.Figure(go.Bar(x=shap_df["기여"], y=shap_df["라벨"], orientation="h",
                           marker_color=[BRICK if x > 0 else "#4F6D7A" for x in shap_df["기여"]]))
    fig.update_xaxes(title=f"{GRADES[pred]} 쪽으로 미는 정도 (log-odds)")
    st.plotly_chart(chart_style(fig, 440), width="stretch")
    st.markdown(f'<div class="note">빨간 막대는 {GRADES[pred]} 가능성을 높인 변수, 파란 막대는 낮춘 변수입니다. '
                'Soft voting은 SHAP 값을 합칠 수 없어 LightGBM 하나로 설명합니다.</div>', unsafe_allow_html=True)
