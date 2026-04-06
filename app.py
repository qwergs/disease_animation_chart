"""
EHEC Crude Incidence Rate — Animated Choropleth Map (South Korea, 2001–2024)

Interactive Streamlit app visualizing annual crude incidence rates of
Enterohemorrhagic E. coli (EHEC) infection at the sigungu (시군구) level.

Data: Korea Disease Control and Prevention Agency (KDCA)
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
from pathlib import Path
import time

# ───────────────────────────────────────────
# Page config
# ───────────────────────────────────────────
st.set_page_config(
    page_title="EHEC Incidence Map — South Korea",
    page_icon="🗺️",
    layout="wide",
)

# ───────────────────────────────────────────
# 한글 폰트 설정 (Streamlit Cloud에서는 NanumGothic 사용)
# ───────────────────────────────────────────
import matplotlib.font_manager as fm
import os

# Streamlit Cloud에 NanumGothic이 있는지 확인, 없으면 설치 시도
FONT_FOUND = False
for font in fm.findSystemFonts():
    if "NanumGothic" in font or "AppleGothic" in font or "Malgun" in font:
        font_name = fm.FontProperties(fname=font).get_name()
        plt.rcParams["font.family"] = font_name
        FONT_FOUND = True
        break

if not FONT_FOUND:
    # Streamlit Cloud (Ubuntu): apt로 설치된 나눔 폰트 사용
    nanum_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    if os.path.exists(nanum_path):
        fm.fontManager.addfont(nanum_path)
        plt.rcParams["font.family"] = fm.FontProperties(fname=nanum_path).get_name()
    else:
        plt.rcParams["font.family"] = "sans-serif"

plt.rcParams["axes.unicode_minus"] = False


# ───────────────────────────────────────────
# 색상 / 분류 유틸 (노트북 원본 로직)
# ───────────────────────────────────────────
def make_white_to_red_cmap(n_classes=6):
    reds = plt.get_cmap("Reds", n_classes)
    colors = [(1, 1, 1, 1)] + [reds(i) for i in range(1, n_classes)]
    return mcolors.ListedColormap(colors)


def compute_bins(values, zero_eps=1e-9):
    """전체 연도 crude_rate 기준 quintile bins 계산"""
    pos = values[np.isfinite(values) & (values > zero_eps)]
    if len(pos) == 0:
        return np.array([0, zero_eps, zero_eps*2, zero_eps*3,
                         zero_eps*4, zero_eps*5, zero_eps*6])
    qs = np.quantile(pos, [0.2, 0.4, 0.6, 0.8, 1.0])
    return np.concatenate(([0.0, zero_eps], qs))


def fmt_tick(x):
    if x == 0:
        return "0"
    if x < 1e-6:
        return f"{x:.1e}"
    return f"{x:.4f}".rstrip("0").rstrip(".")


# ───────────────────────────────────────────
# 데이터 로딩 (캐시)
# ───────────────────────────────────────────
@st.cache_data
def load_data():
    data_dir = Path(__file__).parent / "data"
    df = pd.read_csv(data_dir / "ehec_yearly_sigungu.csv")
    df["year"] = df["year"].astype(int)
    return df


@st.cache_resource
def load_geojson():
    data_dir = Path(__file__).parent / "data"
    gdf = gpd.read_file(data_dir / "korea_sigungu.geojson")
    return gdf


# ───────────────────────────────────────────
# 단일 연도 지도 렌더링
# ───────────────────────────────────────────
def render_map(gdf, df_year, year, edges, cmap, norm, value_col="crude_rate"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    # 지도 데이터 병합
    d = df_year[["region", value_col]].copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    merged = gdf.merge(d, on="region", how="left")

    # 지도 그리기
    merged.plot(
        ax=ax,
        column=value_col,
        cmap=cmap,
        norm=norm,
        linewidth=0.3,
        edgecolor="black",
        missing_kwds=dict(color="lightgrey"),
    )

    ax.set_title(
        f"EHEC 조발생률 (Crude Rate per 100,000) — {year}년",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    ax.axis("off")

    # 컬러바
    cax = fig.add_axes([0.88, 0.25, 0.025, 0.5])
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, boundaries=edges)
    cb.set_ticks(edges)
    cb.set_ticklabels([fmt_tick(t) for t in edges], fontsize=9)
    cb.set_label("Crude Rate (per 100,000)", fontsize=11)

    plt.tight_layout(rect=[0, 0, 0.87, 1])
    return fig


# ───────────────────────────────────────────
# 메인 앱
# ───────────────────────────────────────────
def main():
    st.title("🦠 EHEC Infection — Animated Choropleth Map")
    st.markdown(
        "**Enterohemorrhagic *E. coli* (EHEC)** 감염증의 연간 조발생률을 "
        "시군구 단위로 시각화합니다. (2001–2024, 대한민국)"
    )

    # 데이터 로딩
    df = load_data()
    gdf = load_geojson()

    years = sorted(df["year"].unique())
    year_min, year_max = years[0], years[-1]

    # 전체 연도 기준 색상 bins 계산 (고정 스케일)
    all_values = df["crude_rate"].values
    edges = compute_bins(all_values)
    cmap = make_white_to_red_cmap(n_classes=6)
    norm = mcolors.BoundaryNorm(edges, ncolors=cmap.N, clip=True)

    # ── 사이드바 ──
    st.sidebar.header("⚙️ Controls")

    mode = st.sidebar.radio(
        "모드 선택",
        ["슬라이더 (수동)", "자동 재생 애니메이션"],
        index=0,
    )

    if mode == "자동 재생 애니메이션":
        speed = st.sidebar.slider("속도 (초/년)", 0.3, 3.0, 1.0, step=0.1)

        if st.sidebar.button("▶ 재생 시작"):
            chart_area = st.empty()
            info_area = st.empty()
            progress = st.progress(0)

            for idx, yr in enumerate(years):
                df_year = df[df["year"] == yr]
                fig = render_map(gdf, df_year, yr, edges, cmap, norm)
                chart_area.pyplot(fig)
                plt.close(fig)

                total_cases = int(df_year["yearly_cases"].sum())
                max_row = df_year.loc[df_year["crude_rate"].idxmax()]
                region_name = max_row["region"]
                max_rate = max_row["crude_rate"]
                info_area.markdown(
                    f"**{yr}년** — 총 {total_cases:,}건 | "
                    f"최고: {region_name} ({max_rate:.4f}/100k)"
                )
                progress.progress((idx + 1) / len(years))
                time.sleep(speed)

            st.sidebar.success("✅ 재생 완료!")
        else:
            st.info("왼쪽 사이드바에서 '▶ 재생 시작' 버튼을 눌러주세요.")

    else:
        # 슬라이더 모드
        selected_year = st.sidebar.slider(
            "연도 선택", year_min, year_max, year_max, step=1
        )

        df_year = df[df["year"] == selected_year]

        # 지도
        fig = render_map(gdf, df_year, selected_year, edges, cmap, norm)
        st.pyplot(fig)
        plt.close(fig)

        # 요약 통계
        col1, col2, col3, col4 = st.columns(4)
        total_cases = int(df_year["yearly_cases"].sum())
        total_pop = df_year["pop"].sum()
        national_rate = (total_cases / total_pop * 100000) if total_pop > 0 else 0
        max_row = df_year.loc[df_year["crude_rate"].idxmax()]

        col1.metric("총 발생건수", f"{total_cases:,}")
        col2.metric("전국 조발생률", f"{national_rate:.4f}")
        col3.metric("최고 지역", max_row["region"])
        col4.metric("최고 발생률", f"{max_row['crude_rate']:.4f}")

        # 상위 10 테이블
        st.subheader(f"📊 {selected_year}년 — 조발생률 상위 10개 시군구")
        top10 = (
            df_year.nlargest(10, "crude_rate")[["region", "yearly_cases", "pop", "crude_rate"]]
            .rename(columns={
                "region": "시군구",
                "yearly_cases": "발생건수",
                "pop": "인구",
                "crude_rate": "조발생률",
            })
        )
        st.dataframe(
            top10.style.format({"인구": "{:,.0f}", "조발생률": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

    # 연도별 전국 추이 차트
    st.markdown("---")
    st.subheader("📈 전국 연간 추이")

    national = df.groupby("year").agg(
        cases=("yearly_cases", "sum"),
        pop=("pop", "sum"),
    ).reset_index()
    national["crude_rate"] = national["cases"] / national["pop"] * 100000

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.bar(national["year"], national["cases"], color="#E74C3C", alpha=0.7, label="발생건수")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Cases", color="#E74C3C")
    ax2.tick_params(axis="y", labelcolor="#E74C3C")

    ax2b = ax2.twinx()
    ax2b.plot(national["year"], national["crude_rate"], "o-", color="#2C3E50", linewidth=2, label="조발생률")
    ax2b.set_ylabel("Crude Rate (per 100,000)", color="#2C3E50")
    ax2b.tick_params(axis="y", labelcolor="#2C3E50")

    ax2.set_title("EHEC 전국 연간 발생 추이 (2001–2024)")
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # Footer
    st.markdown("---")
    st.caption(
        "Data: Korea Disease Control and Prevention Agency (KDCA) | "
        "Crude Rate = (Cases / Population) × 100,000"
    )


if __name__ == "__main__":
    main()
