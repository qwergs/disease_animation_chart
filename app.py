"""
EHEC Crude Incidence Rate — Animated Choropleth Map (South Korea, 2001–2024)

Interactive Streamlit app visualizing annual crude incidence rates of
Enterohemorrhagic E. coli (EHEC) infection at the sigungu level.

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
# Font setup
# ───────────────────────────────────────────
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


# ───────────────────────────────────────────
# Color / classification utils
# ───────────────────────────────────────────
def make_white_to_red_cmap(n_classes=6):
    reds = plt.get_cmap("Reds", n_classes)
    colors = [(1, 1, 1, 1)] + [reds(i) for i in range(1, n_classes)]
    return mcolors.ListedColormap(colors)


def compute_bins(values, zero_eps=1e-9):
    """Compute quintile bins based on all years' crude_rate"""
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
# Data loading (cached)
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
# Map rendering (single year)
# ───────────────────────────────────────────
def render_map(gdf, df_year, year, edges, cmap, norm, value_col="crude_rate"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    d = df_year[["region", value_col]].copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    merged = gdf.merge(d, on="region", how="left")

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
        f"EHEC Crude Incidence Rate (per 100,000) — {year}",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    ax.axis("off")

    # Colorbar
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
# Main app
# ───────────────────────────────────────────
def main():
    st.title("🦠 EHEC Infection — Animated Choropleth Map")
    st.markdown(
        "Interactive visualization of annual crude incidence rates of "
        "**Enterohemorrhagic *E. coli* (EHEC)** infection "
        "at the district level in South Korea (2001–2024)."
    )

    # Load data
    df = load_data()
    gdf = load_geojson()

    years = sorted(df["year"].unique())
    year_min, year_max = years[0], years[-1]

    # Compute fixed color bins across all years
    all_values = df["crude_rate"].values
    edges = compute_bins(all_values)
    cmap = make_white_to_red_cmap(n_classes=6)
    norm = mcolors.BoundaryNorm(edges, ncolors=cmap.N, clip=True)

    # ── Sidebar ──
    st.sidebar.header("⚙️ Controls")

    mode = st.sidebar.radio(
        "Select Mode",
        ["Slider (Manual)", "Auto-play Animation"],
        index=0,
    )

    if mode == "Auto-play Animation":
        speed = st.sidebar.slider("Speed (sec/year)", 0.3, 3.0, 1.0, step=0.1)

        if st.sidebar.button("▶ Start Playback"):
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
                    f"**{yr}** — Total: {total_cases:,} cases | "
                    f"Highest: {region_name} ({max_rate:.4f}/100k)"
                )
                progress.progress((idx + 1) / len(years))
                time.sleep(speed)

            st.sidebar.success("✅ Playback complete!")
        else:
            st.info("Click '▶ Start Playback' in the sidebar to begin the animation.")

    else:
        # Slider mode
        selected_year = st.sidebar.slider(
            "Select Year", year_min, year_max, year_max, step=1
        )

        df_year = df[df["year"] == selected_year]

        # Map
        fig = render_map(gdf, df_year, selected_year, edges, cmap, norm)
        st.pyplot(fig)
        plt.close(fig)

        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        total_cases = int(df_year["yearly_cases"].sum())
        total_pop = df_year["pop"].sum()
        national_rate = (total_cases / total_pop * 100000) if total_pop > 0 else 0
        max_row = df_year.loc[df_year["crude_rate"].idxmax()]

        col1.metric("Total Cases", f"{total_cases:,}")
        col2.metric("National Crude Rate", f"{national_rate:.4f}")
        col3.metric("Highest Region", max_row["region"])
        col4.metric("Highest Rate", f"{max_row['crude_rate']:.4f}")

        # Top 10 table
        st.subheader(f"📊 {selected_year} — Top 10 Districts by Crude Rate")
        top10 = (
            df_year.nlargest(10, "crude_rate")[["region", "yearly_cases", "pop", "crude_rate"]]
            .rename(columns={
                "region": "District",
                "yearly_cases": "Cases",
                "pop": "Population",
                "crude_rate": "Crude Rate",
            })
        )
        st.dataframe(
            top10.style.format({"Population": "{:,.0f}", "Crude Rate": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

    # National trend chart
    st.markdown("---")
    st.subheader("📈 National Annual Trend")

    national = df.groupby("year").agg(
        cases=("yearly_cases", "sum"),
        pop=("pop", "sum"),
    ).reset_index()
    national["crude_rate"] = national["cases"] / national["pop"] * 100000

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.bar(national["year"], national["cases"], color="#E74C3C", alpha=0.7, label="Cases")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Cases", color="#E74C3C")
    ax2.tick_params(axis="y", labelcolor="#E74C3C")

    ax2b = ax2.twinx()
    ax2b.plot(national["year"], national["crude_rate"], "o-", color="#2C3E50", linewidth=2, label="Crude Rate")
    ax2b.set_ylabel("Crude Rate (per 100,000)", color="#2C3E50")
    ax2b.tick_params(axis="y", labelcolor="#2C3E50")

    ax2.set_title("EHEC National Annual Incidence Trend (2001–2024)")
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
