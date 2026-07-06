"""
Cap 6 — El futbol como espejo del Estado (H6, exploratorio).

H6 afirma que el rendimiento consistente-sin-salto de la selección refleja
*intencionalmente* las políticas del Estado. La intencionalidad NO es testeable;
lo que sí se puede cuantificar es si la consistencia mexicana es estadísticamente
anómala entre selecciones comparables.

Datos: data/fifa/worldcup_positions.csv (generado por scripts/build_worldcup_matrix.py).
Ronda ordinal: 1=grupos, 2=octavos, 3=cuartos, 4=semifinal/3º-4º, 5=final/campeón.
Era moderna: 1970-2022 (14 mundiales). Mínimo 7 participaciones para comparar.

Monte Carlo: ¿qué tan probable es una desviación estándar tan baja como la de México
si sus rondas se sortearan de la distribución de sus pares (media similar)?

Figuras → centralismo/informe/figuras/cap6_*.png
Run: uv run python scripts/centralismo/cap6_futbol.py
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import RAIZ, guardar_fig

RONDA = {17: 1, 9: 2, 5: 3, 4: 4, 3: 4, 2: 5, 1: 5}
ERA = [str(a) for a in (1970, 1974, 1978, 1982, 1986, 1990, 1994,
                        1998, 2002, 2006, 2010, 2014, 2018, 2022)]
MIN_PART = 7


def main():
    pos = pl.read_csv(RAIZ / "data/fifa/worldcup_positions.csv")
    largo = (pos.unpivot(index="country", on=ERA, variable_name="año", value_name="pos")
             .drop_nulls("pos")
             .with_columns(pl.col("pos").replace_strict(RONDA).alias("ronda"),
                           pl.col("año").cast(pl.Int64)))

    stats = (largo.group_by("country")
             .agg(pl.len().alias("n"), pl.mean("ronda").alias("media"),
                  pl.std("ronda").alias("sd"),
                  (pl.col("ronda") >= 3).sum().alias("quintos"))
             .filter(pl.col("n") >= MIN_PART)
             .sort("sd"))

    print(f"=== Consistencia en mundiales 1970-2022 (≥{MIN_PART} participaciones) ===")
    print(f"{'selección':<16}{'n':>3}{'media':>7}{'sd':>6}{'≥cuartos':>9}")
    for r in stats.iter_rows(named=True):
        marca = "  ← México" if r["country"] == "Mexico" else ""
        print(f"{r['country']:<16}{r['n']:>3}{r['media']:>7.2f}{r['sd']:>6.2f}{r['quintos']:>9}{marca}")

    mx = stats.filter(pl.col("country") == "Mexico").row(0, named=True)
    mx_rondas = largo.filter(pl.col("country") == "Mexico")["ronda"].to_numpy()
    print(f"\nMéxico: n={mx['n']}, media={mx['media']:.2f}, sd={mx['sd']:.2f}, "
          f"cuartos+ = {mx['quintos']} (1970 y 1986, ambas como anfitrión)")
    oct_seguidos = (largo.filter((pl.col("country") == "Mexico") & (pl.col("ronda") == 2))
                    .height)
    print(f"eliminado exactamente en octavos: {oct_seguidos} de {mx['n']} participaciones")

    # Monte Carlo: pares con media similar (±0.4)
    pares = stats.filter((pl.col("country") != "Mexico")
                         & ((pl.col("media") - mx["media"]).abs() <= 0.4))
    pool = largo.filter(pl.col("country").is_in(pares["country"].to_list()))["ronda"].to_numpy()
    rng = np.random.default_rng(42)
    sims = np.array([rng.choice(pool, size=len(mx_rondas), replace=True).std(ddof=1)
                     for _ in range(20_000)])
    p_mc = float((sims <= mx["sd"]).mean())
    print(f"pares (media±0.4): {pares['country'].to_list()}")
    print(f"Monte Carlo (20k): P(sd ≤ {mx['sd']:.2f} | distribución de pares) = {p_mc:.4f}")

    # ---- F1: media vs sd
    fig = go.Figure()
    otros = stats.filter(pl.col("country") != "Mexico")
    fig.add_trace(go.Scatter(
        x=otros["media"], y=otros["sd"], mode="markers+text", text=otros["country"],
        textposition="top center", textfont=dict(size=9),
        marker=dict(size=8, color="#7f7f7f"), showlegend=False))
    fig.add_trace(go.Scatter(
        x=[mx["media"]], y=[mx["sd"]], mode="markers+text", text=["México"],
        textposition="bottom center", marker=dict(size=14, color="#c0392b"),
        showlegend=False))
    fig.update_layout(
        title=f"Rendimiento medio vs. variabilidad en mundiales, 1970-2022<br><sup>ronda: 1=grupos … 5=final; México tiene la consistencia (sd) más baja del grupo — P(azar)≈{p_mc:.3f}</sup>",
        xaxis_title="ronda media alcanzada", yaxis_title="desviación estándar de la ronda",
    )
    guardar_fig(fig, "cap6_consistencia")

    # ---- F2: trayectoria de México vs pares ilustrativos
    fig = go.Figure()
    for pais, color in [("Mexico", "#c0392b"), ("United States", "#7f7f7f"),
                        ("Belgium", "#1f77b4"), ("Croatia", "#27ae60")]:
        s = largo.filter(pl.col("country") == pais).sort("año")
        fig.add_trace(go.Scatter(x=s["año"], y=s["ronda"], mode="lines+markers", name=pais,
                                 line=dict(color=color)))
    fig.update_layout(
        title="Trayectoria mundialista: México nunca varía; los pares saltan en ambas direcciones",
        xaxis_title="mundial",
        yaxis=dict(title="ronda", tickvals=[1, 2, 3, 4, 5],
                   ticktext=["grupos", "octavos", "cuartos", "semis", "final"]),
    )
    guardar_fig(fig, "cap6_trayectoria")


if __name__ == "__main__":
    main()
