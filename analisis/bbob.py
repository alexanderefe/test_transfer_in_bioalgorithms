"""
Análisis de la parrilla BBOB (opción A de README_BBOB.md): 12 algoritmos ×
24 funciones × D ∈ {2, 3, 5, 10, 20, 40} × 15 instancias, presupuesto 10⁴·D.

Misma metodología de rankings que CEC 2022 (setup §6–§8), POR DIMENSIÓN
(no se promedia entre dimensiones, igual que en CEC no se promedia entre
configuraciones):

  - error final = f − f(x*) con el presupuesto completo; < 1e-8 cuenta como 0;
  - error medio de las 15 instancias por (algoritmo × función);
  - rango por función (empate si |Δ| < 1e-8) y rango promedio sobre las 24
    funciones, y también por cada uno de los 5 grupos de funciones de COCO;
  - Friedman (k = 12, N = 24) y Diferencia Crítica de Nemenyi (CD ≈ 3.40);
  - Wilcoxon de rangos con signo, propuesto contra cada algoritmo (y PSO
    contra DE), por función, emparejado por instancia (15 pares);
  - tasa de éxito: instancias que llegan a error < 1e-8;
  - diagnóstico del middleware (activaciones, Canal A/B, abortos W₁).

Figuras: CD, mapa de rangos, rangos y éxito contra D, boxplots y
convergencia (mediana de las 15 instancias). Con `--ppdata`, copia además
las ECDF agregadas de `cocopp` (análisis estándar de COCO).

    python analisis/bbob.py --resultados resultados_bbob
    python analisis/bbob.py --resultados resultados_bbob --ppdata resultados_bbob/ppdata

Salida: `<resultados>/analisis/{tablas,figuras}/` + `LEEME.md`.
No necesita `cocoex` (corre en Colab sin instalar COCO).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import friedmanchisquare, wilcoxon

from analisis.boxplots import COLOR_GRUPO, _cajas, _leyenda_grupos
from analisis.cd_nemenyi import diferencia_critica, dibujar_cd
from analisis.datos import (
    ALGORITMOS_PRINCIPALES, GRUPO_ALGORITMO, NOMBRE_ALGORITMO, REFERENCIA_SELECCIONADOS,
)
from analisis.rankings import rankear_columna
from problems.bbob_meta import (
    DIMENSIONES_BBOB, FUNCIONES_BBOB, GRUPO_BBOB, GRUPOS_BBOB, INSTANCIAS_BBOB, NOMBRES_BBOB,
)
from problems.cec2022_wrapper import FRACCIONES_CHECKPOINT, UMBRAL_ERROR_CERO

ALPHA = 0.05
ALGORITMOS_BBOB = ALGORITMOS_PRINCIPALES + REFERENCIA_SELECCIONADOS
PARES_WILCOXON = tuple(("middleware", a) for a in ALGORITMOS_BBOB if a != "middleware") \
    + (("pso", "de"),)
TINTA = "#0b0b0b"


def nombre(a: str) -> str:
    return NOMBRE_ALGORITMO.get(a, a)


# ── datos ───────────────────────────────────────────────────────────────────

def cargar_bbob(ruta: str | Path, permitir_incompleto: bool = False) -> pd.DataFrame:
    """Carga `corridas.parquet` de BBOB y valida que la parrilla esté completa,
    toda en estado ok y con un solo presupuesto. Agrega `instancia` (=
    idx_semilla + 1) y `error_u` (error con < 1e-8 colapsado a 0)."""
    df = pd.read_parquet(ruta)
    df = df[df["benchmark"] == "BBOB"].copy()
    if df.empty:
        raise ValueError(f"{ruta}: no hay corridas BBOB.")
    problemas = []
    if (df["estado"] != "ok").any():
        problemas.append(f"{(df['estado'] != 'ok').sum()} corridas no 'ok'")
        df = df[df["estado"] == "ok"]
    presupuestos = sorted((df["max_fes"] // df["dim"]).unique())
    if len(presupuestos) > 1:
        raise ValueError(f"Mezcla de presupuestos (·D): {presupuestos}. "
                         f"Usar una carpeta de resultados por presupuesto.")
    algoritmos = [a for a in ALGORITMOS_BBOB if a in set(df["algoritmo"])]
    esperado = len(algoritmos) * len(FUNCIONES_BBOB) * len(DIMENSIONES_BBOB) * len(INSTANCIAS_BBOB)
    if len(df) != esperado:
        problemas.append(f"{len(df)} corridas, se esperaban {esperado}")
    if problemas and not permitir_incompleto:
        raise ValueError("; ".join(problemas) + " (usar --permitir-incompleto para analizar igual)")
    for p in problemas:
        print("Aviso:", p)
    df["instancia"] = df["idx_semilla"] + 1
    df["error_u"] = np.where(df["error"] < UMBRAL_ERROR_CERO, 0.0, df["error"])
    df.attrs["presupuesto"] = int(presupuestos[0])
    df.attrs["algoritmos"] = algoritmos
    return df


def algoritmos_de(df: pd.DataFrame) -> list[str]:
    return df.attrs.get("algoritmos") or [a for a in ALGORITMOS_BBOB if a in set(df["algoritmo"])]


def dims_de(df: pd.DataFrame) -> list[int]:
    return [d for d in DIMENSIONES_BBOB if d in set(df["dim"])]


# ── tablas ──────────────────────────────────────────────────────────────────

def error_medio(df: pd.DataFrame, dim: int) -> pd.DataFrame:
    """Algoritmo × función: error medio de las instancias (con umbral 1e-8)."""
    sub = df[df["dim"] == dim]
    t = sub.groupby(["algoritmo", "funcion"])["error_u"].mean().unstack("funcion")
    return t.reindex(index=algoritmos_de(df)).dropna(axis=1)


def rangos(df: pd.DataFrame, dim: int) -> pd.DataFrame:
    """Algoritmo × función: rango 1 (mejor) .. k, empate si |Δ| < 1e-8."""
    return error_medio(df, dim).apply(rankear_columna, axis=0)


def tabla_rangos_promedio(df: pd.DataFrame) -> pd.DataFrame:
    """Algoritmo × D: rango promedio sobre las 24 funciones."""
    t = pd.DataFrame({d: rangos(df, d).mean(axis=1) for d in dims_de(df)})
    t.columns.name = "dim"
    return t


def tabla_rangos_por_grupo(df: pd.DataFrame) -> pd.DataFrame:
    """Formato largo: dim, grupo, algoritmo, rango promedio en el grupo."""
    filas = []
    for d in dims_de(df):
        r = rangos(df, d)
        for g, fs in GRUPOS_BBOB.items():
            cols = [f for f in fs if f in r.columns]
            for a, v in r[cols].mean(axis=1).items():
                filas.append({"dim": d, "grupo": g, "algoritmo": a, "rango_promedio": v})
    return pd.DataFrame(filas)


def tabla_friedman(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for d in dims_de(df):
        e = error_medio(df, d)
        k, n = e.shape
        stat, p = friedmanchisquare(*[e.loc[a].values for a in e.index])
        filas.append({"dim": d, "k": k, "n_funciones": n, "estadistico": float(stat),
                      "p_valor": float(p), "rechaza_h0": bool(p < ALPHA),
                      "cd_nemenyi": diferencia_critica(k, n)})
    return pd.DataFrame(filas)


def tabla_exito(df: pd.DataFrame) -> pd.DataFrame:
    """Algoritmo × D: fracción de las (24 funciones × 15 instancias) resueltas
    a error < 1e-8."""
    t = (df.assign(ok=df["error_u"] == 0)
         .groupby(["algoritmo", "dim"])["ok"].mean().unstack("dim"))
    return t.reindex(index=algoritmos_de(df))


def tabla_exito_por_funcion(df: pd.DataFrame) -> pd.DataFrame:
    """Formato largo: dim, funcion, algoritmo, nº de instancias resueltas (de 15)."""
    return (df.assign(ok=df["error_u"] == 0)
            .groupby(["dim", "funcion", "algoritmo"])["ok"].sum()
            .rename("instancias_resueltas").reset_index())


def wilcoxon_bew(df: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon de rangos con signo, por (dim, función), emparejado por
    instancia (15 pares), sobre el error con umbral 1e-8 (dos errores bajo
    1e-8 son iguales: sin el umbral, diferencias de 1e-12 contarían como
    significativas). Veredicto desde la perspectiva de `algoritmo_a`:
    p ≥ 0.05 → equal; si no, better/worse según qué media es menor."""
    filas = []
    for d in dims_de(df):
        sub = df[df["dim"] == d]
        for f in FUNCIONES_BBOB:
            par = sub[sub["funcion"] == f].pivot(index="instancia", columns="algoritmo",
                                                  values="error_u")
            for a, b in PARES_WILCOXON:
                if a not in par or b not in par:
                    continue
                x, y = par[a].values, par[b].values
                if np.allclose(x, y, rtol=0, atol=UMBRAL_ERROR_CERO):
                    p, v = 1.0, "equal"
                else:
                    p = float(wilcoxon(x, y).pvalue)
                    v = "equal" if p >= ALPHA else ("better" if x.mean() < y.mean() else "worse")
                filas.append({"dim": d, "funcion": f, "algoritmo_a": a, "algoritmo_b": b,
                              "p_valor": p, "veredicto": v})
    return pd.DataFrame(filas)


def resumen_wilcoxon(tabla: pd.DataFrame) -> pd.DataFrame:
    """Conteo better/equal/worse (de 24 funciones) por dim y par."""
    return (tabla.groupby(["dim", "algoritmo_a", "algoritmo_b"])["veredicto"]
            .value_counts().unstack(fill_value=0)
            .reindex(columns=["better", "equal", "worse"], fill_value=0).reset_index())


def tabla_suplementaria(df: pd.DataFrame) -> pd.DataFrame:
    """dim, funcion, algoritmo: media, mediana, desviación, mín, P25, P75, máx
    del error final (sin umbral) de las 15 instancias."""
    g = df.groupby(["dim", "funcion", "algoritmo"])["error"]
    return pd.DataFrame({
        "media": g.mean(), "mediana": g.median(), "desv": g.std(),
        "minimo": g.min(), "q1": g.quantile(0.25), "q3": g.quantile(0.75), "maximo": g.max(),
    }).reset_index()


def tabla_middleware(prop: pd.DataFrame) -> pd.DataFrame:
    """Diagnóstico del middleware por dimensión (media de las 24 × 15 corridas)."""
    prop = prop[prop["run_id"].str.startswith("bbob__")]
    cols = ["n_activaciones_fase2", "n_activaciones_fase3", "n_canal_a", "n_canal_b",
            "n_abortos_wasserstein", "n_instancias_inyectadas", "frac_primera_activacion"]
    t = prop.groupby("dim")[cols].mean()
    t["corridas_sin_activacion"] = prop.groupby("dim")["n_activaciones_fase2"].apply(
        lambda s: int((s == 0).sum()))
    # R² del subrogado, uno por activación de Fase 2. La media no sirve: unos
    # pocos R² muy negativos (modelo inútil, transferencia bloqueada) la
    # arrastran a −10² o menos. Se reporta mediana y % bloqueado (R² < 0).
    def _r2(s):
        return np.array([v for lista in s for v in (list(lista) if lista is not None else [])],
                        dtype=float)
    r2 = prop.groupby("dim")["r2_fase2"].apply(_r2)
    t["r2_mediana"] = r2.apply(lambda v: float(np.median(v)) if len(v) else np.nan)
    t["pct_r2_negativo"] = r2.apply(lambda v: float(np.mean(v < 0) * 100) if len(v) else np.nan)
    return t.reset_index()


# ── figuras ─────────────────────────────────────────────────────────────────

def _estilo(a: str, i: int) -> dict:
    """Color por grupo y, dentro del grupo, distinto marcador y trazo."""
    marcadores = "osD^v"
    trazos = ["-", "--", ":", "-.", (0, (5, 1))]
    # El propuesto, más grueso y encima: su rojo se parece al naranja de GA.
    destacado = a == "middleware"
    return dict(color=COLOR_GRUPO[GRUPO_ALGORITMO.get(a, "principal")],
                marker=marcadores[i % len(marcadores)], linestyle=trazos[i % len(trazos)],
                linewidth=2.6 if destacado else 1.3, zorder=5 if destacado else 2)


def _indice_en_grupo(algoritmos: list[str]) -> dict[str, int]:
    vistos: dict[str, int] = {}
    salida = {}
    for a in algoritmos:
        g = GRUPO_ALGORITMO.get(a, "principal")
        salida[a] = vistos.get(g, 0)
        vistos[g] = salida[a] + 1
    return salida


def figura_cd(df: pd.DataFrame, dim: int, ruta: Path) -> None:
    r = rangos(df, dim)
    k, n = r.shape
    dibujar_cd(r.mean(axis=1).to_dict(), diferencia_critica(k, n),
               f"BBOB D={dim} — {df.attrs['presupuesto']}·D", ruta)


def figura_mapa_rangos(df: pd.DataFrame, dim: int, ruta: Path) -> None:
    """Mapa de calor algoritmo × función (rango; 1 = mejor), filas por rango
    promedio y columnas separadas por grupo de COCO."""
    r = rangos(df, dim)
    r = r.loc[r.mean(axis=1).sort_values().index]
    k = len(r)
    fig, ax = plt.subplots(figsize=(13, 0.42 * k + 1.6))
    im = ax.imshow(r.values, cmap="RdYlGn_r", vmin=1, vmax=k, aspect="auto")
    for i in range(k):
        for j in range(r.shape[1]):
            v = r.values[i, j]
            ax.text(j, i, f"{v:g}", ha="center", va="center", fontsize=7,
                    color="white" if v <= 2 or v >= k - 1 else TINTA)
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"{nombre(a)}  ({r.loc[a].mean():.2f})" for a in r.index], fontsize=8)
    ax.set_xticks(range(r.shape[1]))
    ax.set_xticklabels([f"f{f}" for f in r.columns], fontsize=8)
    for fs in GRUPOS_BBOB.values():
        ax.axvline(list(r.columns).index(fs[-1]) + 0.5, color="white", linewidth=2)
    ax.set_title(f"BBOB D={dim}: rango por función (1 = mejor; entre paréntesis, rango promedio)",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_contra_dim(tabla: pd.DataFrame, titulo: str, eje_y: str, ruta: Path,
                      invertir: bool = False, porcentaje: bool = False) -> None:
    """Una línea por algoritmo a lo largo de D (tabla algoritmo × D)."""
    fig, ax = plt.subplots(figsize=(9, 5.2))
    idx = _indice_en_grupo(list(tabla.index))
    for a, fila in tabla.iterrows():
        y = fila.values * (100 if porcentaje else 1)
        ax.plot(range(len(fila)), y, label=nombre(a), markersize=5, **_estilo(a, idx[a]))
    ax.set_xticks(range(tabla.shape[1]))
    ax.set_xticklabels([f"D={d}" for d in tabla.columns])
    ax.set_ylabel(eje_y)
    if invertir:
        ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.set_title(titulo, fontsize=11)
    ax.legend(ncol=1, fontsize=8, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_boxplots(df: pd.DataFrame, dim: int, ruta: Path) -> None:
    """Panel 4 × 6 (24 funciones): error final de las 15 instancias, log."""
    algoritmos = algoritmos_de(df)
    sub = df[df["dim"] == dim]
    fig, ejes = plt.subplots(4, 6, figsize=(24, 15))
    for f, ax in zip(FUNCIONES_BBOB, ejes.flat):
        g = sub[sub["funcion"] == f]
        datos = [np.maximum(g.loc[g["algoritmo"] == a, "error"].to_numpy(float), UMBRAL_ERROR_CERO)
                 for a in algoritmos]
        _cajas(ax, datos, algoritmos, log=True, fuente_etiquetas=6.5)
        ax.set_title(f"f{f} {NOMBRES_BBOB[f]}", fontsize=9)
    _leyenda_grupos(fig, algoritmos, loc="upper center", bbox_to_anchor=(0.5, 0.995), fontsize=11)
    fig.suptitle(f"BBOB D={dim}: error final f − f(x*) de las 15 instancias "
                 f"(piso 1e-8, presupuesto {df.attrs['presupuesto']}·D)", y=0.965, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def curvas_convergencia(df: pd.DataFrame, dim: int, funcion: int) -> dict[str, np.ndarray]:
    """Mediana de las instancias del error best-so-far en cada fracción del
    presupuesto (FRACCIONES_CHECKPOINT), con piso 1e-8."""
    sub = df[(df["dim"] == dim) & (df["funcion"] == funcion)]
    salida = {}
    for a, g in sub.groupby("algoritmo"):
        m = np.vstack([np.asarray(c, dtype=float) for c in g["error_checkpoints"]])
        salida[a] = np.maximum(np.median(m, axis=0), UMBRAL_ERROR_CERO)
    return salida


def figura_convergencia(df: pd.DataFrame, dim: int, ruta: Path) -> None:
    algoritmos = algoritmos_de(df)
    idx = _indice_en_grupo(algoritmos)
    x = np.asarray(FRACCIONES_CHECKPOINT) * df.attrs["presupuesto"] * dim
    fig, ejes = plt.subplots(4, 6, figsize=(24, 15))
    for f, ax in zip(FUNCIONES_BBOB, ejes.flat):
        curvas = curvas_convergencia(df, dim, f)
        for a in algoritmos:
            if a in curvas:
                est = _estilo(a, idx[a])
                ax.plot(x, curvas[a], color=est["color"], linestyle=est["linestyle"],
                        linewidth=est["linewidth"] * 0.9, zorder=est["zorder"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"f{f} {NOMBRES_BBOB[f]}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
    manejadores = [Line2D([], [], label=nombre(a), color=_estilo(a, idx[a])["color"],
                          linestyle=_estilo(a, idx[a])["linestyle"],
                          linewidth=_estilo(a, idx[a])["linewidth"])
                   for a in algoritmos]
    fig.legend(handles=manejadores, loc="upper center", ncol=len(algoritmos), frameon=False,
               bbox_to_anchor=(0.5, 0.995), fontsize=10)
    fig.suptitle(f"BBOB D={dim}: convergencia, mediana de 15 instancias "
                 f"(eje x: evaluaciones; eje y: error best-so-far, piso 1e-8)", y=0.965, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


# ── cocopp ──────────────────────────────────────────────────────────────────

# Sufijo de cocopp para cada grupo de funciones de COCO -> nombre del archivo copiado.
GRUPO_ECDF_COCOPP = {
    "noiselessall": "todas",              # f1–f24
    "separ": "separables",                # f1–f5
    "lcond": "cond_moderado",             # f6–f9
    "hcond": "cond_alto",                 # f10–f14
    "multi": "multimodales_adecuada",     # f15–f19
    "mult2": "multimodales_debil",        # f20–f24
}


def copiar_ecdf_cocopp(ppdata: Path, destino: Path) -> int:
    """Copia de la salida de cocopp (varios algoritmos) las ECDF agregadas
    `pprldmany_<D>D_<grupo>.{png,svg}`: sobre las 24 funciones y sobre cada
    grupo, una por dimensión (36). Las ECDF por función (144) quedan en
    ppdata. Las renombra `ecdf_dDD_<grupo>.<ext>`. Devuelve cuántas copió."""
    destino.mkdir(parents=True, exist_ok=True)
    n = 0
    for ext in ("png", "svg"):
        for fig in sorted(Path(ppdata).rglob(f"pprldmany_*D_*.{ext}")):
            dim_txt, grupo = fig.stem.split("_")[1:3]
            if grupo not in GRUPO_ECDF_COCOPP:
                continue
            shutil.copy2(fig, destino / f"ecdf_d{int(dim_txt[:-1]):02d}_{GRUPO_ECDF_COCOPP[grupo]}.{ext}")
            n += 1
        if n:
            break                         # con PNG alcanza; SVG solo si no hay PNG
    return n


# ── LEEME ───────────────────────────────────────────────────────────────────

LEEME = """\
# Análisis BBOB: cómo leer cada tabla y figura

Parrilla: 12 algoritmos × 24 funciones × D ∈ {{2, 3, 5, 10, 20, 40}} × 15
instancias = {n_corridas:,} corridas, presupuesto **{presupuesto}·D**
evaluaciones (opción A de `README_BBOB.md`). Generado con
`python analisis/bbob.py --resultados <carpeta>`.

**Error:** `f − f(x*)` al agotar el presupuesto (best-so-far). Un error bajo
`1e-8` cuenta como 0 (convención CEC, y objetivo final de COCO).
**Repeticiones:** en COCO las 15 instancias reemplazan a las semillas: son
versiones desplazadas y rotadas de la misma función, una corrida por
instancia.

Todo es **por dimensión**: no se promedia entre dimensiones (igual que en
CEC 2022 no se promedia entre configuraciones dim × MaxFES).

## Tablas (`tablas/`)

| Archivo | Contenido |
|---|---|
| `rangos_promedio.csv` | Algoritmo × D: rango promedio sobre las 24 funciones (1 = mejor, k = 12 = peor). Es la tabla principal. |
| `rangos_dDD.csv` / `error_medio_dDD.csv` | Algoritmo × función: rango y error medio de las 15 instancias, en la dimensión DD. El rango se asigna por función sobre el error medio, con empate si la diferencia es < 1e-8 (rango promedio de las posiciones empatadas). |
| `rangos_por_grupo.csv` | Rango promedio dentro de cada grupo de funciones de COCO: separables (f1–f5), condicionamiento moderado (f6–f9), condicionamiento alto y unimodales (f10–f14), multimodales con estructura global adecuada (f15–f19) y débil (f20–f24). |
| `friedman.csv` | Friedman por dimensión (k algoritmos, N = 24 funciones). `rechaza_h0` = hay diferencia significativa (α = 0.05). `cd_nemenyi` = diferencia crítica de los diagramas CD. |
| `wilcoxon_bew.csv` / `wilcoxon_resumen.csv` | Wilcoxon de rangos con signo, propuesto contra cada algoritmo (y PSO contra DE), por (dim, función), emparejado por instancia (15 pares). better / equal / worse desde el primer algoritmo: `equal` si p ≥ 0.05; si no, gana el de menor error medio. Se usa el error con umbral 1e-8. El resumen cuenta de 24 funciones por dimensión. |
| `exito.csv` | Algoritmo × D: fracción de las 360 corridas (24 × 15) que llegan a error < 1e-8. |
| `exito_por_funcion.csv` | Instancias resueltas (de 15) por dim, función y algoritmo. |
| `suplementario.csv` | Media, mediana, desviación, mínimo, P25, P75 y máximo del error final (sin umbral) por dim, función y algoritmo. |
| `middleware_diagnostico.csv` | Por dimensión, media de 360 corridas del propuesto: activaciones de Fase 2 y 3, Canal A, Canal B, abortos por la barrera de Wasserstein, instancias inyectadas, fracción del presupuesto en la primera activación, corridas sin ninguna activación, y mediana y % negativo del R² del modelo subrogado (R² < 0 bloquea la transferencia; la media no se reporta porque unos pocos R² muy negativos la distorsionan). |

## Figuras (`figuras/`)

| Archivo | Cómo leerla |
|---|---|
| `cd_dDD.png` | Diagrama de diferencia crítica (Nemenyi): cada algoritmo en su rango promedio. Una barra une algoritmos cuya diferencia de rango es menor que la CD, que no se distinguen estadísticamente. |
| `mapa_rangos_dDD.png` | Rango de cada algoritmo (filas, ordenadas por rango promedio) en cada función (columnas). Verde = mejor. Las líneas blancas separan los 5 grupos de funciones. |
| `rangos_contra_dim.png` | Rango promedio de cada algoritmo en cada D (eje invertido: arriba = mejor). Muestra cómo escala cada algoritmo con la dimensión. |
| `exito_contra_dim.png` | % de corridas resueltas a 1e-8 en cada D. |
| `boxplot_dDD.png` | Panel de las 24 funciones: distribución del error final de las 15 instancias por algoritmo, en escala log con piso 1e-8. Colores por grupo de algoritmos. |
| `convergencia_dDD.png` | Panel de las 24 funciones: mediana de las 15 instancias del error best-so-far contra las evaluaciones consumidas (ejes log). |
| `cocopp/ecdf_dDD_<grupo>.png` | Si se generó con `--ppdata`: ECDF agregadas de `cocopp` (grupo `todas` = f1–f24, o uno de los 5 grupos). La curva *best 2009* es la envolvente de los mejores algoritmos de BBOB 2009 (archivo de datos de COCO). **El presupuesto real es 10⁴·D (x = log₁₀ = 4, la cruz de cada curva): a la derecha, `cocopp` dibuja reinicios simulados, no evaluaciones hechas.** Fracción de pares (función, objetivo de precisión 10² … 10⁻⁸) alcanzados contra evaluaciones / D, sobre todas las funciones o un grupo, por dimensión. Más arriba y más a la izquierda = mejor. Es la figura estándar con que se publica en BBOB. |

## Advertencias

- Los 12 algoritmos usan la **configuración congelada de CEC 2022**, sin
  re-ajustar para BBOB: BBOB mide la generalización de esa configuración.
- El presupuesto 10⁴·D es chico en D = 2–5 (2·10⁴ a 5·10⁴ evaluaciones).
  El middleware gasta el 15 % en warm-up y ventanas del 10 % por ciclo,
  así que en esas dimensiones alcanza a actuar pocas veces
  (`middleware_diagnostico.csv`).
- Los rangos de BBOB y de CEC 2022 no se comparan entre sí (otro
  protocolo, otras funciones): cada benchmark tiene su propio ranking.
"""


# ── main ────────────────────────────────────────────────────────────────────

def generar_todo(resultados: str | Path, ppdata: str | Path | None = None,
                 permitir_incompleto: bool = False, figuras: bool = True) -> Path:
    base = Path(resultados)
    df = cargar_bbob(base / "corridas.parquet", permitir_incompleto)
    salida = base / "analisis"
    T, F = salida / "tablas", salida / "figuras"
    T.mkdir(parents=True, exist_ok=True)
    F.mkdir(parents=True, exist_ok=True)
    print(f"BBOB, presupuesto {df.attrs['presupuesto']}·D: {len(df):,} corridas, "
          f"{len(algoritmos_de(df))} algoritmos")

    for d in dims_de(df):
        error_medio(df, d).to_csv(T / f"error_medio_d{d:02d}.csv")
        rangos(df, d).to_csv(T / f"rangos_d{d:02d}.csv")
    rp = tabla_rangos_promedio(df)
    rp.to_csv(T / "rangos_promedio.csv")
    tabla_rangos_por_grupo(df).to_csv(T / "rangos_por_grupo.csv", index=False)
    fr = tabla_friedman(df)
    fr.to_csv(T / "friedman.csv", index=False)
    ex = tabla_exito(df)
    ex.to_csv(T / "exito.csv")
    tabla_exito_por_funcion(df).to_csv(T / "exito_por_funcion.csv", index=False)
    w = wilcoxon_bew(df)
    w.to_csv(T / "wilcoxon_bew.csv", index=False)
    resumen_wilcoxon(w).to_csv(T / "wilcoxon_resumen.csv", index=False)
    tabla_suplementaria(df).to_csv(T / "suplementario.csv", index=False)
    ruta_prop = base / "corridas_propuesto.parquet"
    if ruta_prop.exists():
        tabla_middleware(pd.read_parquet(ruta_prop)).to_csv(
            T / "middleware_diagnostico.csv", index=False)
    print("  tablas ->", T)

    if figuras:
        for d in dims_de(df):
            figura_cd(df, d, F / f"cd_d{d:02d}.png")
            figura_mapa_rangos(df, d, F / f"mapa_rangos_d{d:02d}.png")
            figura_boxplots(df, d, F / f"boxplot_d{d:02d}.png")
            figura_convergencia(df, d, F / f"convergencia_d{d:02d}.png")
            print(f"  figuras D={d}")
        figura_contra_dim(rp, f"BBOB: rango promedio sobre 24 funciones ({df.attrs['presupuesto']}·D)",
                          "rango promedio (1 = mejor)", F / "rangos_contra_dim.png", invertir=True)
        figura_contra_dim(ex, "BBOB: corridas resueltas a error < 1e-8",
                          "% de las 360 corridas (24 funciones × 15 instancias)",
                          F / "exito_contra_dim.png", porcentaje=True)
    if ppdata is not None:
        n = copiar_ecdf_cocopp(Path(ppdata), F / "cocopp")
        print(f"  {n} ECDF de cocopp -> {F / 'cocopp'}")

    (salida / "LEEME.md").write_text(
        LEEME.format(n_corridas=len(df), presupuesto=df.attrs["presupuesto"]), encoding="utf-8")

    print("\nRango promedio por dimensión (1 = mejor):")
    orden = rp.mean(axis=1).sort_values().index
    print(rp.loc[orden].rename(index=nombre).round(2).to_string())
    print("\nFriedman:")
    print(fr[["dim", "estadistico", "p_valor", "rechaza_h0", "cd_nemenyi"]].round(4).to_string(index=False))
    print(f"\n[OK] análisis BBOB en {salida}")
    return salida


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resultados", default="resultados_bbob")
    ap.add_argument("--ppdata", default=None,
                    help="carpeta de salida de cocopp (para copiar sus ECDF agregadas)")
    ap.add_argument("--permitir-incompleto", action="store_true")
    ap.add_argument("--sin-figuras", action="store_true")
    args = ap.parse_args()
    generar_todo(args.resultados, args.ppdata, args.permitir_incompleto,
                 figuras=not args.sin_figuras)


if __name__ == "__main__":
    main()
