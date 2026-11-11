"""
Competidores externos (D-8, docs/configuracion_experimental.md): corren
SOLOS, como medida comparativa frente al middleware. Cada uno expone
`correr(problema, max_fes, semilla) -> None` (el resultado queda en el
contador y los checkpoints de `problema`) y una constante `VERSION` para
la columna `version` de §11.2 (setup §10).

    id corto   -> (módulo, nombre para tablas/figuras, familia §11.2)


Algoritmos de referencia por familia (docs/competidores.md §8): no
compiten directamente con el middleware, sirven para ubicarlo frente a las
familias de algoritmos bioinspirados (evolutivos, enjambre, sistemas
inmunes). `REFERENCIA` lista sus ids. En la familia inmune, GCO reemplaza
a opt-IA (papers de acceso cerrado; docs/competidores.md §8.3).
"""

from experimentos.competidores import (
    abc_karaboga, clonalg_aisp, clpso, cmaes_ipop, ga_pymoo, gco, gwo, jade, jde,
    madde, nlshade_rsp, opt_ainet,
)

COMPETIDORES = {
    "cmaes": (cmaes_ipop, "IPOP-CMA-ES", "distribucion"),
    "madde": (madde, "MadDE", "DE"),
    "nlshadersp": (nlshade_rsp, "NL-SHADE-RSP", "DE"),
    # ── referencia por familia ──
    "ga": (ga_pymoo, "GA", "evolutivo"),
    "jde": (jde, "jDE", "evolutivo"),
    "jade": (jade, "JADE", "evolutivo"),
    "clpso": (clpso, "CLPSO", "enjambre"),
    "gwo": (gwo, "GWO", "enjambre"),
    "abc": (abc_karaboga, "ABC", "enjambre"),
    "clonalg": (clonalg_aisp, "CLONALG", "inmune"),
    "optainet": (opt_ainet, "opt-aiNet", "inmune"),
    "gco": (gco, "GCO", "inmune"),
}

REFERENCIA = ("ga", "jde", "jade", "clpso", "gwo", "abc", "clonalg", "optainet", "gco")
