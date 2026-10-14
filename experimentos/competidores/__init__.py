"""
Competidores externos (D-8, docs/configuracion_experimental.md): corren
SOLOS, como medida comparativa frente al middleware. Cada uno expone
`correr(problema, max_fes, semilla) -> None` (el resultado queda en el
contador y los checkpoints de `problema`) y una constante `VERSION` para
la columna `version` de §11.2 (setup §10).

    id corto   -> (módulo, nombre para tablas/figuras, familia §11.2)
"""

from experimentos.competidores import cmaes_ipop, madde, nlshade_rsp

COMPETIDORES = {
    "cmaes": (cmaes_ipop, "IPOP-CMA-ES", "distribucion"),
    "madde": (madde, "MadDE", "DE"),
    "nlshadersp": (nlshade_rsp, "NL-SHADE-RSP", "DE"),
}
