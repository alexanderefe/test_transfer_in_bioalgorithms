"""
Metadatos de la suite `bbob` de COCO, sin depender de `cocoex`: los usa
`analisis/bbob.py` (que también corre en Colab, donde no hace falta
instalar COCO). `problems/bbob_wrapper.py` los reexporta.
"""

FUNCIONES_BBOB = tuple(range(1, 25))
DIMENSIONES_BBOB = (2, 3, 5, 10, 20, 40)
INSTANCIAS_BBOB = tuple(range(1, 16))      # instancias clásicas 2009 (1..15)

# Nombre de cada función (Hansen et al., 2009, "Real-parameter black-box
# optimization benchmarking 2009: noiseless functions definitions").
NOMBRES_BBOB = {
    1: "Sphere", 2: "Ellipsoidal separable", 3: "Rastrigin separable",
    4: "Bueche-Rastrigin", 5: "Linear slope",
    6: "Attractive sector", 7: "Step ellipsoidal", 8: "Rosenbrock original",
    9: "Rosenbrock rotated",
    10: "Ellipsoidal", 11: "Discus", 12: "Bent cigar", 13: "Sharp ridge",
    14: "Different powers",
    15: "Rastrigin", 16: "Weierstrass", 17: "Schaffer F7",
    18: "Schaffer F7 mal condicionada", 19: "Griewank-Rosenbrock F8F2",
    20: "Schwefel x*sin(x)", 21: "Gallagher 101 picos", 22: "Gallagher 21 picos",
    23: "Katsuura", 24: "Lunacek bi-Rastrigin",
}

# Grupos estándar de COCO, para el análisis por grupo de funciones.
GRUPOS_BBOB = {
    "separables": tuple(range(1, 6)),
    "cond. moderado": tuple(range(6, 10)),
    "cond. alto, unimodales": tuple(range(10, 15)),
    "multimodales, estructura adecuada": tuple(range(15, 20)),
    "multimodales, estructura débil": tuple(range(20, 25)),
}
GRUPO_BBOB = {f: g for g, fs in GRUPOS_BBOB.items() for f in fs}
