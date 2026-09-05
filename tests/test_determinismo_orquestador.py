"""
Criterio de aceptación de D-2 (docs/configuracion_experimental.md):
el planificador cooperativo determinista debe dar EXACTAMENTE el mismo
resultado en corridas repetidas, dada la misma (función, dim, MaxFES,
semilla) — sin la variación por scheduling de hilos que tenía la versión
con threading.

Corre la misma configuración dos veces y compara el resumen completo.
También confirma que el reparto de FES entre A y B es 50/50 (por diseño
del round-robin con poblaciones iguales, ya no una aproximación medida).
"""

import sys
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROYECTO))

from problems.cec2022_wrapper import ProblemaCEC2022
from bioalgorithms.pso import PSO
from bioalgorithms.de import DE
from middleware.orquestador import Orquestador

FUNCION = 11
DIM = 10
MAX_FES = 30_000
SEMILLA_PSO = 42
SEMILLA_DE = 500


def correr():
    problema = ProblemaCEC2022(numero_funcion=FUNCION, ndim=DIM)
    alg_a = PSO(problema, 30, problema.ndim, problema.limites, semilla=SEMILLA_PSO,
                w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, max_fes=MAX_FES)
    alg_b = DE(problema, 30, problema.ndim, problema.limites, semilla=SEMILLA_DE,
               F=0.2, CR=0.3, max_fes=MAX_FES)
    orq = Orquestador(
        algoritmo_a=alg_a, algoritmo_b=alg_b, limites=problema.limites,
        problema=problema, max_fes=MAX_FES, frecuencia_monitoreo_fes=600,
        directorio_log="logs", nombre_log="prueba_determinismo",
        max_epochs_fastshap=5, silencioso=True,
    )
    return orq.ejecutar()


def main():
    r1 = correr()
    r2 = correr()

    campos_exactos = [
        "error_sistema", "fitness_final_algoritmo_a", "fitness_final_algoritmo_b",
        "fes_consumidas_total", "fes_consumidas_a", "fes_consumidas_b",
        "n_generaciones_a", "n_generaciones_b",
        "n_activaciones_fase2", "n_activaciones_fase3",
        "n_transferencias_canal_a", "n_transferencias_canal_b",
        "n_transferencias_abortadas_wasserstein",
        "n_reentrenamientos_por_cambio_fuente",
    ]

    print(f"{'campo':38s} {'corrida 1':>15s} {'corrida 2':>15s}  igual")
    todo_igual = True
    for campo in campos_exactos:
        v1, v2 = getattr(r1, campo), getattr(r2, campo)
        igual = v1 == v2
        todo_igual &= igual
        print(f"{campo:38s} {v1!r:>15} {v2!r:>15}  {'OK' if igual else 'DIFERENTE'}")

    assert todo_igual, "Corridas idénticas dieron resultados distintos — D-2 no está resuelto."

    # Reparto 50/50 (± 1 generación por el guard de sobrepaso: pop=30 cada uno)
    diff_fes = abs(r1.fes_consumidas_a - r1.fes_consumidas_b)
    print(f"\nfes_consumidas_a={r1.fes_consumidas_a}  fes_consumidas_b={r1.fes_consumidas_b}  "
          f"diferencia={diff_fes}")
    assert diff_fes <= 30, (
        f"El reparto debería ser ~50/50 con el round-robin (pop=30); "
        f"diferencia observada: {diff_fes}"
    )

    print("\n[OK] D-2 validado: misma semilla -> mismo resultado, reparto 50/50.")


if __name__ == "__main__":
    main()
