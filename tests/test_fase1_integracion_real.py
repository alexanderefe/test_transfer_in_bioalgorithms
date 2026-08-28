"""
Integración Fase 0 + Fase 1 — Score real sobre PSO y DE.

Ejecuta PSO (fuente) y DE (objetivo) sobre la misma función F11/D=20 del
benchmark CEC 2022 ya usada en la validación de Fase 0, pero esta vez
calculando el Score compuesto S en cada iteración con la implementación
formal de Fase 1 (FIR + DWD, ecuaciones 2 y 4 del documento).

Esto NO es todavía el middleware completo: es la verificación de que el
Score, ya validado sobre casos sintéticos, se comporta de forma coherente
sobre el comportamiento real de los algoritmos. Es el puente entre Fase 0
y Fase 1 antes de construir Fase 2 (Extracción).
"""

import sys
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROYECTO))

from problems.cec2022_wrapper import ProblemaCEC2022
from bioalgorithms.pso import PSO
from bioalgorithms.de import DE
from middleware.deteccion import calcular_score, fuente_en_condiciones_optimas


def ejecutar_con_score(algoritmo, n_iteraciones: int, limites):
    historial_fitness = []
    resultados_score = []
    for _ in range(n_iteraciones):
        snapshot = algoritmo.ejecutar_iteracion()
        historial_fitness.append(snapshot.mejor_fitness_historico)
        resultado = calcular_score(
            historial_fitness=historial_fitness,
            poblacion_actual=snapshot.poblacion,
            limites=limites,
            iteracion_actual=snapshot.iteracion,
            max_iteraciones=n_iteraciones,
        )
        resultados_score.append(resultado)
    return historial_fitness, resultados_score


def main():
    problema = ProblemaCEC2022(numero_funcion=11, ndim=20)
    print(f"Problema: {problema}\n")

    n_individuos = 30
    n_iteraciones = 300
    semilla = 42

    print("=" * 70)
    print("PSO (FUENTE) — Score compuesto a lo largo de la ejecución")
    print("=" * 70)
    pso = PSO(problema, n_individuos, problema.ndim, problema.limites,
              n_iteraciones, semilla=semilla)
    _, scores_pso = ejecutar_con_score(pso, n_iteraciones, problema.limites)

    for it in [20, 60, 100, 150, 200, 300]:
        r = scores_pso[it - 1]
        print(f"  it={it:4d} | S={r.score:.4f} | FIR={r.fir:.4f} | DWD={r.dwd:.4f} "
              f"| estancado={r.estancado} | activo={r.activo}")

    print()
    print("=" * 70)
    print("DE (OBJETIVO) — Score compuesto a lo largo de la ejecución")
    print("=" * 70)
    de = DE(problema, n_individuos, problema.ndim, problema.limites,
            n_iteraciones, semilla=semilla, F=0.2, CR=0.3)
    _, scores_de = ejecutar_con_score(de, n_iteraciones, problema.limites)

    for it in [20, 60, 100, 150, 200, 300]:
        r = scores_de[it - 1]
        print(f"  it={it:4d} | S={r.score:.4f} | FIR={r.fir:.4f} | DWD={r.dwd:.4f} "
              f"| estancado={r.estancado} | activo={r.activo}")

    # Punto clave: ¿en qué iteración el Score de DE cruza el umbral de
    # estancamiento por primera vez de forma sostenida (10 iteraciones
    # consecutivas estancado)?
    print()
    print("-" * 70)
    print("Primer cruce sostenido de estancamiento en DE (>=10 iter consecutivas)")
    print("-" * 70)
    consecutivas = 0
    iteracion_deteccion = None
    for r in scores_de:
        if r.activo and r.estancado:
            consecutivas += 1
            if consecutivas >= 10 and iteracion_deteccion is None:
                iteracion_deteccion = r.iteracion - 9
        else:
            consecutivas = 0

    if iteracion_deteccion:
        print(f"DE detectado en estancamiento sostenido desde la iteración "
              f"{iteracion_deteccion} (de {n_iteraciones} totales).")
        print("\n✅ El Score detecta el estancamiento de DE con margen "
              "suficiente para que el middleware pueda intervenir "
              "(Fase 2 en adelante) antes de agotar el presupuesto de "
              "iteraciones.")
    else:
        print("⚠️  No se detectó un cruce sostenido de estancamiento en DE. "
              "Esto contradice lo observado en Fase 0 y debe investigarse "
              "antes de continuar.")

    # ------------------------------------------------------------------
    # Verificación del criterio "fuente en condiciones óptimas" (Opción B
    # acordada: fitness_fuente < fitness_objetivo), usando el fitness real
    # registrado en cada iteración por ambos algoritmos.
    # ------------------------------------------------------------------
    print()
    print("-" * 70)
    print("Criterio 'fuente en condiciones óptimas' (PSO vs DE, fitness real)")
    print("-" * 70)
    fitness_pso_por_iter = [s.mejor_fitness_historico for s in pso.historial]
    fitness_de_por_iter = [s.mejor_fitness_historico for s in de.historial]

    for it in [20, 60, 100, 150, 200, 300]:
        f_pso = fitness_pso_por_iter[it - 1]
        f_de = fitness_de_por_iter[it - 1]
        optimo = fuente_en_condiciones_optimas(f_pso, f_de)
        print(f"  it={it:4d} | fitness_PSO={f_pso:10.4f} | fitness_DE={f_de:10.4f} "
              f"| ¿PSO en condiciones óptimas como fuente?={optimo}")

    #print("\nEste criterio (no el Score S) es el que debe usarse en la Fase 2 "
    #      "para verificar la condición 'algoritmo de origen funciona en "
    #      "condiciones óptimas', exigida junto con S_objetivo < 0.2 según "
    #      "la sección 4.2.2 del documento de tesis.")


if __name__ == "__main__":
    main()