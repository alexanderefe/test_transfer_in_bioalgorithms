"""
Validación de la Fase 1 (criterio de aceptación) — casos sintéticos.

Antes de conectar el Score compuesto S a los algoritmos reales (PSO/DE),
se valida sobre historiales SINTÉTICOS donde el resultado esperado se
conoce por diseño:

1. Historial con estancamiento inducido (población congelada, fitness sin
   mejora) -> S debe caer bajo 0.2 de forma consistente.
2. Historial con progreso activo simulado (población dispersa, fitness
   mejorando de forma sostenida) -> S debe mantenerse sobre 0.2.
3. Test de robustez numérica: S nunca debe salir de [0,1] bajo entradas
   extremas (fitness negativo, población con varianza cero, etc.).
"""

import sys
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROYECTO))

import numpy as np
from middleware.deteccion import calcular_score, UMBRAL_ESTANCAMIENTO


def caso_estancamiento_inducido():
    """
    Simula 200 iteraciones donde la población queda congelada en torno a
    un punto fijo (sin dispersión) y el fitness deja de mejorar tras la
    iteración 50. El Score debe detectar esto como estancamiento.
    """
    print("=" * 70)
    print("CASO 1 — Estancamiento inducido (población congelada)")
    print("=" * 70)

    rng = np.random.default_rng(0)
    limites = np.array([[-100.0, 100.0]] * 10)
    n_individuos = 30
    n_dimensiones = 10
    max_iteraciones = 200

    historial_fitness = []
    punto_fijo = rng.uniform(-50, 50, size=n_dimensiones)

    resultados = []
    for it in range(1, max_iteraciones + 1):
        if it <= 50:
            # Fase inicial: progreso simulado (fitness decreciente).
            mejor_fitness = 1000.0 / it
            ruido = rng.normal(0, 0.5, size=(n_individuos, n_dimensiones))
            poblacion = punto_fijo + ruido * (max(50 - it, 1) / 50) * 30
        else:
            # Fase de estancamiento: fitness fijo, población sin dispersión
            # (todos los individuos casi idénticos al punto fijo).
            mejor_fitness = 1000.0 / 50  # congelado
            ruido = rng.normal(0, 0.01, size=(n_individuos, n_dimensiones))
            poblacion = punto_fijo + ruido

        historial_fitness.append(mejor_fitness)
        resultado = calcular_score(
            historial_fitness=historial_fitness,
            poblacion_actual=poblacion,
            limites=limites,
            iteracion_actual=it,
            max_iteraciones=max_iteraciones,
        )
        resultados.append(resultado)

    # Verificación: en el último 20% de iteraciones (claramente estancadas
    # y con el módulo ya activo), el Score debe estar bajo el umbral en
    # la gran mayoría de los casos.
    tramo_final = [r for r in resultados if r.iteracion > 150]
    proporcion_estancado = sum(r.estancado for r in tramo_final) / len(tramo_final)

    print(f"Iteración 60  -> S={resultados[59].score:.4f} "
          f"(FIR={resultados[59].fir:.4f}, DWD={resultados[59].dwd:.4f}, "
          f"estancado={resultados[59].estancado})")
    print(f"Iteración 100 -> S={resultados[99].score:.4f} "
          f"(FIR={resultados[99].fir:.4f}, DWD={resultados[99].dwd:.4f}, "
          f"estancado={resultados[99].estancado})")
    print(f"Iteración 200 -> S={resultados[-1].score:.4f} "
          f"(FIR={resultados[-1].fir:.4f}, DWD={resultados[-1].dwd:.4f}, "
          f"estancado={resultados[-1].estancado})")
    print(f"\nProporción de iteraciones estancadas en tramo final (>150): "
          f"{proporcion_estancado:.2%}")

    aprobado = proporcion_estancado >= 0.95
    print(f"\n{'✅ APROBADO' if aprobado else '❌ FALLÓ'}: "
          f"se exige >=95% de detección en tramo claramente estancado.")
    return aprobado


def caso_progreso_activo():
    """
    Simula 200 iteraciones con progreso sostenido: el fitness mejora de
    forma constante y la población mantiene dispersión razonable. El
    Score NO debe marcar estancamiento en la mayoría de las iteraciones
    (una vez activo, tras el 15%).
    """
    print()
    print("=" * 70)
    print("CASO 2 — Progreso activo simulado")
    print("=" * 70)

    rng = np.random.default_rng(1)
    limites = np.array([[-100.0, 100.0]] * 10)
    n_individuos = 30
    n_dimensiones = 10
    max_iteraciones = 200

    historial_fitness = []
    resultados = []
    for it in range(1, max_iteraciones + 1):
        # Fitness decreciente sostenido (mejora constante, nunca se estanca).
        mejor_fitness = 5000.0 / (1 + 0.05 * it)
        # Población con dispersión moderada-alta mantenida en el tiempo.
        poblacion = rng.uniform(-80, 80, size=(n_individuos, n_dimensiones))

        historial_fitness.append(mejor_fitness)
        resultado = calcular_score(
            historial_fitness=historial_fitness,
            poblacion_actual=poblacion,
            limites=limites,
            iteracion_actual=it,
            max_iteraciones=max_iteraciones,
        )
        resultados.append(resultado)

    tramo_activo = [r for r in resultados if r.activo]
    proporcion_no_estancado = sum(not r.estancado for r in tramo_activo) / len(tramo_activo)

    print(f"Iteración 60  -> S={resultados[59].score:.4f} "
          f"(FIR={resultados[59].fir:.4f}, DWD={resultados[59].dwd:.4f})")
    print(f"Iteración 150 -> S={resultados[149].score:.4f} "
          f"(FIR={resultados[149].fir:.4f}, DWD={resultados[149].dwd:.4f})")
    print(f"\nProporción de iteraciones NO marcadas como estancadas "
          f"(módulo activo): {proporcion_no_estancado:.2%}")

    aprobado = proporcion_no_estancado >= 0.95
    print(f"\n{'✅ APROBADO' if aprobado else '❌ FALLÓ'}: "
          f"se exige >=95% de no-falsos-positivos durante progreso activo.")
    return aprobado


def caso_robustez_numerica():
    """
    Verifica que el Score nunca sale de [0,1] bajo entradas extremas:
    fitness negativos, población con varianza cero, fitness igual a cero.
    """
    print()
    print("=" * 70)
    print("CASO 3 — Robustez numérica ante entradas extremas")
    print("=" * 70)

    limites = np.array([[-100.0, 100.0]] * 5)
    casos_extremos = [
        ("fitness negativos constantes",
         [-500.0] * 25, np.zeros((10, 5))),
        ("fitness cero",
         [0.0] * 25, np.zeros((10, 5))),
        ("población idéntica (varianza cero) con fitness positivo",
         [100.0 - i * 0.001 for i in range(25)], np.full((10, 5), 3.0)),
        ("fitness con salto abrupto enorme",
         [1e10] * 24 + [1e-10], np.random.default_rng(2).uniform(-100, 100, (10, 5))),
    ]

    todos_validos = True
    for nombre, historial, poblacion in casos_extremos:
        resultado = calcular_score(
            historial_fitness=historial,
            poblacion_actual=poblacion,
            limites=limites,
            iteracion_actual=len(historial),
            max_iteraciones=100,
        )
        valido = 0.0 <= resultado.score <= 1.0
        todos_validos &= valido
        print(f"  [{('OK' if valido else 'FALLA')}] {nombre}: "
              f"S={resultado.score:.6f} (FIR={resultado.fir:.4f}, DWD={resultado.dwd:.4f})")

    print(f"\n{'✅ APROBADO' if todos_validos else '❌ FALLÓ'}: "
          f"S debe permanecer en [0,1] en todos los casos extremos.")
    return todos_validos


def main():
    print(f"Umbral de estancamiento configurado: S < {UMBRAL_ESTANCAMIENTO}\n")
    resultados = [
        caso_estancamiento_inducido(),
        caso_progreso_activo(),
        caso_robustez_numerica(),
    ]

    print()
    print("=" * 70)
    print("RESUMEN FASE 1 — CRITERIO DE ACEPTACIÓN")
    print("=" * 70)
    if all(resultados):
        print("✅ Los 3 casos pasaron. El módulo de detección (Fase 1) "
              "queda validado para integrarse con los algoritmos reales.")
    else:
        print("⚠️  Al menos un caso falló. Revisar la formalización antes "
              "de continuar con la integración.")


if __name__ == "__main__":
    main()