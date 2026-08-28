"""
Validación de la Fase 2 (criterio de aceptación) — FastSHAP real sobre PSO.

Ejecuta PSO sobre F11/D=20 (mismo caso ya validado en Fase 0 y Fase 1),
acumula su historial, y aplica la Fase 2 completa: modelo subrogado
XGBoost + FastSHAP oficial + extracción de instancias de élite y
parámetros de escape.

Criterios de aceptación (acordados en el plan metodológico original):
1. El modelo subrogado debe alcanzar R² > 0.7 en datos de validación.
2. FastSHAP debe asignar mayor contribución (|ϕ| promedio) a la variable
   que se sabe, por diseño de un caso sintético controlado, que más
   influye en el fitness.
"""

import sys
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROYECTO))

import numpy as np

from problems.cec2022_wrapper import ProblemaCEC2022
from bioalgorithms.pso import PSO
from middleware.extraccion import extraer_conocimiento, entrenar_modelo_subrogado


def caso_real_pso_f11():
    """
    Ejecuta PSO sobre F11/D=20 durante 80 iteraciones (suficientes para
    superar la ventana de 50 requerida por la Fase 2) y aplica extracción
    de conocimiento sobre el historial acumulado.
    """
    print("=" * 70)
    print("CASO 1 — Extracción real sobre historial de PSO (F11, D=20)")
    print("=" * 70)

    problema = ProblemaCEC2022(numero_funcion=11, ndim=20)
    n_individuos = 30
    n_iteraciones = 80
    semilla = 42

    pso = PSO(problema, n_individuos, problema.ndim, problema.limites,
              n_iteraciones, semilla=semilla)

    for _ in range(n_iteraciones):
        pso.ejecutar_iteracion()

    historial_poblacional = [s.poblacion for s in pso.historial]
    historial_fitness_individual = [s.fitness for s in pso.historial]
    historial_mejor_fitness = [s.mejor_fitness_historico for s in pso.historial]
    historial_hiperparametros = [s.hiperparametros for s in pso.historial]

    print(f"Historial acumulado: {len(pso.historial)} iteraciones, "
          f"{n_individuos} individuos, {problema.ndim} dimensiones.")
    print(f"Hiperparámetros en iteración 1:  {historial_hiperparametros[0]}")
    print(f"Hiperparámetros en iteración 80: {historial_hiperparametros[-1]}")

    resultado = extraer_conocimiento(
        historial_poblacional=historial_poblacional,
        historial_fitness_individual=historial_fitness_individual,
        historial_mejor_fitness=historial_mejor_fitness,
        historial_hiperparametros=historial_hiperparametros,
        n_dimensiones=problema.ndim,
    )

    print(f"\nR² del modelo subrogado (XGBoost): {resultado.r2_modelo_subrogado:.4f}")
    print(f"Instancias de élite extraídas: {resultado.instancias_elite.shape}")
    print(f"Fitness de las instancias de élite: {resultado.fitness_instancias_elite}")
    print(f"Parámetros de escape: {resultado.parametros_escape}")

    # Verificación de calidad real: las instancias de élite deben estar
    # dentro del percentil de mejor fitness de la ventana analizada (no
    # solo tener buena atribución de Shapley en valor absoluto).
    todos_los_fitness_ventana = np.concatenate(historial_fitness_individual[-50:])
    umbral_percentil_20 = np.percentile(todos_los_fitness_ventana, 20)
    todas_dentro_del_buen_percentil = bool(np.all(
        resultado.fitness_instancias_elite <= umbral_percentil_20
    ))
    print(f"Umbral del percentil 20 de fitness (ventana): {umbral_percentil_20:.4f}")
    print(f"¿Todas las instancias de élite están dentro del buen percentil? "
          f"{todas_dentro_del_buen_percentil}")

    r2_aprobado = resultado.r2_modelo_subrogado > 0.7
    print(f"\n{'✅' if r2_aprobado else '❌'} Criterio R² > 0.7: "
          f"{'APROBADO' if r2_aprobado else 'FALLÓ'} "
          f"(obtenido: {resultado.r2_modelo_subrogado:.4f})")
    print(f"{'✅' if todas_dentro_del_buen_percentil else '❌'} Criterio de "
          f"calidad real de instancias seleccionadas: "
          f"{'APROBADO' if todas_dentro_del_buen_percentil else 'FALLÓ'}")

    return (r2_aprobado and todas_dentro_del_buen_percentil), resultado


def caso_sintetico_variable_dominante():
    """
    Caso de control: genera datos sintéticos donde se SABE, por
    construcción, que la variable 0 domina completamente el fitness
    (las demás variables son ruido sin relación con el resultado).
    FastSHAP debe asignarle a esa variable la mayor contribución |ϕ|
    promedio entre todas las dimensiones.
    """
    print()
    print("=" * 70)
    print("CASO 2 — Variable dominante conocida (control sintético)")
    print("=" * 70)

    rng = np.random.default_rng(7)
    n_dimensiones = 5
    n_iteraciones_sinteticas = 60
    n_individuos = 30

    historial_poblacional = []
    historial_fitness_individual = []
    historial_mejor_fitness = []
    historial_hiperparametros = []
    mejor_global = np.inf

    for it in range(n_iteraciones_sinteticas):
        poblacion = rng.uniform(-10, 10, size=(n_individuos, n_dimensiones))
        # Fitness depende CASI exclusivamente de la variable 0 (peso 10x
        # mayor que las demás, que son ruido), de forma determinista.
        fitness = (10.0 * poblacion[:, 0] ** 2
                   + 0.01 * np.sum(poblacion[:, 1:] ** 2, axis=1))
        historial_poblacional.append(poblacion)
        historial_fitness_individual.append(fitness)
        mejor_global = min(mejor_global, fitness.min())
        historial_mejor_fitness.append(mejor_global)
        # Hiperparámetros sintéticos simples (no provienen de un algoritmo
        # real en este caso de control, solo se simula un esquema de
        # decrecimiento similar al de PSO para poder verificar que la
        # función recupera correctamente el valor vigente en el índice
        # exacto del salto de mejora).
        historial_hiperparametros.append({"w_sintetico": 0.9 - it * 0.005})

    resultado = extraer_conocimiento(
        historial_poblacional=historial_poblacional,
        historial_fitness_individual=historial_fitness_individual,
        historial_mejor_fitness=historial_mejor_fitness,
        historial_hiperparametros=historial_hiperparametros,
        n_dimensiones=n_dimensiones,
    )

    contribucion_promedio_abs = np.mean(np.abs(resultado.valores_shapley_elite), axis=0)
    print(f"R² del modelo subrogado: {resultado.r2_modelo_subrogado:.4f}")
    print(f"Fitness de las instancias de élite seleccionadas: "
          f"{resultado.fitness_instancias_elite}")
    print(f"Contribución |ϕ| promedio por variable (élite): {contribucion_promedio_abs}")
    print(f"Parámetros de escape: {resultado.parametros_escape}")

    # Verificación de la corrección: el w_sintetico recuperado debe
    # corresponder EXACTAMENTE al valor calculado en la iteración relativa
    # reportada, según la fórmula sintética usada para generar los datos
    # (0.9 - it * 0.005), tomando en cuenta el recorte a la ventana de
    # las últimas 50 iteraciones que aplica internamente extraer_conocimiento.
    it_relativa = resultado.parametros_escape["iteracion_relativa_escape"]
    it_absoluta_en_ventana = n_iteraciones_sinteticas - 50 + it_relativa
    w_esperado = 0.9 - it_absoluta_en_ventana * 0.005
    w_recuperado = resultado.parametros_escape["hiperparametros"]["w_sintetico"]
    coincide = abs(w_esperado - w_recuperado) < 1e-9
    print(f"w_sintetico esperado en esa iteración: {w_esperado:.6f}")
    print(f"w_sintetico recuperado por la función:  {w_recuperado:.6f}")
    print(f"{'✅' if coincide else '❌'} Criterio de recuperación exacta de "
          f"hiperparámetros en el momento del escape: "
          f"{'APROBADO' if coincide else 'FALLÓ'}")

    # Verificación adicional (criterio corregido, Opción C): las instancias
    # de élite deben tener fitness real BAJO, no solo buena atribución de
    # Shapley. Se compara contra la VENTANA reciente (últimas 50
    # iteraciones), que es exactamente la misma ventana que usa
    # extraer_conocimiento() internamente — comparar contra el historial
    # completo (60 iteraciones) introduciría un desajuste artificial, ya
    # que las primeras 10 iteraciones (fuera de ventana) no participan
    # en absoluto en la selección real.
    fitness_ventana_real = np.concatenate(historial_fitness_individual[-50:])
    umbral_percentil_20 = np.percentile(fitness_ventana_real, 20)
    todas_dentro_del_buen_percentil = np.all(
        resultado.fitness_instancias_elite <= umbral_percentil_20
    )
    print(f"Umbral del percentil 20 de fitness (todo el historial): "
          f"{umbral_percentil_20:.4f}")
    print(f"¿Todas las instancias de élite están dentro del buen percentil? "
          f"{todas_dentro_del_buen_percentil}")

    variable_mas_importante = int(np.argmax(contribucion_promedio_abs))
    aprobado_atribucion = variable_mas_importante == 0
    print(f"\nVariable identificada como más importante: {variable_mas_importante} "
          f"(esperada: 0)")
    print(f"{'✅' if aprobado_atribucion else '❌'} Criterio de atribución correcta: "
          f"{'APROBADO' if aprobado_atribucion else 'FALLÓ'}")
    print(f"{'✅' if todas_dentro_del_buen_percentil else '❌'} Criterio de "
          f"calidad real de las instancias seleccionadas: "
          f"{'APROBADO' if todas_dentro_del_buen_percentil else 'FALLÓ'}")

    aprobado = aprobado_atribucion and todas_dentro_del_buen_percentil and coincide
    return aprobado


def main():
    aprobado_r2, _ = caso_real_pso_f11()
    aprobado_atribucion = caso_sintetico_variable_dominante()

    print()
    print("=" * 70)
    print("RESUMEN FASE 2 — CRITERIO DE ACEPTACIÓN")
    print("=" * 70)
    if aprobado_r2 and aprobado_atribucion:
        print("✅ Ambos casos pasaron. El módulo de extracción (Fase 2) "
              "queda validado, usando la implementación oficial de "
              "FastSHAP (iancovert/fastshap).")
    else:
        print("⚠️  Al menos un caso falló. Revisar antes de continuar a Fase 3.")


if __name__ == "__main__":
    main()