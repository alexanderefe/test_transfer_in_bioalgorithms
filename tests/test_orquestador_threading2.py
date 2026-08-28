"""
Prueba del orquestador con threading (Solución A) — dos variantes de PSO.

Usa F1/D=10 con dos variantes de PSO:
- PSO A: bien afinado (fuente, converge bien)
- PSO B: mal afinado (objetivo, se estanca por inercia alta y coeficientes bajos)

Este es el mismo escenario validado en test_fase3_transferencia_exitosa.py,
donde se confirmó que:
- Las poblaciones tienen distribuciones compatibles (mismo tipo de algoritmo)
- El filtro MMD aprueba las instancias de élite
- Canal A puede transferir hiperparámetros (ambos usan {w, c1, c2})
- El algoritmo objetivo mejoró de 302.97 a 300.09
"""

import sys
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROYECTO))

from problems.cec2022_wrapper import ProblemaCEC2022
from bioalgorithms.pso import PSO
from middleware.orquestador import Orquestador


def main():
    problema = ProblemaCEC2022(numero_funcion=11, ndim=20)
    n_iteraciones = 2000

    # PSO bien afinado como algoritmo A (fuente: converge mejor en F11/D=20)
    alg_a = PSO(problema, 30, problema.ndim, problema.limites,
                n_iteraciones, semilla=42,
                w_max=0.9, w_min=0.4, c1=2.0, c2=2.0)

    # PSO mal afinado como algoritmo B (objetivo: inercia alta fija,
    # coeficientes bajos → queda atrapado en óptimos locales en F11/D=20)
    # semilla=500 verificada: PSO A mejor en 498/500 iter, varianza B≈1.37
    alg_b = PSO(problema, 30, problema.ndim, problema.limites,
                n_iteraciones, semilla=500,
                w_max=0.6, w_min=0.6, c1=0.8, c2=0.8)

    orq = Orquestador(
        algoritmo_a=alg_a,
        algoritmo_b=alg_b,
        limites=problema.limites,
        n_iteraciones=n_iteraciones,
        frecuencia_monitoreo=10,
        directorio_log="logs",
        nombre_log="prueba_threading_pso_pso",
        max_epochs_fastshap=10,
    )

    resumen = orq.ejecutar()

    print(f"\n{'='*50}")
    print(f"Fase 2 activada: {resumen.n_activaciones_fase2} veces")
    print(f"Fase 3 activada: {resumen.n_activaciones_fase3} veces")
    print(f"Canal A:    {resumen.n_transferencias_canal_a}")
    print(f"Canal B:    {resumen.n_transferencias_canal_b}")
    print(f"Abortadas:  {resumen.n_transferencias_abortadas_wasserstein}")
    print(f"Fitness A:  {resumen.fitness_final_algoritmo_a:.4f}")
    print(f"Fitness B:  {resumen.fitness_final_algoritmo_b:.4f}")
    print(f"Tiempo total:    {resumen.tiempo_total_segundos:.2f}s")
    print(f"Tiempo pausado:  {resumen.tiempo_pausado_en_middleware_segundos:.2f}s")


if __name__ == "__main__":
    main()