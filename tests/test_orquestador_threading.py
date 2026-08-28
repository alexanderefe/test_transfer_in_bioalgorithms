"""
Prueba del orquestador con threading (Solución A).

Usa F11/D=20 con 2000 iteraciones y max_epochs_fastshap=10 para que:
- Los algoritmos tarden ~8s en completar sus iteraciones
- Fase 2 tarde ~3-4s (reducido de 50 a 10 epochs)
- Sea posible ver múltiples ciclos de detección → extracción → transferencia
  → cierre → nueva detección dentro de la misma ejecución.
"""

import sys
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROYECTO))

from problems.cec2022_wrapper import ProblemaCEC2022
from bioalgorithms.pso import PSO
from bioalgorithms.de import DE
from middleware.orquestador import Orquestador


def main():
    problema = ProblemaCEC2022(numero_funcion=11, ndim=20)
    n_iteraciones = 2000

    # PSO bien afinado como algoritmo A
    alg_a = PSO(problema, 30, problema.ndim, problema.limites,
                n_iteraciones, semilla=0,
                w_max=0.9, w_min=0.4, c1=2.0, c2=2.0)

    # DE conservador como algoritmo B (propenso a estancarse)
    alg_b = DE(problema, 30, problema.ndim, problema.limites,
               n_iteraciones, semilla=42, F=0.2, CR=0.3)

    orq = Orquestador(
        algoritmo_a=alg_a,
        algoritmo_b=alg_b,
        limites=problema.limites,
        n_iteraciones=n_iteraciones,
        frecuencia_monitoreo=10,
        directorio_log="logs",
        nombre_log="prueba_threading_f11",
        max_epochs_fastshap=10,  # reducido para que Fase 2 sea ~3-4s
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