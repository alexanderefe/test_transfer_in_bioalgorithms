"""
Prueba del orquestador con threading (Solución A) usando MaxFES como
criterio de parada — la vía exigida por el setup experimental de la tesis.

A diferencia de test_orquestador_threading.py (que corre por número fijo de
iteraciones), aquí:

- El criterio de parada es el número TOTAL de evaluaciones de la función
  objetivo (MaxFES), agregado sobre los dos algoritmos que corren en
  paralelo. Lo cuenta ProblemaCEC2022.fes (contador thread-safe compartido).
- La inercia decreciente de PSO se calcula contra su cuota estimada del
  presupuesto (MaxFES * fraccion_presupuesto, 0.5 por defecto — exacto,
  ver D-2: el planificador determinista reparte 50/50).
- El warm-up del 15%, la ventana de Canal A y el cooldown se miden en
  evaluaciones agregadas, no en iteraciones.

MAX_FES = 120_000 ≈ 2000 generaciones * 60 FES/generación (30 individuos por
algoritmo × 2 algoritmos), para que el tiempo de corrida sea comparable al
test por iteraciones y se alcancen a ver varios ciclos de detección →
extracción → transferencia.
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
    MAX_FES = 120_000
    POP = 30

    # PSO bien afinado como algoritmo A. `semilla` DEBE pasarse como keyword
    # y `max_iteraciones` se omite: en la vía MaxFES el 5º posicional
    # (max_iteraciones) chocaría con `semilla`.
    alg_a = PSO(problema, POP, problema.ndim, problema.limites, semilla=0,
                w_max=0.9, w_min=0.4, c1=2.0, c2=2.0,
                max_fes=MAX_FES)  # fraccion_presupuesto = 0.5 por defecto (D-2)

    # DE conservador como algoritmo B (propenso a estancarse)
    alg_b = DE(problema, POP, problema.ndim, problema.limites, semilla=42,
               F=0.2, CR=0.3, max_fes=MAX_FES)

    orq = Orquestador(
        algoritmo_a=alg_a,
        algoritmo_b=alg_b,
        limites=problema.limites,
        problema=problema,
        max_fes=MAX_FES,
        frecuencia_monitoreo_fes=600,   # ≈ cada 10 generaciones agregadas
        directorio_log="logs",
        nombre_log="prueba_threading_maxfes",
        max_epochs_fastshap=10,
    )

    resumen = orq.ejecutar()

    print(f"\n{'='*50}")
    print(f"MaxFES:              {resumen.max_fes}")
    print(f"FES consumidas:      {resumen.fes_consumidas_total} "
          f"(A={resumen.fes_consumidas_a}, B={resumen.fes_consumidas_b})")
    print(f"Generaciones:        {resumen.n_iteraciones_totales}")
    print(f"Fase 2 activada:     {resumen.n_activaciones_fase2} veces")
    print(f"Fase 3 activada:     {resumen.n_activaciones_fase3} veces")
    print(f"Canal A:             {resumen.n_transferencias_canal_a}")
    print(f"Canal B:             {resumen.n_transferencias_canal_b}")
    print(f"Abortadas:           {resumen.n_transferencias_abortadas_wasserstein}")
    print(f"Fitness A:           {resumen.fitness_final_algoritmo_a:.4f}")
    print(f"Fitness B:           {resumen.fitness_final_algoritmo_b:.4f}")
    print(f"Mejor del sistema:   {resumen.mejor_fitness_sistema:.4f}")
    print(f"Óptimo f(x*):        {resumen.optimo_conocido:.4f}")
    print(f"Error del sistema:   {resumen.error_sistema:.4e}  (0 = óptimo alcanzado)")
    print(f"Tiempo total:        {resumen.tiempo_total_segundos:.2f}s")

    # ── Verificaciones ───────────────────────────────────────────────────
    margen = 2 * POP  # sobrepaso máximo tolerado: ~1 generación por hilo
    assert resumen.fes_consumidas_total <= MAX_FES + margen, (
        f"Sobrepaso excesivo: {resumen.fes_consumidas_total} > "
        f"{MAX_FES} + {margen}"
    )
    assert resumen.fes_consumidas_total >= MAX_FES - margen, (
        f"Se detuvo demasiado pronto: {resumen.fes_consumidas_total} < "
        f"{MAX_FES} - {margen}"
    )
    assert (resumen.fes_consumidas_a + resumen.fes_consumidas_b
            == resumen.fes_consumidas_total), (
        "El contador global no coincide con la suma de contadores por algoritmo."
    )
    assert resumen.mejor_fitness_sistema == min(
        resumen.fitness_final_algoritmo_a, resumen.fitness_final_algoritmo_b
    )
    assert resumen.optimo_conocido == 2600.0, "F11 tiene f(x*) = 2600"
    assert abs(resumen.error_sistema
               - (resumen.mejor_fitness_sistema - 2600.0)) < 1e-6
    assert resumen.error_sistema >= 0.0
    print("\n✅ Verificaciones OK")


if __name__ == "__main__":
    main()
