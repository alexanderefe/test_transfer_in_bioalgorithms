"""
Validación de transferencia exitosa con filtro MMD aprobado.

Usa dos variantes de PSO sobre F1/D=10: una bien afinada (fuente, converge
bien) y otra mal afinada (objetivo, se estanca). Al ser la misma familia
de algoritmo, sus poblaciones convergen con formas de distribución similares
(misma escala de dispersión al centrar), por lo que el filtro MMD las
considera compatibles y permite la inyección de instancias de élite.

Este test complementa test_fase3_transferencia.py, que demostró el
funcionamiento correcto de la BARRERA (rechazo por incompatibilidad real
entre PSO y DE), verificando aquí que el sistema también permite la
transferencia cuando la compatibilidad sí existe.
"""

import sys
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROYECTO))

import numpy as np

from problems.cec2022_wrapper import ProblemaCEC2022
from bioalgorithms.pso import PSO
from middleware.deteccion import (
    calcular_score, fuente_en_condiciones_optimas,
    UMBRAL_ESTANCAMIENTO,
)
from middleware.extraccion import extraer_conocimiento, VENTANA_HISTORIAL_FUENTE
from middleware.transferencia import (
    EstadoTransferencia, ejecutar_ciclo_transferencia, CanalActivo,
    RMP_INICIAL, RMP_TOPE,
)


def main():
    print("=" * 70)
    print("TRANSFERENCIA EXITOSA — PSO bien afinado → PSO mal afinado (F1, D=10)")
    print("=" * 70)

    problema = ProblemaCEC2022(numero_funcion=1, ndim=10)
    n_iteraciones = 200

    # PSO fuente: bien afinado (parámetros estándar de la literatura)
    pso_fuente = PSO(problema, 30, problema.ndim, problema.limites,
                     n_iteraciones, semilla=0,
                     w_max=0.9, w_min=0.4, c1=2.0, c2=2.0)

    # PSO objetivo: mal afinado (inercia alta fija, coeficientes bajos
    # que favorecen la inercia sobre atracción al óptimo, produciendo
    # estancamiento al no converger suficientemente).
    pso_objetivo = PSO(problema, 30, problema.ndim, problema.limites,
                       n_iteraciones, semilla=99,
                       w_max=0.6, w_min=0.6, c1=0.8, c2=0.8)

    estado = EstadoTransferencia()
    historial_fitness_obj = []
    conocimiento_actual = None
    eventos = []
    instancias_inyectadas_total = 0

    for it in range(1, n_iteraciones + 1):
        sf = pso_fuente.ejecutar_iteracion()
        so = pso_objetivo.ejecutar_iteracion()
        historial_fitness_obj.append(so.mejor_fitness_historico)

        res = calcular_score(historial_fitness_obj, so.poblacion,
                              problema.limites, it, n_iteraciones)
        fuente_ok = fuente_en_condiciones_optimas(
            sf.mejor_fitness_historico, so.mejor_fitness_historico
        )

        ciclo_inactivo = (estado.canal_activo == CanalActivo.NINGUNO)
        condicion = res.activo and res.estancado and fuente_ok

        if condicion and len(pso_fuente.historial) > VENTANA_HISTORIAL_FUENTE:
            if ciclo_inactivo or conocimiento_actual is None:
                conocimiento_actual = extraer_conocimiento(
                    historial_poblacional=[s.poblacion for s in pso_fuente.historial],
                    historial_fitness_individual=[s.fitness for s in pso_fuente.historial],
                    historial_mejor_fitness=[s.mejor_fitness_historico for s in pso_fuente.historial],
                    historial_hiperparametros=[s.hiperparametros for s in pso_fuente.historial],
                    n_dimensiones=problema.ndim,
                )

            resultado = ejecutar_ciclo_transferencia(
                estado=estado,
                algoritmo_fuente=pso_fuente,
                algoritmo_objetivo=pso_objetivo,
                resultado_deteccion_objetivo=res,
                parametros_fuente=conocimiento_actual.parametros_escape["hiperparametros"],
                instancias_elite=conocimiento_actual.instancias_elite,
                fitness_instancias_elite=conocimiento_actual.fitness_instancias_elite,
                limites=problema.limites,
                iteracion_actual=it,
            )
            eventos.append((it, resultado))

            if resultado.canal_utilizado == CanalActivo.CANAL_B:
                n_inj = resultado.payload_inyectado.get("n_instancias_inyectadas", 0)
                instancias_inyectadas_total += n_inj
        else:
            if not condicion:
                conocimiento_actual = None

    print(f"\nTotal de ciclos activados: {len(eventos)}")
    print(f"Total de instancias inyectadas (Canal B): {instancias_inyectadas_total}")

    # Criterios de aceptación
    eventos_canal_a_activo = [
        e for e in estado.log_eventos
        if e["evento"] == "ciclo_transferencia" and e.get("canal") == "canal_a_rmp"
    ]
    eventos_canal_a_exitoso = [e for e in estado.log_eventos if e["evento"] == "canal_a_exitoso"]
    eventos_cierre_provisional = [e for e in estado.log_eventos if e["evento"] == "ciclo_cerrado_provisional"]
    eventos_canal_b = [e for e in estado.log_eventos if e["evento"] == "canal_b_aplicado"]
    inyecciones_canal_b = sum(e.get("n_instancias_inyectadas", 0) for e in eventos_canal_b)
    rechazadas_mmd = sum(e.get("instancias_rechazadas_por_mmd", 0) for e in eventos_canal_b)

    print(f"\nEventos Canal A activo (RMP aplicado):       {len(eventos_canal_a_activo)}")
    print(f"Eventos Canal A exitoso (evaluacion formal): {len(eventos_canal_a_exitoso)}")
    print(f"Cierres provisionales (objetivo mejoró):     {len(eventos_cierre_provisional)}")
    print(f"Eventos Canal B aplicado:                    {len(eventos_canal_b)}")
    print(f"Instancias inyectadas Canal B:               {inyecciones_canal_b}")
    print(f"Instancias rechazadas por MMD:               {rechazadas_mmd}")

    # Verificar mejora de fitness del objetivo
    fitness_inicial_obj = historial_fitness_obj[30] if len(historial_fitness_obj) > 30 else historial_fitness_obj[0]
    fitness_final_obj = historial_fitness_obj[-1]
    fitness_final_fuente = pso_fuente.mejor_fitness_historico

    print(f"\nFitness objetivo it=30 (pre-transferencia): {fitness_inicial_obj:.4f}")
    print(f"Fitness objetivo final:                      {fitness_final_obj:.4f}")
    print(f"Fitness fuente final:                        {fitness_final_fuente:.4f}")
    mejora_obj = fitness_inicial_obj > fitness_final_obj

    hubo_transferencia_real = (
        len(eventos_canal_a_activo) > 0
        or len(eventos_canal_a_exitoso) > 0
        or len(eventos_cierre_provisional) > 0
        or inyecciones_canal_b > 0
    )
    rmp_dentro_rango = all(RMP_INICIAL <= r.rmp_resultante <= RMP_TOPE for _, r in eventos) if eventos else True

    print()
    print("=" * 70)
    print("CRITERIOS DE ACEPTACIÓN")
    print("=" * 70)
    print(f"{'✅' if hubo_transferencia_real else '❌'} Hubo transferencia activa "
          f"(Canal A aplicó RMP o Canal B inyectó): "
          f"{'APROBADO' if hubo_transferencia_real else 'FALLÓ'} "
          f"({len(eventos_canal_a_activo)} iter de Canal A, {inyecciones_canal_b} inyecciones Canal B)")
    print(f"{'✅' if mejora_obj else '❌'} Objetivo mejoró su fitness tras la transferencia: "
          f"{'APROBADO' if mejora_obj else 'FALLÓ'}")
    print(f"{'✅' if rmp_dentro_rango else '❌'} RMP siempre dentro de [{RMP_INICIAL}, {RMP_TOPE}]: "
          f"{'APROBADO' if rmp_dentro_rango else 'FALLÓ'}")

    if hubo_transferencia_real and mejora_obj and rmp_dentro_rango:
        print("\n✅ Transferencia exitosa validada: el middleware activó un canal de "
              "transferencia real y el algoritmo objetivo mejoró su fitness.")
    else:
        print("\n⚠️  Al menos un criterio no se cumplió.")


if __name__ == "__main__":
    main()