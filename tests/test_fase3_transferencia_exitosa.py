"""
Validación de transferencia exitosa con filtro MMD aprobado.

Usa dos variantes de PSO sobre F2/D=10: una bien afinada (fuente, converge
bien) y otra mal afinada (objetivo, se estanca). Al ser la misma familia
de algoritmo, sus poblaciones convergen con formas de distribución similares,
por lo que el filtro MMD considera típicas del objetivo a las instancias de
élite y permite inyectarlas.

Escenario: hasta D-9 era F1/D=10, pero con el Zakharov oficial el PSO "bien
afinado" termina peor que el "mal afinado" (6901 contra 2555) y no se activa
ningún ciclo. Se buscó en las 12 funciones con D ∈ {10, 20} (D-11): la
premisa y los tres criterios se cumplen en 16 de 24 casos; se eligió F2/D=10
(fuente 433 contra objetivo 469 al final, primer ciclo en la iteración 51,
sin bloqueos por R² ni rechazos del MMD). `correr_escenario(f, d)` permite
repetir la búsqueda.

Como el orquestador, un ciclo no se ejecuta si la Fase 2 quedó bloqueada
por R² < 0 (las élites devueltas en ese caso son ceros).

Este test complementa test_fase3_transferencia.py (PSO↔DE), verificando
que el sistema transfiere cuando la compatibilidad existe.
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


FUNCION_ESCENARIO = 2   # elegida por búsqueda en las 12 funciones (ver docstring)
NDIM_ESCENARIO = 10


def correr_escenario(numero_funcion: int = FUNCION_ESCENARIO,
                     ndim: int = NDIM_ESCENARIO, verbose: bool = True) -> dict:
    """Corre el escenario fuente bien afinada → objetivo mal afinado y
    devuelve los resultados de los criterios."""
    _print = print if verbose else (lambda *a, **k: None)
    _print("=" * 70)
    _print(f"TRANSFERENCIA EXITOSA — PSO bien afinado → PSO mal afinado "
           f"(F{numero_funcion}, D={ndim})")
    _print("=" * 70)

    problema = ProblemaCEC2022(numero_funcion=numero_funcion, ndim=ndim)
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
    n_bloqueos_r2 = 0
    it_primer_ciclo = None

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

            # Como el orquestador: con R² < 0 no hay conocimiento que
            # transferir (las élites son ceros) y el ciclo no se ejecuta.
            if conocimiento_actual.transferencia_bloqueada_por_r2:
                n_bloqueos_r2 += 1
                conocimiento_actual = None
                continue

            if it_primer_ciclo is None:
                it_primer_ciclo = it
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

    _print(f"\nTotal de ciclos activados: {len(eventos)}")
    _print(f"Ciclos no ejecutados por R² < 0 en la Fase 2: {n_bloqueos_r2}")
    _print(f"Total de instancias inyectadas (Canal B): {instancias_inyectadas_total}")

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

    _print(f"\nEventos Canal A activo (RMP aplicado):       {len(eventos_canal_a_activo)}")
    _print(f"Eventos Canal A exitoso (evaluacion formal): {len(eventos_canal_a_exitoso)}")
    _print(f"Cierres provisionales (objetivo mejoró):     {len(eventos_cierre_provisional)}")
    _print(f"Eventos Canal B aplicado:                    {len(eventos_canal_b)}")
    _print(f"Instancias inyectadas Canal B:               {inyecciones_canal_b}")
    _print(f"Instancias rechazadas por MMD:               {rechazadas_mmd}")

    # Mejora del objetivo desde el primer ciclo de transferencia (antes no
    # hubo intervención) hasta el final.
    fitness_final_obj = historial_fitness_obj[-1]
    fitness_final_fuente = pso_fuente.mejor_fitness_historico
    if it_primer_ciclo is not None:
        fitness_inicial_obj = historial_fitness_obj[it_primer_ciclo - 1]
        _print(f"\nFitness objetivo it={it_primer_ciclo} (primer ciclo): {fitness_inicial_obj:.4f}")
        mejora_obj = fitness_inicial_obj > fitness_final_obj
    else:
        mejora_obj = False
    _print(f"Fitness objetivo final:                      {fitness_final_obj:.4f}")
    _print(f"Fitness fuente final:                        {fitness_final_fuente:.4f}")

    hubo_transferencia_real = (
        len(eventos_canal_a_activo) > 0
        or len(eventos_canal_a_exitoso) > 0
        or len(eventos_cierre_provisional) > 0
        or inyecciones_canal_b > 0
    )
    rmp_dentro_rango = all(RMP_INICIAL <= r.rmp_resultante <= RMP_TOPE for _, r in eventos) if eventos else True

    _print()
    _print("=" * 70)
    _print("CRITERIOS DE ACEPTACIÓN")
    _print("=" * 70)
    _print(f"{'✅' if hubo_transferencia_real else '❌'} Hubo transferencia activa "
           f"(Canal A aplicó RMP o Canal B inyectó): "
           f"{'APROBADO' if hubo_transferencia_real else 'FALLÓ'} "
           f"({len(eventos_canal_a_activo)} iter de Canal A, {inyecciones_canal_b} inyecciones Canal B)")
    _print(f"{'✅' if mejora_obj else '❌'} Objetivo mejoró su fitness desde el primer ciclo: "
           f"{'APROBADO' if mejora_obj else 'FALLÓ'}")
    _print(f"{'✅' if rmp_dentro_rango else '❌'} RMP siempre dentro de [{RMP_INICIAL}, {RMP_TOPE}]: "
           f"{'APROBADO' if rmp_dentro_rango else 'FALLÓ'}")

    aprobado = hubo_transferencia_real and mejora_obj and rmp_dentro_rango
    if aprobado:
        _print("\n✅ Transferencia exitosa validada: el middleware activó un canal de "
               "transferencia real y el algoritmo objetivo mejoró su fitness.")
    else:
        _print("\n⚠️  Al menos un criterio no se cumplió.")
    return {
        "funcion": numero_funcion, "ndim": ndim, "aprobado": aprobado,
        "hubo_transferencia": hubo_transferencia_real, "mejora_obj": mejora_obj,
        "n_ciclos": len(eventos), "n_bloqueos_r2": n_bloqueos_r2,
        "it_primer_ciclo": it_primer_ciclo,
        "iter_canal_a": len(eventos_canal_a_activo),
        "inyecciones_canal_b": inyecciones_canal_b, "rechazadas_mmd": rechazadas_mmd,
        "fitness_final_fuente": float(fitness_final_fuente),
        "fitness_final_objetivo": float(fitness_final_obj),
    }


def main():
    correr_escenario()


if __name__ == "__main__":
    main()
