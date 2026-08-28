"""
Validación de la Fase 0 (criterio de aceptación).

Antes de construir el middleware, hay que demostrar dos cosas sobre el
par de algoritmos elegido:

1. PSO (fuente) alcanza buen desempeño de forma consistente.
2. DE (objetivo), con configuración conservadora, efectivamente se estanca
   en al menos una función multimodal del CEC 2022 — es decir, su
   diversidad poblacional decae y su fitness deja de mejorar antes de
   acabarse el presupuesto de iteraciones.

Si DE no se estanca, el par no es adecuado para validar el middleware,
porque no habría ningún problema que transferir conocimiento pueda
resolver. Este script corrobora ese supuesto antes de avanzar a la Fase 1.
"""

import sys
from pathlib import Path

# Agrega la raíz del proyecto (Fase 3\) al sys.path calculándola desde la
# ubicación de este mismo archivo. Esto permite ejecutar el script con
# "python .\tests\test_fase0_algoritmos_base.py" desde cualquier carpeta,
# sin depender de PYTHONPATH ni de desde dónde se invoque el comando.
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROYECTO))

import numpy as np
from problems.cec2022_wrapper import ProblemaCEC2022
from bioalgorithms.pso import PSO
from bioalgorithms.de import DE


def ejecutar_algoritmo(algoritmo, n_iteraciones: int):
    """Ejecuta n_iteraciones y retorna el historial de mejor fitness."""
    mejores = []
    for _ in range(n_iteraciones):
        snapshot = algoritmo.ejecutar_iteracion()
        mejores.append(snapshot.mejor_fitness_historico)
    return mejores


def diversidad_dimension_wise(poblacion: np.ndarray) -> float:
    """
    Versión simplificada de Dimension-Wise Diversity (ecuación 2 del
    documento de tesis), usada aquí solo para diagnóstico de la Fase 0,
    NO es todavía la implementación formal de Fase 1.
    """
    mediana = np.median(poblacion, axis=0)
    rango = poblacion.max(axis=0) - poblacion.min(axis=0)
    rango[rango == 0] = 1.0  # evitar división por cero
    dispersión = np.abs(poblacion - mediana) / rango
    return float(np.mean(dispersión))


def main():
    # F11 es función de composición (máxima multimodalidad del benchmark),
    # con D=20 para aumentar la dificultad del espacio de búsqueda y
    # forzar estancamiento real en un DE con parámetros conservadores.
    problema = ProblemaCEC2022(numero_funcion=11, ndim=20)
    print(f"Problema seleccionado: {problema}")
    print(f"Óptimo global conocido (bias): {problema.optimo_global}\n")

    n_individuos = 30
    n_iteraciones = 300
    semilla = 42

    print("=" * 60)
    print("PSO (algoritmo FUENTE)")
    print("=" * 60)
    pso = PSO(
        funcion_objetivo=problema,
        n_individuos=n_individuos,
        n_dimensiones=problema.ndim,
        limites=problema.limites,
        max_iteraciones=n_iteraciones,
        semilla=semilla,
    )
    historial_pso = ejecutar_algoritmo(pso, n_iteraciones)
    print(f"Fitness inicial:  {historial_pso[0]:.4f}")
    print(f"Fitness final:    {historial_pso[-1]:.4f}")
    print(f"Mejora total:     {historial_pso[0] - historial_pso[-1]:.4f}")
    diversidad_final_pso = diversidad_dimension_wise(pso.poblacion)
    print(f"Diversidad final (DWD simplificada): {diversidad_final_pso:.4f}")

    print()
    print("=" * 60)
    print("DE (algoritmo OBJETIVO) — configuración conservadora")
    print("=" * 60)
    de = DE(
        funcion_objetivo=problema,
        n_individuos=n_individuos,
        n_dimensiones=problema.ndim,
        limites=problema.limites,
        max_iteraciones=n_iteraciones,
        semilla=semilla,
        F=0.2,   # muy bajo: mutación diferencial casi nula, poca exploración
        CR=0.3,  # bajo: poco intercambio de información entre dimensiones
    )
    historial_de = ejecutar_algoritmo(de, n_iteraciones)
    print(f"Fitness inicial:  {historial_de[0]:.4f}")
    print(f"Fitness final:    {historial_de[-1]:.4f}")
    print(f"Mejora total:     {historial_de[0] - historial_de[-1]:.4f}")
    diversidad_final_de = diversidad_dimension_wise(de.poblacion)
    print(f"Diversidad final (DWD simplificada): {diversidad_final_de:.4f}")

    # Diagnóstico de estancamiento: ¿en qué punto dejó de mejorar DE?
    print()
    print("-" * 60)
    print("Diagnóstico de estancamiento de DE (ventana de 20 iteraciones)")
    print("-" * 60)
    ventana = 20
    for i in range(ventana, n_iteraciones, ventana):
        mejora_ventana = historial_de[i - ventana] - historial_de[i]
        print(f"Iteración {i:4d} | mejora en últimas {ventana} iter: {mejora_ventana:.6f}")

    print()
    print("=" * 60)
    print("CRITERIO DE ACEPTACIÓN FASE 0")
    print("=" * 60)
    mejora_ultima_ventana_de = historial_de[-ventana] - historial_de[-1]
    # Estancamiento relativo: la mejora de la ventana es insignificante
    # comparada con la distancia que aún falta recorrer hasta el óptimo
    # conocido. Esto es más robusto que un umbral absoluto fijo, porque
    # la escala de fitness varía mucho entre funciones del benchmark.
    distancia_al_optimo = historial_de[-1] - problema.optimo_global
    mejora_relativa = mejora_ultima_ventana_de / max(distancia_al_optimo, 1e-9)
    estancado = mejora_relativa < 0.05  # mejora menor al 5% de lo que falta
    print(f"¿DE se estancó en las últimas {ventana} iteraciones? "
          f"{'SÍ' if estancado else 'NO'} "
          f"(mejora={mejora_ultima_ventana_de:.6f}, "
          f"distancia al óptimo={distancia_al_optimo:.6f}, "
          f"mejora relativa={mejora_relativa:.4f})")

    # El criterio relevante no es que PSO le gane a DE en valor absoluto,
    # sino que exista margen real de mejora sin explotar: si DE está
    # estancado lejos del óptimo conocido, hay espacio para que la
    # transferencia de conocimiento (vía middleware) lo desbloquee.
    margen_de_mejora = distancia_al_optimo > 1.0
    print(f"¿Existe margen de mejora respecto al óptimo conocido? "
          f"{'SÍ' if margen_de_mejora else 'NO'} "
          f"(DE={historial_de[-1]:.4f}, óptimo={problema.optimo_global})")

    if estancado and margen_de_mejora:
        print("\n✅ Par fuente-objetivo VALIDADO para continuar con Fase 1.")
    else:
        print("\n⚠️  El par no muestra el patrón esperado. Revisar parámetros "
              "(F, CR de DE, o número de iteraciones) antes de continuar.")


if __name__ == "__main__":
    main()