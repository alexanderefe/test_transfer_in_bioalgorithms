"""
Validación de la barrera de seguridad (Wasserstein + MMD corregido).

Reproduce los casos de control ya explorados manualmente durante el
diseño de esta fase, esta vez contra las funciones reales del módulo
middleware/barrera_seguridad.py, para confirmar que el comportamiento
documentado se mantiene de forma reproducible.
"""

import sys
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROYECTO))

import numpy as np

from problems.cec2022_wrapper import ProblemaCEC2022
from bioalgorithms.pso import PSO
from bioalgorithms.de import DE
from middleware.barrera_seguridad import (
    calcular_wasserstein,
    calcular_diametro_maximo,
    evaluar_barrera_wasserstein,
    calcular_mmd,
    calcular_umbral_mmd_dinamico,
    filtrar_instancias_por_mmd,
)


def caso_wasserstein_real():
    print("=" * 70)
    print("CASO 1 — Wasserstein real, PSO vs DE (F11, D=20)")
    print("=" * 70)

    problema = ProblemaCEC2022(numero_funcion=11, ndim=20)
    pso = PSO(problema, 30, problema.ndim, problema.limites, 300, semilla=42)
    de = DE(problema, 30, problema.ndim, problema.limites, 300, semilla=42,
            F=0.2, CR=0.3)

    diametro = calcular_diametro_maximo(problema.limites)
    print(f"Diámetro máximo del espacio (D=20, rango [-100,100]): {diametro:.4f}")

    resultados_esperados_aprox = {20: 343.99, 100: 172.34, 300: 130.74}
    todo_coherente = True
    for it in [20, 100, 300]:
        while pso.iteracion_actual < it:
            pso.ejecutar_iteracion()
        while de.iteracion_actual < it:
            de.ejecutar_iteracion()

        resultado = evaluar_barrera_wasserstein(
            pso.poblacion, de.poblacion, problema.limites
        )
        fraccion = resultado.distancia_wasserstein / diametro
        print(f"  it={it:4d} | W1={resultado.distancia_wasserstein:8.4f} "
              f"| fracción={fraccion:.4f} | umbral={resultado.umbral_wasserstein:.4f} "
              f"| permitida={resultado.transferencia_permitida}")

        # Verificación de coherencia: el valor debe estar en el mismo
        # orden de magnitud que lo observado durante el diseño manual.
        esperado = resultados_esperados_aprox[it]
        diferencia_relativa = abs(resultado.distancia_wasserstein - esperado) / esperado
        if diferencia_relativa > 0.05:
            todo_coherente = False

    print(f"\n{'✅' if todo_coherente else '❌'} Coherencia con valores observados "
          f"durante el diseño manual: {'APROBADO' if todo_coherente else 'FALLÓ'}")
    return todo_coherente


def caso_mmd_control():
    print()
    print("=" * 70)
    print("CASO 2 — MMD corregido, casos de control conocidos")
    print("=" * 70)

    rng = np.random.default_rng(0)

    poblacion_concentrada = np.tile(np.array([5.0] * 20), (30, 1)) + rng.normal(0, 0.01, (30, 20))
    poblacion_dispersa = rng.uniform(-100, 100, (30, 20))
    mmd_disimil = calcular_mmd(poblacion_concentrada, poblacion_dispersa)
    print(f"MMD (concentrada vs dispersa, claramente disímiles): {mmd_disimil:.6f}")

    pob_a = rng.uniform(-100, 100, (30, 20))
    pob_b = rng.uniform(-100, 100, (30, 20))
    mmd_similar = calcular_mmd(pob_a, pob_b)
    print(f"MMD (dispersa vs dispersa, semillas distintas, similares): {mmd_similar:.6f}")

    # El criterio de aceptación es relativo, no de valores absolutos
    # exactos: el caso disímil debe dar un MMD claramente mayor (al
    # menos 5x) que el caso similar, confirmando que el kernel corregido
    # discrimina correctamente entre ambos escenarios.
    razon = mmd_disimil / max(mmd_similar, 1e-9)
    discrimina_correctamente = razon > 5.0
    print(f"Razón disímil/similar: {razon:.2f}x")
    print(f"{'✅' if discrimina_correctamente else '❌'} Criterio de discriminación "
          f"correcta (razón > 5x): "
          f"{'APROBADO' if discrimina_correctamente else 'FALLÓ'}")

    return discrimina_correctamente


def caso_umbral_dinamico_y_filtro():
    print()
    print("=" * 70)
    print("CASO 3 — Umbral dinámico de MMD y filtro de instancias")
    print("=" * 70)

    problema = ProblemaCEC2022(numero_funcion=11, ndim=20)
    pso = PSO(problema, 30, problema.ndim, problema.limites, 300, semilla=42)
    de = DE(problema, 30, problema.ndim, problema.limites, 300, semilla=42,
            F=0.2, CR=0.3)
    for _ in range(100):
        pso.ejecutar_iteracion()
        de.ejecutar_iteracion()

    umbral = calcular_umbral_mmd_dinamico(pso.poblacion)
    print(f"Umbral dinámico de MMD (3x percentil 95 del ruido de referencia "
          f"de la población fuente): {umbral:.6f}")

    # Caso de control: instancias tomadas DIRECTAMENTE de la población
    # objetivo (deberían ser compatibles casi siempre, MMD bajo).
    instancias_compatibles = de.poblacion[:5]
    aprobadas_compatibles = filtrar_instancias_por_mmd(
        instancias_compatibles, pso.poblacion, de.poblacion
    )
    print(f"Instancias tomadas de la propia población objetivo -> "
          f"aprobadas: {aprobadas_compatibles.sum()}/5")

    # Caso de control: instancias claramente fuera de rango/distribución
    # (en una esquina extrema del espacio de búsqueda).
    rng = np.random.default_rng(1)
    instancias_incompatibles = np.tile(
        problema.limites[:, 1], (5, 1)
    )  # esquina extrema del espacio: todas las dimensiones en su máximo
    aprobadas_incompatibles = filtrar_instancias_por_mmd(
        instancias_incompatibles, pso.poblacion, de.poblacion
    )
    print(f"Instancias en la esquina extrema del espacio -> "
          f"aprobadas: {aprobadas_incompatibles.sum()}/5")

    filtro_funciona = (aprobadas_compatibles.sum() >= 4) and (aprobadas_incompatibles.sum() <= 1)
    print(f"\n{'✅' if filtro_funciona else '❌'} Criterio de filtrado correcto: "
          f"{'APROBADO' if filtro_funciona else 'FALLÓ'}")
    return filtro_funciona


def main():
    aprobados = [
        caso_wasserstein_real(),
        caso_mmd_control(),
        caso_umbral_dinamico_y_filtro(),
    ]

    print()
    print("=" * 70)
    print("RESUMEN — BARRERA DE SEGURIDAD (Wasserstein + MMD)")
    print("=" * 70)
    if all(aprobados):
        print("✅ Los 3 casos pasaron. La barrera de seguridad queda validada "
              "para integrarse con el módulo de transferencia (Fase 3).")
    else:
        print("⚠️  Al menos un caso falló. Revisar antes de continuar.")


if __name__ == "__main__":
    main()