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
    calcular_referencia_mmd,
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

    # Recalibrado sobre el evaluador CEC 2022 oficial (D-9). Con opfunu eran
    # {20: 343.99, 100: 172.34, 300: 130.74}: PSO y DE convergían hacia la
    # misma región; con la F11 oficial quedan en cuencas distintas (~0.44
    # del diámetro, ver D-11).
    resultados_esperados_aprox = {20: 403.36, 100: 399.21, 300: 387.91}
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
        # orden de magnitud que lo observado con el evaluador oficial (D-9).
        esperado = resultados_esperados_aprox[it]
        diferencia_relativa = abs(resultado.distancia_wasserstein - esperado) / esperado
        if diferencia_relativa > 0.05:
            todo_coherente = False

    print(f"\n{'✅' if todo_coherente else '❌'} Coherencia con valores de referencia "
          f"(evaluador oficial, D-9): {'APROBADO' if todo_coherente else 'FALLÓ'}")
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
    print("CASO 3 — Filtro MMD por instancia (forma, no ubicación)")
    print("=" * 70)

    # Reescrito en D-11 (cierra D-1e): el filtro decide por instancia; las
    # candidatas se centran en la media de la fuente (de ahí salen las
    # élites de la Fase 2) y el ancho del kernel y el umbral (máximo del
    # nulo leave-one-out) salen del OBJETIVO. Con el objetivo colapsado el
    # filtro no es aplicable y aprueba todo.
    problema = ProblemaCEC2022(numero_funcion=11, ndim=20)
    pso = PSO(problema, 30, problema.ndim, problema.limites, 300, semilla=42)
    for _ in range(100):
        pso.ejecutar_iteracion()
    fuente = pso.poblacion.copy()
    centro = fuente.mean(axis=0)
    referencia = calcular_referencia_mmd(fuente)

    def filtrar(candidatas, objetivo):
        return filtrar_instancias_por_mmd(candidatas, fuente, objetivo,
                                          problema.limites,
                                          referencia=referencia)

    # (a) Objetivo con la misma forma que la fuente pero trasladado: la
    # ubicación la controla Wasserstein, el MMD debe aprobar todo.
    a = filtrar(fuente[:5], fuente + 300.0)
    print(f"(a) Misma forma, objetivo trasladado +300   -> aprobadas: {a.sum()}/5")

    # (b) Una élite en la esquina extrema del espacio y cuatro normales:
    # se rechaza solo la de la esquina (decisión por instancia).
    candidatas_b = fuente[:5].copy()
    candidatas_b[0] = problema.limites[:, 1]
    b = filtrar(candidatas_b, fuente)
    print(f"(b) Esquina extrema en la posición 0       -> aprobadas: "
          f"{b.astype(int).tolist()}")

    # (c) Cinco copias de la esquina (varianza cero, antiguo D-1e): cada
    # una se evalúa sola, todas rechazadas.
    c = filtrar(np.tile(problema.limites[:, 1], (5, 1)), fuente)
    print(f"(c) Cinco copias de la esquina extrema      -> aprobadas: {c.sum()}/5")

    # (d) Objetivo 5x más concentrado que la fuente (sin colapsar): las
    # élites quedan fuera de su forma y se rechazan.
    d = filtrar(fuente[:5], centro + 0.2 * (fuente - centro))
    print(f"(d) Objetivo 5x más concentrado             -> aprobadas: {d.sum()}/5")

    # (e) Objetivo 5x más disperso: las élites caen dentro de él.
    e = filtrar(fuente[:5], centro + 5.0 * (fuente - centro))
    print(f"(e) Objetivo 5x más disperso                -> aprobadas: {e.sum()}/5")

    # (f) Objetivo colapsado a un punto: filtro no aplicable, aprueba
    # todo (incluida la esquina extrema de (b)).
    f = filtrar(candidatas_b, centro + 1e-6 * (fuente - centro))
    print(f"(f) Objetivo colapsado (MMD no aplicable)   -> aprobadas: {f.sum()}/5")

    filtro_funciona = (a.all() and not b[0] and b[1:].all()
                       and not c.any() and d.sum() <= 1 and e.all()
                       and f.all())
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