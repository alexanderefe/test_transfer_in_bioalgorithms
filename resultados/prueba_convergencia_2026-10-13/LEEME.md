# Prueba de los gráficos de convergencia (2026-10-13)

Figuras de prueba de `analisis/convergencia.py`, generadas con 144 corridas
reales pero **reducidas**: los 6 algoritmos × 12 funciones × D=10 ×
**MaxFES = 5 000** × **2 semillas** (idx 0 y 1). No son resultados del
experimento: sirven solo para ver el formato de las figuras. Las del
experimento salen de `python analisis/main.py` sobre la parrilla completa
(51 semillas, tiers 5·10⁴ … 5·10⁷).

| Archivo | Contenido |
|---|---|
| `panel.png` | 12 funciones, error f − f(x*), mediana + banda intercuartil |
| `panel_fitness.png` | ídem, en fitness f |
| `f06.png`, `f11.png` | una función, mediana + banda intercuartil |
| `f11_s01.png`, `f11_s01_fitness.png` | F11, semilla 1: curva de cada algoritmo (error / fitness) |
| `f11_s01_solo.png` | ídem, generada con un DataFrame que solo trae la semilla 1 |
