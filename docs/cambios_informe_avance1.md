# Registro de cambios — `Avance 1/4.trabajo.tex` (y archivos relacionados)

Registro de todo lo hecho sobre el informe LaTeX en esta sesión de trabajo, para dar
continuidad si se retoma en otra conversación. Los números de sección citados abajo
son los que tenía el documento al cierre de esta sesión — pueden correrse si se
agregan o quitan secciones después.

## Archivos tocados

- `Avance 1/0.main.tex`: agregado `\usepackage{tikz}` +
  `\usetikzlibrary{arrows.meta, positioning, calc}`; descomentado
  `\input{5.anexos}` (antes estaba desactivado).
- `Avance 1/4.trabajo.tex`: el grueso de los cambios (detalle abajo).
- `Avance 1/5.anexos.tex`: reescrito — se descartó casi toda la plantilla
  genérica del profesor (Requerimientos, Diseño de software, Pruebas,
  Despliegue, Investigación del Problema, Diseño de la Solución) por no
  aplicar a esta tesis cuantitativa. Solo queda `Apéndice A. Validación de la
  Solución` con `A.1 Datos Recolectados` (2 tablas landscape con el detalle
  de ranking por función, D=10 y D=20) y `A.2 Análisis Descriptivo`
  (referencia al CSV de 288 filas de material suplementario, no reproducido
  íntegro por extensión).
- `Avance 1/refs.bib`: 6 entradas nuevas — `kennedy1995`, `storn1997`,
  `friedman1937`, `wilcoxon1945`, `demsar2006`, `shaffer1986`.
- `Avance 1/images/analisis/`: carpeta nueva con 10 PNG copiados de
  `resultados/analisis/figuras/` (8 diagramas CD + 2 gráficos de barras de
  rangos), usados en la sección de Análisis de Resultados.
- `.vscode/settings.json`: recipe de LaTeX Workshop
  `pdflatex → bibtex → pdflatex × 2` (sin `latexmk`, evita la dependencia de
  Perl).

## Secciones nuevas agregadas

Todas basadas en la batería de **14 688 corridas ya ejecutada**
(`git:892f6d2|eval:numba`, MaxFES∈{5 000, 50 000, 500 000, 5 000 000}), no en
el protocolo futuro post-D-6.

### §4.7 "Preguntas de Investigación" (dentro de Definición del Problema)

Pregunta Principal + 4 Preguntas Específicas, con el texto literal recuperado
de una versión anterior del informe que el usuario compartió como PDF
(`informe_avance_2_farias_alexander.pdf`). Antes el documento citaba "PP" y
"PEᵢ" en las tablas de variables sin haberlas definido en ningún lado.

### §8 "Implementación"

- **8.1 Esquema de Alto Nivel**: las 5 capas del pipeline
  (`problems/ → bioalgorithms/ → middleware/ → experimentos/ → analisis/`),
  incluyendo el backend Numba validado por paridad numérica contra `opfunu`.
  Incluye **Figura 1** (diagrama TikZ de arquitectura, ver más abajo).
- **8.2 Descripción del Procedimiento**: la grilla real ejecutada (3
  algoritmos × 12 funciones × 2 dim × 4 MaxFES × 51 semillas), el
  procedimiento de `corrida.py`, paralelización/resumibilidad de `grid.py`
  (`ProcessPoolExecutor`), manejo de fallos (0 de 14 688), consolidación vía
  `consolidar.py`. Incluye **Figura 2** (flowchart TikZ del procedimiento).
- **8.3 Requisitos de Hardware**: referencia a la sección de hardware
  existente + dato real calculado desde el parquet: **358.3 horas-CPU**
  acumuladas, **86.1 horas de reloj reales** (medidas desde `timestamp_utc`,
  sin fechas específicas en el texto por pedido del usuario), paralelismo
  efectivo ≈4.2× sobre 7 workers.

### §9 "Pruebas y Experimentos"

- **9.1 Comparación entre lo Planificado y lo Ejecutado**: tabla con 4 filas
  (criterio de parada N=2000→MaxFES; repeticiones 30→51 semillas; grupos de
  comparación ablación→D-4 baseline; escala ≈2 880→14 688 corridas). La cifra
  "≈2 880" es una estimación calculada por mí, no un número explícito del
  documento original — se lo advertí al usuario.
- **9.2 Elementos que se Mantuvieron sin Cambios**: benchmark CEC2022, D∈{10,20},
  α=0.05, hiperparámetros de PSO/DE, umbral S<0.2.

### §10 "Análisis de Resultados"

- **10.1 Rangos Promedio y Posición Final**: tabla resumen (8 configuraciones)
  + **Figura 3** (gráficos de barras de rangos D10/D20, PNG existentes).
- **10.2 Significancia Estadística**: tabla de Friedman (8 configs, todas
  rechazan H₀) + tabla de % significativo según Shaffer + **Figura 4** (grilla
  de los 8 diagramas de Diferencia Crítica/Nemenyi, PNG existentes). Incluye
  la discusión de que en D10/fes=5000 las 3 comparaciones son significativas
  pero en D20/fes=5000 el 0% lo es (el rechazo de Friedman ahí viene de
  PSO-vs-DE, no del propuesto).
- **10.3 Comparación por Pares (Wilcoxon)**: tabla condensada b/e/w para los
  3 pares, por configuración.
- **10.4 Diagnóstico del Comportamiento del Middleware**: tabla D-1d (%
  corridas sin activación de Fase 2 por MaxFES — 35.4% a fes=5000, <4% desde
  fes=50000), tabla de uso Canal A/B (Canal A = 0 en las 4 896 corridas,
  resultado esperado por el diseño heterogéneo PSO+DE), tabla de R² del
  modelo subrogado por función (mediana como lectura robusta, medias
  distorsionadas por outliers extremos en f8/f9).

### §11 "Discusión" (separada de Análisis de Resultados a pedido del usuario)

- **11.1 Discusión de los Resultados**: tabla de robustez (media/std del
  rango por algoritmo a través de las 8 configs) + los 4 puntos de discusión
  obligatoria de `docs/setup_experimental.md` §9 (dónde destaca/es inferior,
  variación D10 vs D20, componente inerte según MaxFES, robustez).
- **11.2 Validación de Hipótesis y Respuesta a las Preguntas de
  Investigación**: veredicto explícito (se rechaza H₀, se acepta H₁
  condicionalmente al presupuesto) + respuesta a PP y PE1–PE4 una por una.
  Para PE3 calculé un dato nuevo no explorado antes: de las 25 254
  transferencias de Canal B intentadas en toda la batería, **0 fueron
  abortadas por la barrera de Wasserstein** (el filtro MMD parece bastar solo
  para este par de algoritmos). Para PE1 se es honesto: no hay *ground truth*
  para medir precisión formal del Score de estancamiento, queda como
  limitación/trabajo futuro.

## Marco Conceptual (§2) — 5 subsecciones nuevas

Conceptos que se usaban extensamente en el resto del trabajo pero nunca se
habían definido:

1. **Optimización por Enjambre de Partículas (PSO)** — ecuaciones de
   velocidad/posición, $w$, $c_1$, $c_2$ (cita: Kennedy & Eberhart 1995).
2. **Evolución Diferencial (DE)** — mutación/cruce/selección, $F$, $CR$,
   variante DE/rand/1/bin (cita: Storn & Price 1997).
3. **Gradient Boosting y XGBoost** — árboles aditivos sobre residuos,
   regularización y aproximación Hessiana (cita: Chen & Guestrin 2016, ya
   existía en `refs.bib`).
4. **Coeficiente de Determinación ($R^2$)** — fórmula, interpretación, por
   qué puede ser negativo (clave para el criterio de bloqueo de transferencia
   ya descrito en Solución Propuesta).
5. **Pruebas Estadísticas No Paramétricas para Comparación de Algoritmos** —
   Friedman, post-hoc de Shaffer, Diagrama de Diferencia Crítica (Nemenyi),
   Wilcoxon de rangos con signo, con sus citas originales.

**Excluido a propósito**: el tamaño del efecto $A_{12}$ de Vargha-Delaney —
se mencionaba como métrica planificada en Métodos del Estudio pero nunca se
calculó en el análisis final, así que no se le agregó marco conceptual.

## Corrección arquitectónica mayor: threading → planificador cooperativo

**Motivo**: la solución propuesta cambió antes de las pruebas. El informe
describía el middleware funcionando con 3 hilos reales de Python
(`threading.Thread` + `threading.Event` para pausar/reanudar), pero el
código real (`middleware/orquestador.py`, decisión **D-2** documentada en su
propio docstring) reemplazó eso por un **planificador cooperativo
determinista de un solo hilo** — un bucle que alterna estrictamente una
generación de A, una de B, e invoca el ciclo del Orquestador entre medio.
Motivo del cambio real: el threading con GIL no daba paralelismo real, y el
scheduling del SO hacía que la misma semilla diera resultados distintos
entre corridas (~70/30 emergente en vez de 50/50), violando la
reproducibilidad exigida por el protocolo. **A pedido del usuario, esta
historia (que se intentó threading y se descartó) NO se narra en el
documento** — simplemente se describe el diseño final como si fuera el
único considerado.

Pasajes corregidos (~15 en total, en Solución Propuesta, Hardware y
software, Limitaciones, Implementación, Casos de Estudio, Análisis de
Resultados y Discusión): "tres hilos"/"Hilo A/B/Middleware" →
"planificador cooperativo"/"Algoritmo A/B + Orquestador"; se eliminó el
párrafo completo "Consideración sobre el GIL"; la razón de diseño "Threading
nativo" (por qué Python) pasó a ser "Determinismo de ejecución"; la
limitación "Pseudo-paralelismo" pasó a ser "Sin concurrencia real entre A y
B" (con el costo real: tiempo de reloj ≈ suma de costos de ambos
algoritmos); toda mención suelta de "ejecutados en paralelo/paralelamente" o
"concurrente" referida a este middleware se cambió a "cooperativo",
"alternado" o "de forma conjunta".

## Diagramas recreados en TikZ (reemplazan PNGs con la arquitectura vieja)

Los 3 PNG originales (`images/arq_middleware.png`, `images/diagrama_secuencia.png`,
`images/uml_clases_legible.png`) mostraban visualmente "Hilo A/B/Middleware"
con eventos de pausa — quedaron obsoletos por el cambio de arquitectura y no
son editables. Se recrearon como TikZ vectorial dentro del propio `.tex`:

1. **Figura "Arquitectura de middleware"** (dentro de §Solución Propuesta):
   banner del planificador cooperativo + columnas Algoritmo A / Orquestador
   (Fases 1–3) / Algoritmo B, conectadas por flechas "historial"/"modifica".
   Columnas espaciadas (Orquestador angostado a 6.6cm, A/B separadas a
   ±8.2cm) para que las etiquetas de las flechas no queden apretadas.

2. **Figura "Diagrama UML de la solución propuesta"**: clases
   `AlgoritmoBioinspirado` (abstracta) → `PSO`/`DE`, `EstadoIteracion`, y los
   5 módulos del middleware (`deteccion.py`, `extraccion.py`,
   `barrera_seguridad.py`, `orquestador.py`, `transferencia.py`). Se
   corrigió la clase `HiloAlgoritmo (Thread)` que ya no existe en el código
   real — `orquestador.py` ahora lista `Orquestador.ejecutar()` (bucle
   cooperativo). Las 4 flechas de dependencia hacia `transferencia.py`
   entran ordenadamente por los costados (2 por la altura del encabezado,
   2 por la altura del cuerpo, para no cruzarse) en vez de atravesar el
   módulo o el encabezado.

3. **Figura "Diagrama de secuencia del ciclo completo"**: se mantuvieron los
   **mismos 5 actores del diseño original** (Algoritmo A/fuente,
   Orquestador, Extracción, Transferencia, Algoritmo B/objetivo — a pedido
   explícito del usuario, que notó que un primer intento con 4 actores
   distintos "desordenaba" el diagrama) pero reemplazando los mensajes de
   pausa/reanudación de hilos por las llamadas directas y síncronas del
   planificador cooperativo.

### Bugs de TikZ encontrados y corregidos en el proceso

- Coordenadas tipo `($(nodo)+(x,y)$)` requieren `\usetikzlibrary{calc}`
  (no estaba cargada) — sin ella, las coordenadas colapsaban silenciosamente
  cerca del origen sin dar error de compilación.
- Cada clase/módulo del UML es en realidad 2 nodos (encabezado + cuerpo).
  Las flechas apuntaban a la costura interna entre ambos (p. ej.
  `habs.south` = fondo del encabezado) en vez del borde exterior real de la
  caja completa (`babs.south` = fondo del cuerpo). Esto hacía que las líneas
  parecieran atravesar las cajas. Regla general para el futuro: usar
  siempre el nodo **cuerpo** para conectar por abajo/lados, y el nodo
  **encabezado** para conectar por arriba.

## Otras correcciones menores

- 2 errores reales de compilación (no solo warnings): caracteres Unicode
  griegos (`μετά`) y el símbolo `⋅` en modo matemático, reemplazados por
  comandos LaTeX (`$\mu\epsilon\tau\acute{\alpha}$`, `\cdot`).
- 14 usos redundantes de `\newline` justo antes de una línea en blanco
  (que ya de por sí termina el párrafo) eliminados — causaban warnings de
  "Underfull hbox" que VSCode marcaba en rojo.
- 3 comillas rectas (`"..."`) que rompían el shorthand de babel en español
  reemplazadas por comillas tipográficas (`"..."`).
- Ejemplo de nombre de archivo de log actualizado (`prueba_threading_f11...`
  → `middleware__f11__d20__fes500000__s00...`, con el patrón real de
  `run_id`).

## Sesión 2 — Diagramas de Fase 1/2/3 a TikZ

Se recrearon como TikZ vectorial (dentro de §4.2 "Fase 1/2/3 - ...") los 3
diagramas de flujo que hasta ahora eran PNG estáticos:

- `images/diagram_deteccion.png` → Figura "Diagrama de flujo de Fase 1 -
  Detección" (`fig:diagram_deteccion`): Score compuesto ($t\geq15\%$ →
  `asignar_roles_dinamicos()` → FIR/DWD en paralelo → Score → umbral
  $S<0.2$ → activar Fase 2).
- `images/diagram_extraccion.png` → Figura "... Fase 2 - Extracción"
  (`fig:diagram_extraccion`): guardia de varianza/$R^2$ → XGBoost →
  FastSHAP → Instancias de Élite / Parámetros de Escape en paralelo → salida
  hacia Fase 3.
- `images/diagram_transferencia.png` → Figura "... Fase 3 - Transferencia"
  (`fig:diagram_transferencia`): barrera de Wasserstein → Canal A (RMP) →
  filtro MMD → Canal B, con las ramas de abortar/cerrar ciclo.

Los 3 PNG originales quedaron en `images/` sin usar (mismo criterio que los 3
PNG de arquitectura de la Sesión 1: no editables, obsoletos frente a la
versión vectorial). Estilo reutilizado de las figuras TikZ ya existentes en
el documento (`inicio`/`proceso`/`decision`/`dato`/`flecha` de la Figura de
procedimiento en §8.2): diamantes naranjas para decisiones, colores por fase
(azul=Fase 1, verde=Fase 2, marrón=Fase 3, calcando la paleta de los PNG
originales) y `resizebox{0.8–0.9\textwidth}` para que cada figura ocupe una
página.

**Bug preexistente encontrado y corregido**: `0.main.tex` cargaba
`\usetikzlibrary{arrows.meta, positioning, calc}` pero nunca
`shapes.geometric`, necesaria para los nodos `diamond`/`ellipse`. Esto hacía
que la Figura 8 (procedimiento de `corrida.py`, agregada en la Sesión 1,
§8.2) fallara en cada compilación con `Package pgfkeys Error: I do not know
the key '/tikz/diamond'`/`'/tikz/ellipse'` — el error no detenía la
compilación en `nonstopmode`, así que el documento igual generaba PDF, pero
esos nodos quedaban rotos/vacíos. Se agregó `shapes.geometric` a la lista de
librerías (línea de `\usetikzlibrary` en `0.main.tex`), lo que arregla tanto
la Figura 8 como las 3 figuras nuevas de esta sesión (todas usan `diamond`
para los rombos de decisión).

**Ajuste de tamaño (mismo día, a pedido del usuario)**: las 3 figuras
ocupaban casi una página completa cada una. Se redujeron los
`\resizebox{...}{!}` de 0.85/0.9/0.8`\textwidth` a 0.6/0.65/0.58`\textwidth`
y se compactó el espaciado vertical entre nodos (~25-30% menos separación,
`inner sep` de las cajas reducido de 2.5mm/2mm a 2mm/1.5mm). Al achicar
también la separación horizontal de las cajas laterales ("Esperar
iteraciones", "Monitor pasivo", "RMP+... Cerrar ciclo") aparecieron
colisiones entre las etiquetas "sí"/"no" y esas cajas (el rombo de decisión
no se achica al mismo ritmo que el resto); se corrigió devolviendo esas 3
cajas a una distancia horizontal similar a la original mientras se mantiene
la compresión vertical y el `resizebox` reducido. Resultado: cada figura
ahora comparte página con texto en vez de dominarla completa.

**Pendiente sin tocar (fuera de alcance de este pedido)**: la compilación
completa (`pdflatex` sobre `0.main.tex`) sigue mostrando 3 errores
preexistentes, ninguno introducido en esta sesión ni relacionado con los
diagramas de fase: `LaTeX Error: Unicode character ①/②/③ (U+2460–U+2462)`,
en los círculos numerados `①`/`②`/`③` de la Figura "Arquitectura de
middleware" (§Solución Propuesta, agregada en la Sesión 1). `pdflatex` los
saltea en `nonstopmode` sin detener la compilación, pero esos 3 glifos no se
renderizan en el PDF final. No se corrigió porque no formaba parte del
pedido de esta sesión.

## Pendiente / a considerar en el futuro

- El tamaño del efecto $A_{12}$ (Vargha-Delaney) sigue mencionado como
  planificado en Métodos del Estudio pero nunca calculado — decisión
  consciente de no agregarle marco conceptual por ahora.
- La decisión D-6 (retiro del tier MaxFES=5 000, nuevos tiers) es para
  pruebas **futuras** — no afecta nada de lo ya escrito, que se basa
  íntegramente en la batería ya ejecutada (con MaxFES=5 000 incluido).
- Si se agregan más secciones nuevas, revisar que no vuelvan a aparecer
  referencias a "hilos"/"paralelo" para el middleware (grep rápido:
  `threading|GIL|Hilo|paralel|concurrente`, excluyendo menciones legítimas
  a `grid.py`/`ProcessPoolExecutor`, que sí es paralelismo real del harness
  experimental, no del middleware).
