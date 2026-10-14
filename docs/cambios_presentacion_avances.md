# Registro de cambios — `Avance 1/presentacion_avances.pptx`

Registro de todo lo hecho sobre la presentación de avances en esta sesión de
trabajo, para dar continuidad si se retoma en otra conversación. La
presentación partió de una plantilla de Google Slides exportada a `.pptx`
(la propuesta de trabajo de título original, 34 diapositivas) y terminó
reemplazada por una presentación de avances de 31 diapositivas con el mismo
estilo visual pero contenido íntegramente nuevo, tomado de
`Avance 1/4.trabajo.tex`.

**Advertencia importante**: `Avance 1/presentacion_avances.pptx` está listado
en `.gitignore`, por lo que no tenía historial de git. Al sobrescribirlo, la
plantilla original (la propuesta de trabajo de título, con las diapositivas
de Marco Conceptual, Estado del Arte, Objetivos, etc.) **ya no es
recuperable** salvo que exista una copia fuera de este repositorio.

## Estilo conservado de la plantilla

Fondo `00293B` (navy) sólido en diapositivas de contenido, foto de fondo
oscura (teclado/manos) atenuada al 55% en portada/divisores/diapositivas de
imagen, títulos en Montserrat ExtraBold blanco, cuerpo en Roboto blanco con
viñeta simple, logo de la Escuela de Informática abajo a la derecha, y un
rótulo "fantasma" (texto grande, mismo color que el fondo, apenas visible)
con el nombre de la sección actual abajo a la izquierda en las diapositivas
de viñetas+imagen.

## Método de edición

Sin LibreOffice disponible en el equipo, se usó automatización COM de
PowerPoint (vía PowerShell) para exportar diapositivas a PNG y hacer QA
visual. La edición de contenido se hizo desabotonando el `.pptx` (ZIP de
XML), duplicando las diapositivas "molde" de la plantilla (divisor de
sección, viñetas simples, viñetas+imagen, índice) con `add_slide.py` del
skill de `pptx`, reescribiendo `<p:sldIdLst>` en `ppt/presentation.xml` para
fijar el orden final, y aplicando el contenido con un script Python propio
(`build_content.py` + `apply_content.py`, ninguno versionado — vivían en
`Avance 1/pptx_work/`, borrado al terminar cada ronda) que localiza
marcadores de texto placeholder (`"Resumen"`, `"Introducción"`,
`"Solución Propuesta"`, etc.) y los reemplaza por regex, preservando el
`pPr`/`rPr` original de cada plantilla.

## Sesión 1 — Construcción inicial (26 diapositivas)

Estructura pedida por el usuario: Introducción (contexto, problema, solución
propuesta, avances previos a las pruebas) → Estrategia de Implementación
(esquema de alto nivel, etapas, patrones y su justificación, hardware y
software) → Pruebas y Experimentos → Análisis de Resultados → Discusión de
Resultados (discusión + validación de hipótesis).

- **Figuras**: en vez de reusar las imágenes viejas de la plantilla (Fase
  1/2/3 con la arquitectura de threading ya descartada), se recortaron en
  alta resolución directamente del PDF ya compilado del informe
  (`0.main.pdf`) la Figura 1 (arquitectura del middleware), Figura 7
  (pipeline de 5 capas) y Figura 8 (procedimiento por corrida) — evita
  discrepancias de fuente entre un render aislado y el documento real (ver
  nota de fuentes más abajo). Los gráficos de rangos promedio y los
  diagramas de Diferencia Crítica se tomaron directo de
  `images/analisis/`, componiendo verticalmente D=10 y D=20 con Pillow en
  una sola imagen por diapositiva.
- **Nota de fuentes**: un primer intento de renderizar la figura de
  arquitectura de forma aislada (script LaTeX propio, sin las fuentes
  `txfonts`/`times` que usa `elsarticle`) mostró texto desbordado y cajas
  solapadas — no es un bug del informe, sino una diferencia de métricas de
  fuente entre Computer Modern (default) y Times. Recortar directamente del
  PDF real evita este problema.
- **Bug de solapamiento de títulos**: los títulos largos (p. ej. "Patrones
  de Implementación", "Diseño Experimental Ejecutado") envolvían a 2 líneas
  dentro de la caja de título angosta heredada de la plantilla (4934400 EMU,
  calibrada para palabras sueltas como "Resumen"), pisando el primer
  renglón de viñetas. Se corrigió ensanchando esa caja a 8200000 EMU de
  forma uniforme en todas las diapositivas de viñetas y de viñetas+imagen
  (función `widen_title69` en `build_content.py`).
- Verificado con `scripts/office/validate.py --original` y QA visual de las
  26 diapositivas exportadas vía PowerShell/COM.

## Sesión 2 — Correcciones del usuario (26 → 31 diapositivas)

El usuario pidió 6 ajustes sobre la v1:

1. **Más detalle por viñeta** — se amplió el punteo de todas las
   diapositivas con cifras y matices adicionales del informe (p. ej.
   desglose de horas-CPU por algoritmo, crecimiento de Canal B por MaxFES),
   sin convertirlo en párrafos.
2. **"Resumen" de toda la presentación**, no solo de pruebas/resultados —
   reescrito en 6 viñetas: contexto → solución → implementación → pruebas →
   resultados → conclusión.
3. **3 diapositivas nuevas** con los diagramas de flujo de Fase 1
   (Detección), Fase 2 (Extracción) y Fase 3 (Transferencia), insertadas
   entre "Solución Propuesta" y "Avances hasta la Fecha". Se reusaron las
   imágenes PNG originales de la plantilla vieja
   (`images/diagram_deteccion.png`, `diagram_extraccion.png`,
   `diagram_transferencia.png`), que siguen siendo fieles al diseño actual
   (Score compuesto, XGBoost+FastSHAP, Canal A/Canal B — nada de esto
   cambió con la corrección de arquitectura threading → planificador
   cooperativo, esa corrección solo afectaba el diagrama de más alto nivel).
4. **"Avances hasta la Fecha" reescrita** para dejar explícito que describe
   lo hecho *antes* de las pruebas: ahora referencia el diseño
   "Planificado" (2.000 iteraciones fijas, 30 repeticiones — el mismo que
   aparece en la diapositiva "Planificado vs. Ejecutado") y aclara que ese
   diseño se reformuló para incorporar MaxFES antes de correr la batería.
5. **Guías de lectura en Análisis de Resultados** — se agregó una viñeta
   "Cómo leerlo" al inicio de "Rangos Promedio y Posición Final"
   (barras = rango promedio, más corta es mejor), "Significancia
   Estadística" (qué prueba cada uno de Friedman/Shaffer/Nemenyi-CD) y
   "Comparación por Pares (Wilcoxon)" (qué significan b/e/w).
6. **Nueva sección final "Trabajo Futuro y Correcciones"** (divisor +
   contenido, agregada también como 6.º ítem del índice): elimina
   MaxFES=5.000 y agrega MaxFES=50.000.000; suma más algoritmos
   bioinspirados y combinaciones vía middleware para dar peso estadístico a
   Nemenyi y validar el Canal A (no evaluado en esta batería porque PSO/DE
   no comparten hiperparámetros); y señala que eso implica mayor costo
   computacional, exigiendo un entorno de cómputo más capaz que la CPU de
   escritorio usada hasta ahora.

### Reconstrucción técnica

Como la plantilla original ya no existía (sobrescrita en la Sesión 1), la
Sesión 2 dupicó las diapositivas "molde" **desde el propio archivo ya
editado** (`slide2.xml`="Resumen", `slide4.xml`="Introducción",
`slide21.xml`="Solución Propuesta" seguían con esos textos ancla sin tocar,
así que servían igual de bien como fuente de duplicación). Se ensancharon
además las cajas de imagen y de viñetas de las diapositivas tipo
"viñetas+imagen" con matching estructural por `id` de shape en vez de por
valor exacto de posición, para que la función siguiera funcionando aunque la
diapositiva de origen ya no estuviera en su estado prístino.

- **Índice**: vuelve al layout original de 3+3 columnas (en la Sesión 1 se
  había recortado a 3+2 al quitar una entrada); ahora incluye "Trabajo
  Futuro y Correcciones" como 6.º ítem.
- **Bug encontrado — recorte de imagen equivocado**: al regenerar
  `img_pipeline.png` para esta ronda se recortó por error la página 40 del
  PDF del informe en vez de la 41 (la Figura 7 real), pegando en la
  diapositiva "Esquema de Alto Nivel" un fragmento de texto de la sección de
  Limitaciones ("7. FastSHAP con épocas reducidas...") en vez del diagrama
  del pipeline. Corregido recortando la página correcta.
- **Bug encontrado — título de Fase 3 envuelto**: "Fase 3 — Transferencia de
  Conocimiento" (38 caracteres) superaba el ancho seguro de la caja de
  título (calibrado empíricamente en ~33-36 caracteres a esa fuente/tamaño)
  y envolvía a 2 líneas, pisando la primera viñeta. Se acortó a
  "Fase 3 — Transferencia".
- Verificado de nuevo con `validate.py --original` (un error preexistente de
  `revisionInfo.xml` — metadato que PowerPoint agrega al abrir el archivo,
  no introducido por estas ediciones — se descarta comparando contra el
  propio archivo base) y QA visual de las 31 diapositivas.

## Pendiente / a considerar en el futuro

- No hay copia de la plantilla original (propuesta de trabajo de título) en
  ningún lado accesible por git, dado que `.pptx` está en `.gitignore`. Si
  se necesita en el futuro, habría que reconstruirla desde el texto plano
  extraído por `markitdown` al inicio de la Sesión 1 (no sería una réplica
  exacta del archivo, sí de su contenido).
- Si se agregan más diapositivas de viñetas+imagen a futuro, revisar que el
  título no supere ~33-36 caracteres a menos que se ensanche la caja
  manualmente (ver `set_image_slide`/`set_slide2_bullets` en
  `build_content.py`, recreado en cada sesión ya que no vive en el
  repositorio).
