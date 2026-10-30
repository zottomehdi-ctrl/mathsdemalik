# DECISIONES — *Claude / Motion Reel 2026*

Showreel de 15 segundos en el que **cada píxel y cada sonido salen de código**. No hay metraje,
imágenes, samples, presets ni modelos de generación de imagen, vídeo o audio. Solo Python:
numpy y scipy para la señal, numba para los filtros, skia-python para el dibujo vectorial,
HarfBuzz para el kerning y ffmpeg para el encoding.

| Entregable | Detalle |
|---|---|
| `claude_showreel_2026.mp4` | 1920×1080, 60 fps, 15,000 s (900 fotogramas), H.264 High 4.2 (CRF 17, yuv420p, etiquetado BT.709) + AAC 320 kb/s 48 kHz estéreo |
| Código | `common.py` (línea de tiempo compartida), `music.py` (música y diseño sonoro), `gfx.py` (texto, trazados, 3D), `scenes.py` (las 7 escenas), `render.py` (motion blur, post y encoding) |

Reproducir (unos 3,5 min de render con 4 núcleos):

```bash
pip install skia-python numba scipy uharfbuzz      # + libegl1 en Linux
python3 music.py build/music.wav                    # escribe también build/music_mono.npy
REEL_AUDIO_NPY=build/music_mono.npy python3 render.py video claude_showreel_2026.mp4 build/music.wav
```

---

## 1. La idea: un punto que se convierte en todo

Un showreel de currículum tiene que demostrar **oficio** (timing, tipografía, sistemas, 3D,
simulación, edición) y **voz propia**, en muy poco tiempo. Elegí como hilo conductor **un único
punto naranja**:

- abre el film,
- se estira y se convierte en una línea de tiempo,
- recorre una curva de animación,
- tacha palabras,
- hace de punto final de *JUST CODE.* y de agujero por el que entramos en el drop,
- vuelve como semilla de las partículas,
- y termina cayendo, con balística real, como el punto de **Claude.**

Cada capítulo enseña una competencia concreta. Un HUD discreto (timecode, número de capítulo,
metrónomo, barra de progreso con las marcas de escena) recuerda el lenguaje de una mesa de
montaje y deja claro que el reel sabe lo que está haciendo.

## 2. Estructura: 128 BPM, 8 compases, exactamente 15 s

A 128 BPM, 8 compases de 4/4 duran **exactamente 15,000 s**: un beat dura 0,46875 s
(28,125 fotogramas). La música y la imagen leen **el mismo archivo** (`common.py`), así que la
sincronía es exacta por construcción, sin tener que ajustar nada a mano.

| Compás | Tiempo | Capítulo | Imagen | Sonido |
|---|---|---|---|---|
| 1–2 | 0:00–3:75 | 01 TIMING | El punto aparece (muelle con rebote) y revela una trama de fondo. Se encoge para coger impulso y se estira en una línea de tiempo graduada de 0 a 900 fotogramas, con un cabezal de lectura que sigue el timecode real. Se posan 8 claves, una por semicorchea. Las claves suben y forman un **editor de curvas** con `cubic-bezier(.83, 0, .17, 1)`, tiradores y papel cebolla, más una lectura lateral del *spacing*. Después el punto tacha FOOTAGE / SAMPLES / PRESETS, se posa como punto final de **JUST CODE.** y la cámara se zambulle en él. | Pad Fa m9 → Re♭maj7, arpegio que se abre (cada nota es una clave), ticks, riser y redoble. Una semicorchea de **silencio** antes del drop. |
| 3 | 3:75–5:63 | 02 TYPOGRAPHY | **CLAUDE** en Inter Display Black a 400 px, sobre naranja, con guías de línea base y altura de mayúsculas. Corte en 3 bandas que se desplazan. Cortinilla circular que invierte los colores al pasar. **Volteo 3D letra a letra** con perspectiva (estilo panel de aeropuerto): CLAUDE → MOTION → DESIGN, las tres de 6 letras. | Drop: kick a negras, clap, bajo house a contratiempo, stabs supersaw. Cada volteo de letra suena como un *clac* de panel. |
| 4 | 5:63–7:50 | 03 SYSTEMS | Rejilla de 16×9 formas, con ondas de morphing polar: círculo → cuadrado → cruz. Cada onda parte de un origen distinto. Luego cada forma se subdivide en 4 y se forma una **esfera en semitono** iluminada que gira. | Campanas cortas en cada onda. |
| 5 | 7:50–9:38 | 04 DIMENSION | Corte gráfico a azul cobalto. El círculo del semitono se convierte en 18 anillos 3D: esfera de meridianos → torsión tipo ratán → **toro**. Las líneas se disuelven en puntos. Un marcador de seguimiento (*call-out*) enseña las coordenadas en directo. | Campanas FM (cascada cuando giran los anillos), soplo ascendente en la disolución. |
| 6 | 9:38–11:25 | 05 PARTICLES | 2 400 partículas **nacen de los puntos de los anillos**. Pasan por un campo de flujo *curl noise* (sin divergencia), forman un vórtice e implosionan en el punto. | 110 destellos granulares afinados en la escala, panoramizados. |
| 7 | 11:25–13:13 | 06 RHYTHM | Montaje en corcheas de **palabras-ventana** (track mattes): TYPE, SHAPE, SPACE, FLOW, SOUND y CODE, cada una rellena de su técnica. SOUND muestra el **espectro FFT real de la mezcla** en ese instante y CODE, **el propio código fuente** del reel. Después CODE tartamudea y se repliega hasta el punto, solo, en silencio. | Cada palabra tiene su sonido (teclas, campana, soplo, sub, bips de datos). Stabs ascendentes y redoble. |
| 8 | 13:13–15:00 | 07 SIGNATURE | Papel. **Claude.** sube desde su línea base, aparecen el filete y *Motion Designer*, y el punto cae del cielo y rebota dos veces con **balística real**: los rebotes caen en beats exactos y la altura viene de g. El pie de página se escribe a máquina. | Acorde final **Fa mayor add9** (tercera de picardía tras Re♭–Do7), sub, crash y tres *bloops* afinados, uno por rebote. |

## 3. Sistema visual

- **Paleta de 4 tintas**: tinta `#0D0D10`, papel `#F4F0E8`, señal `#FF5620` y cobalto
  `#2838FF`. Los fundidos de color se interpolan en **OKLab** para que sean perceptualmente
  uniformes.
- **Tipografía**: Inter Display Black para los titulares, Inter Display Medium para el subtítulo y
  JetBrains Mono para todo lo técnico. Las dos son fuentes libres (OFL) **ya instaladas en la
  máquina**. Son contornos vectoriales, no imágenes. El texto se compone con **HarfBuzz** para
  tener el kerning real (LA, AV…) y cada glifo se convierte en trazado para poder animarlo por
  separado.
- **Lenguaje de movimiento**: una curva firma (`cubic-bezier(.83,0,.17,1)`, muy lenta al
  arrancar y muy decidida al llegar), muelles amortiguados analíticos para todo lo que «aterriza»,
  anticipación antes de los gestos grandes y *squash & stretch* en el punto.
- **Motion blur real** por supermuestreo temporal: de 7 a 16 subimágenes por fotograma con
  obturador de 180°, 7 059 renders en total. Las subimágenes se acumulan en 16 bits para que las
  estelas no hagan bandas.
- **Cámara**: sacudidas con ruido suave y un golpe de zoom en cada impacto de la partitura, con
  aberración cromática breve y radial. Halo (bloom) con umbral de luminancia solo en las escenas
  de luz (3D y partículas), viñeta y un grano fino correlacionado.
- **Encoding**: matriz BT.709 explícita en la conversión y en las etiquetas, para que el naranja
  sea el mismo en todos los reproductores. `faststart` para que arranque al instante en web.

## 4. Sistema sonoro

- **Tonalidad**: Fa menor. Intro Fa m9 → Re♭maj7. Drop Fa m9 – Re♭maj7 – La♭maj7 – Mi♭add9.
  Cierre Re♭maj7 → Do7 → **Fa mayor add9**: la cadencia final resuelve hacia la luz, igual que la
  imagen pasa al papel.
- **Síntesis desde cero**:
  - osciladores de sierra **PolyBLEP** (sin aliasing);
  - filtros de variable de estado TPT con frecuencia de corte modulada muestra a muestra
    (compilados con numba);
  - supersaw de 7 voces;
  - campanas y piano eléctrico por **FM**;
  - charles metálicos tipo 808 (6 osciladores cuadrados inarmónicos);
  - kick con envolvente de tono y clic;
  - clap de 3 ráfagas más cola;
  - soplos (ruido con paso banda barrido), risers y sub-booms.
- **Mezcla**:
  - delay ping-pong a corchea con puntillo en el arpegio;
  - reverb por convolución con una **respuesta al impulso sintética** (ruido multibanda con
    caídas distintas);
  - sidechain disparado por el kick;
  - cada bus se nivela con un **objetivo RMS** medido en su zona activa: mezclar sin oír exige
    medir;
  - EQ shelving, saturación suave y limitador con anticipación.
- **Master**: **−11,5 LUFS** integrados (BS.1770-4 con puertas, comprobado con ffmpeg
  `ebur128`), true peak ≈ **−1,9 dBFS** y el limitador prácticamente sin trabajar. El drop tiene
  margen para pegar.
- **Sincronía de diseño sonoro**: cada gesto visual tiene su sonido, calculado con la misma
  fórmula que la imagen. El ejemplo más claro: mientras el punto recorre la curva del editor, un
  oscilador sigue exactamente esa curva de easing en tono. El *clac* de cada letra suena en el
  instante en que el panel pasa los 90°, y los tres *bloops* finales caen en los impactos de la
  balística.

## 5. Lo que descarté

- **Diseñar una tipografía propia en código.** Era tentador («hasta la fuente es código»), pero
  ponía en riesgo la calidad. Inter con kerning de HarfBuzz da una tipografía profesional, y la
  tipografía es lo primero que juzga un director de arte.
- **Renderizar en Chromium/WebGL sin interfaz.** Lo descarté por determinismo y control.
  skia-python es el mismo motor de dibujo, pero fotograma a fotograma, en paralelo y sin
  sorpresas de GPU.
- **Atravesar el agujero del toro con la cámara.** Mirado desde su eje, los anillos se ven de
  canto y el toro parecía un estallido de rayos, sin lectura. Lo sustituí por una disolución en
  puntos que luego **son** las partículas: mejor transición y más honesta.
- **«NO FOOTAGE» escrito con la palabra NO.** Tachar las palabras con el punto es un gesto
  gráfico más fuerte y con menos texto.
- **Pantalla partida en 4 para el resumen final.** La sustituí por las palabras-ventana: se lee
  más rápido y es más tipográfico.
- **Escena de líquidos (metaballs).** Una técnica más diluía el arco narrativo; en 15 s cada
  capítulo tiene que respirar.
- **Melodía principal.** Habría competido con el diseño sonoro. La melodía vive en el arpegio,
  las campanas y los destellos.
- **Montaje estroboscópico a pantalla completa en corcheas.** Más de 3 destellos/s es un riesgo
  de fotosensibilidad. El montaje alterna contenido sobre un fondo oscuro constante.
- **Fundido a negro al final.** El último fotograma es la firma, porque es el que la gente pausa.
- **Grano fuerte.** Hacía explotar el bitrate (127 MB). Ahora es fino y correlacionado.
- **Cabezal de lectura barriendo la intro.** Lo sustituí por un cabezal atado al timecode real
  del film.

## 6. Iteraciones

1. **Plan**: estructura de 8 compases, paleta, guion beat a beat y lista de sincronías.
2. **Audio v1**: mi primer estimador de loudness era erróneo. Normalicé a «−10,5» y en realidad
   salió **−8,1 LUFS con ~5 dB de limitación** (aplastado). Lo corregí con BS.1770-4 completo:
   −11,5 LUFS y el limitador ya solo protege.
3. **Audio v2**: un análisis por octavas mostró demasiado sub (63 Hz a +12,7 dB respecto a
   500 Hz) y demasiado aire (16 kHz a −5,9 dB).
   - Shelves en el master, charles con paso bajo a 14 kHz y −2 dB, bajo −1,5 dB.
   - El acorde final tenía un hueco en 125 Hz: añadí un Fa2.
   - Riser −3 dB y redoble −2 dB para que el drop contraste más.
4. **Hoja de contactos 1**, nueve defectos corregidos:
   - las claves no estaban sobre la curva (rebote del muelle);
   - la etiqueta bezier tapaba un tirador;
   - la cortinilla era demasiado rápida y se leía como un corte;
   - la esfera en semitono no estaba limpia antes del corte;
   - el toro era ilegible;
   - las partículas parecían ruido: puse un campo más amplio y estelas de 5 segmentos;
   - la textura CODE estaba borrosa y la «E» vacía;
   - el punto de *Claude.* estaba demasiado lejos: ahora ocupa exactamente el lugar del glifo del
     punto de Inter;
   - los anillos de SPACE eran demasiado finos.
5. **Tiras fotograma a fotograma** de las transiciones:
   - CLAUDE aparecía demasiado lento tras el drop;
   - los subtítulos de la firma eran visibles antes de tiempo (fallo de máscara);
   - la aberración cromática dejaba un borde amarillo;
   - el motion blur hacía escalones en los gestos más rápidos: subí a 16 subimágenes en el drop
     y en la firma.
6. **Render completo v1**: se rompió en el efecto máquina de escribir (cadena vacía pasada a
   HarfBuzz) y pesaba 127 MB. Corregí el fallo y pasé a grano fino con CRF 17: unos 40 MB.
7. **v2 → v3**: halo con umbral de luminancia en 3D y partículas, y la etiqueta del marcador de seguimiento
   fija en pantalla (en movimiento era ilegible).
8. **Verificación final**: ffprobe (900 fotogramas, 15,000 s y BT.709). Correlación cruzada entre
   la pista AAC del MP4 y el WAV maestro: **0 muestras de desfase**.

## 7. Tiempo total

**≈ 53 minutos de reloj**, de 10:27 a 11:20 UTC (del primer comando al commit), en una sola
sesión.

| Fase | Aprox. |
|---|---|
| Entorno (skia-python + libEGL, numba, HarfBuzz) y pruebas de API | 13 min |
| Música: composición, síntesis y dos pasadas de mezcla y masterización | 12 min |
| Motor gráfico, 7 escenas y primera hoja de contactos | 12 min |
| Revisiones: hojas de contactos, tiras, 3 renders completos de ~3,5 min y verificaciones | 13 min |
| DECISIONES.md y entrega | 3 min |
