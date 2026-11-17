# Guion de presentación — versión de trabajo (6 diapositivas)

**No presentar como cierre final hasta completar test y análisis de segmentos.** Cada diapositiva propone una idea, una visual y una frase oral. Los números son de validación.

1. **Pregunta y población.** ¿Entre préstamos Lending Club ya aceptados de 36 meses, podemos ordenar cuáles tienen mayor riesgo de impago? Visual: flujo de aceptado → pagado/incumplido. Decir: no predice solicitudes rechazadas.
2. **Datos y control de fugas.** 456 348 casos terminales; G1 origen, G2 oferta del prestamista, G3 pagos posteriores prohibidos. Visual: línea temporal de emisión y resultado. Ejemplo oral: recuperaciones cobradas después del impago no estaban disponibles al conceder el préstamo.
3. **Validación.** Train 244 943 hasta junio 2014; valid 90 615 en julio–diciembre 2014; test 120 790 enero–junio 2015 reservado. Visual: tres bloques cronológicos. Advertencia oral: las etiquetas tardías hacen esto retrospectivo, no una simulación histórica estricta.
4. **Comparación G1.** Tabla pequeña Dummy 0.5000, logística 0.6548, árbol 0.6323, bosque 0.6560, HGB 0.6665 en ROC-AUC. Visual: barras con eje explícito 0.50–0.70 y valores; no recortar para exagerar sin avisar. Decir: AUC es capacidad de ordenar, no tasa de aciertos.
5. **Errores y lectura.** Bosque G1 a umbral 0.5: 30 186 falsos positivos, 4 996 falsos negativos; precisión 20.54 %, recall 60.97 %. HGB G1 no detecta positivos al umbral 0.5; muestra por qué el umbral importa. Visual: matriz de confusión con etiquetas claras. G1+G2 mejora HGB a 0.6854, pero es otra información disponible.
6. **Conclusión y trabajo que falta.** Hay señal sobre el dummy, mejora limitada entre familias, sesgo de aceptados y corte de etiquetas supuesto. Siguiente: segmentos, política de umbral, test final una sola vez y actualización del informe. No recomendar uso crediticio real.

**Pregunta probable:** «¿Por qué no presentan test?» Respuesta: aún no se fijó conjuntamente el modelo ni el protocolo de disponibilidad de etiquetas; reportar validación como test sería engañoso. «¿Por qué no usar recoveries?» Porque ocurre después del resultado que se pretende anticipar.
