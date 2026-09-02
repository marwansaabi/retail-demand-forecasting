# -*- coding: utf-8 -*-
"""Build the explainer PDF for the retail demand forecasting project."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)

HERE = Path(__file__).parent
ASSETS = HERE / "assets"
OUT = HERE / "Proyecto_Demand_Forecasting_explicado.pdf"

ACCENT = colors.HexColor("#2563eb")
INK = colors.HexColor("#1e293b")
MUTED = colors.HexColor("#64748b")
RULE = colors.HexColor("#dde3ea")
BOXBG = colors.HexColor("#f4f7fb")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", parent=ss["Title"], fontName="Helvetica-Bold",
                            fontSize=25, leading=30, textColor=INK, alignment=0),
    "subtitle": ParagraphStyle("subtitle", parent=ss["Normal"], fontSize=12.5,
                               leading=17, textColor=MUTED, spaceBefore=6),
    "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold",
                         fontSize=15, leading=19, textColor=ACCENT,
                         spaceBefore=17, spaceAfter=7),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                         fontSize=11.5, leading=15, textColor=INK,
                         spaceBefore=11, spaceAfter=4),
    "body": ParagraphStyle("body", parent=ss["Normal"], fontSize=9.8, leading=14.6,
                           textColor=INK, alignment=TA_JUSTIFY, spaceAfter=6),
    "small": ParagraphStyle("small", parent=ss["Normal"], fontSize=8.6, leading=12.4,
                            textColor=MUTED, spaceAfter=4),
    "cell": ParagraphStyle("cell", parent=ss["Normal"], fontSize=8.7, leading=12),
    "cellb": ParagraphStyle("cellb", parent=ss["Normal"], fontSize=8.7, leading=12,
                            fontName="Helvetica-Bold"),
    "q": ParagraphStyle("q", parent=ss["Normal"], fontSize=9.8, leading=13.6,
                        fontName="Helvetica-Bold", textColor=ACCENT, spaceAfter=3),
    "code": ParagraphStyle("code", parent=ss["Normal"], fontName="Courier",
                           fontSize=8.4, leading=11.6, textColor=INK),
}


def P(t, s="body"):
    return Paragraph(t, S[s])


def rule(space=4):
    t = Table([[""]], colWidths=[168 * mm], rowHeights=[0.7])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), RULE)]))
    return [Spacer(1, space), t, Spacer(1, space + 2)]


def callout(title, text):
    inner = [P(f"<b>{title}</b>", "body"), P(text, "body")]
    t = Table([[inner]], colWidths=[168 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BOXBG),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LINEBEFORE", (0, 0), (0, -1), 2.5, ACCENT),
    ]))
    return KeepTogether([Spacer(1, 3), t, Spacer(1, 8)])


def table(rows, widths, header=True, highlight_row=None):
    data = []
    for i, r in enumerate(rows):
        style = "cellb" if (header and i == 0) else "cell"
        data.append([Paragraph(str(c), S[style]) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
    ]
    if header:
        cmds += [("BACKGROUND", (0, 0), (-1, 0), BOXBG),
                 ("LINEBELOW", (0, 0), (-1, 0), 0.9, ACCENT)]
    if highlight_row is not None:
        cmds.append(("BACKGROUND", (0, highlight_row), (-1, highlight_row),
                     colors.HexColor("#eaf1fe")))
    t.setStyle(TableStyle(cmds))
    return t


def figure(name, width=168 * mm, caption=None):
    img = Image(str(ASSETS / name))
    img._restrictSize(width, 200 * mm)
    out = [Spacer(1, 4), img]
    if caption:
        out += [Spacer(1, 3), P(caption, "small")]
    return KeepTogether(out + [Spacer(1, 8)])


def qa(question, answer):
    return KeepTogether([P(question, "q"), P(answer, "body"), Spacer(1, 5)])


# ----------------------------------------------------------------- content
story = []

story += [
    Spacer(1, 8),
    P("Previsión de demanda en retail", "title"),
    P("Qué hace el proyecto, por qué está construido así, y cómo defenderlo "
      "en una entrevista técnica.", "subtitle"),
    Spacer(1, 10),
]
story += rule()
story += [
    P("<b>Repositorio:</b> github.com/marwansaabi/retail-demand-forecasting &nbsp;·&nbsp; "
      "<b>Stack:</b> Python, LightGBM, pandas, Streamlit, Plotly", "small"),
]
story += rule(2)

# 1
story += [P("1. Qué hace, en 30 segundos", "h1")]
story += [P(
    "El proyecto predice <b>cuántas unidades de cada producto se van a vender "
    "cada semana, con cuatro semanas de antelación</b>, a partir del histórico "
    "real de ventas de una tienda online. Se entrena un modelo de machine "
    "learning (LightGBM) y se compara contra cuatro métodos simples de "
    "referencia para comprobar que realmente aporta algo.")]
story += [P(
    "Resultado: el modelo gana a la mejor referencia por un <b>4,8%</b>. Es un "
    "margen modesto, y esa modestia es justo lo que hace creíble el proyecto: "
    "en previsión de demanda nadie gana por goleada, y quien lo afirma casi "
    "siempre ha filtrado información del futuro en su modelo.")]

story += [callout(
    "Por qué esto le interesa a una empresa como Inditex",
    "Toda la cadena de suministro del retail depende de esta pregunta. Si "
    "predices de más, acumulas stock que acaba en rebajas o se destruye. Si "
    "predices de menos, hay rotura de stock y ventas perdidas. Un punto "
    "porcentual de mejora en la previsión se traduce en millones de euros en "
    "inventario. Es, literalmente, el problema central de datos del sector.")]

# 2
story += [P("2. Los datos", "h1")]
story += [P(
    "Se usa el dataset público <b>Online Retail II</b> (repositorio UCI): "
    "<b>1.067.371 líneas de factura</b> reales de una tienda online británica "
    "entre diciembre de 2009 y diciembre de 2011. Cada línea es \"esta factura, "
    "este producto, esta cantidad, este precio, este día\".")]
story += [P("<b>Limpieza aplicada</b>", "h2")]
story += [table([
    ["Qué se elimina", "Por qué"],
    ["Facturas que empiezan por <font face='Courier'>C</font>",
     "Son cancelaciones, con cantidades negativas. Si se dejan, anulan demanda real."],
    ["Cantidades y precios <= 0", "Ajustes contables, no ventas."],
    ["Códigos no-producto (<font face='Courier'>POST</font>, "
     "<font face='Courier'>BANK CHARGES</font>...)",
     "Portes, comisiones y ajustes manuales; no son artículos de catálogo."],
], [58 * mm, 110 * mm])]
story += [Spacer(1, 6), P(
    "Tras limpiar quedan 1.036.877 líneas. Se agregan a <b>unidades por producto "
    "y semana</b> y se conservan las 50 referencias de mayor volumen: "
    "<b>5.200 observaciones sobre 104 semanas completas</b>.")]


# 3
story += [P("3. Las tres decisiones que definen el proyecto", "h1")]
story += [P(
    "Estas tres son las que de verdad determinaron el resultado, mucho más que "
    "la elección del algoritmo. Si te preguntan algo en una entrevista, será "
    "sobre esto.")]

story += [P("3.1. Semanal, no diario", "h2")]
story += [P(
    "El primer intento fue predecir demanda <b>diaria</b>. Falló, y el "
    "diagnóstico fue revelador: este negocio es <b>mayorista</b>. La mediana "
    "son 12 unidades al día, pero el máximo es de <b>80.995 unidades en un solo "
    "día</b> — un único cliente haciendo un pedido enorme. El <b>0,4% de los "
    "días concentra el 21% del volumen total</b>.")]
story += [P(
    "Ningún modelo puede adivinar, desde el histórico de ventas, el día exacto "
    "en que un cliente decide hacer un pedido de 80.000 unidades. A escala "
    "diaria eso es ruido irreducible. Al agregar a semanal, ese ruido de "
    "<i>timing</i> se absorbe, y además coincide con cómo se planifica el "
    "reaprovisionamiento en la realidad.")]
story += [figure("daily_vs_weekly.png",
                 caption="La misma referencia a las dos escalas. Arriba, picos "
                         "de pedidos individuales imposibles de anticipar; abajo, "
                         "la señal que sí tiene estructura aprendible.")]

story += [P("3.2. Las semanas sin ventas son ceros, no huecos", "h2")]
story += [P(
    "El registro de facturas solo contiene los días en que hubo venta. Si un "
    "producto no se vendió una semana, esa semana simplemente no aparece. Pero "
    "<b>no vender nada no es un dato ausente: es un cero</b>. Dejarlo como "
    "hueco inflaría al alza todas las medias móviles y sesgaría el modelo "
    "entero. El 12% de las observaciones son ceros explícitos añadidos en la "
    "preparación.")]

story += [P("3.3. El objetivo se modela como log(1+unidades)", "h2")]
story += [P(
    "Con el objetivo en su escala original, el modelo <b>perdía</b> contra una "
    "simple media móvil (wMAPE 0,62 frente a 0,59). La distribución de demanda "
    "está muy sesgada a la derecha: unas pocas semanas con pedidos enormes "
    "dominan la función de pérdida y arrastran hacia arriba todas las demás "
    "predicciones. Transformar el objetivo con logaritmo (y deshacer la "
    "transformación al predecir) fue lo que dio la vuelta al resultado.")]

story += [callout(
    "El detalle metodológico que más vale en una entrevista",
    "Esa decisión se tomó midiendo sobre un <b>periodo de desarrollo separado</b> "
    "(nov 2010 – mar 2011), no sobre el periodo de test que se reporta. Elegir "
    "configuraciones mirando el test es la versión sutil de entrenar sobre el "
    "test: infla el resultado y no se sostiene en producción. En el periodo de "
    "desarrollo, log obtuvo 0,768 frente a 0,803 — por eso se eligió.")]


# 4
story += [P("4. Cómo funciona el modelo", "h1")]
story += [P(
    "Se usa <b>LightGBM</b>, un algoritmo de <i>gradient boosting</i>: construye "
    "cientos de árboles de decisión pequeños, cada uno corrigiendo los errores "
    "del anterior. Es el estándar de facto en previsión de demanda tabular.")]

story += [P("Un solo modelo global para todos los productos", "h2")]
story += [P(
    "En lugar de entrenar 50 modelos (uno por referencia), se entrena "
    "<b>uno solo</b> con todas, incluyendo la identidad del producto como una "
    "variable categórica más. El motivo: cada serie tiene solo ~100 semanas, "
    "que es muy poco. Al juntarlas, un producto de poca rotación puede "
    "aprovechar el patrón estacional aprendido de los productos con mucha venta.")]

story += [P("Las variables (features)", "h2")]
story += [table([
    ["Grupo", "Qué contiene"],
    ["Retardos", "Unidades vendidas hace 1, 2, 3, 4, 8, 13 y 52 semanas"],
    ["Ventanas móviles", "Media, desviación típica y proporción de ceros a 4, 8, 13 y 26 semanas"],
    ["Tendencia", "Media de 4 semanas dividida entre la de 13; mismo periodo del año anterior"],
    ["Señales de mercado", "Precio, nº de facturas y nº de clientes: último valor, media de 4 semanas, y ratio frente a su norma de 13 semanas"],
    ["Calendario", "Mes, trimestre, semana del año y del mes, festivos británicos en la semana, semanas hasta Navidad, codificaciones cíclicas seno/coseno"],
    ["Horizonte", "Qué paso se está prediciendo (1 a 4), para que un mismo modelo module su incertidumbre según la distancia"],
], [36 * mm, 132 * mm])]
story += [Spacer(1, 6), P(
    "El ratio de precio frente a su propia norma de 13 semanas es lo más "
    "parecido a un indicador de promoción que permiten estos datos: un producto "
    "que cotiza por debajo de su precio habitual suele estar en campaña.")]

story += [P("5. Cómo se valida (la parte crítica)", "h1")]
story += [P("La regla que gobierna todo el código", "h2")]
story += [P(
    "Una variable que describe la semana objetivo <b>solo puede usar "
    "información disponible en el momento de hacer la previsión</b>. Los "
    "retardos y las medias móviles se leen siempre en el origen de previsión, "
    "nunca en la semana que se quiere predecir. La única excepción son los "
    "datos de calendario, porque de verdad se conocen de antemano: hoy ya "
    "sabemos que Navidad cae en la semana 52.")]
story += [P(
    "Saltarse esto es la forma clásica de construir un modelo con un backtest "
    "espectacular y cero capacidad predictiva real.")]

story += [P("Backtesting de origen móvil", "h2")]
story += [P(
    "Una única división entrenamiento/test en una serie temporal te informa "
    "sobre el tiempo que hizo un mes concreto. En su lugar se avanzan <b>8 "
    "orígenes de previsión</b> por el calendario, separados 4 semanas, "
    "prediciendo 4 semanas cada vez. En cada origen <b>el modelo se reentrena "
    "desde cero</b> usando únicamente los datos anteriores a ese punto.")]
story += [figure("rolling_origin.png",
                 caption="Cada fila es un fold. El bloque gris es lo que el "
                         "modelo puede ver; el azul, las 4 semanas que debe "
                         "predecir sin haberlas visto nunca.")]


# 6
story += [P("6. Las métricas, explicadas", "h1")]
story += [table([
    ["Métrica", "Qué mide", "Por qué está aquí"],
    ["<b>wMAPE</b>", "Error absoluto total dividido entre la demanda total. Un 55,8% significa que la suma de errores equivale al 55,8% de las unidades realmente vendidas.",
     "Es la que se usa en planificación de demanda. El MAPE clásico se dispara al infinito con las semanas de venta cero, y aquí el 12% lo son."],
    ["<b>MASE</b>", "Error dividido por el del método ingenuo (\"la próxima semana será como esta\") medido dentro del entrenamiento.",
     "Por debajo de 1 significa que bates a ese método ingenuo. Es comparable entre productos de volúmenes muy distintos."],
    ["<b>MAE</b>", "Error medio en unidades, sin signo.", "Interpretable directamente en unidades de producto."],
    ["<b>RMSE</b>", "Como el MAE pero penalizando mucho más los errores grandes.",
     "Se reporta por transparencia, no como titular: aquí está dominado por un puñado de pedidos mayoristas."],
], [22 * mm, 76 * mm, 70 * mm])]

# 7
story += [P("7. Resultados", "h1")]
story += [table([
    ["Modelo", "MAE", "RMSE", "wMAPE", "MASE"],
    ["<b>LightGBM</b>", "<b>219,5</b>", "456,2", "<b>55,8%</b>", "<b>0,783</b>"],
    ["Media móvil (13 sem.)", "230,6", "449,4", "58,6%", "0,829"],
    ["Media móvil (4 sem.)", "233,1", "444,4", "59,2%", "0,809"],
    ["Ingenuo (última semana)", "290,9", "530,5", "73,9%", "0,988"],
    ["Ingenuo estacional (52 sem.)", "362,1", "772,0", "92,0%", "1,237"],
], [56 * mm, 26 * mm, 26 * mm, 30 * mm, 30 * mm], highlight_row=1)]
story += [figure("results.png")]

story += [P("Cómo interpretarlo con honestidad", "h2")]
story += [P(
    "El modelo gana en MAE, wMAPE y MASE. <b>Pierde en RMSE</b> frente a las "
    "medias móviles, y eso no es un descuido: la transformación logarítmica "
    "impide deliberadamente que el modelo persiga los pedidos gigantes, lo que "
    "cuesta error cuadrático y compra precisión en el 99% de semanas normales. "
    "Saber explicar ese compromiso vale más que ocultarlo.")]
story += [P(
    "El ingenuo estacional (comparar con la misma semana del año anterior) es "
    "con diferencia el peor. Tiene sentido: con solo dos años de datos, la "
    "referencia interanual es una sola observación por semana, demasiado "
    "ruidosa para fiarse de ella.")]


# 8
story += [P("8. El código, archivo por archivo", "h1")]
story += [table([
    ["Archivo", "Responsabilidad"],
    ["<font face='Courier'>data/prepare_data.py</font>",
     "Del Excel bruto al panel semanal limpio: limpieza, agregación, ceros explícitos, descarte de semanas parciales."],
    ["<font face='Courier'>utils/features.py</font>",
     "Construcción de variables. Todo indexado por posición de periodo, no por días, así que funciona igual en diario o semanal."],
    ["<font face='Courier'>utils/models.py</font>",
     "Los cuatro métodos de referencia y la clase del modelo LightGBM."],
    ["<font face='Courier'>utils/backtest.py</font>",
     "Validación de origen móvil y cálculo de las cuatro métricas."],
    ["<font face='Courier'>run_backtest.py</font>",
     "Reproduce la tabla de resultados y cachea la salida."],
    ["<font face='Courier'>app.py</font>",
     "Dashboard en Streamlit: comparativa, previsión por producto, importancia de variables y metodología."],
], [46 * mm, 122 * mm])]
story += [Spacer(1, 6), P(
    "El backtest es <b>determinista</b>: dos ejecuciones seguidas producen "
    "exactamente los mismos números. Está verificado.", "body")]

# 9
story += [P("9. Preguntas que te pueden hacer, y cómo responderlas", "h1")]
story += [P(
    "Estas son las preguntas razonables que haría alguien técnico al leer el "
    "proyecto. Vale la pena que las respuestas sean tuyas, no memorizadas.", "small")]
story += [Spacer(1, 5)]

story += [qa("\"Solo mejoras un 4,8% sobre una media móvil. ¿Merece la pena?\"",
    "Sí, y hay dos motivos. Primero, a escala de un retailer grande ese "
    "porcentaje sobre el inventario total es mucho dinero. Segundo, y más "
    "importante: ese 4,8% es real. Las referencias contra las que compito no "
    "son de paja — una media móvil de 13 semanas es un rival duro cuando la "
    "demanda es un nivel que deriva lentamente más ruido. Un proyecto que "
    "presumiera de un 90% de mejora estaría casi con seguridad filtrando datos.")]

story += [qa("\"¿Cómo sabes que no hay fuga de datos?\"",
    "Por construcción. Las variables de histórico se calculan sobre una serie "
    "truncada en el origen de previsión: la función que las construye "
    "físicamente no recibe datos posteriores. Las de calendario sí describen la "
    "semana objetivo, pero son conocibles de antemano. Y en cada uno de los 8 "
    "folds el modelo se reentrena desde cero solo con datos anteriores al origen.")]

story += [qa("\"¿Por qué LightGBM y no una red neuronal o ARIMA?\"",
    "ARIMA modela una serie cada vez, y aquí cada serie tiene solo ~100 puntos: "
    "muy poco, y además no aprovecha lo que comparten las 50 referencias. Una "
    "red neuronal necesitaría bastante más volumen de datos para justificarse. "
    "El gradient boosting sobre variables tabulares es el punto óptimo para "
    "este tamaño, y es lo que se usa en la industria para esto.")]

story += [qa("\"¿Qué harías si tuvieras que llevarlo a producción?\"",
    "Tres cosas. Predecir intervalos en vez de un valor puntual, porque una "
    "decisión de reposición necesita un cuantil asociado a un nivel de servicio, "
    "no una media. Reconciliación jerárquica: el total es mucho más predecible "
    "que cualquier producto individual, así que prever arriba y repartir hacia "
    "abajo suele batir al enfoque bottom-up. Y modelos específicos de demanda "
    "intermitente (Croston, TSB) para la cola larga de referencias que solo "
    "venden algunas semanas.")]

story += [qa("\"¿Qué te limitó más?\"",
    "La ausencia de variables explicativas de verdad. No hay datos de "
    "promociones, ni de campañas, ni de competencia, ni de stock disponible. "
    "Reconstruí una señal aproximada de promoción con el ratio del precio "
    "frente a su norma de 13 semanas, pero es un sustituto pobre. Con datos "
    "reales de promoción y de precio planificado, el margen sobre las "
    "referencias sería bastante mayor.")]

story += [qa("\"¿Por qué el modelo pierde en RMSE?\"",
    "Porque es una consecuencia buscada de modelar en escala logarítmica. El "
    "RMSE castiga mucho los errores grandes, y los errores grandes aquí son "
    "los pedidos mayoristas atípicos. Al no perseguirlos, acierto más en las "
    "semanas normales y fallo por más en esas pocas. Dado que el objetivo es "
    "planificar reposición, prefiero ese compromiso.")]


# 10
story += [KeepTogether([P("10. Limitaciones honestas", "h1"), table([
    ["Limitación", "Implicación"],
    ["Solo 104 semanas de histórico",
     "Apenas dos ciclos anuales. La estacionalidad interanual se estima con muy pocos datos, y por eso el ingenuo estacional rinde tan mal."],
    ["Datos mayoristas, no de tienda física",
     "El patrón de pedidos grandes y esporádicos no es el mismo que el de una tienda de retail al consumidor final."],
    ["Sin promociones ni precio planificado",
     "En la industria el precio futuro se conoce (lo fijas tú). Aquí solo se puede usar el precio pasado."],
    ["50 referencias, las de más volumen",
     "La cola larga de productos con venta esporádica se comporta distinto y necesitaría modelos específicos."],
], [50 * mm, 118 * mm])])]

story += [Spacer(1, 10)]
story += rule()
story += [P(
    "Marwan El Saabi &nbsp;·&nbsp; github.com/marwansaabi/retail-demand-forecasting "
    "&nbsp;·&nbsp; Datos: UCI Online Retail II", "small")]


# ------------------------------------------------------------------- build
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.6)
    canvas.setFillColor(MUTED)
    canvas.drawString(21 * mm, 12 * mm, "Previsión de demanda en retail")
    canvas.drawRightString(A4[0] - 21 * mm, 12 * mm, str(doc.page))
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(21 * mm, 15.5 * mm, A4[0] - 21 * mm, 15.5 * mm)
    canvas.restoreState()


doc = BaseDocTemplate(str(OUT), pagesize=A4,
                      leftMargin=21 * mm, rightMargin=21 * mm,
                      topMargin=18 * mm, bottomMargin=20 * mm,
                      title="Previsión de demanda en retail - proyecto explicado",
                      author="Marwan El Saabi")
frame = Frame(doc.leftMargin, doc.bottomMargin,
              doc.width, doc.height, id="main")
doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=footer)])
doc.build(story)
print("PDF generado:", OUT.name)
