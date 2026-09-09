"""
BOT $200 V8 - CON NOTIFICACIÓN A TELEGRAM
===========================================
Todo lo de V6, más:

8. Envía la alerta del modo en vivo a Telegram usando tu propio bot
   (creado con @BotFather). Se eligió Telegram en vez de WhatsApp porque
   la API gratuita de WhatsApp (CallMeBot) puede tener cupo lleno y no
   siempre está disponible; Telegram es gratis, oficial y sin límites de
   ese tipo.
   Requiere las variables de entorno TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID.
   Si no están configuradas, el bot simplemente no manda el mensaje pero
   sigue funcionando normal (imprime en consola y guarda el CSV).

--- Docstring original de V6 ---
Todo lo de V5, más un motor de backtest reescrito para eliminar los sesgos
que inflaban el resultado anterior:

1. SIN ventana artificial de 20 días: cada operación se sostiene hasta tocar
   stop o take-profit (con un límite de seguridad de 120 días, reportado aparte
   si se alcanza, para no simular una posición "colgada" para siempre).
2. UNA posición a la vez: si ya hay una operación abierta, se ignoran nuevas
   señales hasta cerrarla — así se simula cómo operarías en la vida real
   con un solo capital, no como si cada señal fuera independiente.
3. Costos reales: se resta comisión + slippage estimado a cada operación
   (configurable con --comision, en % por lado de la operación).
4. Validación fuera de muestra: el histórico se divide en un tramo de
   "entrenamiento" (primer 70%) y uno de "prueba" (últimos 30%), y se
   reportan métricas por separado. Si el resultado solo se sostiene en
   el tramo de entrenamiento, es una señal de sobreajuste (overfitting).
5. Métricas de trader real: win rate, expectancy, profit factor,
   max drawdown y curva de equity simulada.

IMPORTANTE: ningún backtest es "perfecto". Este es más honesto que el
anterior, pero sigue sin incluir impuestos, no modela huecos de apertura
(gaps) más allá de lo que ya captura el High/Low diario, y el periodo
histórico disponible puede no representar todos los regímenes de mercado
posibles. Trátalo como evidencia, no como garantía.
"""

import sys
import time
import argparse
import csv
import os
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
import pandas as pd
import yfinance as yf

TICKER = "SPY"
LOG_FILE = "senales_spy.csv"
MONTO_MXN = 200

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


# ---------------------------------------------------------------------------
# Notificación por Telegram — falla silenciosamente si no está configurado,
# para que el bot nunca se caiga por un problema de notificación
# ---------------------------------------------------------------------------
def enviar_telegram(mensaje):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("ℹ️ Telegram no configurado (faltan TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID) — se omite el envío.")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        datos = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": mensaje}).encode("utf-8")
        req = urllib.request.Request(
            url, data=datos, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
        print("✅ Alerta enviada a Telegram.")
    except Exception as e:
        print(f"⚠️ No se pudo enviar la alerta a Telegram: {e}")
LIMITE_DIAS_OPERACION = 120  # salvaguarda: si no toca stop ni TP en este plazo, se cierra a mercado


# ---------------------------------------------------------------------------
# Descarga de datos con reintentos
# ---------------------------------------------------------------------------
def descargar_datos(periodo="1y", intentos=3, espera=5):
    for intento in range(1, intentos + 1):
        try:
            datos = yf.download(
                TICKER, period=periodo, interval="1d",
                auto_adjust=True, progress=False
            )
            if not datos.empty:
                if isinstance(datos.columns, pd.MultiIndex):
                    datos.columns = datos.columns.get_level_values(0)
                return datos
            print(f"⚠️ Intento {intento}: datos vacíos, reintentando...")
        except Exception as e:
            print(f"⚠️ Intento {intento} falló: {e}")
        time.sleep(espera)
    return None


# ---------------------------------------------------------------------------
# Indicadores
# ---------------------------------------------------------------------------
def calcular_indicadores(datos):
    datos = datos.copy()
    datos["EMA20"] = datos["Close"].ewm(span=20, adjust=False).mean()
    datos["EMA50"] = datos["Close"].ewm(span=50, adjust=False).mean()
    datos["EMA200"] = datos["Close"].ewm(span=200, adjust=False).mean()

    delta = datos["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    datos["RSI"] = 100 - (100 / (1 + rs))

    high_low = datos["High"] - datos["Low"]
    high_close = (datos["High"] - datos["Close"].shift()).abs()
    low_close = (datos["Low"] - datos["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    datos["ATR14"] = tr.rolling(window=14).mean()

    datos["Vol_Avg20"] = datos["Volume"].rolling(window=20).mean()

    return datos


def evaluar_señal(fila):
    precio = fila["Close"]
    ema20, ema50, ema200 = fila["EMA20"], fila["EMA50"], fila["EMA200"]
    rsi = fila["RSI"]
    atr = fila["ATR14"]
    vol_hoy, vol_avg = fila["Volume"], fila["Vol_Avg20"]

    condicion_macro = precio > ema200 and ema20 > ema50
    condicion_momentum = 40 < rsi < 65
    condicion_volumen = vol_hoy > vol_avg

    if condicion_macro and condicion_momentum and condicion_volumen:
        stop = precio - 1.5 * atr
        tp = precio + 3.0 * atr
        return "COMPRAR", stop, tp
    elif precio < ema200:
        return "NO_COMPRAR", None, None
    elif rsi >= 65:
        return "ESPERAR", None, None
    else:
        return "NEUTRAL", None, None


def registrar_señal(fecha, precio, señal, stop, tp, rsi, atr):
    existe = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not existe:
            writer.writerow(["fecha", "precio", "señal", "stop", "take_profit", "rsi", "atr"])
        writer.writerow([fecha, f"{precio:.2f}", señal,
                          f"{stop:.2f}" if stop else "",
                          f"{tp:.2f}" if tp else "",
                          f"{rsi:.2f}", f"{atr:.2f}"])


# ---------------------------------------------------------------------------
# Modo en vivo
# ---------------------------------------------------------------------------
def correr_en_vivo():
    print("🤖 BOT $200 V8")
    print("==================================")
    print("Descargando datos de SPY...")

    datos = descargar_datos(periodo="1y")
    if datos is None:
        print("❌ No se pudieron obtener datos tras varios intentos. Revisa tu conexión o instala curl_cffi.")
        sys.exit(1)

    datos = calcular_indicadores(datos)
    fila = datos.iloc[-1]
    fecha_dato = datos.index[-1]

    señal, stop, tp = evaluar_señal(fila)

    print(f"\n📅 Última vela: {fecha_dato.strftime('%Y-%m-%d')}")
    print("📊 RADIOGRAFÍA")
    print("----------------------------")
    print(f"Precio Actual SPY:       ${fila['Close']:.2f} USD")
    print(f"EMA200:                  ${fila['EMA200']:.2f} ({'POR ARRIBA 🟢' if fila['Close'] > fila['EMA200'] else 'POR ABAJO 🔴'})")
    print(f"RSI:                     {fila['RSI']:.2f}")
    print(f"ATR14 (volatilidad):     ${fila['ATR14']:.2f}")
    print(f"Volumen vs promedio 20d: {'ALTO 📈' if fila['Volume'] > fila['Vol_Avg20'] else 'BAJO 📉'}")

    print("\n🚨 ALERTA:")
    print("----------------------------------------")
    fecha_str = fecha_dato.strftime("%Y-%m-%d")
    if señal == "COMPRAR":
        print("🔔 🟢 COMPRAR")
        print(f"   - Compra ~${MONTO_MXN} MXN de SPY (convierte a USD con el tipo de cambio del día)")
        print(f"   - Stop-Loss dinámico (1.5x ATR):  ${stop:.2f}")
        print(f"   - Take-Profit dinámico (3x ATR):  ${tp:.2f}")
        mensaje_tg = (
            f"🤖 BOT SPY [{fecha_str}]\n"
            f"🟢 COMPRAR\n"
            f"Precio: ${fila['Close']:.2f} USD\n"
            f"Stop-Loss: ${stop:.2f}\n"
            f"Take-Profit: ${tp:.2f}\n"
            f"RSI: {fila['RSI']:.1f} | ATR: ${fila['ATR14']:.2f}"
        )
    elif señal == "NO_COMPRAR":
        print("🔔 🔴 ZONA DE PELIGRO / NO COMPRAR — precio bajo la EMA200")
        mensaje_tg = f"🤖 BOT SPY [{fecha_str}]\n🔴 NO COMPRAR — precio bajo EMA200 (${fila['Close']:.2f} vs ${fila['EMA200']:.2f})"
    elif señal == "ESPERAR":
        print("🔔 🟡 ESPERAR — RSI cerca de sobrecompra")
        mensaje_tg = f"🤖 BOT SPY [{fecha_str}]\n🟡 ESPERAR — RSI en {fila['RSI']:.1f} (zona de sobrecompra)"
    else:
        print("🔔 🟡 NEUTRAL — faltan condiciones de volumen o momentum")
        mensaje_tg = f"🤖 BOT SPY [{fecha_str}]\n🟡 NEUTRAL — sin condiciones suficientes hoy"

    registrar_señal(fecha_str, fila["Close"], señal, stop, tp, fila["RSI"], fila["ATR14"])
    print(f"\n📝 Señal registrada en {LOG_FILE}")
    print("⚠️ Esto es una alerta informativa, no ejecuta órdenes ni es asesoría financiera.")

    enviar_telegram(mensaje_tg)


# ---------------------------------------------------------------------------
# Motor de simulación: una posición a la vez, sin ventana artificial
# ---------------------------------------------------------------------------
def simular_tramo(datos, comision_pct):
    """
    Recorre 'datos' vela por vela. Solo puede haber una operación abierta.
    Devuelve la lista de operaciones cerradas (retorno neto ya con costos)
    y cuántas se cerraron por límite de tiempo en vez de stop/TP.
    """
    operaciones = []
    cerradas_por_tiempo = 0
    en_posicion = False
    precio_entrada = stop_actual = tp_actual = None
    dia_entrada = 0

    for i in range(len(datos)):
        fila = datos.iloc[i]

        if en_posicion:
            dias_transcurridos = i - dia_entrada
            salida = None
            if fila["Low"] <= stop_actual:
                salida = stop_actual
            elif fila["High"] >= tp_actual:
                salida = tp_actual
            elif dias_transcurridos >= LIMITE_DIAS_OPERACION:
                salida = fila["Close"]
                cerradas_por_tiempo += 1

            if salida is not None:
                retorno_bruto = (salida - precio_entrada) / precio_entrada
                # comisión aplicada en la entrada y en la salida
                retorno_neto = retorno_bruto - (2 * comision_pct)
                operaciones.append(retorno_neto)
                en_posicion = False
            continue  # mientras esté en posición, no evalúa nuevas señales

        señal, stop, tp = evaluar_señal(fila)
        if señal == "COMPRAR":
            en_posicion = True
            precio_entrada = fila["Close"]
            stop_actual = stop
            tp_actual = tp
            dia_entrada = i

    return operaciones, cerradas_por_tiempo


def calcular_metricas(operaciones, cerradas_por_tiempo, etiqueta):
    print(f"\n📈 {etiqueta}")
    print("-" * 50)
    if not operaciones:
        print("No se cerraron operaciones en este tramo.")
        return

    ganadoras = [r for r in operaciones if r > 0]
    perdedoras = [r for r in operaciones if r <= 0]
    win_rate = len(ganadoras) / len(operaciones) * 100

    ganancia_promedio = sum(ganadoras) / len(ganadoras) if ganadoras else 0
    perdida_promedio = sum(perdedoras) / len(perdedoras) if perdedoras else 0
    expectancy = (win_rate / 100 * ganancia_promedio) + ((1 - win_rate / 100) * perdida_promedio)

    suma_ganancias = sum(ganadoras) if ganadoras else 0
    suma_perdidas = abs(sum(perdedoras)) if perdedoras else 0
    profit_factor = (suma_ganancias / suma_perdidas) if suma_perdidas > 0 else float("inf")

    # Curva de equity y max drawdown (retornos compuestos, base 1.0)
    equity = 1.0
    pico = 1.0
    max_drawdown = 0.0
    for r in operaciones:
        equity *= (1 + r)
        pico = max(pico, equity)
        drawdown = (equity - pico) / pico
        max_drawdown = min(max_drawdown, drawdown)

    retorno_total_compuesto = (equity - 1) * 100

    print(f"Operaciones cerradas:     {len(operaciones)}  ({cerradas_por_tiempo} cerradas por límite de {LIMITE_DIAS_OPERACION} días)")
    print(f"Win rate:                 {win_rate:.1f}%")
    print(f"Retorno promedio/op:      {sum(operaciones)/len(operaciones)*100:.2f}%")
    print(f"Ganancia promedio:        {ganancia_promedio*100:.2f}%")
    print(f"Pérdida promedio:         {perdida_promedio*100:.2f}%")
    print(f"Expectancy (por op):      {expectancy*100:.2f}%")
    print(f"Profit factor:            {profit_factor:.2f}  (>1 = gana más de lo que pierde)")
    print(f"Máximo drawdown:          {max_drawdown*100:.2f}%")
    print(f"Retorno total compuesto:  {retorno_total_compuesto:.2f}%  (reinvirtiendo cada operación)")


# ---------------------------------------------------------------------------
# Backtest con split entrenamiento/prueba
# ---------------------------------------------------------------------------
def correr_backtest(periodo="5y", comision_pct=0.001):
    print(f"🔬 BACKTEST RIGUROSO sobre los últimos {periodo} de SPY")
    print(f"   (comisión + slippage asumidos: {comision_pct*100:.2f}% por lado)")
    print("=" * 60)

    datos = descargar_datos(periodo=periodo)
    if datos is None:
        print("❌ No se pudieron obtener datos.")
        sys.exit(1)

    datos = calcular_indicadores(datos).dropna().reset_index(drop=True)

    corte = int(len(datos) * 0.7)
    entrenamiento = datos.iloc[:corte]
    prueba = datos.iloc[corte:]

    print(f"Rango total:         {len(datos)} velas")
    print(f"Entrenamiento (70%): {len(entrenamiento)} velas")
    print(f"Prueba (30%):        {len(prueba)} velas")

    ops_train, ct_train = simular_tramo(entrenamiento, comision_pct)
    calcular_metricas(ops_train, ct_train, "TRAMO DE ENTRENAMIENTO (primer 70% del historial)")

    ops_test, ct_test = simular_tramo(prueba, comision_pct)
    calcular_metricas(ops_test, ct_test, "TRAMO DE PRUEBA / OUT-OF-SAMPLE (últimos 30%, no visto)")

    print("\n" + "=" * 60)
    if ops_train and ops_test:
        wr_train = len([r for r in ops_train if r > 0]) / len(ops_train) * 100
        wr_test = len([r for r in ops_test if r > 0]) / len(ops_test) * 100
        diferencia = abs(wr_train - wr_test)
        if diferencia > 15:
            print(f"⚠️ El win rate cambia {diferencia:.1f} puntos entre entrenamiento y prueba.")
            print("   Eso es señal de que el resultado puede depender mucho del periodo,")
            print("   no de un edge consistente. Trátalo con cautela.")
        else:
            print(f"✅ El win rate se mantiene razonablemente estable entre tramos ({diferencia:.1f} pts de diferencia).")
            print("   Es una señal (no una garantía) de que el comportamiento es más consistente.")

    print("\n⚠️ Sigue sin incluir impuestos ni modelar gaps más allá del High/Low diario.")
    print("   Ningún backtest garantiza resultados futuros. Úsalo como evidencia, no como certeza.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true", help="Corre el backtest en vez del modo en vivo")
    parser.add_argument("--periodo", default="5y", help="Periodo histórico para el backtest (ej. 5y, 10y, max)")
    parser.add_argument("--comision", type=float, default=0.001,
                         help="Comisión + slippage estimado por lado, como fracción (0.001 = 0.1%%)")
    args = parser.parse_args()

    if args.backtest:
        correr_backtest(periodo=args.periodo, comision_pct=args.comision)
    else:
        correr_en_vivo()
