#!/usr/bin/env python3
"""
EJEMPLO DE USO RÁPIDO DEL SISTEMA DE TRADING MEJORADO

Este script muestra cómo usar el sistema para:
1. Entrenamiento inicial
2. Análisis de un ticker
3. Monitoreo en tiempo real
"""

from trading_system_improved import (
    SistemaTradingTicker,
    TradingConfig
)
import time

# ============================================
# EJEMPLO 1: ANÁLISIS COMPLETO DE UN TICKER
# ============================================

def ejemplo_analisis_completo():
    """Análisis completo de BTC: entrenamiento + backtest + señal actual"""
    
    print("=" * 80)
    print("EJEMPLO 1: ANÁLISIS COMPLETO DE BTC-USD")
    print("=" * 80)
    
    # Crear sistema para BTC
    sistema = SistemaTradingTicker("BTC-USD")
    
    # 1. Descargar datos
    if not sistema.descargar_datos():
        print("❌ Error descargando datos")
        return
    
    # 2. Entrenar modelos
    if not sistema.entrenar_modelos():
        print("❌ Error entrenando modelos")
        return
    
    # 3. Ejecutar backtest
    if not sistema.ejecutar_backtest():
        print("❌ Error en backtest")
        return
    
    # 4. Evaluar viabilidad
    viable, criterios = sistema.es_viable()
    
    print(f"\n{'='*80}")
    print(f"RESULTADO: {'✅ VIABLE' if viable else '❌ NO VIABLE'}")
    print(f"Criterios cumplidos: {criterios}/6")
    print(f"{'='*80}")
    
    # 5. Si es viable, analizar señal actual
    if viable:
        print("\n🔍 Analizando condiciones actuales...")
        señal = sistema.analizar_tiempo_real()
        
        if señal:
            mostrar_señal(señal)
            sistema.guardar_modelos()
        else:
            print("✅ No hay señales en este momento")
    
    return sistema, viable


# ============================================
# EJEMPLO 2: ANÁLISIS RÁPIDO (SOLO TIEMPO REAL)
# ============================================

def ejemplo_analisis_rapido(ticker="ETH-USD"):
    """
    Análisis rápido usando modelos ya entrenados
    (Asume que ya ejecutaste el sistema completo antes)
    """
    
    print("=" * 80)
    print(f"EJEMPLO 2: ANÁLISIS RÁPIDO DE {ticker}")
    print("=" * 80)
    
    sistema = SistemaTradingTicker(ticker)
    
    # Intentar cargar modelos existentes
    from pathlib import Path
    path_modelos = TradingConfig.MODELOS_DIR / ticker
    
    if not path_modelos.exists():
        print(f"⚠️ No hay modelos entrenados para {ticker}")
        print("Ejecuta primero ejemplo_analisis_completo()")
        return None
    
    # Descargar solo datos recientes
    sistema.descargar_datos()
    
    # Analizar
    señal = sistema.analizar_tiempo_real()
    
    if señal:
        mostrar_señal(señal)
    else:
        print(f"✅ {ticker}: Sin señales actualmente")
    
    return señal


# ============================================
# EJEMPLO 3: MONITOREO CONTINUO
# ============================================

def ejemplo_monitoreo_continuo(tickers=["BTC-USD", "ETH-USD"], intervalo_minutos=60):
    """
    Monitorea múltiples tickers continuamente
    
    Args:
        tickers: Lista de tickers a monitorear
        intervalo_minutos: Frecuencia de revisión
    """
    
    print("=" * 80)
    print(f"EJEMPLO 3: MONITOREO CONTINUO")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Intervalo: {intervalo_minutos} minutos")
    print("=" * 80)
    print("\n⚠️ Presiona Ctrl+C para detener\n")
    
    iteracion = 0
    
    try:
        while True:
            iteracion += 1
            print(f"\n{'='*80}")
            print(f"ITERACIÓN {iteracion} - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}")
            
            señales_detectadas = []
            
            for ticker in tickers:
                print(f"\n🔍 Analizando {ticker}...")
                
                try:
                    señal = ejemplo_analisis_rapido(ticker)
                    if señal and señal['confianza'] >= TradingConfig.UMBRAL_CONFIANZA_MIN:
                        señales_detectadas.append(señal)
                except Exception as e:
                    print(f"❌ Error en {ticker}: {e}")
            
            # Resumen
            if señales_detectadas:
                print(f"\n{'='*80}")
                print(f"🚨 {len(señales_detectadas)} SEÑALES DETECTADAS")
                print(f"{'='*80}")
                for señal in señales_detectadas:
                    print(f"\n{señal['ticker']}: {señal['señal']} (Confianza: {señal['confianza']:.0%})")
            else:
                print(f"\n✅ Sin señales en esta iteración")
            
            # Esperar
            print(f"\n⏳ Próxima revisión en {intervalo_minutos} minutos...")
            time.sleep(intervalo_minutos * 60)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Monitoreo detenido por el usuario")


# ============================================
# EJEMPLO 4: BATCH PROCESSING DE TODOS LOS TICKERS
# ============================================

def ejemplo_procesar_todos():
    """Procesa todos los tickers configurados y muestra resumen"""
    
    print("=" * 80)
    print("EJEMPLO 4: PROCESAMIENTO COMPLETO DE TODOS LOS TICKERS")
    print("=" * 80)
    
    resultados = {}
    tickers_viables = []
    
    for ticker in TradingConfig.ACTIVOS:
        print(f"\n{'='*80}")
        print(f"Procesando {ticker}...")
        print(f"{'='*80}")
        
        sistema = SistemaTradingTicker(ticker)
        
        # Pipeline completo
        if sistema.descargar_datos():
            if sistema.entrenar_modelos():
                if sistema.ejecutar_backtest():
                    viable, criterios = sistema.es_viable()
                    
                    resultados[ticker] = {
                        'viable': viable,
                        'criterios': criterios,
                        'metricas': sistema.metricas_backtest
                    }
                    
                    if viable:
                        tickers_viables.append(ticker)
                        sistema.guardar_modelos()
    
    # Resumen final
    print(f"\n{'='*80}")
    print("RESUMEN FINAL")
    print(f"{'='*80}")
    print(f"\nTickers procesados: {len(resultados)}/{len(TradingConfig.ACTIVOS)}")
    print(f"Tickers viables: {len(tickers_viables)}")
    
    if tickers_viables:
        print(f"\n✅ TICKERS VIABLES:")
        for ticker in tickers_viables:
            m = resultados[ticker]['metricas']
            print(f"\n  {ticker}:")
            print(f"    Win Rate: {m['tasa_exito']:.1%}")
            print(f"    Profit Factor: {m['profit_factor']:.2f}")
            print(f"    Retorno Total: {m['retorno_total']:.2%}")
            print(f"    Sharpe Ratio: {m['sharpe_ratio']:.2f}")
    
    return resultados, tickers_viables


# ============================================
# UTILIDADES
# ============================================

def mostrar_señal(señal):
    """Formatea y muestra una señal de trading"""
    
    print(f"\n{'='*80}")
    print(f"🚨 SEÑAL DE TRADING - {señal['ticker']}")
    print(f"{'='*80}")
    
    print(f"\n📅 Fecha: {señal['fecha']}")
    print(f"💰 Precio actual: ${señal['precio']:,.2f}")
    print(f"🎯 Dirección: {señal['señal']}")
    
    print(f"\n📊 CONFIANZA:")
    print(f"  Probabilidad: {señal['probabilidad']:.1%}")
    print(f"  Confianza: {señal['confianza']:.1%}")
    
    print(f"\n💰 GESTIÓN DE RIESGO:")
    print(f"  🛑 Stop Loss: ${señal['stop_loss']:,.2f} ({abs(señal['stop_loss']/señal['precio']-1)*100:.2f}%)")
    print(f"  🎯 Take Profit: ${señal['take_profit']:,.2f} ({abs(señal['take_profit']/señal['precio']-1)*100:.2f}%)")
    print(f"  ⚖️ Ratio R:R: {señal['ratio_rr']:.2f}:1")
    
    print(f"\n📈 CONTEXTO TÉCNICO:")
    print(f"  RSI: {señal.get('rsi', 'N/A'):.1f}")
    print(f"  Tendencia: {señal.get('tendencia', 'N/A')}")
    
    print(f"\n🔮 PREDICCIONES POR HORIZONTE:")
    for horizonte, pred in señal.get('predicciones_detalle', {}).items():
        direccion = "📈 ALCISTA" if pred['prediccion'] == 1 else "📉 BAJISTA"
        print(f"  {horizonte}h: {direccion} (Confianza: {pred['confianza']:.1%})")
    
    # Recomendación
    if señal['confianza'] >= 0.70 and señal['ratio_rr'] >= 2.0:
        recomendacion = "🟢 SEÑAL FUERTE - CONSIDERAR OPERACIÓN"
    elif señal['confianza'] >= 0.60 and señal['ratio_rr'] >= 1.5:
        recomendacion = "🟡 SEÑAL MODERADA - MONITOREAR"
    else:
        recomendacion = "🔴 SEÑAL DÉBIL - ESPERAR MEJOR OPORTUNIDAD"
    
    print(f"\n💡 RECOMENDACIÓN: {recomendacion}")
    print(f"{'='*80}")


def enviar_alerta_telegram(señal, bot_token=None, chat_id=None):
    """
    Envía alerta de trading por Telegram
    
    Args:
        señal: Diccionario con información de la señal
        bot_token: Token del bot de Telegram
        chat_id: ID del chat donde enviar
    """
    import os
    import requests
    
    if not bot_token:
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not chat_id:
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("⚠️ Credenciales de Telegram no configuradas")
        return False
    
    mensaje = f"""
🚨 *SEÑAL DE TRADING*

🪙 *{señal['ticker']}*
📅 {señal['fecha'].strftime('%Y-%m-%d %H:%M')}

💰 *Precio:* ${señal['precio']:,.2f}
🎯 *Dirección:* {señal['señal']}
📊 *Confianza:* {señal['confianza']:.0%}

*NIVELES:*
🛑 SL: ${señal['stop_loss']:,.2f}
🎯 TP: ${señal['take_profit']:,.2f}
⚖️ R:R: {señal['ratio_rr']:.2f}:1

RSI: {señal.get('rsi', 'N/A'):.0f}
Tendencia: {señal.get('tendencia', 'N/A')}
"""
    
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": mensaje,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error enviando a Telegram: {e}")
        return False


# ============================================
# MENÚ INTERACTIVO
# ============================================

def menu_principal():
    """Menú interactivo para ejecutar ejemplos"""
    
    while True:
        print("\n" + "=" * 80)
        print("SISTEMA DE TRADING - MENÚ DE EJEMPLOS")
        print("=" * 80)
        print("\n1. Análisis completo de un ticker (entrenamiento + backtest + señal)")
        print("2. Análisis rápido (solo señal actual)")
        print("3. Monitoreo continuo de múltiples tickers")
        print("4. Procesar todos los tickers configurados")
        print("5. Mostrar configuración actual")
        print("0. Salir")
        
        opcion = input("\n👉 Selecciona una opción: ")
        
        if opcion == "1":
            ticker = input("Ticker a analizar (ej: BTC-USD): ").upper()
            sistema = SistemaTradingTicker(ticker)
            ejemplo_analisis_completo()
        
        elif opcion == "2":
            ticker = input("Ticker a analizar (ej: ETH-USD): ").upper()
            ejemplo_analisis_rapido(ticker)
        
        elif opcion == "3":
            tickers_input = input("Tickers separados por comas (ej: BTC-USD,ETH-USD): ")
            tickers = [t.strip().upper() for t in tickers_input.split(",")]
            intervalo = int(input("Intervalo en minutos (ej: 60): "))
            ejemplo_monitoreo_continuo(tickers, intervalo)
        
        elif opcion == "4":
            confirmar = input("⚠️ Esto puede tomar varios minutos. ¿Continuar? (s/n): ")
            if confirmar.lower() == 's':
                ejemplo_procesar_todos()
        
        elif opcion == "5":
            print("\n" + "=" * 80)
            print("CONFIGURACIÓN ACTUAL")
            print("=" * 80)
            fechas = TradingConfig.get_fechas()
            print(f"\nTickers: {', '.join(TradingConfig.ACTIVOS)}")
            print(f"Intervalo: {TradingConfig.INTERVALO}")
            print(f"Horizontes: {TradingConfig.HORIZONTES} horas")
            print(f"Período entrenamiento: {TradingConfig.DIAS_ENTRENAMIENTO} días")
            print(f"Período backtest: {TradingConfig.DIAS_BACKTEST} días")
            print(f"Umbral confianza: {TradingConfig.UMBRAL_CONFIANZA_MIN:.0%}")
            print(f"Ratio R:R mínimo: {TradingConfig.RATIO_MINIMO_RR}")
        
        elif opcion == "0":
            print("\n👋 ¡Hasta luego!")
            break
        
        else:
            print("\n❌ Opción inválida")


# ============================================
# PUNTO DE ENTRADA
# ============================================

if __name__ == "__main__":
    import sys
    
    # Si se ejecuta sin argumentos, mostrar menú
    if len(sys.argv) == 1:
        menu_principal()
    
    # Si se pasa un ticker como argumento, hacer análisis rápido
    elif len(sys.argv) == 2:
        ticker = sys.argv[1].upper()
        print(f"\n🚀 Análisis rápido de {ticker}")
        ejemplo_analisis_rapido(ticker)
    
    # Modo batch
    elif sys.argv[1] == "--batch":
        ejemplo_procesar_todos()
    
    # Modo monitor
    elif sys.argv[1] == "--monitor":
        tickers = sys.argv[2].split(",") if len(sys.argv) > 2 else ["BTC-USD", "ETH-USD"]
        intervalo = int(sys.argv[3]) if len(sys.argv) > 3 else 60
        ejemplo_monitoreo_continuo(tickers, intervalo)
    
    else:
        print("Uso:")
        print("  python ejemplo_uso.py                    # Menú interactivo")
        print("  python ejemplo_uso.py BTC-USD            # Análisis rápido")
        print("  python ejemplo_uso.py --batch            # Procesar todos")
        print("  python ejemplo_uso.py --monitor BTC-USD,ETH-USD 60  # Monitoreo continuo")
