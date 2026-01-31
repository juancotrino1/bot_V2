#!/usr/bin/env python3
"""
EJEMPLO DE USO - SISTEMA V3 OPTIMIZADO
======================================
Scripts rápidos para usar el sistema sin complicaciones
"""

from trading_system_v3_optimized import (
    TradingConfig, 
    SistemaTicker, 
    DataCache,
    ejecutar_sistema_completo,
    logger
)
import logging

# ============================================
# USO BÁSICO: EJECUTAR TODO
# ============================================

def ejemplo_1_ejecutar_todo():
    """
    Forma más simple: ejecuta todo el sistema con configuración por defecto
    """
    print("\n" + "="*80)
    print("EJEMPLO 1: EJECUTAR SISTEMA COMPLETO")
    print("="*80 + "\n")
    
    # Ejecutar con defaults
    resultados = ejecutar_sistema_completo()
    
    # Los resultados ya se muestran en consola automáticamente
    # Pero puedes acceder a ellos programáticamente:
    
    for ticker, data in resultados.items():
        if data['viable']:
            print(f"\n✅ {ticker} es viable para trading")
            print(f"   Win Rate: {data['metricas']['win_rate']:.1%}")
            print(f"   Profit Factor: {data['metricas']['pf']:.2f}")


# ============================================
# USO INTERMEDIO: CONFIGURACIÓN PERSONALIZADA
# ============================================

def ejemplo_2_config_personalizada():
    """
    Configuración personalizada para trading conservador
    """
    print("\n" + "="*80)
    print("EJEMPLO 2: CONFIGURACIÓN CONSERVADORA")
    print("="*80 + "\n")
    
    # Crear config personalizada
    config = TradingConfig(
        # Solo los mejores activos
        ACTIVOS=["BTC-USD", "ETH-USD"],
        
        # Horizontes más cortos para más señales
        HORIZONTES=[1, 2, 4],
        
        # Umbrales más estrictos
        UMBRAL_PROBABILIDAD_MIN=0.70,  # Solo señales muy confiables
        UMBRAL_CONFIANZA_MIN=0.65,
        
        # Risk management conservador
        MULTIPLICADOR_SL=2.5,  # SL más amplio
        MULTIPLICADOR_TP=4.0,  # TP más ambicioso
        RATIO_MINIMO_RR=2.0,   # Mínimo 2:1 R:R
        
        # Logging detallado
        LOG_LEVEL="DEBUG"
    )
    
    # Ejecutar
    resultados = ejecutar_sistema_completo(config, paralelo=False)
    
    return resultados


# ============================================
# USO AVANZADO: ANÁLISIS DE UN TICKER
# ============================================

def ejemplo_3_ticker_individual(ticker="BTC-USD"):
    """
    Análisis detallado de un solo ticker
    """
    print("\n" + "="*80)
    print(f"EJEMPLO 3: ANÁLISIS INDIVIDUAL - {ticker}")
    print("="*80 + "\n")
    
    config = TradingConfig()
    cache = DataCache(config.CACHE_DIR)
    
    # Crear sistema
    sistema = SistemaTicker(ticker, config, cache)
    
    # Pipeline completo
    if not sistema.descargar_datos():
        print(f"❌ Error descargando {ticker}")
        return None
    
    if not sistema.entrenar_modelos():
        print(f"❌ Error entrenando {ticker}")
        return None
    
    if not sistema.backtest():
        print(f"❌ Error en backtest {ticker}")
        return None
    
    # Evaluar
    viable, criterios = sistema.es_viable()
    
    print(f"\n{'='*80}")
    print(f"RESULTADO: {'✅ VIABLE' if viable else '❌ NO VIABLE'}")
    print(f"Criterios cumplidos: {criterios}/6")
    print(f"{'='*80}")
    
    # Mostrar métricas detalladas
    if sistema.metricas_bt:
        m = sistema.metricas_bt
        print(f"\n📊 MÉTRICAS DE BACKTEST:")
        print(f"   Operaciones: {m['n_ops']}")
        print(f"   Win Rate: {m['win_rate']:.1%}")
        print(f"   Profit Factor: {m['pf']:.2f}")
        print(f"   Retorno Total: {m['ret_total']:.2%}")
        print(f"   Retorno Promedio: {m['ret_promedio']:.2%}")
        print(f"   Sharpe Ratio: {m['sharpe']:.2f}")
        print(f"   Max Drawdown: {m['max_dd']:.2%}")
    
    # Análisis actual
    if viable:
        print(f"\n🔍 Analizando condiciones actuales...")
        senal = sistema.analizar_actual()
        
        if senal:
            mostrar_senal_detallada(senal)
            sistema.guardar_modelos()
        else:
            print("   ✅ Sin señales en este momento")
    
    return sistema


# ============================================
# USO EXPERTO: MONITOREO CONTINUO
# ============================================

def ejemplo_4_monitoreo_continuo(intervalo_minutos=60):
    """
    Monitoreo continuo cada X minutos
    """
    import time
    from datetime import datetime
    
    print("\n" + "="*80)
    print(f"EJEMPLO 4: MONITOREO CONTINUO")
    print(f"Intervalo: {intervalo_minutos} minutos")
    print("⚠️  Presiona Ctrl+C para detener")
    print("="*80 + "\n")
    
    config = TradingConfig(
        ACTIVOS=["BTC-USD", "ETH-USD", "SOL-USD"],
        LOG_LEVEL="INFO"
    )
    
    iteracion = 0
    
    try:
        while True:
            iteracion += 1
            
            print(f"\n{'='*80}")
            print(f"⏰ ITERACIÓN {iteracion} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}\n")
            
            # Ejecutar análisis
            resultados = ejecutar_sistema_completo(config, paralelo=True)
            
            # Resumen de señales
            senales_activas = []
            for ticker, data in resultados.items():
                if data.get('senal') and data['senal']['confianza'] >= config.UMBRAL_CONFIANZA_MIN:
                    senal = data['senal']
                    # Filtro mean reversion
                    if senal.get('evento_mr') == senal['senal']:
                        senales_activas.append(senal)
            
            if senales_activas:
                print(f"\n🚨 {len(senales_activas)} SEÑALES ACTIVAS:")
                for senal in senales_activas:
                    print(f"   {senal['ticker']}: {senal['senal']} (Conf: {senal['confianza']:.1%})")
            else:
                print("\n✅ Sin señales en esta iteración")
            
            # Esperar
            print(f"\n⏳ Próxima revisión en {intervalo_minutos} minutos...")
            time.sleep(intervalo_minutos * 60)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Monitoreo detenido por usuario")


# ============================================
# UTILIDADES
# ============================================

def mostrar_senal_detallada(senal):
    """Muestra señal de forma legible"""
    
    print(f"\n{'='*80}")
    print(f"🚨 SEÑAL DE TRADING - {senal['ticker']}")
    print(f"{'='*80}")
    
    print(f"\n📅 Fecha: {senal['fecha'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💰 Precio: ${senal['precio']:,.2f}")
    print(f"🎯 Dirección: {senal['senal']}")
    
    print(f"\n📊 CONFIANZA:")
    print(f"   Probabilidad: {senal['prob']:.1%}")
    print(f"   Confianza: {senal['confianza']:.1%}")
    
    print(f"\n💰 GESTIÓN DE RIESGO:")
    riesgo_pct = abs(senal['sl'] / senal['precio'] - 1) * 100
    reward_pct = abs(senal['tp'] / senal['precio'] - 1) * 100
    
    print(f"   🛑 Stop Loss: ${senal['sl']:,.2f} ({riesgo_pct:.2f}%)")
    print(f"   🎯 Take Profit: ${senal['tp']:,.2f} ({reward_pct:.2f}%)")
    print(f"   ⚖️  Ratio R:R: {senal['rr']:.2f}:1")
    
    print(f"\n📈 CONTEXTO TÉCNICO:")
    print(f"   RSI: {senal['rsi']:.1f}")
    print(f"   Tendencia: {senal['tendencia']}")
    print(f"   Z-Score MR: {senal['z_mr']:.2f}")
    
    # Predicciones por horizonte
    if senal.get('preds'):
        print(f"\n🔮 PREDICCIONES POR HORIZONTE:")
        for h, pred in senal['preds'].items():
            direccion = "📈 LONG" if pred['prediccion'] == 1 else "📉 SHORT"
            print(f"   {h}h: {direccion} (Conf: {pred['confianza']:.1%})")
    
    # Recomendación
    if senal['confianza'] >= 0.70 and senal['rr'] >= 2.0:
        rec = "🟢 SEÑAL FUERTE - Considerar operación"
    elif senal['confianza'] >= 0.60 and senal['rr'] >= 1.5:
        rec = "🟡 SEÑAL MODERADA - Monitorear"
    else:
        rec = "🔴 SEÑAL DÉBIL - Esperar mejor oportunidad"
    
    print(f"\n💡 RECOMENDACIÓN: {rec}")
    print(f"{'='*80}\n")


def mostrar_comparacion_tickers(resultados):
    """Compara performance de múltiples tickers"""
    
    print("\n" + "="*80)
    print("📊 COMPARACIÓN DE TICKERS")
    print("="*80 + "\n")
    
    # Filtrar viables
    viables = {t: r for t, r in resultados.items() if r.get('viable')}
    
    if not viables:
        print("❌ No hay tickers viables para comparar")
        return
    
    # Ordenar por profit factor
    ranking = sorted(
        viables.items(), 
        key=lambda x: x[1]['metricas']['pf'], 
        reverse=True
    )
    
    print(f"{'Ticker':<12} {'Win Rate':<12} {'PF':<8} {'Retorno':<12} {'Sharpe':<8}")
    print("-" * 80)
    
    for ticker, data in ranking:
        m = data['metricas']
        print(f"{ticker:<12} {m['win_rate']:>10.1%}  {m['pf']:>6.2f}  {m['ret_total']:>10.2%}  {m['sharpe']:>6.2f}")
    
    print("\n" + "="*80)


# ============================================
# MENÚ INTERACTIVO
# ============================================

def menu_principal():
    """Menú interactivo para ejecutar ejemplos"""
    
    while True:
        print("\n" + "="*80)
        print("🚀 SISTEMA DE TRADING V3 - MENÚ DE EJEMPLOS")
        print("="*80)
        print("\n1. Ejecutar sistema completo (todos los tickers)")
        print("2. Configuración conservadora (solo BTC y ETH)")
        print("3. Analizar ticker individual")
        print("4. Monitoreo continuo")
        print("5. Comparar performance de tickers")
        print("6. Modo debug (logging detallado)")
        print("0. Salir")
        
        opcion = input("\n👉 Selecciona opción: ").strip()
        
        try:
            if opcion == "1":
                ejemplo_1_ejecutar_todo()
            
            elif opcion == "2":
                ejemplo_2_config_personalizada()
            
            elif opcion == "3":
                ticker = input("Ticker (ej: BTC-USD): ").strip().upper()
                ejemplo_3_ticker_individual(ticker)
            
            elif opcion == "4":
                intervalo = int(input("Intervalo en minutos (ej: 60): "))
                ejemplo_4_monitoreo_continuo(intervalo)
            
            elif opcion == "5":
                print("\nEjecutando análisis completo...")
                resultados = ejecutar_sistema_completo()
                mostrar_comparacion_tickers(resultados)
            
            elif opcion == "6":
                logger.setLevel(logging.DEBUG)
                print("\n✅ Modo DEBUG activado")
                ejemplo_1_ejecutar_todo()
            
            elif opcion == "0":
                print("\n👋 ¡Hasta luego!")
                break
            
            else:
                print("\n❌ Opción inválida")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Operación cancelada")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.error("Error en menú", exc_info=True)


# ============================================
# PUNTO DE ENTRADA
# ============================================

if __name__ == "__main__":
    import sys
    
    # Si se ejecuta sin argumentos, mostrar menú
    if len(sys.argv) == 1:
        menu_principal()
    
    # Argumentos de línea de comandos
    elif sys.argv[1] == "--full":
        ejemplo_1_ejecutar_todo()
    
    elif sys.argv[1] == "--conservative":
        ejemplo_2_config_personalizada()
    
    elif sys.argv[1] == "--ticker":
        ticker = sys.argv[2] if len(sys.argv) > 2 else "BTC-USD"
        ejemplo_3_ticker_individual(ticker)
    
    elif sys.argv[1] == "--monitor":
        intervalo = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        ejemplo_4_monitoreo_continuo(intervalo)
    
    elif sys.argv[1] == "--help":
        print("""
Uso:
  python ejemplo_uso_v3.py                    # Menú interactivo
  python ejemplo_uso_v3.py --full             # Sistema completo
  python ejemplo_uso_v3.py --conservative     # Config conservadora
  python ejemplo_uso_v3.py --ticker BTC-USD   # Analizar ticker
  python ejemplo_uso_v3.py --monitor 60       # Monitoreo cada 60 min
        """)
    
    else:
        print(f"❌ Opción desconocida: {sys.argv[1]}")
        print("Usa --help para ver opciones disponibles")
