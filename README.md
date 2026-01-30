# 🚀 Sistema de Trading Algorítmico para Criptomonedas

Sistema mejorado de trading algorítmico con Machine Learning, validación walk-forward y backtesting riguroso.

## 📋 Índice

- [Características](#características)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso Rápido](#uso-rápido)
- [Configuración](#configuración)
- [Documentación](#documentación)
- [Advertencias](#advertencias)

## ✨ Características

- ✅ **Machine Learning robusto**: Random Forest con validación walk-forward
- ✅ **Backtesting riguroso**: Métricas profesionales (Sharpe, Profit Factor, Drawdown)
- ✅ **Trading en tiempo real**: Sistema listo para producción
- ✅ **Gestión de riesgo**: Stop-loss y take-profit basados en ATR
- ✅ **Multi-ticker**: Procesa múltiples criptomonedas independientemente
- ✅ **Persistencia**: Guarda y carga modelos entrenados
- ✅ **Monitoreo continuo**: Escanea mercado automáticamente

## 🔧 Requisitos

### Software
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Hardware Recomendado
- 4 GB RAM mínimo
- Conexión a internet estable

## 📦 Instalación

### 1. Clonar o descargar el repositorio

```bash
# Si usas git
git clone https://github.com/tu-usuario/trading-system.git
cd trading-system

# O descarga y descomprime el ZIP
```

### 2. Crear entorno virtual (recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Contenido de `requirements.txt`:**
```
yfinance>=0.2.28
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
requests>=2.31.0
```

### 4. Verificar instalación

```bash
python -c "import yfinance, sklearn, pandas; print('✅ Instalación exitosa')"
```

## 🚀 Uso Rápido

### Opción 1: Menú Interactivo

```bash
python ejemplo_uso.py
```

Esto abrirá un menú con las siguientes opciones:
1. Análisis completo de un ticker
2. Análisis rápido (solo señal actual)
3. Monitoreo continuo
4. Procesar todos los tickers
5. Ver configuración

### Opción 2: Análisis de un Ticker Específico

```bash
# Análisis rápido de Bitcoin
python ejemplo_uso.py BTC-USD

# Análisis rápido de Ethereum
python ejemplo_uso.py ETH-USD
```

### Opción 3: Procesamiento Batch

```bash
# Procesar todos los tickers configurados
python ejemplo_uso.py --batch
```

### Opción 4: Monitoreo Continuo

```bash
# Monitorear BTC y ETH cada 60 minutos
python ejemplo_uso.py --monitor BTC-USD,ETH-USD 60
```

### Opción 5: Usar el Sistema Directamente

```python
from trading_system_improved import SistemaTradingTicker

# Crear sistema para Bitcoin
sistema = SistemaTradingTicker("BTC-USD")

# 1. Descargar datos
sistema.descargar_datos()

# 2. Entrenar modelos
sistema.entrenar_modelos()

# 3. Ejecutar backtest
sistema.ejecutar_backtest()

# 4. Evaluar viabilidad
viable, criterios = sistema.es_viable()
print(f"Viable: {viable}, Criterios: {criterios}/6")

# 5. Analizar tiempo real
if viable:
    señal = sistema.analizar_tiempo_real()
    if señal:
        print(f"Señal: {señal['señal']}")
        print(f"Confianza: {señal['confianza']:.0%}")
```

## ⚙️ Configuración

### Modificar Tickers

Edita `trading_system_improved.py`:

```python
class TradingConfig:
    ACTIVOS = [
        "BTC-USD",   # Bitcoin
        "ETH-USD",   # Ethereum
        "SOL-USD",   # Solana
        # Añade más aquí
    ]
```

### Ajustar Parámetros de Riesgo

```python
class TradingConfig:
    MULTIPLICADOR_SL = 2.0      # Stop-loss (2× ATR)
    MULTIPLICADOR_TP = 3.0      # Take-profit (3× ATR)
    RATIO_MINIMO_RR = 1.5       # Ratio riesgo/recompensa mínimo
    MAX_RIESGO_POR_OPERACION = 0.02  # 2% del capital por operación
```

### Cambiar Umbrales de Trading

```python
class TradingConfig:
    UMBRAL_PROBABILIDAD_MIN = 0.65  # Probabilidad mínima para operar
    UMBRAL_CONFIANZA_MIN = 0.60     # Confianza mínima del modelo
```

### Ajustar Períodos de Datos

```python
class TradingConfig:
    DIAS_ENTRENAMIENTO = 365  # Datos para entrenar (1 año)
    DIAS_VALIDACION = 90      # Datos para validar (3 meses)
    DIAS_BACKTEST = 30        # Datos para backtest (1 mes)
```

## 📊 Interpretación de Resultados

### Métricas de Backtesting

```
RESULTADOS BACKTESTING:
  Operaciones totales: 28
  Win Rate: 60.7%           # ✅ >50% es bueno
  Profit Factor: 1.85       # ✅ >1.3 es rentable
  Retorno total: +4.2%      # ✅ Positivo es bueno
  Max Drawdown: -8.3%       # ✅ <20% es aceptable
  Sharpe Ratio: 0.87        # ✅ >0.5 es bueno
```

### Criterios de Viabilidad

El sistema evalúa 6 criterios:
1. ✅ Tasa de éxito > 50%
2. ✅ Retorno total positivo
3. ✅ Profit Factor > 1.2
4. ✅ Max Drawdown < 25%
5. ✅ Mínimo 15 operaciones
6. ✅ Sharpe Ratio > 0

**Umbral:** Se requieren mínimo 4/6 criterios para considerar un ticker viable.

### Señales de Trading

```
🚨 SEÑAL DE TRADING - BTC-USD

📅 Fecha: 2026-01-29 19:30:00
💰 Precio actual: $45,678.00
🎯 Dirección: LONG

📊 CONFIANZA:
  Probabilidad: 72.3%        # Chance de éxito
  Confianza: 68.5%           # Certeza del modelo

💰 GESTIÓN DE RIESGO:
  🛑 Stop Loss: $44,890.00   # -1.7%
  🎯 Take Profit: $47,254.00 # +3.4%
  ⚖️ Ratio R:R: 2.0:1        # Ganas $2 por cada $1 arriesgado
```

## 📁 Estructura de Archivos

```
trading-system/
│
├── trading_system_improved.py   # Sistema principal mejorado
├── ejemplo_uso.py               # Ejemplos de uso y menú interactivo
├── requirements.txt             # Dependencias de Python
├── README.md                    # Este archivo
│
├── ANALISIS_MEJORAS.md         # Análisis técnico detallado
├── COMPARACION_SISTEMAS.md     # Comparación original vs mejorado
│
└── modelos_trading/            # Modelos entrenados (se crea automáticamente)
    ├── BTC-USD/
    │   ├── modelo_1h.pkl
    │   ├── modelo_2h.pkl
    │   └── ...
    └── ETH-USD/
        └── ...
```

## 🔐 Integración con Telegram (Opcional)

Para recibir alertas por Telegram:

### 1. Crear un bot

1. Habla con [@BotFather](https://t.me/BotFather) en Telegram
2. Ejecuta `/newbot` y sigue las instrucciones
3. Guarda el **token** que te da

### 2. Obtener tu Chat ID

1. Envía un mensaje a tu bot
2. Visita: `https://api.telegram.org/bot<TU_TOKEN>/getUpdates`
3. Busca tu `chat_id` en el JSON

### 3. Configurar variables de entorno

```bash
# Windows (CMD)
set TELEGRAM_BOT_TOKEN=tu_token_aqui
set TELEGRAM_CHAT_ID=tu_chat_id_aqui

# Linux/Mac
export TELEGRAM_BOT_TOKEN=tu_token_aqui
export TELEGRAM_CHAT_ID=tu_chat_id_aqui
```

### 4. Usar en el código

```python
from ejemplo_uso import enviar_alerta_telegram

señal = sistema.analizar_tiempo_real()
if señal and señal['confianza'] > 0.65:
    enviar_alerta_telegram(señal)
```

## 🧪 Testing y Validación

### Paper Trading (OBLIGATORIO antes de dinero real)

1. **Ejecutar sistema en modo simulación:**
   ```bash
   python ejemplo_uso.py --monitor BTC-USD,ETH-USD 60
   ```

2. **Registrar TODAS las señales:**
   - Fecha y hora
   - Precio de entrada
   - Stop-loss y take-profit
   - Resultado real (después de ejecutar)

3. **Validar por 30 días mínimo:**
   - Comparar performance real vs backtesting
   - Si difieren mucho (>20%), NO usar con dinero real

### Checklist Antes de Producción

- [ ] Sistema ejecutado exitosamente en todos los tickers
- [ ] Al menos 3 tickers identificados como viables
- [ ] Paper trading por mínimo 30 días
- [ ] Performance paper trading aceptable (Win rate >50%, PF >1.2)
- [ ] Gestión de riesgo definida (máximo 2% por operación)
- [ ] Stop-loss automáticos configurados en exchange
- [ ] Sistema de monitoreo y alertas funcionando

## ⚠️ ADVERTENCIAS IMPORTANTES

### 🔴 RIESGO FINANCIERO

- ⚠️ Trading de criptomonedas es **ALTAMENTE RIESGOSO**
- ⚠️ Puedes **PERDER TODO** tu capital invertido
- ⚠️ Este sistema **NO GARANTIZA** ganancias
- ⚠️ Performance pasado **NO PREDICE** resultados futuros

### 🔴 LIMITACIONES TÉCNICAS

1. **No incluye costos de transacción**
   - Fees de exchange (~0.1-0.5% por operación)
   - Slippage en órdenes de mercado
   - Restar ~0.3% a los retornos esperados

2. **Sensible a cambios de mercado**
   - Modelos pueden fallar en nuevos regímenes
   - Re-entrenar cada 1-2 semanas
   - Monitorear performance continuamente

3. **Requiere liquidez**
   - Solo usar en pares principales
   - Verificar volumen antes de operar

4. **Latencia y ejecución**
   - Sistema genera señales, TÚ ejecutas
   - Precio real puede diferir del teórico
   - Considerar órdenes limitadas vs mercado

### 🔴 RESPONSABILIDAD

- Este software se proporciona "tal cual"
- Sin garantías de ningún tipo
- El autor NO se hace responsable por pérdidas
- Usa bajo tu propio riesgo

## 📚 Documentación Adicional

- **[ANALISIS_MEJORAS.md](ANALISIS_MEJORAS.md)**: Análisis técnico detallado de las mejoras
- **[COMPARACION_SISTEMAS.md](COMPARACION_SISTEMAS.md)**: Comparación sistema original vs mejorado
- Código fuente está extensamente comentado

## 🐛 Solución de Problemas

### Error: "No hay datos disponibles"

**Causa:** yfinance no puede descargar datos para ese ticker.

**Solución:**
1. Verificar que el ticker existe en Yahoo Finance
2. Verificar conexión a internet
3. Intentar con otro ticker (ej: BTC-USD, ETH-USD)

### Error: "Datos insuficientes para backtesting"

**Causa:** No hay suficientes velas horarias en el período.

**Solución:**
1. Aumentar `DIAS_BACKTEST` a 60 o más
2. Usar datos de mayor timeframe (ej: 4h en lugar de 1h)

### Accuracy muy bajo (~50%)

**Causa:** Modelo no está aprendiendo patrones reales.

**Solución:**
1. Aumentar `DIAS_ENTRENAMIENTO` a 730 (2 años)
2. Probar con otros tickers
3. Ajustar hiperparámetros del Random Forest

### ModuleNotFoundError

**Causa:** Dependencias no instaladas.

**Solución:**
```bash
pip install -r requirements.txt
```

## 📧 Soporte

- **Issues:** Abre un issue en GitHub
- **Mejoras:** Pull requests son bienvenidos
- **Consultas:** Consulta la documentación primero

## 📜 Licencia

Este proyecto es para **uso educativo y de investigación únicamente**.

NO es:
- ❌ Asesoramiento financiero
- ❌ Garantía de ganancias
- ❌ Recomendación de inversión

**Consulta con un asesor financiero profesional antes de operar.**

---

## 🚀 Inicio Rápido en 3 Pasos

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Analizar Bitcoin
python ejemplo_uso.py BTC-USD

# 3. Si es viable, monitorear continuamente
python ejemplo_uso.py --monitor BTC-USD 60
```

---

**Versión:** 2.0 Mejorado  
**Última Actualización:** Enero 2026  
**Estado:** Beta - Testing recomendado  

**¡Buen trading! 🚀📈**
