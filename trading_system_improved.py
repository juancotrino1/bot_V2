import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import os
import requests
import warnings
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
from pathlib import Path
import json

def cargar_ultima_senal():
    if os.path.exists("ultima_senal.json"):
        with open("ultima_senal.json") as f:
            return json.load(f)
    return None

def guardar_ultima_senal(senal):
    with open("ultima_senal.json", "w") as f:
        json.dump(senal, f)

def enviar_telegram(mensaje):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    print("DEBUG token:", "OK" if token else "NONE")
    print("DEBUG chat_id:", chat_id)

    if not token or not chat_id:
        print("⚠️ Telegram no configurado")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, data={"chat_id": chat_id, "text": mensaje})

    print("📨 Telegram status:", r.status_code)
    print("📨 Telegram response:", r.text)


warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN REALISTA
# ============================================

class TradingConfig:
    """Configuración centralizada del sistema"""
    
    # Timezone
    TIMEZONE = pytz.timezone('America/Bogota')
    
    # Períodos de tiempo
    INTERVALO = "1h"
    DIAS_ENTRENAMIENTO = 180  # 6 meses de datos históricos (reducido)
    DIAS_VALIDACION = 60      # 2 meses para validación
    DIAS_BACKTEST = 30        # 1 mes para backtesting final
    
    # Activos
    ACTIVOS = [
        "BTC-USD"
    ]
    
    # Parámetros técnicos
    VENTANA_VOLATILIDAD = 24
    VENTANA_TENDENCIA = 50
    ATR_PERIODO = 14
    RSI_PERIODO = 14
    
    # Horizontes de predicción REALISTAS
    HORIZONTES = [6, 12, 24]  # En horas
    
    # Gestión de riesgo REALISTA
    MULTIPLICADOR_SL = 2.0
    MULTIPLICADOR_TP = 3.0
    RATIO_MINIMO_RR = 1.5
    
    # Validación
    N_FOLDS_WF = 3
    MIN_MUESTRAS_ENTRENAMIENTO = 500
    MIN_MUESTRAS_CLASE = 50  # Aumentado
    
    # Umbrales REALISTAS
    UMBRAL_PROBABILIDAD_MIN = 0.60
    UMBRAL_CONFIANZA_MIN = 0.60
    
    # Persistencia
    MODELOS_DIR = Path("modelos_trading")
    
    @classmethod
    def get_fechas(cls):
        """Calcula fechas del sistema"""
        now = datetime.now(cls.TIMEZONE)
        return {
            'actual': now,
            'inicio_entrenamiento': now - timedelta(days=cls.DIAS_ENTRENAMIENTO + cls.DIAS_VALIDACION + cls.DIAS_BACKTEST),
            'inicio_validacion': now - timedelta(days=cls.DIAS_VALIDACION + cls.DIAS_BACKTEST),
            'inicio_backtest': now - timedelta(days=cls.DIAS_BACKTEST)
        }


# ============================================
# CÁLCULO DE INDICADORES SIN DATA LEAKAGE
# ============================================

class IndicadoresTecnicos:
    """Calcula indicadores técnicos SIN look-ahead bias"""
    
    @staticmethod
    def calcular_rsi(precios, periodo=14):
        """RSI sin look-ahead"""
        delta = precios.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=periodo, min_periods=periodo).mean()
        avg_loss = loss.rolling(window=periodo, min_periods=periodo).mean()
        
        # Primer cálculo correcto
        for i in range(periodo, len(precios)):
            if i == periodo:
                continue
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (periodo-1) + gain.iloc[i]) / periodo
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (periodo-1) + loss.iloc[i]) / periodo
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    @staticmethod
    def calcular_atr(df, periodo=14):
        """ATR sin look-ahead"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=periodo, min_periods=periodo).mean()
        return atr.fillna(method='bfill')
    
    @staticmethod
    def calcular_features_sin_leakage(df):
        """Calcula features SIN data leakage - usando solo datos pasados"""
        df = df.copy()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # 1. Retornos pasados (NO futuros)
        df['retorno_1h'] = close.pct_change(1)
        df['retorno_6h'] = close.pct_change(6)
        df['retorno_12h'] = close.pct_change(12)
        
        # 2. Volatilidad pasada
        df['volatilidad_24h'] = df['retorno_1h'].rolling(window=24, min_periods=12).std()
        
        # 3. RSI (solo pasado)
        df['RSI'] = IndicadoresTecnicos.calcular_rsi(close, TradingConfig.RSI_PERIODO)
        
        # 4. ATR (solo pasado)
        df['ATR'] = IndicadoresTecnicos.calcular_atr(df, TradingConfig.ATR_PERIODO)
        df['ATR_pct'] = df['ATR'] / close
        
        # 5. Medias móviles (solo pasado)
        df['SMA_20'] = close.rolling(window=20, min_periods=10).mean()
        df['SMA_50'] = close.rolling(window=50, min_periods=25).mean()
        df['EMA_12'] = close.ewm(span=12, min_periods=6).mean()
        
        # 6. Diferencias con medias móviles
        df['dist_sma_20'] = (close - df['SMA_20']) / df['SMA_20']
        df['dist_sma_50'] = (close - df['SMA_50']) / df['SMA_50']
        
        # 7. Tendencia simple
        df['tendencia_sma'] = (df['SMA_20'] > df['SMA_50']).astype(int)
        
        # 8. Bollinger Bands (solo pasado)
        df['bb_middle'] = close.rolling(window=20, min_periods=10).mean()
        df['bb_std'] = close.rolling(window=20, min_periods=10).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, 1)
        df['bb_position'] = df['bb_position'].clip(0, 1)
        
        # 9. Momentum pasado
        df['momentum_6h'] = close / close.shift(6) - 1
        
        # 10. Volumen relativo
        df['volumen_relativo'] = volume / volume.rolling(window=24, min_periods=12).mean()
        
        # 11. Rango de precio
        df['rango_hl'] = (high - low) / close
        df['body_size'] = abs(close - df['Open']) / close
        
        # 12. Ratio volumen/precio
        df['volume_price_ratio'] = df['volumen_relativo'] / (abs(df['retorno_1h']) + 0.0001)
        
        # 13. Características temporales simples
        df['hora_dia'] = df.index.hour
        
        return df


# ============================================
# ETIQUETADO CORREGIDO
# ============================================

class EtiquetadoDatos:
    """Crea etiquetas realistas"""
    
    @staticmethod
    def crear_etiquetas_realistas(df, horizonte):
        """Etiquetas realistas sin leakage"""
        # Precio futuro
        precio_futuro = df['Close'].shift(-horizonte)
        
        # Retorno futuro
        retorno_futuro = (precio_futuro / df['Close']) - 1
        
        # Umbral dinámico basado en volatilidad
        volatilidad = df['retorno_1h'].rolling(24).std().fillna(0.01)
        umbral = volatilidad * 2  # 2 desviaciones estándar
        
        # Etiqueta: 1 si retorno > umbral, 0 si < -umbral, NaN en medio
        etiqueta = pd.Series(np.nan, index=df.index)
        etiqueta[retorno_futuro > umbral] = 1
        etiqueta[retorno_futuro < -umbral] = 0
        
        return etiqueta, retorno_futuro
    
    @staticmethod
    def preparar_dataset_seguro(df, horizonte):
        """Prepara dataset COMPLETAMENTE sin look-ahead"""
        
        # 1. Calcular features (solo con datos pasados)
        df_features = IndicadoresTecnicos.calcular_features_sin_leakage(df)
        
        # 2. Crear etiquetas (usando futuro)
        etiqueta, retorno_futuro = EtiquetadoDatos.crear_etiquetas_realistas(df_features, horizonte)
        
        # 3. Asegurar que features vienen ANTES de las etiquetas
        df_features[f'label_{horizonte}h'] = etiqueta
        df_features[f'return_{horizonte}h'] = retorno_futuro
        
        # 4. Features seleccionadas (solo las seguras)
        safe_features = [
            'RSI', 'ATR_pct', 'volatilidad_24h',
            'dist_sma_20', 'dist_sma_50', 'tendencia_sma',
            'bb_position', 'bb_std',
            'momentum_6h', 'retorno_1h', 'retorno_6h', 'retorno_12h',
            'volumen_relativo', 'rango_hl', 'body_size',
            'volume_price_ratio', 'hora_dia'
        ]
        
        # Filtrar features disponibles
        features_disponibles = [f for f in safe_features if f in df_features.columns]
        
        return df_features, features_disponibles


# ============================================
# MODELO SIMPLE Y ROBUSTO
# ============================================

class ModeloSimple:
    """Modelo simple sin overfitting"""
    
    def __init__(self, horizonte, ticker):
        self.horizonte = horizonte
        self.ticker = ticker
        self.modelo = None
        self.scaler = None
        self.features = None
    
    def entrenar(self, df, features, label_col):
        """Entrenamiento robusto con validación estricta"""
        
        # Filtrar datos completos
        df_completo = df.dropna(subset=[label_col] + features)
        
        if len(df_completo) < 1000:
            print(f"    ⚠️ Muy pocos datos: {len(df_completo)}")
            return False
        
        X = df_completo[features]
        y = df_completo[label_col]
        
        # Balance de clases
        class_counts = y.value_counts()
        if len(class_counts) < 2:
            print(f"    ⚠️ Solo una clase: {class_counts.to_dict()}")
            return False
        
        print(f"    📊 Clases: {class_counts.to_dict()}")
        
        # Validación walk-forward real
        tscv = TimeSeriesSplit(n_splits=5)
        accuracies = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Escalar
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Modelo simple
            modelo = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            modelo.fit(X_train_scaled, y_train)
            
            # Predicción
            y_pred = modelo.predict(X_val_scaled)
            acc = accuracy_score(y_val, y_pred)
            accuracies.append(acc)
        
        acc_promedio = np.mean(accuracies)
        acc_std = np.std(accuracies)
        
        print(f"    📈 Accuracy: {acc_promedio:.2%} (±{acc_std:.2%})")
        
        if acc_promedio < 0.55:  # Mínimo 55%
            print(f"    ❌ Accuracy muy baja")
            return False
        
        # Entrenar modelo final
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.modelo = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.modelo.fit(X_scaled, y)
        self.features = features
        
        return True
    
    def predecir(self, df_actual):
        """Predicción segura"""
        if self.modelo is None:
            return None
        
        # Verificar features
        features_faltantes = [f for f in self.features if f not in df_actual.columns]
        if features_faltantes:
            return None
        
        X = df_actual[self.features].iloc[[-1]]
        X_scaled = self.scaler.transform(X)
        
        prediccion = self.modelo.predict(X_scaled)[0]
        probabilidad = self.modelo.predict_proba(X_scaled)[0]
        
        return {
            'prediccion': int(prediccion),
            'probabilidad_positiva': probabilidad[1],
            'probabilidad_negativa': probabilidad[0],
            'confianza': max(probabilidad)
        }


# ============================================
# BACKTESTING REALISTA
# ============================================

class BacktesterRealista:
    """Backtesting realista con métricas honestas"""
    
    def __init__(self, df, modelos, ticker):
        self.df = df
        self.modelos = modelos
        self.ticker = ticker
        self.operaciones = []
    
    def ejecutar(self, fecha_inicio):
        """Ejecuta backtesting realista"""
        df_backtest = self.df[self.df.index >= fecha_inicio].copy()
        
        print(f"  📊 Período backtest: {len(df_backtest)} velas")
        
        for i in range(50, len(df_backtest) - 24):  # Dejar margen
            idx = df_backtest.index[i]
            
            # Datos disponibles hasta este momento
            df_historia = df_backtest.iloc[:i+1].copy()
            
            # Obtener predicción de cada modelo
            señales = []
            confianzas = []
            
            for horizonte, modelo in self.modelos.items():
                pred = modelo.predecir(df_historia)
                if pred:
                    señales.append(pred['prediccion'])
                    confianzas.append(pred['confianza'])
            
            if len(señales) < 2:  # Mínimo 2 modelos
                continue
            
            # Consenso: mayoría simple
            señal_final = 1 if sum(señales) > len(señales) / 2 else 0
            confianza_promedio = np.mean(confianzas)
            
            # Filtrar por confianza
            if confianza_promedio < TradingConfig.UMBRAL_CONFIANZA_MIN:
                continue
            
            # Simular operación
            operacion = self.simular_operacion_simple(df_backtest, i, señal_final, confianza_promedio)
            if operacion:
                self.operaciones.append(operacion)
        
        if not self.operaciones:
            print("  ⚠️ No se generaron operaciones")
            return None
        
        return self.calcular_metricas_honestas()
    
    def simular_operacion_simple(self, df, idx_pos, señal, confianza):
        """Simulación simple y realista"""
        entrada = df.iloc[idx_pos]
        precio_entrada = entrada['Close']
        atr = entrada.get('ATR', precio_entrada * 0.02)
        
        # Stop Loss y Take Profit
        if señal == 1:  # LONG
            sl = precio_entrada - (atr * TradingConfig.MULTIPLICADOR_SL)
            tp = precio_entrada + (atr * TradingConfig.MULTIPLICADOR_TP)
        else:  # SHORT
            sl = precio_entrada + (atr * TradingConfig.MULTIPLICADOR_SL)
            tp = precio_entrada - (atr * TradingConfig.MULTIPLICADOR_TP)
        
        # Verificar R:R
        riesgo = abs(precio_entrada - sl)
        recompensa = abs(tp - precio_entrada)
        ratio_rr = recompensa / riesgo if riesgo > 0 else 0
        
        if ratio_rr < TradingConfig.RATIO_MINIMO_RR:
            return None
        
        # Simular 24 horas máximo
        resultado = 'TIEMPO'
        retorno = 0
        velas = 0
        
        for j in range(1, min(24, len(df) - idx_pos - 1)):
            precio_actual = df.iloc[idx_pos + j]['Close']
            
            if señal == 1:  # LONG
                if precio_actual >= tp:
                    resultado = 'TP'
                    retorno = (tp - precio_entrada) / precio_entrada
                    velas = j
                    break
                elif precio_actual <= sl:
                    resultado = 'SL'
                    retorno = (sl - precio_entrada) / precio_entrada
                    velas = j
                    break
            else:  # SHORT
                if precio_actual <= tp:
                    resultado = 'TP'
                    retorno = (precio_entrada - tp) / precio_entrada
                    velas = j
                    break
                elif precio_actual >= sl:
                    resultado = 'SL'
                    retorno = (precio_entrada - sl) / precio_entrada
                    velas = j
                    break
        
        if resultado == 'TIEMPO':
            precio_final = df.iloc[idx_pos + 23]['Close']
            if señal == 1:
                retorno = (precio_final - precio_entrada) / precio_entrada
            else:
                retorno = (precio_entrada - precio_final) / precio_entrada
            velas = 23
        
        return {
            'fecha': df.index[idx_pos],
            'direccion': 'LONG' if señal == 1 else 'SHORT',
            'precio': precio_entrada,
            'resultado': resultado,
            'retorno': retorno,
            'velas': velas,
            'confianza': confianza,
            'ratio_rr': ratio_rr
        }
    
    def calcular_metricas_honestas(self):
        """Métricas realistas"""
        df_ops = pd.DataFrame(self.operaciones)
        
        n_ops = len(df_ops)
        if n_ops == 0:
            return None
        
        retornos = df_ops['retorno']
        ganadoras = retornos > 0
        
        # Métricas básicas
        win_rate = ganadoras.mean()
        retorno_total = retornos.sum()
        retorno_promedio = retornos.mean()
        
        # Drawdown
        equity_curve = (1 + retornos).cumprod()
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_dd = drawdown.min()
        
        # Profit Factor
        ganancias = retornos[retornos > 0].sum()
        perdidas = abs(retornos[retornos < 0].sum())
        pf = ganancias / perdidas if perdidas > 0 else np.inf
        
        # Sharpe ratio aproximado
        sharpe = retornos.mean() / retornos.std() if retornos.std() > 0 else 0
        
        return {
            'n_operaciones': n_ops,
            'win_rate': win_rate,
            'retorno_total': retorno_total,
            'retorno_promedio': retorno_promedio,
            'max_drawdown': max_dd,
            'profit_factor': pf,
            'sharpe_ratio': sharpe,
            'mejor_operacion': retornos.max(),
            'peor_operacion': retornos.min(),
            'expectativa': retornos.mean() * win_rate - abs(retornos.mean()) * (1 - win_rate)
        }, df_ops


# ============================================
# SISTEMA PRINCIPAL
# ============================================

class SistemaTradingReal:
    """Sistema realista de trading"""
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.modelos = {}
        self.fechas = TradingConfig.get_fechas()
        self.datos = None
    
    def descargar_datos(self):
        """Descarga datos"""
        try:
            df = yf.download(
                self.ticker,
                start=self.fechas['inicio_entrenamiento'],
                end=self.fechas['actual'],
                interval=TradingConfig.INTERVALO,
                progress=False
            )
            
            if df.empty:
                return False
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            self.datos = df
            print(f"  ✅ {len(df)} velas descargadas")
            return True
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False
    
    def entrenar(self):
        """Entrena modelos realistas"""
        df_train = self.datos[self.datos.index < self.fechas['inicio_backtest']].copy()
        
        print(f"  📊 Entrenamiento: {len(df_train)} velas")
        
        for horizonte in TradingConfig.HORIZONTES:
            print(f"\n  🔄 Horizonte {horizonte}h")
            
            # Preparar dataset seguro
            df_prep, features = EtiquetadoDatos.preparar_dataset_seguro(df_train, horizonte)
            label_col = f'label_{horizonte}h'
            
            # Entrenar modelo
            modelo = ModeloSimple(horizonte, self.ticker)
            if modelo.entrenar(df_prep, features, label_col):
                self.modelos[horizonte] = modelo
                print(f"    ✅ Modelo entrenado")
            else:
                print(f"    ❌ Falló entrenamiento")
        
        return len(self.modelos) > 0
    
    def backtest(self):
        """Backtesting realista"""
        if not self.modelos:
            return False
        
        # Preparar datos completos
        df_completo, _ = EtiquetadoDatos.preparar_dataset_seguro(self.datos, TradingConfig.HORIZONTES[0])
        
        # Ejecutar backtest
        backtester = BacktesterRealista(df_completo, self.modelos, self.ticker)
        resultado = backtester.ejecutar(self.fechas['inicio_backtest'])
        
        if resultado is None:
            return False
        
        metricas, df_ops = resultado
        self.metricas = metricas
        
        # Mostrar resultados
        print(f"\n  📊 RESULTADOS REALISTAS:")
        print(f"    Operaciones: {metricas['n_operaciones']}")
        print(f"    Win Rate: {metricas['win_rate']:.2%}")
        print(f"    Retorno Total: {metricas['retorno_total']:.2%}")
        print(f"    Retorno Promedio: {metricas['retorno_promedio']:.2%}")
        print(f"    Max Drawdown: {metricas['max_drawdown']:.2%}")
        print(f"    Profit Factor: {metricas['profit_factor']:.2f}")
        print(f"    Sharpe Ratio: {metricas['sharpe_ratio']:.2f}")
        print(f"    Expectativa: {metricas['expectativa']:.4f}")
        
        return True
    
    def evaluar(self):
        """Evaluación realista"""
        if not hasattr(self, 'metricas'):
            return False, 0
        
        m = self.metricas
        criterios = []
        
        # Criterios REALISTAS
        criterios.append(m['win_rate'] > 0.50)           # Win rate > 50%
        criterios.append(m['retorno_total'] > 0)        # Retorno positivo
        criterios.append(m['profit_factor'] > 1.2)      # PF > 1.2
        criterios.append(abs(m['max_drawdown']) < 0.20) # DD < 20%
        criterios.append(m['n_operaciones'] >= 15)      # Mínimo 15 ops
        criterios.append(m['expectativa'] > 0)          # Expectativa positiva
        
        cumplidos = sum(criterios)
        viable = cumplidos >= 4  # Al menos 4 de 6
        
        return viable, cumplidos


# ============================================
# EJECUCIÓN
# ============================================

def main():
    print("🔍 SISTEMA DE TRADING - VERSIÓN REALISTA")
    print("=" * 60)
    print("ADVERTENCIA: Esta versión elimina data leakage y overfitting")
    print("=" * 60)
    
    config = TradingConfig()
    fechas = config.get_fechas()
    
    print(f"\n📅 Período de análisis:")
    print(f"  Entrenamiento: {fechas['inicio_entrenamiento'].date()}")
    print(f"  Backtest: {fechas['inicio_backtest'].date()} a {fechas['actual'].date()}")
    
    for ticker in TradingConfig.ACTIVOS:
        print(f"\n{'='*60}")
        print(f"PROCESANDO: {ticker}")
        print('='*60)
        
        sistema = SistemaTradingReal(ticker)
        
        # 1. Datos
        if not sistema.descargar_datos():
            continue
        
        # 2. Entrenamiento
        if not sistema.entrenar():
            print("  ❌ Falló entrenamiento")
            continue
        
        # 3. Backtest
        if not sistema.backtest():
            print("  ❌ Falló backtest")
            continue
        
        # 4. Evaluación
        viable, criterios = sistema.evaluar()
        
        print(f"\n  📈 Evaluación:")
        print(f"    Criterios cumplidos: {criterios}/6")
        print(f"    Sistema viable: {'✅ SÍ' if viable else '❌ NO'}")
        
        if viable:
            print(f"\n  🎯 SISTEMA VIABLE DETECTADO!")
            print(f"    Considerar uso en tiempo real con monitoreo estricto")
        else:
            print(f"\n  ⚠️ Sistema no viable en condiciones actuales")
    
    print(f"\n{'='*60}")
    print("ANÁLISIS COMPLETADO")
    print("=" * 60)

if __name__ == "__main__":
    main()
