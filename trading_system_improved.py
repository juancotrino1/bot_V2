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
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
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
# CONFIGURACIÓN MEJORADA
# ============================================

class TradingConfig:
    """Configuración centralizada del sistema"""
    
    # Timezone
    TIMEZONE = pytz.timezone('America/Bogota')
    
    # Períodos de tiempo (CORREGIDO)
    INTERVALO = "1h"
    DIAS_ENTRENAMIENTO = 365  # 1 año de datos históricos
    DIAS_VALIDACION = 90      # 3 meses para validación
    DIAS_BACKTEST = 30        # 1 mes para backtesting final
    
    # Activos
    ACTIVOS = [
        "BTC-USD"
    ]
    
    # Parámetros técnicos
    VENTANA_VOLATILIDAD = 24  # 24 horas
    VENTANA_TENDENCIA = 50
    VENTANA_RAPIDA = 12
    ATR_PERIODO = 14
    RSI_PERIODO = 14
    
    # Horizontes de predicción (CORREGIDO - más cortos)
    HORIZONTES = [4, 8, 12, 24, 48]  # En horas
    
    # Gestión de riesgo
    MULTIPLICADOR_SL = 2.0
    MULTIPLICADOR_TP = 3.0
    RATIO_MINIMO_RR = 1.5
    MAX_RIESGO_POR_OPERACION = 0.02  # 2% del capital
    
    # Validación
    N_FOLDS_WF = 3
    MIN_MUESTRAS_ENTRENAMIENTO = 500
    MIN_MUESTRAS_CLASE = 20
    
    # Umbrales de trading
    UMBRAL_PROBABILIDAD_MIN = 0.65
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
# CÁLCULO DE INDICADORES (MEJORADO)
# ============================================

class IndicadoresTecnicos:
    """Calcula indicadores técnicos sin look-ahead bias"""
    
    @staticmethod
    def calcular_rsi(precios, periodo=14):
        """RSI robusto"""
        delta = precios.diff()
        ganancia = delta.where(delta > 0, 0).rolling(window=periodo, min_periods=periodo//2).mean()
        perdida = (-delta.where(delta < 0, 0)).rolling(window=periodo, min_periods=periodo//2).mean()
        
        # Evitar división por cero
        perdida = perdida.replace(0, 1e-10)
        rs = ganancia / perdida
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    @staticmethod
    def calcular_atr(df, periodo=14):
        """Average True Range"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        close_prev = close.shift(1)
        
        tr = pd.concat([
            high - low,
            (high - close_prev).abs(),
            (low - close_prev).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window=periodo, min_periods=periodo//2).mean()
        return atr.fillna(method='bfill')
    
    @staticmethod
    def calcular_bollinger_bands(precios, ventana=20, num_std=2):
        """Bandas de Bollinger"""
        sma = precios.rolling(window=ventana, min_periods=ventana//2).mean()
        std = precios.rolling(window=ventana, min_periods=ventana//2).std()
        
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        
        # Posición relativa en las bandas (0 = banda inferior, 1 = banda superior)
        bb_position = (precios - lower) / (upper - lower)
        bb_position = bb_position.clip(0, 1).fillna(0.5)
        
        return upper, lower, bb_position
    
    @staticmethod
    def calcular_features(df):
        """Calcula todas las features para el modelo"""
        df = df.copy()
        
        # Asegurar columnas simples
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df.get('Volume', pd.Series(1, index=df.index))
        
        # 1. Retornos
        df['retorno_1h'] = close.pct_change(1)
        df['retorno_4h'] = close.pct_change(4)
        df['retorno_24h'] = close.pct_change(24)
        
        # 2. Volatilidad
        df['volatilidad_24h'] = df['retorno_1h'].rolling(24, min_periods=12).std()
        df['volatilidad_normalizada'] = df['volatilidad_24h'] / df['volatilidad_24h'].rolling(100, min_periods=50).mean()
        
        # 3. Indicadores técnicos
        df['RSI'] = IndicadoresTecnicos.calcular_rsi(close, TradingConfig.RSI_PERIODO)
        df['ATR'] = IndicadoresTecnicos.calcular_atr(df, TradingConfig.ATR_PERIODO)
        df['ATR_pct'] = df['ATR'] / close
        
        # 4. Medias móviles y tendencia
        df['SMA_12'] = close.rolling(12, min_periods=6).mean()
        df['SMA_50'] = close.rolling(50, min_periods=25).mean()
        df['EMA_12'] = close.ewm(span=12, min_periods=6).mean()
        
        df['dist_sma_12'] = (close - df['SMA_12']) / df['SMA_12']
        df['dist_sma_50'] = (close - df['SMA_50']) / df['SMA_50']
        df['tendencia'] = (df['SMA_12'] > df['SMA_50']).astype(int)
        
        # 5. Bollinger Bands
        bb_upper, bb_lower, bb_pos = IndicadoresTecnicos.calcular_bollinger_bands(close)
        df['BB_position'] = bb_pos
        df['BB_width'] = (bb_upper - bb_lower) / close
        
        # 6. Momentum
        df['momentum_4h'] = close / close.shift(4) - 1
        df['momentum_aceleracion'] = df['retorno_1h'].diff()
        
        # 7. Volumen
        df['volumen_relativo'] = volume / volume.rolling(24, min_periods=12).mean()
        
        # 8. Rango de precio
        df['rango_hl'] = (high - low) / close
        df['body_size'] = abs(close - df['Open']) / close
        
        # 9. Features de contexto
        df['hora_dia'] = df.index.hour
        df['es_apertura_ny'] = ((df['hora_dia'] >= 13) & (df['hora_dia'] <= 15)).astype(int)
        
        # 10. Z-scores para detección de anomalías
        for col in ['retorno_1h', 'volatilidad_24h', 'volumen_relativo']:
            if col in df.columns:
                media = df[col].rolling(100, min_periods=50).mean()
                std = df[col].rolling(100, min_periods=50).std()
                df[f'{col}_zscore'] = (df[col] - media) / (std + 1e-10)
        
        return df


# ============================================
# ETIQUETADO DE DATOS (CORREGIDO)
# ============================================

class EtiquetadoDatos:
    """Crea etiquetas para entrenamiento"""
    
    @staticmethod
    def calcular_retorno_futuro(df, horizonte):
        """Calcula retorno futuro real"""
        return df['Close'].shift(-horizonte) / df['Close'] - 1
    
    @staticmethod
    def crear_etiquetas_direccion(df, horizonte, umbral_movimiento=0.005):
        """
        Etiqueta binaria: 1 si hay movimiento significativo alcista, 0 si bajista
        Se ignoran movimientos pequeños (< umbral)
        """
        retorno_futuro = EtiquetadoDatos.calcular_retorno_futuro(df, horizonte)
        
        # Clasificación triple: LONG (1), SHORT (0), NEUTRAL (NaN)
        etiqueta = pd.Series(np.nan, index=df.index)
        etiqueta[retorno_futuro > umbral_movimiento] = 1
        etiqueta[retorno_futuro < -umbral_movimiento] = 0
        
        return etiqueta, retorno_futuro
    
    @staticmethod
    def preparar_dataset_ml(df, horizonte):
        """Prepara dataset completo para ML"""
        # Calcular features
        df = IndicadoresTecnicos.calcular_features(df)
        
        # Crear etiquetas
        etiqueta, retorno_futuro = EtiquetadoDatos.crear_etiquetas_direccion(df, horizonte)
        df[f'etiqueta_{horizonte}h'] = etiqueta
        df[f'retorno_futuro_{horizonte}h'] = retorno_futuro
        
        # Features para el modelo (sin look-ahead bias)
        features = [
            'RSI', 'ATR_pct', 'volatilidad_24h', 'volatilidad_normalizada',
            'dist_sma_12', 'dist_sma_50', 'tendencia',
            'BB_position', 'BB_width',
            'momentum_4h', 'momentum_aceleracion',
            'volumen_relativo', 'rango_hl', 'body_size',
            'retorno_1h', 'retorno_4h', 'retorno_24h',
            'retorno_1h_zscore', 'volatilidad_24h_zscore', 'volumen_relativo_zscore',
            'es_apertura_ny'
        ]
        
        # Filtrar solo features disponibles
        features_disponibles = [f for f in features if f in df.columns]
        
        return df, features_disponibles


# ============================================
# MODELO DE MACHINE LEARNING (MEJORADO)
# ============================================

class ModeloPrediccion:
    """Modelo de ML para predicción de dirección"""
    
    def __init__(self, horizonte, ticker):
        self.horizonte = horizonte
        self.ticker = ticker
        self.modelo = None
        self.scaler = None
        self.features = None
        self.metricas_validacion = {}
    
    def entrenar_walk_forward(self, df, features, etiqueta_col):
        """Entrenamiento con validación walk-forward"""
        
        # Filtrar datos válidos
        df_valido = df.dropna(subset=[etiqueta_col] + features).copy()
        
        if len(df_valido) < TradingConfig.MIN_MUESTRAS_ENTRENAMIENTO:
            print(f"    ⚠️ Datos insuficientes: {len(df_valido)} < {TradingConfig.MIN_MUESTRAS_ENTRENAMIENTO}")
            return False
        
        X = df_valido[features]
        y = df_valido[etiqueta_col]
        
        # Verificar balance de clases
        if y.sum() < TradingConfig.MIN_MUESTRAS_CLASE or (len(y) - y.sum()) < TradingConfig.MIN_MUESTRAS_CLASE:
            print(f"    ⚠️ Clases desbalanceadas: Positivos={y.sum()}, Negativos={len(y)-y.sum()}")
            return False
        
        # Walk-forward validation
        tscv = TimeSeriesSplit(n_splits=TradingConfig.N_FOLDS_WF)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Escalar
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Entrenar modelo
            modelo = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            modelo.fit(X_train_scaled, y_train)
            
            # Evaluar
            y_pred = modelo.predict(X_val_scaled)
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            
            scores.append({'accuracy': acc, 'precision': prec, 'recall': rec})
        
        # Métricas promedio
        self.metricas_validacion = {
            'accuracy': np.mean([s['accuracy'] for s in scores]),
            'precision': np.mean([s['precision'] for s in scores]),
            'recall': np.mean([s['recall'] for s in scores]),
            'n_folds': len(scores)
        }
        
        print(f"      ✅ Accuracy: {self.metricas_validacion['accuracy']:.2%}, "
              f"Precision: {self.metricas_validacion['precision']:.2%}, "
              f"Recall: {self.metricas_validacion['recall']:.2%}")
        
        # Entrenar modelo final con todos los datos
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.modelo = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
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
        """Realiza predicción en datos nuevos"""
        if self.modelo is None:
            return None
        
        # Asegurar que tenemos todas las features
        if not all(f in df_actual.columns for f in self.features):
            return None
        
        X = df_actual[self.features].iloc[[-1]]  # Última fila
        X_scaled = self.scaler.transform(X)
        
        prediccion_clase = self.modelo.predict(X_scaled)[0]
        probabilidades = self.modelo.predict_proba(X_scaled)[0]
        
        return {
            'prediccion': int(prediccion_clase),
            'probabilidad_positiva': probabilidades[1],
            'probabilidad_negativa': probabilidades[0],
            'confianza': max(probabilidades)
        }
    
    def guardar(self, path):
        """Guarda el modelo"""
        if self.modelo is None:
            return False
        
        modelo_data = {
            'modelo': self.modelo,
            'scaler': self.scaler,
            'features': self.features,
            'metricas': self.metricas_validacion,
            'horizonte': self.horizonte,
            'ticker': self.ticker
        }
        
        joblib.dump(modelo_data, path)
        return True
    
    @classmethod
    def cargar(cls, path):
        """Carga un modelo guardado"""
        modelo_data = joblib.load(path)
        
        instancia = cls(modelo_data['horizonte'], modelo_data['ticker'])
        instancia.modelo = modelo_data['modelo']
        instancia.scaler = modelo_data['scaler']
        instancia.features = modelo_data['features']
        instancia.metricas_validacion = modelo_data['metricas']
        
        return instancia


# ============================================
# BACKTESTING (MEJORADO)
# ============================================

class Backtester:
    """Ejecuta backtesting riguroso"""
    
    def __init__(self, df, modelos, ticker):
        self.df = df
        self.modelos = modelos  # Dict de modelos por horizonte
        self.ticker = ticker
        self.operaciones = []
    
    def simular_operacion(self, idx, señal_long, prob, features_row):
        """Simula una operación completa"""
        precio_entrada = self.df.loc[idx, 'Close']
        atr = self.df.loc[idx, 'ATR']
        
        # Determinar dirección
        direccion = 'LONG' if señal_long else 'SHORT'
        
        # Calcular niveles
        if señal_long:
            stop_loss = precio_entrada - (TradingConfig.MULTIPLICADOR_SL * atr)
            take_profit = precio_entrada + (TradingConfig.MULTIPLICADOR_TP * atr)
        else:
            stop_loss = precio_entrada + (TradingConfig.MULTIPLICADOR_SL * atr)
            take_profit = precio_entrada - (TradingConfig.MULTIPLICADOR_TP * atr)
        
        riesgo = abs(precio_entrada - stop_loss)
        recompensa = abs(take_profit - precio_entrada)
        ratio_rr = recompensa / riesgo if riesgo > 0 else 0
        
        # Filtro R:R
        if ratio_rr < TradingConfig.RATIO_MINIMO_RR:
            return None
        
        # Simular resultado (mirar hacia adelante máximo 24 horas)
        idx_pos = self.df.index.get_loc(idx)
        max_ventana = min(24, len(self.df) - idx_pos - 1)
        
        if max_ventana < 4:
            return None
        
        precios_futuros = self.df.iloc[idx_pos:idx_pos + max_ventana + 1]['Close'].values
        
        # Determinar resultado
        resultado = 'TIEMPO'
        velas_hasta_cierre = max_ventana
        retorno = 0
        
        for i, precio in enumerate(precios_futuros[1:], 1):
            if señal_long:
                if precio >= take_profit:
                    resultado = 'TP'
                    velas_hasta_cierre = i
                    retorno = recompensa / precio_entrada
                    break
                elif precio <= stop_loss:
                    resultado = 'SL'
                    velas_hasta_cierre = i
                    retorno = -riesgo / precio_entrada
                    break
            else:  # SHORT
                if precio <= take_profit:
                    resultado = 'TP'
                    velas_hasta_cierre = i
                    retorno = recompensa / precio_entrada
                    break
                elif precio >= stop_loss:
                    resultado = 'SL'
                    velas_hasta_cierre = i
                    retorno = -riesgo / precio_entrada
                    break
        
        # Si llegamos hasta el final sin hit
        if resultado == 'TIEMPO':
            precio_cierre = precios_futuros[velas_hasta_cierre]
            if señal_long:
                retorno = (precio_cierre - precio_entrada) / precio_entrada
            else:
                retorno = (precio_entrada - precio_cierre) / precio_entrada
        
        return {
            'fecha': idx,
            'ticker': self.ticker,
            'direccion': direccion,
            'precio_entrada': precio_entrada,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'ratio_rr': ratio_rr,
            'probabilidad': prob,
            'resultado': resultado,
            'retorno': retorno,
            'velas_hasta_cierre': velas_hasta_cierre
        }
    
    def ejecutar(self, fecha_inicio, umbral_prob=0.65):
        """Ejecuta backtesting completo"""
        df_backtest = self.df[self.df.index >= fecha_inicio].copy()
        
        if len(df_backtest) < 100:
            print(f"  ⚠️ Datos insuficientes para backtesting: {len(df_backtest)} velas")
            return None
        
        print(f"  📊 Backtesting: {df_backtest.index[0]} a {df_backtest.index[-1]} ({len(df_backtest)} velas)")
        
        # Iterar sobre cada vela
        for idx in df_backtest.index[:-24]:  # Dejar margen para simulación
            predicciones = {}
            
            # Obtener predicciones de todos los horizontes
            for horizonte, modelo in self.modelos.items():
                pred = modelo.predecir(df_backtest.loc[:idx])
                if pred:
                    predicciones[horizonte] = pred
            
            if not predicciones:
                continue
            
            # Consenso de modelos
            probs_positivas = [p['probabilidad_positiva'] for p in predicciones.values()]
            prob_promedio = np.mean(probs_positivas)
            confianza_promedio = np.mean([p['confianza'] for p in predicciones.values()])
            
            # Decidir señal
            señal_long = prob_promedio > 0.5
            
            # Filtros
            if confianza_promedio < TradingConfig.UMBRAL_CONFIANZA_MIN:
                continue
            
            if max(probs_positivas) < umbral_prob and max([1-p for p in probs_positivas]) < umbral_prob:
                continue
            
            # Simular operación
            operacion = self.simular_operacion(
                idx, 
                señal_long, 
                prob_promedio,
                df_backtest.loc[idx]
            )
            
            if operacion:
                self.operaciones.append(operacion)
        
        if not self.operaciones:
            print(f"  ⚠️ No se generaron operaciones en backtesting")
            return None
        
        return self.calcular_metricas()
    
    def calcular_metricas(self):
        """Calcula métricas de rendimiento"""
        df_ops = pd.DataFrame(self.operaciones)
        
        n_ops = len(df_ops)
        n_tp = (df_ops['resultado'] == 'TP').sum()
        n_sl = (df_ops['resultado'] == 'SL').sum()
        
        retornos = df_ops['retorno']
        operaciones_ganadoras = retornos > 0
        
        metricas = {
            'n_operaciones': n_ops,
            'tasa_exito': operaciones_ganadoras.sum() / n_ops,
            'hit_tp_rate': n_tp / n_ops,
            'hit_sl_rate': n_sl / n_ops,
            'retorno_total': retornos.sum(),
            'retorno_promedio': retornos.mean(),
            'retorno_mediano': retornos.median(),
            'mejor_operacion': retornos.max(),
            'peor_operacion': retornos.min(),
            'profit_factor': abs(retornos[retornos > 0].sum() / retornos[retornos < 0].sum()) if (retornos < 0).any() else np.inf,
            'max_drawdown': self._calcular_max_drawdown(retornos),
            'sharpe_ratio': retornos.mean() / retornos.std() if retornos.std() > 0 else 0,
        }
        
        return metricas, df_ops
    
    def _calcular_max_drawdown(self, retornos):
        """Calcula drawdown máximo"""
        equity_curve = (1 + retornos).cumprod()
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown.min()


# ============================================
# SISTEMA COMPLETO POR TICKER
# ============================================

class SistemaTradingTicker:
    """Sistema completo de trading para un ticker"""
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.modelos = {}
        self.fechas = TradingConfig.get_fechas()
        self.df_historico = None
        self.metricas_backtest = None
    
    def descargar_datos(self):
        """Descarga datos históricos"""
        print(f"\n{'='*80}")
        print(f"📥 DESCARGANDO {self.ticker}")
        print(f"{'='*80}")
        
        try:
            df = yf.download(
                self.ticker,
                start=self.fechas['inicio_entrenamiento'],
                end=self.fechas['actual'],
                interval=TradingConfig.INTERVALO,
                progress=False
            )
            
            if df.empty:
                print(f"  ❌ No hay datos disponibles")
                return False
            
            # Limpiar MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df = df.dropna()
            
            self.df_historico = df
            print(f"  ✅ Descargado: {len(df)} velas")
            print(f"  📅 Rango: {df.index[0]} a {df.index[-1]}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False
    
    def entrenar_modelos(self):
        """Entrena modelos para todos los horizontes"""
        print(f"\n🎯 ENTRENANDO MODELOS - {self.ticker}")
        print("-" * 80)
        
        if self.df_historico is None:
            return False
        
        # Usar datos hasta inicio de backtest para entrenamiento
        df_train = self.df_historico[self.df_historico.index < self.fechas['inicio_backtest']].copy()
        
        print(f"  📊 Datos entrenamiento: {len(df_train)} velas")
        
        modelos_entrenados = 0
        
        for horizonte in TradingConfig.HORIZONTES:
            print(f"\n  🔄 Horizonte {horizonte}h...")
            
            # Preparar dataset
            df_prep, features = EtiquetadoDatos.preparar_dataset_ml(df_train, horizonte)
            etiqueta_col = f'etiqueta_{horizonte}h'
            
            # Entrenar modelo
            modelo = ModeloPrediccion(horizonte, self.ticker)
            if modelo.entrenar_walk_forward(df_prep, features, etiqueta_col):
                self.modelos[horizonte] = modelo
                modelos_entrenados += 1
        
        print(f"\n  ✅ Modelos entrenados: {modelos_entrenados}/{len(TradingConfig.HORIZONTES)}")
        
        return modelos_entrenados > 0
    
    def ejecutar_backtest(self):
        """Ejecuta backtesting"""
        print(f"\n🔬 BACKTESTING - {self.ticker}")
        print("-" * 80)
        
        if not self.modelos:
            print("  ❌ No hay modelos entrenados")
            return False
        
        # Preparar datos completos (incluye período de backtest)
        df_completo, _ = EtiquetadoDatos.preparar_dataset_ml(
            self.df_historico, 
            TradingConfig.HORIZONTES[0]
        )
        
        # Ejecutar backtest
        backtester = Backtester(df_completo, self.modelos, self.ticker)
        resultado = backtester.ejecutar(self.fechas['inicio_backtest'])
        
        if resultado is None:
            return False
        
        metricas, df_ops = resultado
        self.metricas_backtest = metricas
        
        # Mostrar resultados
        print(f"\n  📊 RESULTADOS:")
        print(f"    Operaciones: {metricas['n_operaciones']}")
        print(f"    Tasa éxito: {metricas['tasa_exito']:.2%}")
        print(f"    Hit TP: {metricas['hit_tp_rate']:.2%}")
        print(f"    Hit SL: {metricas['hit_sl_rate']:.2%}")
        print(f"    Retorno total: {metricas['retorno_total']:.2%}")
        print(f"    Retorno promedio: {metricas['retorno_promedio']:.2%}")
        print(f"    Profit Factor: {metricas['profit_factor']:.2f}")
        print(f"    Max Drawdown: {metricas['max_drawdown']:.2%}")
        print(f"    Sharpe Ratio: {metricas['sharpe_ratio']:.2f}")
        
        return True
    
    def es_viable(self):
        """Evalúa si el sistema es viable para trading"""
        if self.metricas_backtest is None:
            return False, 0
        
        m = self.metricas_backtest
        criterios = []
        
        # Criterio 1: Tasa de éxito > 50%
        criterios.append(m['tasa_exito'] > 0.50)
        
        # Criterio 2: Retorno total positivo
        criterios.append(m['retorno_total'] > 0)
        
        # Criterio 3: Profit factor > 1.2
        criterios.append(m['profit_factor'] > 1.2)
        
        # Criterio 4: Drawdown controlado
        criterios.append(abs(m['max_drawdown']) < 0.25)
        
        # Criterio 5: Suficientes operaciones
        criterios.append(m['n_operaciones'] >= 15)
        
        # Criterio 6: Sharpe ratio positivo
        criterios.append(m['sharpe_ratio'] > 0)
        
        criterios_cumplidos = sum(criterios)
        viable = criterios_cumplidos >= 4
        
        return viable, criterios_cumplidos
    
    def analizar_tiempo_real(self):
        if not self.modelos:
            return None

        try:
            df_reciente = yf.download(
                self.ticker,
                start=self.fechas['actual'] - timedelta(days=7),
                end=self.fechas['actual'],
                interval=TradingConfig.INTERVALO,
                progress=False
            )

            if df_reciente.empty:
                return None

            if isinstance(df_reciente.columns, pd.MultiIndex):
                df_reciente.columns = df_reciente.columns.get_level_values(0)

            df_reciente = df_reciente[['Open', 'High', 'Low', 'Close', 'Volume']]
            df_reciente = IndicadoresTecnicos.calcular_features(df_reciente)

            # === MEAN REVERSION ===
            df_reciente["ret_log"] = np.log(df_reciente["Close"] / df_reciente["Close"].shift(1))
            window = 72
            df_reciente["mu"] = df_reciente["ret_log"].rolling(window).mean()
            df_reciente["sigma"] = df_reciente["ret_log"].rolling(window).std()
            df_reciente["sigma"] = df_reciente["sigma"].replace(0, np.nan)
            df_reciente["z_mr"] = (df_reciente["ret_log"] - df_reciente["mu"]) / df_reciente["sigma"]

            z_actual = df_reciente["z_mr"].iloc[-1]
            if pd.isna(z_actual) or np.isinf(z_actual):
                z_actual = 0

            evento = "NO"
            if z_actual > 2.2:
                evento = "MR SHORT"
            elif z_actual < -2.2:
                evento = "MR LONG"

            # === PREDICCIONES ===
            predicciones = {}
            for horizonte, modelo in self.modelos.items():
                pred = modelo.predecir(df_reciente)
                if pred:
                    predicciones[horizonte] = pred

            if not predicciones:
                return None

            probs_positivas = [p['probabilidad_positiva'] for p in predicciones.values()]
            prob_promedio = np.mean(probs_positivas)
            confianza_promedio = np.mean([p['confianza'] for p in predicciones.values()])

            señal = "LONG" if prob_promedio > 0.5 else "SHORT"
            prob_real = prob_promedio if señal == "LONG" else 1 - prob_promedio

            ultima_vela = df_reciente.iloc[-1]
            precio = ultima_vela['Close']
            atr = ultima_vela['ATR']

            if pd.isna(atr) or atr <= 0:
                return None

            min_dist = precio * 0.002
            atr = max(atr, min_dist)

            if señal == 'LONG':
                sl = precio - TradingConfig.MULTIPLICADOR_SL * atr
                tp = precio + TradingConfig.MULTIPLICADOR_TP * atr
            else:
                sl = precio + TradingConfig.MULTIPLICADOR_SL * atr
                tp = precio - TradingConfig.MULTIPLICADOR_TP * atr

            ratio_rr = abs(tp - precio) / abs(precio - sl)
            if ratio_rr < TradingConfig.RATIO_MINIMO_RR:
                return None

            return {
                'ticker': self.ticker,
                'fecha': datetime.now(TradingConfig.TIMEZONE),
                'precio': precio,
                'señal': señal,
                'probabilidad': prob_real,
                'confianza': confianza_promedio,
                'stop_loss': sl,
                'take_profit': tp,
                'ratio_rr': ratio_rr,
                'predicciones_detalle': predicciones,
                'rsi': ultima_vela.get('RSI', 50),
                'tendencia': 'ALCISTA' if ultima_vela.get('tendencia', 0) == 1 else 'BAJISTA',
                'z_mr': float(z_actual),
                'evento_mr': evento,
             }
    
        except Exception as e:
            print(f"  ❌ Error análisis tiempo real: {e}")
            return None
    
    def guardar_modelos(self):
        """Guarda modelos entrenados"""
        if not self.modelos:
            return False
        
        path_ticker = TradingConfig.MODELOS_DIR / self.ticker
        path_ticker.mkdir(parents=True, exist_ok=True)
        
        for horizonte, modelo in self.modelos.items():
            path_modelo = path_ticker / f"modelo_{horizonte}h.pkl"
            modelo.guardar(path_modelo)
        
        print(f"  💾 Modelos guardados en {path_ticker}")
        return True


# ============================================
# FUNCIÓN PRINCIPAL
# ============================================

def main():

    print("🚀 SISTEMA DE TRADING MEJORADO")
    print("=" * 80)
    
    fechas = TradingConfig.get_fechas()
    print(f"\n📅 Configuración temporal:")
    print(f"  Actual: {fechas['actual'].date()}")
    print(f"  Entrenamiento desde: {fechas['inicio_entrenamiento'].date()}")
    print(f"  Backtest desde: {fechas['inicio_backtest'].date()}")
    print(f"  Intervalo: {TradingConfig.INTERVALO}")
    print(f"  Horizontes: {TradingConfig.HORIZONTES} horas")
    
    resultados_globales = {}
    
    # Procesar cada ticker
    for ticker in TradingConfig.ACTIVOS:
        sistema = SistemaTradingTicker(ticker)
        
        # 1. Descargar datos
        if not sistema.descargar_datos():
            continue
        
        # 2. Entrenar modelos
        if not sistema.entrenar_modelos():
            print(f"  ❌ No se pudieron entrenar modelos para {ticker}")
            continue
        
        # 3. Backtest
        if not sistema.ejecutar_backtest():
            print(f"  ❌ Backtest fallido para {ticker}")
            continue
        
        # 4. Evaluar viabilidad
        viable, criterios = sistema.es_viable()
        
        print(f"\n{'='*80}")
        print(f"📊 EVALUACIÓN - {ticker}")
        print(f"{'='*80}")
        print(f"  Criterios cumplidos: {criterios}/6")
        print(f"  Viable: {'✅ SÍ' if viable else '❌ NO'}")
        
        # 5. Análisis tiempo real (solo si es viable)
        señal_actual = None
        
        if viable:
           señal_actual = sistema.analizar_tiempo_real()

           if (señal_actual and
               señal_actual['confianza'] >= TradingConfig.UMBRAL_CONFIANZA_MIN and
               señal_actual['probabilidad'] >= TradingConfig.UMBRAL_PROBABILIDAD_MIN):

 

               print(f"\n  🚨 SEÑAL DETECTADA:")
               print(f"    Dirección: {señal_actual['señal']}")
               print(f"    Probabilidad: {señal_actual['probabilidad']:.2%}")
               print(f"    Confianza: {señal_actual['confianza']:.2%}")
               print(f"    Precio: ${señal_actual['precio']:,.2f}")
               print(f"    SL: ${señal_actual['stop_loss']:,.2f}")
               print(f"    TP: ${señal_actual['take_profit']:,.2f}")
               print(f"    R:R: {señal_actual['ratio_rr']:.2f}")

               # 🔁 Control de repetición
               ultima = cargar_ultima_senal()
               if ultima and ultima["ticker"] == ticker and ultima["señal"] == señal_actual["señal"]:
                   print("🔁 Señal repetida. No se envía.")
               else:
                   fecha = señal_actual['fecha'].strftime("%Y-%m-%d %H:%M")

                   enviar_telegram(
                       f"📊 SEÑAL {ticker}\n"
                       f"🕒 Fecha: {fecha}\n"
                       f"⏱ TF: {TradingConfig.INTERVALO}\n"
                       f"📈 Tendencia: {señal_actual['tendencia']}\n"
                       f"📊 RSI: {señal_actual['rsi']:.1f}\n\n"
                       f"Dirección: {señal_actual['señal']}\n"
                       f"Probabilidad: {señal_actual['probabilidad']:.2%}\n"
                       f"Confianza: {señal_actual['confianza']:.2%}\n\n"
                       f"🎯 Entrada: {señal_actual['precio']:.2f}\n"
                       f"🛑 SL: {señal_actual['stop_loss']:.2f}\n"
                       f"🎯 TP: {señal_actual['take_profit']:.2f}\n"
                       f"⚖️ R:R: {señal_actual['ratio_rr']:.2f}\n"
                       f"📐 Mean Reversion: {señal_actual['evento_mr']}\n"
                       f"📐 Z-score: {señal_actual['z_mr']:.2f}\n\n"
                   )

                   guardar_ultima_senal({
                       "ticker": ticker,
                       "señal": señal_actual["señal"],
                       "fecha": str(señal_actual["fecha"])
                   })

        # 6. Guardar modelos
        if viable:
            sistema.guardar_modelos()
        
        resultados_globales[ticker] = {
            'viable': viable,
            'criterios': criterios,
            'metricas': sistema.metricas_backtest,
            'señal_actual': señal_actual
        }
    
    # Resumen final
    print(f"\n{'='*80}")
    print("📊 RESUMEN GLOBAL")
    print(f"{'='*80}")
    
    viables = [t for t, r in resultados_globales.items() if r['viable']]
    
    print(f"\n  Tickers procesados: {len(resultados_globales)}")
    print(f"  Tickers viables: {len(viables)}")
    
    if viables:
        print(f"\n  ✅ TICKERS VIABLES:")
        for ticker in viables:
            r = resultados_globales[ticker]
            m = r['metricas']
            print(f"    {ticker}: Retorno {m['retorno_total']:.2%}, "
                  f"Win rate {m['tasa_exito']:.2%}, "
                  f"PF {m['profit_factor']:.2f}")
    
    return resultados_globales


if __name__ == "__main__":
    main()
