import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import warnings
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from scipy import stats
import talib
from pathlib import Path
import json
import requests
import os

warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN SIMPLIFICADA Y ROBUSTA
# ============================================

class TradingConfigOptimizado:
    """Configuración optimizada para trading real"""
    
    # Timezone
    TIMEZONE = pytz.timezone('America/Bogota')
    
    # Períodos (más realista)
    INTERVALO = "1h"
    DIAS_ENTRENAMIENTO = 180  # 6 meses (mercado crypto cambia rápido)
    DIAS_VALIDACION = 30      # 1 mes
    DIAS_BACKTEST = 30        # 1 mes
    
    # Activos principales (menos es más)
    ACTIVOS = [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"
    ]
    
    # Solo 2 horizontes estratégicos
    HORIZONTES = [6, 24]  # 6h para scalping, 24h para swing
    
    # Gestión de riesgo MEJORADA
    MULTIPLICADOR_SL = 1.5    # Más ajustado
    MULTIPLICADOR_TP = 2.0    # Ratio 1:1.33
    MIN_RR_RATIO = 1.3
    MAX_RIESGO_POR_OPERACION = 0.01  # 1%
    
    # Filtros de mercado
    VOLATILIDAD_MAX = 0.15    # Excluir periodos de alta volatilidad
    RSI_OVERBOUGHT = 75
    RSI_OVERSOLD = 25
    MIN_VOLUME_RATIO = 0.7    # Volumen mínimo relativo
    
    # Umbrales de trading CONSERVADORES
    UMBRAL_PROB_MIN = 0.70    # Más exigente
    UMBRAL_CONFIANZA_MIN = 0.65
    
    # Modelo
    N_FOLDS = 3
    MIN_MUESTRAS = 1000
    
    @classmethod
    def get_fechas(cls):
        now = datetime.now(cls.TIMEZONE)
        return {
            'actual': now,
            'inicio': now - timedelta(days=cls.DIAS_ENTRENAMIENTO + cls.DIAS_VALIDACION + cls.DIAS_BACKTEST),
            'fin_backtest': now - timedelta(days=cls.DIAS_BACKTEST)
        }

# ============================================
# FEATURES ROBUSTAS SIN LOOK-AHEAD BIAS
# ============================================

class FeatureEngineerRobusto:
    """Features simples pero efectivas"""
    
    @staticmethod
    def calcular_features(df):
        df = df.copy()
        
        # 1. Precios básicos
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # 2. Retornos (solo pasados)
        df['ret_1h'] = close.pct_change(1)
        df['ret_6h'] = close.pct_change(6)
        df['ret_24h'] = close.pct_change(24)
        
        # 3. Volatilidad (HISTÓRICA, no futura)
        df['vol_6h'] = df['ret_1h'].rolling(6).std()
        df['vol_24h'] = df['ret_1h'].rolling(24).std()
        df['vol_ratio'] = df['vol_6h'] / df['vol_24h'].replace(0, 1e-10)
        
        # 4. Indicadores técnicos básicos (sin futuro)
        # RSI
        df['rsi'] = talib.RSI(close, timeperiod=14)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        # Medias móviles
        df['ema_12'] = talib.EMA(close, timeperiod=12)
        df['ema_26'] = talib.EMA(close, timeperiod=26)
        df['sma_50'] = talib.SMA(close, timeperiod=50)
        
        df['ema_diff'] = (df['ema_12'] - df['ema_26']) / close
        df['above_sma50'] = (close > df['sma_50']).astype(int)
        
        # 5. Patrones de velas (solo pasado)
        df['doji'] = (abs(close - df['Open']) / (high - low).replace(0, 1e-10) < 0.1).astype(int)
        
        # 6. Volumen
        df['volume_sma'] = volume.rolling(20).mean()
        df['volume_ratio'] = volume / df['volume_sma'].replace(0, 1e-10)
        
        # 7. Rango de precio
        df['range_pct'] = (high - low) / close
        df['body_pct'] = abs(close - df['Open']) / close
        
        # 8. Soporte y resistencia simplificado
        df['high_20'] = high.rolling(20).max()
        df['low_20'] = low.rolling(20).min()
        df['near_resistance'] = ((close - df['high_20']) / close).abs() < 0.02
        df['near_support'] = ((close - df['low_20']) / close).abs() < 0.02
        
        # 9. Hora del día (sesgo de mercado)
        df['hour'] = df.index.hour
        df['is_london_open'] = ((df['hour'] >= 8) & (df['hour'] <= 16)).astype(int)
        
        # 10. Market regime (tendencia + volatilidad)
        df['trend_strength'] = abs(df['ema_12'] - df['ema_26']) / close
        df['is_trending'] = (df['trend_strength'] > df['trend_strength'].rolling(50).mean()).astype(int)
        
        return df

# ============================================
# ETIQUETADO MEJORADO
# ============================================

class EtiquetadorInteligente:
    """Etiquetas con filtros de calidad"""
    
    @staticmethod
    def crear_etiquetas(df, horizonte):
        """
        Crea etiquetas binarias con filtros:
        1. Excluye periodos de alta volatilidad
        2. Solo etiqueta movimientos significativos
        3. Considera el contexto de mercado
        """
        close = df['Close']
        
        # Retorno futuro
        future_return = close.shift(-horizonte) / close - 1
        
        # Filtrar por volatilidad
        current_vol = df['vol_24h'].rolling(24).mean()
        high_vol_threshold = current_vol.quantile(0.8)
        
        # Filtrar por volumen
        low_volume = df['volume_ratio'] < TradingConfigOptimizado.MIN_VOLUME_RATIO
        
        # Etiqueta inicial
        labels = pd.Series(np.nan, index=df.index)
        
        # Condiciones para LONG
        long_cond = (
            (future_return > 0.01) &  # Movimiento mínimo 1%
            (current_vol < high_vol_threshold) &  # Volatilidad no muy alta
            (~low_volume) &  # Volumen decente
            (df['rsi'] < TradingConfigOptimizado.RSI_OVERBOUGHT)  # No sobrecomprado
        )
        
        # Condiciones para SHORT
        short_cond = (
            (future_return < -0.01) &  # Movimiento mínimo -1%
            (current_vol < high_vol_threshold) &
            (~low_volume) &
            (df['rsi'] > TradingConfigOptimizado.RSI_OVERSOLD)  # No sobrevendido
        )
        
        labels[long_cond] = 1
        labels[short_cond] = 0
        
        return labels, future_return

# ============================================
# MODELO SIMPLIFICADO Y ROBUSTO
# ============================================

class ModeloTradingRobusto:
    """Modelo más simple pero con validación rigurosa"""
    
    def __init__(self, ticker, horizonte):
        self.ticker = ticker
        self.horizonte = horizonte
        self.model = None
        self.scaler = None
        self.features = None
        self.metrics = {}
        self.feature_importances_ = None
        
    def get_best_features(self):
        """Features probadamente efectivas"""
        return [
            'rsi', 'macd_hist', 'ema_diff', 'above_sma50',
            'vol_ratio', 'range_pct', 'body_pct',
            'volume_ratio', 'trend_strength', 'is_trending',
            'near_resistance', 'near_support', 'hour'
        ]
    
    def entrenar(self, df):
        """Entrenamiento con validación robusta"""
        # Preparar datos
        df_features = FeatureEngineerRobusto.calcular_features(df)
        labels, _ = EtiquetadorInteligente.crear_etiquetas(df_features, self.horizonte)
        
        # Features
        feature_list = self.get_best_features()
        available_features = [f for f in feature_list if f in df_features.columns]
        
        X = df_features[available_features].copy()
        y = labels.copy()
        
        # Eliminar NaNs
        valid_idx = X.notna().all(axis=1) & y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < TradingConfigOptimizado.MIN_MUESTRAS:
            print(f"  ⚠️ Muestras insuficientes: {len(X)}")
            return False
        
        # Balance de clases
        class_ratio = y.mean()
        if class_ratio < 0.3 or class_ratio > 0.7:
            print(f"  ⚠️ Clases desbalanceadas: {class_ratio:.2%} positivas")
            # Podemos usar SMOTE o ajustar pesos
        
        # Time Series Cross Validation
        tscv = TimeSeriesSplit(n_splits=TradingConfigOptimizado.N_FOLDS)
        
        best_score = 0
        best_model = None
        best_scaler = None
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Escalar
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Entrenar modelo simple
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                min_samples_split=50,
                min_samples_leaf=25,
                subsample=0.8,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluar
            y_pred = model.predict(X_val_scaled)
            accuracy = (y_pred == y_val).mean()
            
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                best_scaler = scaler
                
                # Calcular métricas detalladas
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                self.metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                    'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                }
        
        print(f"    ✅ Best CV Accuracy: {best_score:.2%}")
        print(f"    📊 Precision: {self.metrics['precision']:.2%}, Recall: {self.metrics['recall']:.2%}")
        
        if best_score < 0.55:  # Umbral mínimo
            print(f"    ❌ Modelo no cumple umbral mínimo (55%)")
            return False
        
        # Entrenar final con todos los datos
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=4,
            min_samples_split=50,
            min_samples_leaf=25,
            subsample=0.8,
            random_state=42
        )
        
        self.model.fit(X_scaled, y)
        self.features = available_features
        self.feature_importances_ = pd.Series(
            self.model.feature_importances_,
            index=available_features
        ).sort_values(ascending=False)
        
        return True
    
    def predecir(self, df):
        """Predicción con filtros de confianza"""
        if self.model is None:
            return None
        
        try:
            # Calcular features
            df_features = FeatureEngineerRobusto.calcular_features(df)
            
            # Última fila
            X = df_features[self.features].iloc[[-1]]
            
            # Verificar que no haya NaNs
            if X.isna().any().any():
                return None
            
            # Escalar y predecir
            X_scaled = self.scaler.transform(X)
            
            proba = self.model.predict_proba(X_scaled)[0]
            prediction = self.model.predict(X_scaled)[0]
            
            # Solo retornar si confianza alta
            confidence = max(proba)
            
            return {
                'prediction': int(prediction),
                'confidence': confidence,
                'proba_long': proba[1],
                'proba_short': proba[0]
            }
            
        except Exception as e:
            print(f"Error en predicción: {e}")
            return None

# ============================================
# GESTIÓN DE RIESGO MEJORADA
# ============================================

class RiskManager:
    """Gestión de riesgo inteligente"""
    
    @staticmethod
    def calcular_stop_take_profit(df, direccion, entry_price):
        """Calcula SL y TP basado en múltiples factores"""
        last_row = df.iloc[-1]
        
        # 1. ATR básico
        atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14).iloc[-1]
        
        # 2. Volatilidad reciente
        recent_vol = df['Close'].pct_change().rolling(24).std().iloc[-1]
        
        # 3. Soporte/resistencia cercana
        recent_high = df['High'].rolling(20).max().iloc[-1]
        recent_low = df['Low'].rolling(20).min().iloc[-1]
        
        # 4. Niveles de precio psicológicos
        price_levels = RiskManager.get_psychological_levels(entry_price)
        
        if direccion == 'LONG':
            # SL: mínimo entre ATR, soporte cercano y nivel psicológico
            sl_candidates = [
                entry_price - (atr * TradingConfigOptimizado.MULTIPLICADOR_SL),
                recent_low,
                price_levels['support']
            ]
            stop_loss = max(sl_candidates)  # El más alto de los soportes
            
            # TP: mínimo entre ATR, resistencia y nivel psicológico
            tp_candidates = [
                entry_price + (atr * TradingConfigOptimizado.MULTIPLICADOR_TP),
                recent_high * 0.98,  # Justo por debajo de resistencia
                price_levels['resistance']
            ]
            take_profit = min(tp_candidates)  # El más bajo de las resistencias
            
        else:  # SHORT
            sl_candidates = [
                entry_price + (atr * TradingConfigOptimizado.MULTIPLICADOR_SL),
                recent_high,
                price_levels['resistance']
            ]
            stop_loss = min(sl_candidates)
            
            tp_candidates = [
                entry_price - (atr * TradingConfigOptimizado.MULTIPLICADOR_TP),
                recent_low * 1.02,  # Justo por encima de soporte
                price_levels['support']
            ]
            take_profit = max(tp_candidates)
        
        # Asegurar distancia mínima
        min_distance = entry_price * 0.002  # 0.2%
        if abs(take_profit - entry_price) < min_distance:
            take_profit = entry_price * (1 + (0.01 if direccion == 'LONG' else -0.01))
        
        if abs(stop_loss - entry_price) < min_distance:
            stop_loss = entry_price * (1 - (0.01 if direccion == 'LONG' else -0.01))
        
        # Calcular ratio R:R
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk': risk,
            'reward': reward,
            'rr_ratio': rr_ratio
        }
    
    @staticmethod
    def get_psychological_levels(price):
        """Niveles de precio psicológicos (00, 50)"""
        base = int(price / 10) * 10
        return {
            'support': base,
            'resistance': base + 10,
            'half': base + 5
        }
    
    @staticmethod
    def filtrar_señal(df, señal_data):
        """Filtra señales basado en múltiples factores"""
        last_row = df.iloc[-1]
        
        # 1. RSI extremo
        if señal_data['direccion'] == 'LONG' and last_row['rsi'] > 70:
            return False, "RSI sobrecomprado"
        if señal_data['direccion'] == 'SHORT' and last_row['rsi'] < 30:
            return False, "RSI sobrevendido"
        
        # 2. Volatilidad muy alta
        if last_row['vol_24h'] > TradingConfigOptimizado.VOLATILIDAD_MAX:
            return False, "Volatilidad muy alta"
        
        # 3. Volumen bajo
        if last_row.get('volume_ratio', 1) < TradingConfigOptimizado.MIN_VOLUME_RATIO:
            return False, "Volumen bajo"
        
        # 4. Tendencia contraria fuerte
        if señal_data['direccion'] == 'LONG' and last_row['above_sma50'] == 0:
            # Podría ser contra-tendencia, ser más estricto
            if señal_data['confidence'] < 0.75:
                return False, "Contra tendencia fuerte"
        
        # 5. Noticias/eventos (implementar si tienes API)
        
        return True, "Señal válida"

# ============================================
# SISTEMA PRINCIPAL OPTIMIZADO
# ============================================

class SistemaTradingOptimizado:
    """Sistema completo optimizado"""
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.modelos = {}
        self.metricas = {}
        
    def descargar_datos(self, start_date, end_date):
        """Descarga datos con manejo de errores"""
        try:
            df = yf.download(
                self.ticker,
                start=start_date,
                end=end_date,
                interval=TradingConfigOptimizado.INTERVALO,
                progress=False,
                auto_adjust=False
            )
            
            if df.empty:
                return None
            
            # Limpiar
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df = df.dropna()
            
            return df
            
        except Exception as e:
            print(f"Error descargando {self.ticker}: {e}")
            return None
    
    def entrenar_y_validar(self):
        """Pipeline completo de entrenamiento"""
        fechas = TradingConfigOptimizado.get_fechas()
        
        # 1. Descargar datos
        df = self.descargar_datos(fechas['inicio'], fechas['fin_backtest'])
        if df is None or len(df) < 1000:
            return False
        
        print(f"\n{'='*60}")
        print(f"🎯 {self.ticker} - Entrenamiento")
        print(f"{'='*60}")
        print(f"Datos: {len(df)} velas ({df.index[0].date()} a {df.index[-1].date()})")
        
        # 2. Entrenar modelos por horizonte
        for horizonte in TradingConfigOptimizado.HORIZONTES:
            print(f"\n  📊 Horizonte {horizonte}h:")
            
            modelo = ModeloTradingRobusto(self.ticker, horizonte)
            if modelo.entrenar(df):
                self.modelos[horizonte] = modelo
                
                # Feature importance
                print(f"    🔝 Top features:")
                for feat, imp in modelo.feature_importances_.head(5).items():
                    print(f"      {feat}: {imp:.3f}")
        
        return len(self.modelos) > 0
    
    def backtest_simple(self):
        """Backtesting simple pero efectivo"""
        if not self.modelos:
            return None
        
        fechas = TradingConfigOptimizado.get_fechas()
        
        # Datos de backtest (último mes)
        df_bt = self.descargar_datos(
            fechas['fin_backtest'] - timedelta(days=5),
            fechas['actual']
        )
        
        if df_bt is None:
            return None
        
        operaciones = []
        capital = 10000
        position = None
        
        for i in range(24, len(df_bt) - max(TradingConfigOptimizado.HORIZONTES)):
            df_slice = df_bt.iloc[:i+1]
            
            # Obtener predicciones
            predicciones = {}
            for horizonte, modelo in self.modelos.items():
                pred = modelo.predecir(df_slice)
                if pred:
                    predicciones[horizonte] = pred
            
            if not predicciones:
                continue
            
            # Consenso
            confianzas = [p['confidence'] for p in predicciones.values()]
            probs_long = [p['proba_long'] for p in predicciones.values()]
            
            avg_confidence = np.mean(confianzas)
            avg_prob_long = np.mean(probs_long)
            
            # Filtros estrictos
            if avg_confidence < TradingConfigOptimizado.UMBRAL_CONFIANZA_MIN:
                continue
            
            # Determinar dirección
            direccion = 'LONG' if avg_prob_long > 0.5 else 'SHORT'
            prob = avg_prob_long if direccion == 'LONG' else 1 - avg_prob_long
            
            if prob < TradingConfigOptimizado.UMBRAL_PROB_MIN:
                continue
            
            # Calcular SL/TP
            entry_price = df_slice['Close'].iloc[-1]
            risk_data = RiskManager.calcular_stop_take_profit(df_slice, direccion, entry_price)
            
            if risk_data['rr_ratio'] < TradingConfigOptimizado.MIN_RR_RATIO:
                continue
            
            # Filtrar señal
            señal_data = {
                'direccion': direccion,
                'confidence': avg_confidence,
                'prob': prob
            }
            
            es_valida, razon = RiskManager.filtrar_señal(df_slice, señal_data)
            if not es_valida:
                continue
            
            # Simular operación (simplificado)
            # Buscar resultado en ventana futura
            future_window = min(48, len(df_bt) - i - 1)
            future_prices = df_bt['Close'].iloc[i+1:i+1+future_window]
            
            sl = risk_data['stop_loss']
            tp = risk_data['take_profit']
            
            resultado = None
            retorno = 0
            
            for j, price in enumerate(future_prices, 1):
                if direccion == 'LONG':
                    if price >= tp:
                        resultado = 'TP'
                        retorno = (tp - entry_price) / entry_price
                        break
                    elif price <= sl:
                        resultado = 'SL'
                        retorno = (sl - entry_price) / entry_price
                        break
                else:  # SHORT
                    if price <= tp:
                        resultado = 'TP'
                        retorno = (entry_price - tp) / entry_price
                        break
                    elif price >= sl:
                        resultado = 'SL'
                        retorno = (entry_price - sl) / entry_price
                        break
            
            if resultado is None:
                # Cierre por tiempo
                final_price = future_prices.iloc[-1]
                resultado = 'TIME'
                if direccion == 'LONG':
                    retorno = (final_price - entry_price) / entry_price
                else:
                    retorno = (entry_price - final_price) / entry_price
            
            # Registrar operación
            operacion = {
                'fecha': df_slice.index[-1],
                'direccion': direccion,
                'entry': entry_price,
                'sl': sl,
                'tp': tp,
                'rr_ratio': risk_data['rr_ratio'],
                'prob': prob,
                'confianza': avg_confidence,
                'resultado': resultado,
                'retorno': retorno,
                'duracion_velas': j if resultado in ['TP', 'SL'] else future_window
            }
            
            operaciones.append(operacion)
            
            # Actualizar posición
            if resultado in ['TP', 'SL', 'TIME']:
                capital *= (1 + retorno * 0.1)  # 10% del capital por operación
                position = None
        
        # Calcular métricas
        if not operaciones:
            return None
        
        df_ops = pd.DataFrame(operaciones)
        
        wins = (df_ops['retorno'] > 0).sum()
        losses = (df_ops['retorno'] <= 0).sum()
        win_rate = wins / len(df_ops) if len(df_ops) > 0 else 0
        
        avg_win = df_ops[df_ops['retorno'] > 0]['retorno'].mean()
        avg_loss = df_ops[df_ops['retorno'] <= 0]['retorno'].mean() if losses > 0 else 0
        
        profit_factor = abs(avg_win * wins / (avg_loss * losses)) if losses > 0 else np.inf
        
        # Sharpe ratio aproximado
        sharpe = df_ops['retorno'].mean() / df_ops['retorno'].std() * np.sqrt(365*24) if df_ops['retorno'].std() > 0 else 0
        
        self.metricas = {
            'total_ops': len(df_ops),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe': sharpe,
            'max_drawdown': self.calcular_max_dd(df_ops['retorno'].cumsum()),
            'total_return': df_ops['retorno'].sum(),
            'avg_rr': df_ops['rr_ratio'].mean()
        }
        
        return df_ops
    
    def calcular_max_dd(self, returns_cum):
        """Calcula máximo drawdown"""
        running_max = returns_cum.expanding().max()
        drawdown = (returns_cum - running_max) / (running_max + 1e-10)
        return drawdown.min()
    
    def analizar_mercado_actual(self):
        """Análisis en tiempo real con múltiples filtros"""
        if not self.modelos:
            return None
        
        try:
            # Obtener datos recientes
            fechas = TradingConfigOptimizado.get_fechas()
            df = self.descargar_datos(
                fechas['actual'] - timedelta(days=7),
                fechas['actual']
            )
            
            if df is None or len(df) < 50:
                return None
            
            # Calcular features
            df_features = FeatureEngineerRobusto.calcular_features(df)
            last_row = df_features.iloc[-1]
            
            # Obtener predicciones de modelos
            predicciones = {}
            for horizonte, modelo in self.modelos.items():
                pred = modelo.predecir(df_features)
                if pred and pred['confidence'] > 0.6:
                    predicciones[horizonte] = pred
            
            if not predicciones:
                return None
            
            # Consenso ponderado por confianza
            confianzas = [p['confidence'] for p in predicciones.values()]
            probs_long = [p['proba_long'] for p in predicciones.values()]
            
            weights = np.array(confianzas) / sum(confianzas)
            prob_long_weighted = np.average(probs_long, weights=weights)
            avg_confidence = np.mean(confianzas)
            
            # Determinar dirección
            direccion = 'LONG' if prob_long_weighted > 0.5 else 'SHORT'
            prob_final = prob_long_weighted if direccion == 'LONG' else 1 - prob_long_weighted
            
            # Filtros estrictos
            if avg_confidence < TradingConfigOptimizado.UMBRAL_CONFIANZA_MIN:
                return None
            
            if prob_final < TradingConfigOptimizado.UMBRAL_PROB_MIN:
                return None
            
            # Gestión de riesgo
            entry_price = last_row['Close']
            risk_data = RiskManager.calcular_stop_take_profit(df_features, direccion, entry_price)
            
            if risk_data['rr_ratio'] < TradingConfigOptimizado.MIN_RR_RATIO:
                return None
            
            # Filtrar señal
            señal_data = {
                'direccion': direccion,
                'confidence': avg_confidence,
                'prob': prob_final
            }
            
            es_valida, razon = RiskManager.filtrar_señal(df_features, señal_data)
            if not es_valida:
                print(f"    ⚠️ Señal filtrada: {razon}")
                return None
            
            # Preparar señal
            señal = {
                'ticker': self.ticker,
                'timestamp': datetime.now(TradingConfigOptimizado.TIMEZONE),
                'direccion': direccion,
                'entry': entry_price,
                'stop_loss': risk_data['stop_loss'],
                'take_profit': risk_data['take_profit'],
                'rr_ratio': risk_data['rr_ratio'],
                'probabilidad': prob_final,
                'confianza': avg_confidence,
                'rsi': last_row.get('rsi', 50),
                'volatilidad': last_row.get('vol_24h', 0),
                'volume_ratio': last_row.get('volume_ratio', 1),
                'trend': 'ALCISTA' if last_row.get('above_sma50', 0) == 1 else 'BAJISTA',
                'filter_reason': razon
            }
            
            return señal
            
        except Exception as e:
            print(f"Error análisis tiempo real {self.ticker}: {e}")
            return None

# ============================================
# EJECUCIÓN PRINCIPAL
# ============================================

def main_optimizado():
    """Función principal optimizada"""
    print("🚀 SISTEMA DE TRADING OPTIMIZADO")
    print("=" * 60)
    
    resultados = {}
    señales = []
    
    for ticker in TradingConfigOptimizado.ACTIVOS[:4]:  # Solo 4 principales para empezar
        print(f"\n📊 Procesando {ticker}...")
        
        sistema = SistemaTradingOptimizado(ticker)
        
        # 1. Entrenar
        if not sistema.entrenar_y_validar():
            print(f"  ❌ Falló entrenamiento")
            continue
        
        # 2. Backtest
        ops = sistema.backtest_simple()
        if ops is None:
            print(f"  ⚠️ No hay operaciones en backtest")
            continue
        
        # 3. Evaluar resultados
        m = sistema.metricas
        print(f"\n  📈 Resultados Backtest:")
        print(f"    Operaciones: {m['total_ops']}")
        print(f"    Win Rate: {m['win_rate']:.2%}")
        print(f"    Profit Factor: {m['profit_factor']:.2f}")
        print(f"    Sharpe: {m['sharpe']:.2f}")
        print(f"    Avg R:R: {m['avg_rr']:.2f}")
        
        # Criterios de viabilidad
        es_viable = (
            m['win_rate'] > 0.55 and
            m['profit_factor'] > 1.5 and
            m['total_ops'] >= 10 and
            m['sharpe'] > 0.5
        )
        
        if es_viable:
            print(f"  ✅ VIABLE para trading")
            
            # 4. Análisis tiempo real
            señal = sistema.analizar_mercado_actual()
            if señal:
                print(f"  🚨 SEÑAL ACTUAL:")
                print(f"    Dirección: {señal['direccion']}")
                print(f"    Prob: {señal['probabilidad']:.2%}")
                print(f"    R:R: {señal['rr_ratio']:.2f}")
                print(f"    RSI: {señal['rsi']:.1f}")
                
                señales.append(señal)
        else:
            print(f"  ❌ NO VIABLE")
        
        resultados[ticker] = {
            'viable': es_viable,
            'metricas': m,
            'señal_actual': señal if es_viable else None
        }
    
    # Enviar señales
    if señales:
        enviar_telegram_señales(señales)
    
    return resultados

def enviar_telegram_señales(señales):
    """Envía señales a Telegram"""
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        return
    
    for señal in señales:
        mensaje = f"""
🚨 SEÑAL DE TRADING
──────────────
Ticker: {señal['ticker']}
Dirección: {señal['direccion']}
Probabilidad: {señal['probabilidad']:.2%}
Confianza: {señal['confianza']:.2%}
──────────────
🎯 Entry: ${señal['entry']:.2f}
🛑 Stop Loss: ${señal['stop_loss']:.2f}
✅ Take Profit: ${señal['take_profit']:.2f}
──────────────
⚖️ R:R Ratio: {señal['rr_ratio']:.2f}
📊 RSI: {señal['rsi']:.1f}
📈 Tendencia: {señal['trend']}
──────────────
⚠️ Filtro: {señal['filter_reason']}
⏰ Hora: {señal['timestamp'].strftime('%Y-%m-%d %H:%M')}
        """
        
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, json={
            "chat_id": chat_id,
            "text": mensaje,
            "parse_mode": "HTML"
        })

if __name__ == "__main__":
    # Ejecutar sistema optimizado
    resultados = main_optimizado()
