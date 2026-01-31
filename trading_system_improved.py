import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import os
import requests
import warnings
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from pathlib import Path
import json
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ============================================
# CONFIGURACIÓN
# ============================================

class TradingConfig:
    TIMEZONE = pytz.timezone('America/Bogota')
    INTERVALO = "1h"
    DIAS_ENTRENAMIENTO = 180
    DIAS_VALIDACION = 60
    DIAS_BACKTEST = 60
    
    ACTIVOS = ["BTC-USD"]
    HORIZONTES = [4, 8, 12, 24]
    
    # Gestión de riesgo
    SL_MULTIPLIER = 2.0
    TP_MULTIPLIER = 3.0
    RATIO_RR_MINIMO = 1.5
    
    # Umbrales
    UMBRAL_PROBABILIDAD = 0.55
    UMBRAL_CONFIANZA = 0.60
    
    # Modelos
    MODELOS = {
        'XGBoost': XGBClassifier(random_state=42, n_jobs=-1, verbosity=0),
        'LightGBM': LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
    }
    
    @classmethod
    def get_fechas(cls):
        now = datetime.now(cls.TIMEZONE)
        return {
            'actual': now,
            'inicio_backtest': now - timedelta(days=cls.DIAS_BACKTEST),
            'inicio_validacion': now - timedelta(days=cls.DIAS_VALIDACION + cls.DIAS_BACKTEST),
            'inicio_entrenamiento': now - timedelta(days=cls.DIAS_ENTRENAMIENTO + cls.DIAS_VALIDACION + cls.DIAS_BACKTEST)
        }

# ============================================
# FEATURE ENGINEERING SIN TA-Lib
# ============================================

class FeatureEngineer:
    """Calcula indicadores sin TA-Lib"""
    
    @staticmethod
    def calcular_rsi(close, periodo=14):
        """Calcula RSI sin TA-Lib"""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    @staticmethod
    def calcular_atr(high, low, close, periodo=14):
        """Calcula ATR sin TA-Lib"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=periodo).mean()
        return atr.fillna(method='bfill')
    
    @staticmethod
    def calcular_macd(close, fast=12, slow=26, signal=9):
        """Calcula MACD sin TA-Lib"""
        exp1 = close.ewm(span=fast, adjust=False).mean()
        exp2 = close.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    @staticmethod
    def calcular_bollinger_bands(close, periodo=20, std_dev=2):
        """Calcula Bollinger Bands sin TA-Lib"""
        sma = close.rolling(window=periodo).mean()
        std = close.rolling(window=periodo).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def calcular_stochastic(high, low, close, periodo=14, smooth_k=3, smooth_d=3):
        """Calcula Estocástico sin TA-Lib"""
        low_min = low.rolling(window=periodo).min()
        high_max = high.rolling(window=periodo).max()
        
        k = 100 * ((close - low_min) / (high_max - low_min))
        k_smooth = k.rolling(window=smooth_k).mean()
        d = k_smooth.rolling(window=smooth_d).mean()
        
        return k_smooth, d
    
    @staticmethod
    def calcular_obv(close, volume):
        """Calcula OBV sin TA-Lib"""
        obv = pd.Series(0, index=close.index)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def calcular_features(df):
        """Calcula todas las features sin TA-Lib"""
        df = df.copy()
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        open_price = df['Open']
        
        # 1. Retornos
        df['retorno_1h'] = close.pct_change(1)
        df['retorno_4h'] = close.pct_change(4)
        df['retorno_12h'] = close.pct_change(12)
        df['retorno_24h'] = close.pct_change(24)
        
        # 2. Volatilidad
        df['volatilidad_24h'] = df['retorno_1h'].rolling(24).std()
        
        # 3. RSI
        df['RSI'] = FeatureEngineer.calcular_rsi(close, 14)
        
        # 4. ATR
        df['ATR'] = FeatureEngineer.calcular_atr(high, low, close, 14)
        df['ATR_pct'] = df['ATR'] / close
        
        # 5. MACD
        macd, signal, hist = FeatureEngineer.calcular_macd(close)
        df['MACD'] = macd
        df['MACD_SIGNAL'] = signal
        df['MACD_HIST'] = hist
        
        # 6. Bollinger Bands
        bb_upper, bb_middle, bb_lower = FeatureEngineer.calcular_bollinger_bands(close)
        df['BB_UPPER'] = bb_upper
        df['BB_MIDDLE'] = bb_middle
        df['BB_LOWER'] = bb_lower
        df['BB_PERCENT'] = (close - bb_lower) / (bb_upper - bb_lower)
        df['BB_WIDTH'] = (bb_upper - bb_lower) / bb_middle
        
        # 7. Estocástico
        stoch_k, stoch_d = FeatureEngineer.calcular_stochastic(high, low, close)
        df['STOCH_K'] = stoch_k
        df['STOCH_D'] = stoch_d
        
        # 8. OBV
        df['OBV'] = FeatureEngineer.calcular_obv(close, volume)
        df['OBV_MA'] = df['OBV'].rolling(20).mean()
        
        # 9. Medias móviles
        df['SMA_20'] = close.rolling(20).mean()
        df['SMA_50'] = close.rolling(50).mean()
        df['EMA_12'] = close.ewm(span=12, adjust=False).mean()
        df['EMA_26'] = close.ewm(span=26, adjust=False).mean()
        
        df['SMA_20_RATIO'] = close / df['SMA_20']
        df['SMA_50_RATIO'] = close / df['SMA_50']
        
        # 10. Tendencias
        df['tendencia_sma'] = (df['SMA_20'] > df['SMA_50']).astype(int)
        df['tendencia_ema'] = (df['EMA_12'] > df['EMA_26']).astype(int)
        
        # 11. Volumen
        df['volumen_relativo'] = volume / volume.rolling(20).mean()
        df['volumen_sma'] = volume.rolling(20).mean()
        
        # 12. Rango de precio
        df['rango_hl'] = (high - low) / close
        df['rango_hl_pct'] = (high - low) / low
        df['body_size'] = abs(close - open_price) / close
        
        # 13. Momento
        df['momentum_4h'] = close / close.shift(4) - 1
        df['momentum_12h'] = close / close.shift(12) - 1
        
        # 14. Características de tiempo
        df['hora_dia'] = df.index.hour
        df['dia_semana'] = df.index.dayofweek
        df['es_finde'] = df['dia_semana'].isin([5, 6]).astype(int)
        
        # 15. Soporte y resistencia
        df['resistance_20'] = high.rolling(20).max()
        df['support_20'] = low.rolling(20).min()
        df['dist_to_res'] = (df['resistance_20'] - close) / close
        df['dist_to_sup'] = (close - df['support_20']) / close
        
        # 16. Z-score de precio
        df['z_score_24h'] = (close - close.rolling(24).mean()) / close.rolling(24).std()
        
        # Llenar NaN
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df

# ============================================
# SISTEMA DE ENTRENAMIENTO
# ============================================

class TradingSystem:
    def __init__(self, ticker):
        self.ticker = ticker
        self.config = TradingConfig()
        self.fechas = self.config.get_fechas()
        self.datos = None
        self.modelos = {}
        self.resultados = {}
    
    def descargar_datos(self):
        """Descarga datos históricos"""
        try:
            df = yf.download(
                self.ticker,
                start=self.fechas['inicio_entrenamiento'],
                end=self.fechas['actual'],
                interval=self.config.INTERVALO,
                progress=False
            )
            
            if df.empty:
                return False
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            self.datos = df
            print(f"✅ Descargadas {len(df)} velas")
            return True
            
        except Exception as e:
            print(f"❌ Error descargando datos: {e}")
            return False
    
    def preparar_features_y_etiquetas(self, df, horizonte):
        """Prepara features y etiquetas para un horizonte específico"""
        # Calcular features
        df_features = FeatureEngineer.calcular_features(df)
        
        # Crear etiquetas (retorno futuro)
        retorno_futuro = df_features['Close'].shift(-horizonte) / df_features['Close'] - 1
        
        # Umbral dinámico basado en volatilidad
        umbral = df_features['volatilidad_24h'] * 1.5
        
        # Crear etiquetas binarias
        etiqueta = pd.Series(0, index=df_features.index)  # Neutral por defecto
        etiqueta[retorno_futuro > umbral] = 1  # Alcista
        etiqueta[retorno_futuro < -umbral] = 0  # Bajista
        
        # Filtrar columnas para el modelo
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'resistance_20', 'support_20']
        features = [col for col in df_features.columns if col not in exclude_cols]
        
        return df_features, etiqueta, features
    
    def entrenar_modelo_para_horizonte(self, horizonte):
        """Entrena un modelo para un horizonte específico"""
        # Dividir datos
        fecha_corte = self.fechas['inicio_backtest']
        df_train = self.datos[self.datos.index < fecha_corte].copy()
        df_test = self.datos[self.datos.index >= fecha_corte].copy()
        
        # Preparar datos de entrenamiento
        df_train_features, y_train, features = self.preparar_features_y_etiquetas(df_train, horizonte)
        X_train = df_train_features[features]
        
        # Filtrar NaN
        mask = y_train.notna()
        X_train = X_train[mask]
        y_train = y_train[mask]
        
        if len(X_train) < 100:
            print(f"  ⚠️ Datos insuficientes para horizonte {horizonte}h: {len(X_train)}")
            return None
        
        # Escalar
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Entrenar múltiples modelos
        mejores_resultados = {'modelo': None, 'accuracy': 0, 'nombre': ''}
        
        for nombre, modelo_base in self.config.MODELOS.items():
            # Validación walk-forward
            tscv = TimeSeriesSplit(n_splits=3)
            accuracies = []
            
            for train_idx, val_idx in tscv.split(X_train_scaled):
                X_t, X_v = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                modelo = modelo_base.__class__(**modelo_base.get_params())
                modelo.fit(X_t, y_t)
                
                y_pred = modelo.predict(X_v)
                acc = accuracy_score(y_v, y_pred)
                accuracies.append(acc)
            
            acc_promedio = np.mean(accuracies)
            
            if acc_promedio > mejores_resultados['accuracy']:
                mejores_resultados = {
                    'modelo': modelo_base,
                    'accuracy': acc_promedio,
                    'nombre': nombre
                }
        
        if mejores_resultados['modelo'] is None:
            return None
        
        # Entrenar modelo final
        modelo_final = mejores_resultados['modelo'].__class__(**mejores_resultados['modelo'].get_params())
        modelo_final.fit(X_train_scaled, y_train)
        
        print(f"  ✅ Horizonte {horizonte}h: {mejores_resultados['nombre']} (Accuracy: {mejores_resultados['accuracy']:.2%})")
        
        return {
            'modelo': modelo_final,
            'scaler': scaler,
            'features': features,
            'accuracy': mejores_resultados['accuracy'],
            'nombre': mejores_resultados['nombre']
        }
    
    def entrenar_todos_modelos(self):
        """Entrena modelos para todos los horizontes"""
        print("\n🎯 ENTRENANDO MODELOS")
        print("=" * 60)
        
        for horizonte in self.config.HORIZONTES:
            print(f"\n🔮 Horizonte {horizonte}h")
            modelo_data = self.entrenar_modelo_para_horizonte(horizonte)
            if modelo_data:
                self.modelos[horizonte] = modelo_data
    
    def ejecutar_backtest(self):
        """Ejecuta backtesting"""
        if not self.modelos:
            print("❌ No hay modelos entrenados")
            return False
        
        print("\n🔬 EJECUTANDO BACKTEST")
        print("=" * 60)
        
        fecha_inicio = self.fechas['inicio_backtest']
        df_backtest = self.datos[self.datos.index >= fecha_inicio].copy()
        
        operaciones = []
        
        # Para cada punto en el backtest
        for i in range(100, len(df_backtest) - 24):  # Dejar margen
            idx = df_backtest.index[i]
            
            # Datos disponibles hasta este punto
            df_historia = df_backtest.iloc[:i+1].copy()
            df_features = FeatureEngineer.calcular_features(df_historia)
            
            # Obtener predicciones de todos los modelos
            predicciones = []
            confianzas = []
            
            for horizonte, modelo_data in self.modelos.items():
                # Preparar datos para predicción
                X_pred = df_features[modelo_data['features']].tail(1)
                
                if not X_pred.empty:
                    X_scaled = modelo_data['scaler'].transform(X_pred)
                    pred = modelo_data['modelo'].predict(X_scaled)[0]
                    proba = modelo_data['modelo'].predict_proba(X_scaled)[0]
                    
                    predicciones.append(pred)
                    confianzas.append(max(proba))
            
            if predicciones:
                # Consenso: mayoría simple
                señal = 1 if sum(predicciones) > len(predicciones) / 2 else 0
                confianza_promedio = np.mean(confianzas)
                
                # Filtrar por confianza
                if confianza_promedio > self.config.UMBRAL_CONFIANZA:
                    # Simular operación
                    operacion = self.simular_operacion(df_backtest, i, señal, confianza_promedio)
                    if operacion:
                        operaciones.append(operacion)
        
        if operaciones:
            self.analizar_resultados(operaciones)
            return True
        
        return False
    
    def simular_operacion(self, df, idx_pos, señal, confianza):
        """Simula una operación de trading"""
        entrada = df.iloc[idx_pos]
        precio = entrada['Close']
        atr = entrada.get('ATR', precio * 0.02) if 'ATR' in df.columns else precio * 0.02
        
        # Calcular SL y TP
        if señal == 1:  # LONG
            sl = precio * (1 - self.config.SL_MULTIPLIER * atr / precio)
            tp = precio * (1 + self.config.TP_MULTIPLIER * atr / precio)
        else:  # SHORT
            sl = precio * (1 + self.config.SL_MULTIPLIER * atr / precio)
            tp = precio * (1 - self.config.TP_MULTIPLIER * atr / precio)
        
        # Calcular ratio R:R
        riesgo = abs(precio - sl)
        recompensa = abs(tp - precio)
        ratio_rr = recompensa / riesgo if riesgo > 0 else 0
        
        if ratio_rr < self.config.RATIO_RR_MINIMO:
            return None
        
        # Simular resultado
        resultado, retorno, velas = 'TIMEOUT', 0, 0
        
        for j in range(1, min(24, len(df) - idx_pos - 1)):
            precio_actual = df.iloc[idx_pos + j]['Close']
            
            if señal == 1:  # LONG
                if precio_actual >= tp:
                    resultado = 'TP'
                    retorno = (tp - precio) / precio
                    velas = j
                    break
                elif precio_actual <= sl:
                    resultado = 'SL'
                    retorno = (sl - precio) / precio
                    velas = j
                    break
            else:  # SHORT
                if precio_actual <= tp:
                    resultado = 'TP'
                    retorno = (precio - tp) / precio
                    velas = j
                    break
                elif precio_actual >= sl:
                    resultado = 'SL'
                    retorno = (precio - sl) / precio
                    velas = j
                    break
        
        if resultado == 'TIMEOUT':
            precio_final = df.iloc[idx_pos + 23]['Close']
            if señal == 1:
                retorno = (precio_final - precio) / precio
            else:
                retorno = (precio - precio_final) / precio
            velas = 23
        
        return {
            'fecha': df.index[idx_pos],
            'direccion': 'LONG' if señal == 1 else 'SHORT',
            'precio': precio,
            'sl': sl,
            'tp': tp,
            'ratio_rr': ratio_rr,
            'confianza': confianza,
            'resultado': resultado,
            'retorno': retorno,
            'velas': velas
        }
    
    def analizar_resultados(self, operaciones):
        """Analiza los resultados del backtest"""
        df_ops = pd.DataFrame(operaciones)
        
        retornos = df_ops['retorno']
        ganadoras = retornos > 0
        
        n_ops = len(df_ops)
        win_rate = ganadoras.mean()
        retorno_total = retornos.sum()
        retorno_promedio = retornos.mean()
        
        # Profit Factor
        ganancias = retornos[retornos > 0].sum()
        perdidas = abs(retornos[retornos < 0].sum())
        pf = ganancias / perdidas if perdidas > 0 else np.inf
        
        # Drawdown
        equity_curve = (1 + retornos).cumprod()
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_dd = drawdown.min()
        
        # Sharpe Ratio
        sharpe = retornos.mean() / retornos.std() if retornos.std() > 0 else 0
        
        self.resultados = {
            'n_operaciones': n_ops,
            'win_rate': win_rate,
            'retorno_total': retorno_total,
            'retorno_promedio': retorno_promedio,
            'profit_factor': pf,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'ganancias': ganancias,
            'perdidas': perdidas,
            'mejor_operacion': retornos.max(),
            'peor_operacion': retornos.min(),
            'operaciones': df_ops
        }
        
        # Mostrar resultados
        print(f"\n📊 RESULTADOS DEL BACKTEST")
        print(f"  Operaciones: {n_ops}")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Retorno Total: {retorno_total:.2%}")
        print(f"  Retorno Promedio: {retorno_promedio:.2%}")
        print(f"  Profit Factor: {pf:.2f}")
        print(f"  Max Drawdown: {max_dd:.2%}")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        
        if 'direccion' in df_ops.columns:
            long_ops = df_ops[df_ops['direccion'] == 'LONG']
            short_ops = df_ops[df_ops['direccion'] == 'SHORT']
            
            if len(long_ops) > 0:
                print(f"  Win Rate LONG: {long_ops['retorno'].gt(0).mean():.2%}")
            if len(short_ops) > 0:
                print(f"  Win Rate SHORT: {short_ops['retorno'].gt(0).mean():.2%}")
    
    def evaluar_sistema(self):
        """Evalúa si el sistema es viable"""
        if not self.resultados:
            print("❌ No hay resultados para evaluar")
            return False
        
        m = self.resultados
        
        criterios = {
            'Win Rate > 50%': m['win_rate'] > 0.50,
            'Profit Factor > 1.2': m['profit_factor'] > 1.2,
            'Operaciones >= 10': m['n_operaciones'] >= 10,
            'Retorno Total > 0': m['retorno_total'] > 0,
            'Sharpe > 0': m['sharpe_ratio'] > 0,
            'Max DD < 25%': abs(m['max_drawdown']) < 0.25
        }
        
        cumplidos = sum(criterios.values())
        total = len(criterios)
        
        print(f"\n📋 EVALUACIÓN DE VIABILIDAD")
        print("=" * 60)
        
        for criterio, cumple in criterios.items():
            print(f"  {'✅' if cumple else '❌'} {criterio}")
        
        print(f"\n  Criterios cumplidos: {cumplidos}/{total}")
        
        viable = cumplidos >= 4
        
        if viable:
            print(f"\n🎉 ¡SISTEMA VIABLE!")
            self.generar_senal_actual()
        else:
            print(f"\n⚠️ Sistema no viable en condiciones actuales")
        
        return viable
    
    def generar_senal_actual(self):
        """Genera señal actual basada en los modelos entrenados"""
        print(f"\n🔮 GENERANDO SEÑAL ACTUAL")
        print("=" * 60)
        
        try:
            # Descargar datos recientes
            df_reciente = yf.download(
                self.ticker,
                start=datetime.now(self.config.TIMEZONE) - timedelta(days=7),
                end=datetime.now(self.config.TIMEZONE),
                interval=self.config.INTERVALO,
                progress=False
            )
            
            if df_reciente.empty:
                print("❌ No hay datos recientes")
                return
            
            # Procesar features
            df_features = FeatureEngineer.calcular_features(df_reciente)
            
            # Obtener predicciones
            predicciones = []
            confianzas = []
            
            for horizonte, modelo_data in self.modelos.items():
                if modelo_data['features'][0] in df_features.columns:
                    X_pred = df_features[modelo_data['features']].tail(1)
                    X_scaled = modelo_data['scaler'].transform(X_pred)
                    
                    pred = modelo_data['modelo'].predict(X_scaled)[0]
                    proba = modelo_data['modelo'].predict_proba(X_scaled)[0]
                    
                    predicciones.append(pred)
                    confianzas.append(max(proba))
            
            if predicciones:
                # Consenso
                señal_promedio = np.mean(predicciones)
                confianza_promedio = np.mean(confianzas)
                
                direccion = "LONG" if señal_promedio > 0.5 else "SHORT"
                probabilidad = señal_promedio if direccion == "LONG" else 1 - señal_promedio
                
                # Calcular niveles
                ultima_vela = df_features.iloc[-1]
                precio = ultima_vela['Close']
                atr = ultima_vela.get('ATR', precio * 0.02)
                
                if direccion == "LONG":
                    sl = precio * (1 - self.config.SL_MULTIPLIER * atr / precio)
                    tp = precio * (1 + self.config.TP_MULTIPLIER * atr / precio)
                else:
                    sl = precio * (1 + self.config.SL_MULTIPLIER * atr / precio)
                    tp = precio * (1 - self.config.TP_MULTIPLIER * atr / precio)
                
                ratio_rr = abs(tp - precio) / abs(precio - sl)
                
                print(f"\n📡 SEÑAL:")
                print(f"  Dirección: {direccion}")
                print(f"  Probabilidad: {probabilidad:.2%}")
                print(f"  Confianza: {confianza_promedio:.2%}")
                print(f"  Precio: ${precio:,.2f}")
                print(f"  Stop Loss: ${sl:,.2f}")
                print(f"  Take Profit: ${tp:,.2f}")
                print(f"  Ratio R:R: {ratio_rr:.2f}")
                print(f"  RSI: {ultima_vela.get('RSI', 0):.1f}")
                print(f"  Volatilidad: {ultima_vela.get('volatilidad_24h', 0)*100:.1f}%")
                
                # Enviar señal si cumple criterios
                if (confianza_promedio > self.config.UMBRAL_CONFIANZA and 
                    probabilidad > self.config.UMBRAL_PROBABILIDAD and
                    ratio_rr > self.config.RATIO_RR_MINIMO):
                    
                    mensaje = (
                        f"🚨 SEÑAL {self.ticker}\n"
                        f"📅 {datetime.now(self.config.TIMEZONE).strftime('%Y-%m-%d %H:%M')}\n"
                        f"📈 Dirección: {direccion}\n"
                        f"🎯 Probabilidad: {probabilidad:.2%}\n"
                        f"🛡️ Confianza: {confianza_promedio:.2%}\n\n"
                        f"💰 Entrada: ${precio:,.2f}\n"
                        f"🛑 Stop Loss: ${sl:,.2f}\n"
                        f"🎯 Take Profit: ${tp:,.2f}\n"
                        f"⚖️ Ratio R:R: {ratio_rr:.2f}"
                    )
                    
                    # Enviar Telegram
                    token = os.getenv("TELEGRAM_TOKEN")
                    chat_id = os.getenv("TELEGRAM_CHAT_ID")
                    
                    if token and chat_id:
                        try:
                            url = f"https://api.telegram.org/bot{token}/sendMessage"
                            requests.post(url, data={"chat_id": chat_id, "text": mensaje})
                            print("✅ Señal enviada por Telegram")
                        except:
                            print("⚠️ Error enviando Telegram")
                    
                    # Guardar última señal
                    guardar_ultima_senal({
                        "ticker": self.ticker,
                        "direccion": direccion,
                        "probabilidad": probabilidad,
                        "fecha": str(datetime.now(self.config.TIMEZONE))
                    })
        
        except Exception as e:
            print(f"❌ Error generando señal: {e}")
    
    def ejecutar(self):
        """Ejecuta el sistema completo"""
        print("🚀 SISTEMA DE TRADING MEJORADO")
        print("=" * 60)
        
        # 1. Descargar datos
        if not self.descargar_datos():
            return
        
        # 2. Entrenar modelos
        self.entrenar_todos_modelos()
        
        if not self.modelos:
            print("❌ No se pudieron entrenar modelos")
            return
        
        # 3. Backtest
        if not self.ejecutar_backtest():
            print("❌ Backtest fallido")
            return
        
        # 4. Evaluar
        self.evaluar_sistema()


# ============================================
# EJECUCIÓN PRINCIPAL
# ============================================

if __name__ == "__main__":
    # Instanciar y ejecutar sistema
    sistema = TradingSystem("BTC-USD")
    sistema.ejecutar()
