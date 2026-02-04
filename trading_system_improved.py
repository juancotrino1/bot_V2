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

warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN CORREGIDA
# ============================================

class TradingConfig:
    """Configuración con parámetros realistas para crypto"""
    
    TIMEZONE = pytz.timezone('America/Bogota')
    
    # Datos
    INTERVALO = "1h"
    DIAS_ENTRENAMIENTO = 365
    DIAS_VALIDACION = 60
    DIAS_BACKTEST = 30
    
    # Activos (solo los más líquidos)
    ACTIVOS = [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"
    ]
    
    # Features
    VENTANA_VOLATILIDAD = 24
    RSI_PERIODO = 14
    ATR_PERIODO = 14
    
    # ✅ HORIZONTE CORTO (máximo 2 horas en crypto)
    HORIZONTES = [1, 2,4,8,12]  # 1h y 2h
    
    # 🎯 GESTIÓN DE RIESGO REALISTA
    STOP_LOSS_PCT = 0.015    # 2.5% (más espacio para ruido)
    TAKE_PROFIT_PCT = 0.03  # 4.0% (ratio 1.6:1)
    RATIO_MINIMO_RR = 1.5    # Más permisivo
    MAX_RIESGO_POR_OPERACION = 0.02  # 2% por operación
    
    # Validación
    N_FOLDS_WF = 5
    MIN_MUESTRAS_ENTRENAMIENTO = 1000
    MIN_MUESTRAS_CLASE = 50
    
    # 🔥 UMBRALES REALISTAS
    UMBRAL_PROBABILIDAD_MIN = 0.72  # 55% (antes 72%)
    UMBRAL_CONFIANZA_MIN = 0.70     # 53% (antes 70%)
    
    # 🎯 UMBRAL DE MOVIMIENTO PARA ETIQUETAS
    UMBRAL_MOVIMIENTO = 0.012  # 1.2% (más estricto para reducir ruido)
    
    # ✅ SIN FILTROS RSI/Z-SCORE (confiar en el modelo)
    # Solo usar en casos extremos para evitar entrar en crashes
    RSI_EXTREME_LOW = 25   # Solo bloquear SHORT si RSI < 15
    RSI_EXTREME_HIGH = 75  # Solo bloquear LONG si RSI > 85
    
    MODELOS_DIR = Path("modelos_trading")
    
    @classmethod
    def get_fechas(cls):
        now = datetime.now(cls.TIMEZONE)
        return {
            'actual': now,
            'inicio_entrenamiento': now - timedelta(days=cls.DIAS_ENTRENAMIENTO + cls.DIAS_VALIDACION + cls.DIAS_BACKTEST),
            'inicio_validacion': now - timedelta(days=cls.DIAS_VALIDACION + cls.DIAS_BACKTEST),
            'inicio_backtest': now - timedelta(days=cls.DIAS_BACKTEST)
        }


# ============================================
# INDICADORES SIN LOOK-AHEAD BIAS
# ============================================

class IndicadoresTecnicos:
    """Features calculadas SOLO con datos pasados"""
    
    @staticmethod
    def calcular_rsi(precios, periodo=14):
        """RSI sin look-ahead"""
        delta = precios.diff()
        ganancia = delta.where(delta > 0, 0).rolling(window=periodo).mean()
        perdida = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
        perdida = perdida.replace(0, 1e-10)
        rs = ganancia / perdida
        return (100 - (100 / (1 + rs))).fillna(50)
    
    @staticmethod
    def calcular_atr(df, periodo=14):
        """ATR sin look-ahead"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        close_prev = close.shift(1)
        
        tr = pd.concat([
            high - low,
            (high - close_prev).abs(),
            (low - close_prev).abs()
        ], axis=1).max(axis=1)
        
        return tr.rolling(window=periodo).mean().fillna(method='bfill')
    
    @staticmethod
    def calcular_features(df):
        """
        ✅ CRÍTICO: Todas las features usan .shift(1) para evitar look-ahead
        En tiempo real, cuando predecimos la vela N, solo conocemos hasta N-1
        """
        df = df.copy()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df.get('Volume', pd.Series(1, index=df.index))
        
        # ✅ RETORNOS PASADOS (shift = 1)
        df['retorno_1h'] = close.pct_change(1).shift(1)
        df['retorno_4h'] = close.pct_change(4).shift(1)
        df['retorno_12h'] = close.pct_change(12).shift(1)
        
        # ✅ VOLATILIDAD PASADA
        retornos = close.pct_change(1)
        df['volatilidad_24h'] = retornos.rolling(24).std().shift(1)
        
        # ✅ RSI PASADO
        rsi_raw = IndicadoresTecnicos.calcular_rsi(close, 14)
        df['RSI'] = rsi_raw.shift(1)
        
        # ✅ MEDIAS MÓVILES PASADAS
        sma_12 = close.rolling(12).mean().shift(1)
        sma_24 = close.rolling(24).mean().shift(1)
        df['SMA_12'] = sma_12
        df['SMA_24'] = sma_24
        
        # Distancia a SMA (usando precio anterior)
        close_prev = close.shift(1)
        df['dist_sma_12'] = (close_prev - sma_12) / close_prev
        
        # ✅ ATR PASADO
        atr = IndicadoresTecnicos.calcular_atr(df, 14)
        df['ATR'] = atr.shift(1)
        df['ATR_pct'] = (atr / close).shift(1)
        
        # ✅ VOLUMEN RELATIVO PASADO
        vol_ma = volume.rolling(24).mean()
        df['volumen_rel'] = (volume / vol_ma).shift(1)
        
        # ✅ RANGO HL PASADO
        rango = (high - low) / close
        df['rango_hl_pct'] = rango.shift(1)
        
        # ✅ TENDENCIA PASADA
        df['tendencia'] = (sma_12 > sma_24).astype(int)
        
        # ✅ MOMENTUM PASADO
        df['momentum_24h'] = (close / close.shift(24) - 1).shift(1)
        
        return df


# ============================================
# ETIQUETADO MEJORADO
# ============================================

class EtiquetadoDatos:
    
    @staticmethod
    def crear_etiquetas_direccion(df, horizonte):
        """
        ✅ ETIQUETADO CORRECTO:
        - Calcula retorno futuro desde AHORA hasta horizonte
        - umbral más estricto (1.2%) para reducir ruido
        - retorno_futuro > 1.2% → LONG (1)
        - retorno_futuro < -1.2% → SHORT (0)
        - entre -1.2% y 1.2% → NaN (no operar, rango sin tendencia clara)
        """
        retorno_futuro = df['Close'].shift(-horizonte) / df['Close'] - 1
        
        etiqueta = pd.Series(np.nan, index=df.index)
        etiqueta[retorno_futuro > TradingConfig.UMBRAL_MOVIMIENTO] = 1   # LONG
        etiqueta[retorno_futuro < -TradingConfig.UMBRAL_MOVIMIENTO] = 0  # SHORT
        # El resto queda NaN (movimientos laterales)
        
        return etiqueta, retorno_futuro
    
    @staticmethod
    def preparar_dataset_ml(df, horizonte):
        """Prepara dataset con features sin look-ahead"""
        df = IndicadoresTecnicos.calcular_features(df)
        etiqueta, retorno_futuro = EtiquetadoDatos.crear_etiquetas_direccion(df, horizonte)
        
        df[f'etiqueta_{horizonte}h'] = etiqueta
        df[f'retorno_futuro_{horizonte}h'] = retorno_futuro
        
        # Features finales (todas ya tienen shift aplicado)
        features = [
            'RSI',
            'volatilidad_24h',
            'dist_sma_12',
            'ATR_pct',
            'tendencia',
            'retorno_1h',
            'retorno_4h',
            'retorno_12h',
            'volumen_rel',
            'rango_hl_pct',
            'momentum_24h'
        ]
        
        features_disponibles = [f for f in features if f in df.columns]
        
        return df, features_disponibles


# ============================================
# MODELO CON VALIDACIÓN ESTRICTA
# ============================================

class ModeloPrediccion:
    
    def __init__(self, horizonte, ticker):
        self.horizonte = horizonte
        self.ticker = ticker
        self.modelo = None
        self.scaler = None
        self.features = None
        self.metricas_validacion = {}
        self.feature_importance = None
    
    def entrenar_walk_forward(self, df, features, etiqueta_col):
        """Walk-forward con gap temporal y validación estricta"""
        df_valido = df.dropna(subset=[etiqueta_col] + features).copy()
        
        if len(df_valido) < TradingConfig.MIN_MUESTRAS_ENTRENAMIENTO:
            print(f"    ⚠️ Datos insuficientes: {len(df_valido)}")
            return False
        
        X = df_valido[features]
        y = df_valido[etiqueta_col]
        
        # Verificar balance de clases
        class_counts = y.value_counts()
        if len(class_counts) < 2 or class_counts.min() < TradingConfig.MIN_MUESTRAS_CLASE:
            print(f"    ⚠️ Clases desbalanceadas: {class_counts.to_dict()}")
            return False
        
        # ✅ VALIDACIÓN: Gap = horizonte para evitar leakage
        gap = self.horizonte  # Gap temporal
        tscv = TimeSeriesSplit(n_splits=TradingConfig.N_FOLDS_WF, gap=gap)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Verificar que no hay overlap temporal
            if len(X_train) > 0 and len(X_val) > 0:
                fecha_ultima_train = X_train.index[-1]
                fecha_primera_val = X_val.index[0]
                # Debug: verificar gap
                # print(f"    Fold {fold}: Gap entre {fecha_ultima_train} y {fecha_primera_val}")
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Modelo más conservador
            modelo = RandomForestClassifier(
                n_estimators=100,      # Más árboles
                max_depth=4,           # Menos profundidad (evitar overfit)
                min_samples_split=100, # Más samples por split
                min_samples_leaf=50,   # Más samples por hoja
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            modelo.fit(X_train_scaled, y_train)
            
            y_pred = modelo.predict(X_val_scaled)
            y_proba = modelo.predict_proba(X_val_scaled)
            
            # Métricas en validación
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            
            scores.append({'accuracy': acc, 'precision': prec, 'recall': rec})
        
        self.metricas_validacion = {
            'accuracy': np.mean([s['accuracy'] for s in scores]),
            'precision': np.mean([s['precision'] for s in scores]),
            'recall': np.mean([s['recall'] for s in scores]),
            'std_accuracy': np.std([s['accuracy'] for s in scores]),
            'n_folds': len(scores)
        }
        
        # ✅ CRITERIO REALISTA: accuracy > 52% (en crypto es difícil)
        if self.metricas_validacion['accuracy'] < 0.52:
            print(f"      ❌ Accuracy muy baja: {self.metricas_validacion['accuracy']:.2%}")
            return False
        
        print(f"      ✅ Acc: {self.metricas_validacion['accuracy']:.2%} "
              f"(±{self.metricas_validacion['std_accuracy']:.2%}), "
              f"Prec: {self.metricas_validacion['precision']:.2%}, "
              f"Rec: {self.metricas_validacion['recall']:.2%}")
        
        # Entrenar modelo final con todos los datos
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.modelo = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_split=100,
            min_samples_leaf=50,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.modelo.fit(X_scaled, y)
        self.features = features
        
        # Guardar feature importance
        self.feature_importance = dict(zip(features, self.modelo.feature_importances_))
        
        return True
    
    def predecir(self, df_actual):
        """
        Predicción en tiempo real
        - probabilidad_positiva = P(LONG)
        - probabilidad_negativa = P(SHORT)
        """
        if self.modelo is None:
            return None
        
        if not all(f in df_actual.columns for f in self.features):
            return None
        
        X = df_actual[self.features].iloc[[-1]]
        
        if X.isnull().any().any():
            return None
        
        X_scaled = self.scaler.transform(X)
        
        prediccion_clase = self.modelo.predict(X_scaled)[0]
        probabilidades = self.modelo.predict_proba(X_scaled)[0]
        
        return {
            'prediccion': int(prediccion_clase),  # 1=LONG, 0=SHORT
            'probabilidad_positiva': probabilidades[1],  # P(LONG)
            'probabilidad_negativa': probabilidades[0],  # P(SHORT)
            'confianza': max(probabilidades)
        }
    
    def guardar(self, path):
        if self.modelo is None:
            return False
        
        modelo_data = {
            'modelo': self.modelo,
            'scaler': self.scaler,
            'features': self.features,
            'metricas': self.metricas_validacion,
            'feature_importance': self.feature_importance,
            'horizonte': self.horizonte,
            'ticker': self.ticker
        }
        
        joblib.dump(modelo_data, path)
        return True
    
    @classmethod
    def cargar(cls, path):
        modelo_data = joblib.load(path)
        
        instancia = cls(modelo_data['horizonte'], modelo_data['ticker'])
        instancia.modelo = modelo_data['modelo']
        instancia.scaler = modelo_data['scaler']
        instancia.features = modelo_data['features']
        instancia.metricas_validacion = modelo_data['metricas']
        instancia.feature_importance = modelo_data.get('feature_importance', {})
        
        return instancia


# ============================================
# BACKTESTING CORREGIDO
# ============================================

class Backtester:
    
    def __init__(self, df, modelos, ticker):
        self.df = df
        self.modelos = modelos
        self.ticker = ticker
        self.operaciones = []
    
    def simular_operacion(self, idx, señal_long, prob, rsi):
        """Simula operación con SL/TP realistas"""
        precio_entrada = self.df.loc[idx, 'Close']
        
        direccion = 'LONG' if señal_long else 'SHORT'
        
        # Niveles basados en % fijo
        if señal_long:
            stop_loss = precio_entrada * (1 - TradingConfig.STOP_LOSS_PCT)
            take_profit = precio_entrada * (1 + TradingConfig.TAKE_PROFIT_PCT)
        else:
            stop_loss = precio_entrada * (1 + TradingConfig.STOP_LOSS_PCT)
            take_profit = precio_entrada * (1 - TradingConfig.TAKE_PROFIT_PCT)
        
        riesgo = abs(precio_entrada - stop_loss)
        recompensa = abs(take_profit - precio_entrada)
        ratio_rr = recompensa / riesgo
        
        if ratio_rr < TradingConfig.RATIO_MINIMO_RR:
            return None
        
        # Simular hasta 24 horas (más tiempo para desarrollarse)
        idx_pos = self.df.index.get_loc(idx)
        max_ventana = min(24, len(self.df) - idx_pos - 1)
        
        if max_ventana < 4:
            return None
        
        precios_futuros = self.df.iloc[idx_pos:idx_pos + max_ventana + 1]['Close'].values
        highs_futuros = self.df.iloc[idx_pos:idx_pos + max_ventana + 1]['High'].values
        lows_futuros = self.df.iloc[idx_pos:idx_pos + max_ventana + 1]['Low'].values
        
        resultado = 'TIEMPO'
        velas_hasta_cierre = max_ventana
        retorno = 0
        
        # Simular vela por vela con high/low
        for i in range(1, len(precios_futuros)):
            high = highs_futuros[i]
            low = lows_futuros[i]
            
            if señal_long:
                # Primero verificar SL (asumimos que low viene antes que high)
                if low <= stop_loss:
                    resultado = 'SL'
                    velas_hasta_cierre = i
                    retorno = -riesgo / precio_entrada
                    break
                # Luego verificar TP
                elif high >= take_profit:
                    resultado = 'TP'
                    velas_hasta_cierre = i
                    retorno = recompensa / precio_entrada
                    break
            else:  # SHORT
                # Primero verificar SL
                if high >= stop_loss:
                    resultado = 'SL'
                    velas_hasta_cierre = i
                    retorno = -riesgo / precio_entrada
                    break
                # Luego verificar TP
                elif low <= take_profit:
                    resultado = 'TP'
                    velas_hasta_cierre = i
                    retorno = recompensa / precio_entrada
                    break
        
        # Si no tocó ni TP ni SL, cerrar al final de la ventana
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
            'rsi': rsi,
            'resultado': resultado,
            'retorno': retorno,
            'velas_hasta_cierre': velas_hasta_cierre
        }
    
    def ejecutar(self, fecha_inicio):
        """
        ✅ LÓGICA SIMPLIFICADA:
        - Si prob_promedio > 0.5 → LONG
        - Si prob_promedio < 0.5 → SHORT
        - Solo filtrar extremos de RSI (< 15 o > 85)
        """
        df_backtest = self.df[self.df.index >= fecha_inicio].copy()
        
        if len(df_backtest) < 100:
            print(f"  ⚠️ Datos insuficientes para backtest")
            return None
        
        print(f"  📊 Periodo: {df_backtest.index[0].date()} a {df_backtest.index[-1].date()}")
        
        for idx in df_backtest.index[:-24]:  # Dejar 24h al final
            predicciones = {}
            
            # Obtener predicciones de todos los horizontes
            for horizonte, modelo in self.modelos.items():
                pred = modelo.predecir(df_backtest.loc[:idx])
                if pred:
                    predicciones[horizonte] = pred
            
            if not predicciones:
                continue
            
            # Promediar probabilidades
            probs_positivas = [p['probabilidad_positiva'] for p in predicciones.values()]
            prob_promedio = np.mean(probs_positivas)
            confianza_promedio = np.mean([p['confianza'] for p in predicciones.values()])
            
            # ✅ FILTRO DE CONFIANZA (umbral más bajo)
            if confianza_promedio < TradingConfig.UMBRAL_CONFIANZA_MIN:
                continue
            
            # ✅ DECIDIR SEÑAL
            if prob_promedio > 0.5:
                señal_long = True
                prob_real = prob_promedio
            else:
                señal_long = False
                prob_real = 1 - prob_promedio
            
            # Verificar que la probabilidad supera el umbral
            if prob_real < TradingConfig.UMBRAL_PROBABILIDAD_MIN:
                continue
            
            # Obtener RSI actual
            rsi = df_backtest.loc[idx, 'RSI']
            if pd.isna(rsi):
                rsi = 50
            
            # ✅ SOLO FILTRAR EXTREMOS (evitar crashes)
            if señal_long and rsi > TradingConfig.RSI_EXTREME_HIGH:
                continue  # No comprar si RSI > 85 (posible crash)
            
            if not señal_long and rsi < TradingConfig.RSI_EXTREME_LOW:
                continue  # No vender si RSI < 15 (posible rebote)
            
            # Simular operación
            operacion = self.simular_operacion(
                idx,
                señal_long,
                prob_real,
                rsi
            )
            
            if operacion:
                self.operaciones.append(operacion)
        
        if not self.operaciones:
            print(f"  ⚠️ No se generaron operaciones en backtest")
            return None
        
        return self.calcular_metricas()
    
    def calcular_metricas(self):
        """Calcula métricas de rendimiento"""
        df_ops = pd.DataFrame(self.operaciones)
        
        n_ops = len(df_ops)
        n_tp = (df_ops['resultado'] == 'TP').sum()
        n_sl = (df_ops['resultado'] == 'SL').sum()
        n_timeout = (df_ops['resultado'] == 'TIEMPO').sum()
        
        retornos = df_ops['retorno']
        operaciones_ganadoras = retornos > 0
        operaciones_perdedoras = retornos < 0
        
        # Calcular profit factor
        ganancias = retornos[retornos > 0].sum()
        perdidas = abs(retornos[retornos < 0].sum())
        profit_factor = ganancias / perdidas if perdidas > 0 else np.inf
        
        metricas = {
            'n_operaciones': n_ops,
            'tasa_exito': operaciones_ganadoras.sum() / n_ops,
            'hit_tp_rate': n_tp / n_ops,
            'hit_sl_rate': n_sl / n_ops,
            'timeout_rate': n_timeout / n_ops,
            'retorno_total': retornos.sum(),
            'retorno_promedio': retornos.mean(),
            'retorno_mediano': retornos.median(),
            'mejor_operacion': retornos.max(),
            'peor_operacion': retornos.min(),
            'promedio_ganador': retornos[operaciones_ganadoras].mean() if operaciones_ganadoras.any() else 0,
            'promedio_perdedor': retornos[operaciones_perdedoras].mean() if operaciones_perdedoras.any() else 0,
            'profit_factor': profit_factor,
            'max_drawdown': self._calcular_max_drawdown(retornos),
            'sharpe_ratio': retornos.mean() / retornos.std() * np.sqrt(365*24) if retornos.std() > 0 else 0,
            'duracion_promedio': df_ops['velas_hasta_cierre'].mean(),
        }
        
        return metricas, df_ops
    
    def _calcular_max_drawdown(self, retornos):
        """Calcula máximo drawdown"""
        equity_curve = (1 + retornos).cumprod()
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown.min()


# ============================================
# SISTEMA COMPLETO CORREGIDO
# ============================================

class SistemaTradingTicker:
    
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
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            
            self.df_historico = df
            print(f"  ✅ Descargado: {len(df)} velas desde {df.index[0].date()}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error descargando datos: {e}")
            return False
    
    def entrenar_modelos(self):
        """Entrena modelos para cada horizonte"""
        print(f"\n🎯 ENTRENANDO MODELOS - {self.ticker}")
        print("-" * 80)
        
        if self.df_historico is None:
            return False
        
        # Datos hasta inicio de backtest
        df_train = self.df_historico[self.df_historico.index < self.fechas['inicio_backtest']].copy()
        
        print(f"  📊 Datos entrenamiento: {len(df_train)} velas")
        print(f"  📅 Periodo: {df_train.index[0].date()} a {df_train.index[-1].date()}")
        
        modelos_entrenados = 0
        
        for horizonte in TradingConfig.HORIZONTES:
            print(f"\n  🔄 Horizonte {horizonte}h...")
            
            df_prep, features = EtiquetadoDatos.preparar_dataset_ml(df_train, horizonte)
            etiqueta_col = f'etiqueta_{horizonte}h'
            
            # Debug: mostrar distribución de etiquetas
            etiquetas = df_prep[etiqueta_col].dropna()
            if len(etiquetas) > 0:
                dist = etiquetas.value_counts()
                print(f"    Distribución: LONG={dist.get(1, 0)}, SHORT={dist.get(0, 0)}")
            
            modelo = ModeloPrediccion(horizonte, self.ticker)
            if modelo.entrenar_walk_forward(df_prep, features, etiqueta_col):
                self.modelos[horizonte] = modelo
                modelos_entrenados += 1
                
                # Mostrar top features
                if modelo.feature_importance:
                    top_features = sorted(modelo.feature_importance.items(), 
                                        key=lambda x: x[1], reverse=True)[:3]
                    print(f"      Top features: {', '.join([f'{k}:{v:.3f}' for k, v in top_features])}")
        
        print(f"\n  ✅ Modelos entrenados: {modelos_entrenados}/{len(TradingConfig.HORIZONTES)}")
        
        return modelos_entrenados > 0
    
    def ejecutar_backtest(self):
        """Ejecuta backtest con los modelos entrenados"""
        print(f"\n🔬 BACKTESTING - {self.ticker}")
        print("-" * 80)
        
        if not self.modelos:
            print("  ❌ No hay modelos disponibles")
            return False
        
        # Preparar datos completos con features
        df_completo, _ = EtiquetadoDatos.preparar_dataset_ml(
            self.df_historico,
            TradingConfig.HORIZONTES[0]
        )
        
        backtester = Backtester(df_completo, self.modelos, self.ticker)
        resultado = backtester.ejecutar(self.fechas['inicio_backtest'])
        
        if resultado is None:
            return False
        
        metricas, df_ops = resultado
        self.metricas_backtest = metricas
        
        # Mostrar resultados
        print(f"\n  📊 RESULTADOS DEL BACKTEST:")
        print(f"    Operaciones totales: {metricas['n_operaciones']}")
        print(f"    Win rate: {metricas['tasa_exito']:.2%}")
        print(f"    TP alcanzado: {metricas['hit_tp_rate']:.2%}")
        print(f"    SL alcanzado: {metricas['hit_sl_rate']:.2%}")
        print(f"    Timeout: {metricas['timeout_rate']:.2%}")
        print(f"    Retorno total: {metricas['retorno_total']:.2%}")
        print(f"    Retorno promedio: {metricas['retorno_promedio']:.2%}")
        print(f"    Retorno mediano: {metricas['retorno_mediano']:.2%}")
        print(f"    Ganancia promedio: {metricas['promedio_ganador']:.2%}")
        print(f"    Pérdida promedio: {metricas['promedio_perdedor']:.2%}")
        print(f"    Profit Factor: {metricas['profit_factor']:.2f}")
        print(f"    Max Drawdown: {metricas['max_drawdown']:.2%}")
        print(f"    Sharpe Ratio: {metricas['sharpe_ratio']:.2f}")
        print(f"    Duración promedio: {metricas['duracion_promedio']:.1f} velas")
        
        # Distribución de resultados
        dist_long = df_ops[df_ops['direccion'] == 'LONG']['retorno'].mean()
        dist_short = df_ops[df_ops['direccion'] == 'SHORT']['retorno'].mean()
        print(f"\n    Retorno LONG: {dist_long:.2%}")
        print(f"    Retorno SHORT: {dist_short:.2%}")
        
        return True
    
    def es_viable(self):
        """Evalúa si el sistema es viable para este ticker"""
        if self.metricas_backtest is None:
            return False, 0
        
        m = self.metricas_backtest
        criterios = []
        
        # Criterios más realistas
        criterios.append(('Win rate > 48%', m['tasa_exito'] > 0.48))
        criterios.append(('Retorno total > 3%', m['retorno_total'] > 0.03))
        criterios.append(('Profit factor > 1.3', m['profit_factor'] > 1.3))
        criterios.append(('Max DD < 20%', abs(m['max_drawdown']) < 0.20))
        criterios.append(('Operaciones >= 10', m['n_operaciones'] >= 10))
        criterios.append(('Sharpe > 0.5', m['sharpe_ratio'] > 0.5))
        
        criterios_cumplidos = sum([c[1] for c in criterios])
        viable = criterios_cumplidos >= 4  # Al menos 4 de 6
        
        print(f"\n  📋 EVALUACIÓN DE CRITERIOS:")
        for nombre, resultado in criterios:
            print(f"    {'✅' if resultado else '❌'} {nombre}")
        
        return viable, criterios_cumplidos
    
    def analizar_tiempo_real(self):
        """Analiza condiciones actuales y genera señal si aplica"""
        if not self.modelos:
            return None
        
        try:
            # Descargar datos recientes
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
            
            # Calcular features (ya incluyen shift)
            df_reciente = IndicadoresTecnicos.calcular_features(df_reciente)
            
            # Obtener predicciones de todos los modelos
            predicciones = {}
            for horizonte, modelo in self.modelos.items():
                pred = modelo.predecir(df_reciente)
                if pred:
                    predicciones[horizonte] = pred
            
            if not predicciones:
                return None
            
            # Promediar probabilidades
            probs_positivas = [p['probabilidad_positiva'] for p in predicciones.values()]
            prob_promedio = np.mean(probs_positivas)
            confianza_promedio = np.mean([p['confianza'] for p in predicciones.values()])
            
            # Filtro de confianza
            if confianza_promedio < TradingConfig.UMBRAL_CONFIANZA_MIN:
                return None
            
            # Decidir señal
            if prob_promedio > 0.5:
                señal = "LONG"
                prob_real = prob_promedio
            else:
                señal = "SHORT"
                prob_real = 1 - prob_promedio
            
            # Verificar umbral de probabilidad
            if prob_real < TradingConfig.UMBRAL_PROBABILIDAD_MIN:
                return None
            
            # Obtener datos actuales
            ultima_vela = df_reciente.iloc[-1]
            precio = ultima_vela['Close']
            rsi = ultima_vela.get('RSI', 50)
            
            if pd.isna(rsi):
                rsi = 50
            
            # Filtrar extremos de RSI
            if señal == "LONG" and rsi > TradingConfig.RSI_EXTREME_HIGH:
                return None
            
            if señal == "SHORT" and rsi < TradingConfig.RSI_EXTREME_LOW:
                return None
            
            # Calcular niveles
            if señal == 'LONG':
                sl = precio * (1 - TradingConfig.STOP_LOSS_PCT)
                tp = precio * (1 + TradingConfig.TAKE_PROFIT_PCT)
            else:
                sl = precio * (1 + TradingConfig.STOP_LOSS_PCT)
                tp = precio * (1 - TradingConfig.TAKE_PROFIT_PCT)
            
            ratio_rr = abs(tp - precio) / abs(precio - sl)
            
            if ratio_rr < TradingConfig.RATIO_MINIMO_RR:
                return None
            
            # Estado del mercado
            estado_rsi = "NEUTRO"
            if rsi < 30:
                estado_rsi = "OVERSOLD"
            elif rsi > 70:
                estado_rsi = "OVERBOUGHT"
            
            tendencia = "ALCISTA" if ultima_vela.get('tendencia', 0) == 1 else "BAJISTA"
            
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
                'rsi': rsi,
                'estado_rsi': estado_rsi,
                'tendencia': tendencia,
            }
        
        except Exception as e:
            print(f"  ❌ Error en análisis: {e}")
            import traceback
            traceback.print_exc()
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
        
        # Guardar métricas
        if self.metricas_backtest:
            path_metricas = path_ticker / "metricas_backtest.json"
            with open(path_metricas, 'w') as f:
                json.dump(self.metricas_backtest, f, indent=2)
        
        print(f"  💾 Modelos guardados en {path_ticker}")
        return True


# ============================================
# UTILIDADES
# ============================================

def cargar_ultima_senal():
    """Carga la última señal enviada"""
    if os.path.exists("ultima_senal.json"):
        with open("ultima_senal.json") as f:
            return json.load(f)
    return None

def guardar_ultima_senal(senal):
    """Guarda la señal enviada"""
    with open("ultima_senal.json", "w") as f:
        json.dump(senal, f, indent=2)

def enviar_telegram(mensaje):
    """Envía mensaje por Telegram"""
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        print("  ⚠️ Telegram no configurado (variables TELEGRAM_TOKEN y TELEGRAM_CHAT_ID)")
        return
    
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, data={"chat_id": chat_id, "text": mensaje}, timeout=10)
        
        if r.status_code == 200:
            print(f"  📨 Mensaje enviado a Telegram")
        else:
            print(f"  ⚠️ Error Telegram: {r.status_code}")
    except Exception as e:
        print(f"  ⚠️ Error enviando Telegram: {e}")


# ============================================
# MAIN
# ============================================

def main():
    """Sistema principal de trading"""
    print("🚀 SISTEMA DE TRADING CORREGIDO")
    print("=" * 80)
    print("✅ Sin look-ahead bias (todas las features con shift)")
    print("✅ Umbrales realistas (55% probabilidad, 53% confianza)")
    print("✅ SL/TP ajustados (2.5% / 4.0%)")
    print("✅ Solo filtros extremos de RSI (<15 o >85)")
    print("✅ Horizontes cortos (1-2 horas)")
    print("=" * 80)
    
    fechas = TradingConfig.get_fechas()
    print(f"\n📅 Configuración:")
    print(f"  Fecha actual: {fechas['actual'].strftime('%Y-%m-%d %H:%M')}")
    print(f"  Periodo backtest: {fechas['inicio_backtest'].date()}")
    print(f"  SL: {TradingConfig.STOP_LOSS_PCT:.1%}, TP: {TradingConfig.TAKE_PROFIT_PCT:.1%}")
    print(f"  Umbral movimiento: {TradingConfig.UMBRAL_MOVIMIENTO:.1%}")
    
    resultados_globales = {}
    
    for ticker in TradingConfig.ACTIVOS:
        sistema = SistemaTradingTicker(ticker)
        
        # 1. Descargar datos
        if not sistema.descargar_datos():
            continue
        
        # 2. Entrenar modelos
        if not sistema.entrenar_modelos():
            continue
        
        # 3. Backtest
        if not sistema.ejecutar_backtest():
            continue
        
        # 4. Evaluar viabilidad
        viable, criterios = sistema.es_viable()
        
        print(f"\n{'='*80}")
        print(f"📊 EVALUACIÓN FINAL - {ticker}")
        print(f"{'='*80}")
        print(f"  Criterios cumplidos: {criterios}/6")
        print(f"  Sistema viable: {'✅ SÍ' if viable else '❌ NO'}")
        
        señal_actual = None
        
        # 5. Si es viable, analizar tiempo real
        if viable:
            print(f"\n  🔍 Analizando condiciones actuales...")
            try:
                señal_actual = sistema.analizar_tiempo_real()
                
                if señal_actual:
                    print(f"\n  🚨 SEÑAL DETECTADA: {señal_actual['señal']}")
                    print(f"    Probabilidad: {señal_actual['probabilidad']:.2%}")
                    print(f"    Confianza: {señal_actual['confianza']:.2%}")
                    print(f"    RSI: {señal_actual['rsi']:.0f} ({señal_actual['estado_rsi']})")
                    print(f"    Tendencia: {señal_actual['tendencia']}")
                    print(f"    Precio: ${señal_actual['precio']:,.2f}")
                    print(f"    Stop Loss: ${señal_actual['stop_loss']:,.2f}")
                    print(f"    Take Profit: ${señal_actual['take_profit']:,.2f}")
                    print(f"    R:R = {señal_actual['ratio_rr']:.2f}")
                    
                    # Verificar si es señal nueva
                    ultima = cargar_ultima_senal()
                    es_nueva = True
                    
                    if ultima:
                        if (ultima.get("ticker") == ticker and 
                            ultima.get("señal") == señal_actual["señal"]):
                            # Verificar si pasaron al menos 4 horas
                            try:
                                fecha_ultima = datetime.fromisoformat(ultima["fecha"])
                                if datetime.now(TradingConfig.TIMEZONE) - fecha_ultima < timedelta(hours=4):
                                    es_nueva = False
                                    print("  🔁 Señal repetida (< 4 horas)")
                            except:
                                pass
                    
                    if es_nueva:
                        # Enviar por Telegram
                        mensaje = (
                            f"🚨 {ticker} - {señal_actual['señal']}\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📊 Probabilidad: {señal_actual['probabilidad']:.1%}\n"
                            f"📊 Confianza: {señal_actual['confianza']:.1%}\n\n"
                            f"💰 Precio: ${señal_actual['precio']:,.2f}\n"
                            f"🛑 Stop Loss: ${señal_actual['stop_loss']:,.2f} "
                            f"({-TradingConfig.STOP_LOSS_PCT:.1%})\n"
                            f"🎯 Take Profit: ${señal_actual['take_profit']:,.2f} "
                            f"({TradingConfig.TAKE_PROFIT_PCT:.1%})\n"
                            f"⚖️ Ratio R:R: {señal_actual['ratio_rr']:.2f}\n\n"
                            f"📈 RSI: {señal_actual['rsi']:.0f} ({señal_actual['estado_rsi']})\n"
                            f"📊 Tendencia: {señal_actual['tendencia']}\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"⏰ {señal_actual['fecha'].strftime('%Y-%m-%d %H:%M')}"
                        )
                        
                        enviar_telegram(mensaje)
                        
                        # Guardar señal
                        guardar_ultima_senal({
                            "ticker": ticker,
                            "señal": señal_actual["señal"],
                            "fecha": señal_actual["fecha"].isoformat(),
                            "precio": señal_actual["precio"]
                        })
                else:
                    print("  ℹ️ No hay señal en este momento")
            
            except Exception as e:
                print(f"  ❌ Error en análisis: {e}")
                import traceback
                traceback.print_exc()
        
        # 6. Guardar modelos si es viable
        if viable:
            sistema.guardar_modelos()
        
        # Guardar resultados
        resultados_globales[ticker] = {
            'viable': viable,
            'criterios': criterios,
            'metricas': sistema.metricas_backtest,
            'señal_actual': señal_actual
        }
    
    # Resumen final
    print(f"\n{'='*80}")
    print("📊 RESUMEN FINAL")
    print(f"{'='*80}")
    
    viables = [t for t, r in resultados_globales.items() if r['viable']]
    con_senal = [t for t, r in resultados_globales.items() if r.get('señal_actual')]
    
    print(f"\n  Activos procesados: {len(resultados_globales)}")
    print(f"  Sistemas viables: {len(viables)}")
    print(f"  Señales activas: {len(con_senal)}")
    
    if viables:
        print(f"\n  ✅ SISTEMAS VIABLES:")
        for ticker in viables:
            r = resultados_globales[ticker]
            m = r['metricas']
            print(f"\n    {ticker}:")
            print(f"      Retorno: {m['retorno_total']:.2%}")
            print(f"      Win rate: {m['tasa_exito']:.1%}")
            print(f"      Profit Factor: {m['profit_factor']:.2f}")
            print(f"      Operaciones: {m['n_operaciones']}")
    
    if con_senal:
        print(f"\n  🚨 SEÑALES ACTIVAS:")
        for ticker in con_senal:
            s = resultados_globales[ticker]['señal_actual']
            print(f"    {ticker}: {s['señal']} @ ${s['precio']:,.2f} (Prob: {s['probabilidad']:.1%})")
    
    print(f"\n{'='*80}")
    print("✅ Proceso completado")
    print(f"{'='*80}\n")
    
    return resultados_globales


if __name__ == "__main__":
    main()
