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
import time
from typing import List, Optional

warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN CON LÍMITES REALISTAS
# ============================================

class TradingConfig:
    """Configuración adaptada a límites de Yahoo Finance"""
    
    TIMEZONE = pytz.timezone('America/Bogota')
    
    # Yahoo Finance limita datos 1h a últimos 730 días
    INTERVALO = "1h"
    
    # Ajustado para límites de Yahoo
    DIAS_ENTRENAMIENTO = 365    # 1 año
    DIAS_VALIDACION = 60        # 2 meses
    DIAS_BACKTEST = 30          # 1 mes
    
    # Total máximo = 455 días (dentro de 730 días)
    
    # Activos (usar símbolos correctos)
    ACTIVOS = [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"
    ]
    
    # Features
    VENTANA_VOLATILIDAD = 24
    RSI_PERIODO = 14
    ATR_PERIODO = 14
    
    # Horizontes cortos
    HORIZONTES = [1, 2, 3]
    
    # Gestión de riesgo
    STOP_LOSS_PCT = 0.015
    TAKE_PROFIT_PCT = 0.030
    RATIO_MINIMO_RR = 1.5
    MAX_RIESGO_POR_OPERACION = 0.02
    
    # Validación
    N_FOLDS_WF = 3
    MIN_MUESTRAS_ENTRENAMIENTO = 300
    MIN_MUESTRAS_CLASE = 20
    
    # Umbrales realistas
    UMBRAL_PROBABILIDAD_MIN = 0.52
    UMBRAL_CONFIANZA_MIN = 0.51
    UMBRAL_MOVIMIENTO = 0.008  # 0.8%
    
    # Filtros RSI
    RSI_EXTREME_LOW = 10
    RSI_EXTREME_HIGH = 90
    
    MODELOS_DIR = Path("modelos_trading")
    
    @classmethod
    def get_fechas(cls):
        now = datetime.now(cls.TIMEZONE)
        
        # Asegurar que no excedamos límites de Yahoo
        fecha_max_retroceso = now - timedelta(days=730)  # Límite de Yahoo
        
        inicio_backtest = now - timedelta(days=cls.DIAS_BACKTEST)
        inicio_validacion = inicio_backtest - timedelta(days=cls.DIAS_VALIDACION)
        inicio_entrenamiento = inicio_validacion - timedelta(days=cls.DIAS_ENTRENAMIENTO)
        
        # Asegurar que no vamos más allá del límite
        if inicio_entrenamiento < fecha_max_retroceso:
            print(f"  ⚠️ Ajustando fechas por límite de Yahoo Finance")
            inicio_entrenamiento = fecha_max_retroceso
            inicio_validacion = inicio_entrenamiento + timedelta(days=cls.DIAS_ENTRENAMIENTO - cls.DIAS_VALIDACION)
        
        return {
            'actual': now,
            'inicio_entrenamiento': inicio_entrenamiento,
            'inicio_validacion': inicio_validacion,
            'inicio_backtest': inicio_backtest,
            'fecha_max_retroceso': fecha_max_retroceso
        }


# ============================================
# DESCARGA INTELIGENTE POR PARTES
# ============================================

class YahooDataDownloader:
    """Descarga datos de Yahoo Finance respetando límites"""
    
    @staticmethod
    def descargar_por_partes(ticker: str, fecha_inicio: datetime, fecha_fin: datetime, 
                            intervalo: str = "1h", max_dias_por_chunk: int = 180) -> pd.DataFrame:
        """
        Descarga datos por partes para evitar límites de Yahoo Finance
        """
        print(f"  📥 Descargando {ticker} ({intervalo})...")
        print(f"    Periodo: {fecha_inicio.date()} a {fecha_fin.date()}")
        
        # Calcular número de chunks necesarios
        dias_totales = (fecha_fin - fecha_inicio).days
        num_chunks = max(1, dias_totales // max_dias_por_chunk + 1)
        
        print(f"    Dividiendo en {num_chunks} chunks...")
        
        chunks = []
        chunk_size = dias_totales / num_chunks
        
        for i in range(num_chunks):
            chunk_start = fecha_inicio + timedelta(days=i * chunk_size)
            chunk_end = fecha_inicio + timedelta(days=(i + 1) * chunk_size)
            
            # Asegurar que no superemos fecha_fin
            if chunk_end > fecha_fin:
                chunk_end = fecha_fin
            
            # Pequeña pausa para evitar rate limiting
            if i > 0:
                time.sleep(0.5)
            
            try:
                print(f"    Chunk {i+1}/{num_chunks}: {chunk_start.date()} a {chunk_end.date()}")
                
                df_chunk = yf.download(
                    ticker,
                    start=chunk_start,
                    end=chunk_end,
                    interval=intervalo,
                    progress=False,
                    threads=False
                )
                
                if not df_chunk.empty:
                    chunks.append(df_chunk)
                    print(f"      ✅ {len(df_chunk)} velas descargadas")
                else:
                    print(f"      ⚠️ Chunk vacío")
                    
            except Exception as e:
                print(f"      ❌ Error en chunk {i+1}: {e}")
                continue
        
        # Combinar chunks
        if chunks:
            df_completo = pd.concat(chunks)
            df_completo = df_completo[~df_completo.index.duplicated(keep='first')]
            df_completo.sort_index(inplace=True)
            
            # Asegurar columnas
            if isinstance(df_completo.columns, pd.MultiIndex):
                df_completo.columns = df_completo.columns.get_level_values(0)
            
            columnas_necesarias = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in columnas_necesarias:
                if col not in df_completo.columns:
                    print(f"    ⚠️ Columna {col} no encontrada")
            
            df_completo = df_completo[columnas_necesarias].dropna()
            
            print(f"    📊 Total: {len(df_completo)} velas")
            return df_completo
        
        return pd.DataFrame()
    
    @staticmethod
    def descargar_con_reintentos(ticker: str, fecha_inicio: datetime, fecha_fin: datetime, 
                                intervalo: str = "1h", max_reintentos: int = 3) -> pd.DataFrame:
        """
        Intenta descargar con reintentos si falla
        """
        for intento in range(max_reintentos):
            try:
                print(f"  Intento {intento+1}/{max_reintentos}...")
                
                df = yf.download(
                    ticker,
                    start=fecha_inicio,
                    end=fecha_fin,
                    interval=intervalo,
                    progress=False,
                    threads=False
                )
                
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
                    return df
                
            except Exception as e:
                print(f"    Intento {intento+1} falló: {e}")
                
                if intento < max_reintentos - 1:
                    wait_time = 2 ** intento  # Backoff exponencial
                    print(f"    Esperando {wait_time} segundos...")
                    time.sleep(wait_time)
                else:
                    print(f"    Todos los intentos fallaron para {ticker}")
        
        return pd.DataFrame()


# ============================================
# INDICADORES TÉCNICOS (sin cambios desde versión anterior)
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
        df['retorno_24h'] = close.pct_change(24).shift(1)
        
        # ✅ VOLATILIDAD PASADA
        retornos = close.pct_change(1)
        df['volatilidad_24h'] = retornos.rolling(24).std().shift(1)
        
        # ✅ RSI PASADO (múltiples)
        for periodo in [7, 14, 21]:
            rsi_raw = IndicadoresTecnicos.calcular_rsi(close, periodo)
            df[f'RSI_{periodo}'] = rsi_raw.shift(1)
        
        # ✅ MEDIAS MÓVILES PASADAS
        for periodo in [12, 24, 50]:
            sma = close.rolling(periodo).mean().shift(1)
            df[f'SMA_{periodo}'] = sma
            df[f'dist_sma_{periodo}'] = (close.shift(1) - sma) / sma
        
        # ✅ EMA PASADAS
        for periodo in [12, 26]:
            ema = close.ewm(span=periodo, adjust=False).mean().shift(1)
            df[f'EMA_{periodo}'] = ema
        
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
        df['tendencia'] = (df['SMA_12'] > df['SMA_24']).astype(int)
        
        # ✅ MOMENTUM PASADO
        df['momentum_24h'] = (close / close.shift(24) - 1).shift(1)
        
        # ✅ BANDAS DE BOLLINGER
        df['SMA_20'] = close.rolling(20).mean().shift(1)
        df['std_20'] = close.rolling(20).std().shift(1)
        df['bb_upper'] = df['SMA_20'] + (df['std_20'] * 2)
        df['bb_lower'] = df['SMA_20'] - (df['std_20'] * 2)
        df['bb_position'] = (close.shift(1) - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df


# ============================================
# ETIQUETADO (sin cambios)
# ============================================

class EtiquetadoDatos:
    
    @staticmethod
    def crear_etiquetas_direccion(df, horizonte):
        """Etiquetado con umbral dinámico"""
        retorno_futuro = df['Close'].shift(-horizonte) / df['Close'] - 1
        
        # Umbral dinámico basado en volatilidad
        volatilidad = df['Close'].pct_change().rolling(24).std().fillna(0.01)
        umbral_dinamico = volatilidad * 1.5
        
        etiqueta = pd.Series(np.nan, index=df.index)
        umbral_usado = umbral_dinamico.clip(lower=0.006, upper=0.012)
        
        etiqueta[retorno_futuro > umbral_usado] = 1   # LONG
        etiqueta[retorno_futuro < -umbral_usado] = 0  # SHORT
        
        return etiqueta, retorno_futuro
    
    @staticmethod
    def preparar_dataset_ml(df, horizonte):
        """Prepara dataset con features sin look-ahead"""
        df = IndicadoresTecnicos.calcular_features(df)
        etiqueta, retorno_futuro = EtiquetadoDatos.crear_etiquetas_direccion(df, horizonte)
        
        df[f'etiqueta_{horizonte}h'] = etiqueta
        df[f'retorno_futuro_{horizonte}h'] = retorno_futuro
        
        # Features finales
        features_base = [
            'RSI_7', 'RSI_14', 'RSI_21',
            'volatilidad_24h',
            'dist_sma_12', 'dist_sma_24', 'dist_sma_50',
            'ATR_pct',
            'tendencia',
            'retorno_1h', 'retorno_4h', 'retorno_12h', 'retorno_24h',
            'volumen_rel',
            'rango_hl_pct',
            'momentum_24h',
            'bb_position'
        ]
        
        features_disponibles = [f for f in features_base if f in df.columns]
        
        return df, features_disponibles


# ============================================
# MODELO (optimizado)
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
        """Walk-forward simplificado"""
        df_valido = df.dropna(subset=[etiqueta_col] + features).copy()
        
        if len(df_valido) < TradingConfig.MIN_MUESTRAS_ENTRENAMIENTO:
            print(f"    ⚠️ Datos insuficientes: {len(df_valido)} (necesita {TradingConfig.MIN_MUESTRAS_ENTRENAMIENTO})")
            return False
        
        X = df_valido[features]
        y = df_valido[etiqueta_col]
        
        # Verificar balance de clases
        class_counts = y.value_counts()
        if len(class_counts) < 2 or class_counts.min() < TradingConfig.MIN_MUESTRAS_CLASE:
            print(f"    ⚠️ Clases desbalanceadas: LONG={class_counts.get(1, 0)}, SHORT={class_counts.get(0, 0)}")
            return False
        
        print(f"    📊 Muestras: {len(X)} | LONG: {class_counts.get(1, 0)} | SHORT: {class_counts.get(0, 0)}")
        
        # Cross-validation simple
        tscv = TimeSeriesSplit(n_splits=3, test_size=300, gap=self.horizonte)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Modelo simple
            modelo = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features=0.5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            modelo.fit(X_train_scaled, y_train)
            
            y_pred = modelo.predict(X_val_scaled)
            y_proba = modelo.predict_proba(X_val_scaled)
            
            # Métricas
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
        
        # Criterio realista
        if self.metricas_validacion['accuracy'] < 0.505:
            print(f"      ❌ Accuracy baja: {self.metricas_validacion['accuracy']:.2%}")
            return False
        
        print(f"      ✅ Acc: {self.metricas_validacion['accuracy']:.2%} | "
              f"Prec: {self.metricas_validacion['precision']:.2%} | "
              f"Rec: {self.metricas_validacion['recall']:.2%}")
        
        # Entrenar modelo final
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.modelo = RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            min_samples_split=15,
            min_samples_leaf=7,
            max_features=0.5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.modelo.fit(X_scaled, y)
        self.features = features
        
        # Feature importance
        self.feature_importance = dict(zip(features, self.modelo.feature_importances_))
        
        # Top features
        if self.feature_importance:
            top_features = sorted(self.feature_importance.items(), 
                                key=lambda x: x[1], reverse=True)[:5]
            print(f"      🏆 Top features:")
            for feat, imp in top_features:
                print(f"        {feat}: {imp:.3f}")
        
        return True
    
    def predecir(self, df_actual):
        """Predicción en tiempo real"""
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
            'prediccion': int(prediccion_clase),
            'probabilidad_positiva': probabilidades[1],
            'probabilidad_negativa': probabilidades[0],
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
# BACKTESTING (optimizado)
# ============================================

class Backtester:
    
    def __init__(self, df, modelos, ticker):
        self.df = df
        self.modelos = modelos
        self.ticker = ticker
        self.operaciones = []
    
    def simular_operacion(self, idx, señal_long, prob, rsi):
        """Simula operación con SL/TP"""
        precio_entrada = self.df.loc[idx, 'Close']
        
        direccion = 'LONG' if señal_long else 'SHORT'
        
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
        
        # Simular hasta 24 horas
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
        
        for i in range(1, len(precios_futuros)):
            high = highs_futuros[i]
            low = lows_futuros[i]
            
            if señal_long:
                if low <= stop_loss:
                    resultado = 'SL'
                    velas_hasta_cierre = i
                    retorno = -TradingConfig.STOP_LOSS_PCT
                    break
                elif high >= take_profit:
                    resultado = 'TP'
                    velas_hasta_cierre = i
                    retorno = TradingConfig.TAKE_PROFIT_PCT
                    break
            else:
                if high >= stop_loss:
                    resultado = 'SL'
                    velas_hasta_cierre = i
                    retorno = -TradingConfig.STOP_LOSS_PCT
                    break
                elif low <= take_profit:
                    resultado = 'TP'
                    velas_hasta_cierre = i
                    retorno = TradingConfig.TAKE_PROFIT_PCT
                    break
        
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
        """Ejecuta backtest"""
        df_backtest = self.df[self.df.index >= fecha_inicio].copy()
        
        if len(df_backtest) < 100:
            print(f"  ⚠️ Datos insuficientes para backtest")
            return None
        
        print(f"  📊 Periodo backtest: {df_backtest.index[0].date()} a {df_backtest.index[-1].date()}")
        print(f"  📈 Velas disponibles: {len(df_backtest)}")
        
        for idx in df_backtest.index[:-24]:
            predicciones = {}
            
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
            
            if confianza_promedio < TradingConfig.UMBRAL_CONFIANZA_MIN:
                continue
            
            if prob_promedio > 0.5:
                señal_long = True
                prob_real = prob_promedio
            else:
                señal_long = False
                prob_real = 1 - prob_promedio
            
            if prob_real < TradingConfig.UMBRAL_PROBABILIDAD_MIN:
                continue
            
            rsi = df_backtest.loc[idx, 'RSI_14'] if 'RSI_14' in df_backtest.columns else 50
            
            # Filtros RSI
            if señal_long and rsi > TradingConfig.RSI_EXTREME_HIGH:
                continue
            
            if not señal_long and rsi < TradingConfig.RSI_EXTREME_LOW:
                continue
            
            operacion = self.simular_operacion(idx, señal_long, prob_real, rsi)
            
            if operacion:
                self.operaciones.append(operacion)
        
        if not self.operaciones:
            print(f"  ⚠️ No se generaron operaciones")
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
        
        # Profit factor
        ganancias = retornos[retornos > 0].sum()
        perdidas = abs(retornos[retornos < 0].sum())
        profit_factor = ganancias / perdidas if perdidas > 0 else np.inf
        
        # Equity curve
        equity_curve = (1 + retornos).cumprod()
        
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
            'promedio_ganador': retornos[retornos > 0].mean() if retornos[retornos > 0].any() else 0,
            'promedio_perdedor': retornos[retornos < 0].mean() if retornos[retornos < 0].any() else 0,
            'profit_factor': profit_factor,
            'max_drawdown': (equity_curve / equity_curve.cummax() - 1).min(),
            'sharpe_ratio': retornos.mean() / retornos.std() * np.sqrt(365*24) if retornos.std() > 0 else 0,
            'equity_final': equity_curve.iloc[-1] if len(equity_curve) > 0 else 1,
            'duracion_promedio': df_ops['velas_hasta_cierre'].mean(),
        }
        
        return metricas, df_ops


# ============================================
# SISTEMA COMPLETO
# ============================================

class SistemaTradingTicker:
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.modelos = {}
        self.fechas = TradingConfig.get_fechas()
        self.df_historico = None
        self.metricas_backtest = None
    
    def descargar_datos(self):
        """Descarga datos históricos respetando límites"""
        print(f"\n{'='*80}")
        print(f"📥 DESCARGANDO {self.ticker}")
        print(f"{'='*80}")
        
        try:
            # Intentar descarga directa primero
            df = YahooDataDownloader.descargar_con_reintentos(
                self.ticker,
                self.fechas['inicio_entrenamiento'],
                self.fechas['actual'],
                TradingConfig.INTERVALO
            )
            
            if df.empty:
                print(f"  ⚠️ Descarga directa falló, intentando por partes...")
                df = YahooDataDownloader.descargar_por_partes(
                    self.ticker,
                    self.fechas['inicio_entrenamiento'],
                    self.fechas['actual'],
                    TradingConfig.INTERVALO
                )
            
            if df.empty:
                print(f"  ❌ No se pudieron descargar datos")
                return False
            
            self.df_historico = df
            print(f"  ✅ Descargado: {len(df)} velas")
            print(f"     Periodo: {df.index[0].date()} a {df.index[-1].date()}")
            print(f"     Precio actual: ${df['Close'].iloc[-1]:,.2f}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error crítico descargando datos: {e}")
            return False
    
    def entrenar_modelos(self):
        """Entrena modelos para cada horizonte"""
        print(f"\n🎯 ENTRENANDO MODELOS - {self.ticker}")
        print("-" * 80)
        
        if self.df_historico is None:
            return False
        
        # Usar datos hasta inicio de backtest
        df_train = self.df_historico[self.df_historico.index < self.fechas['inicio_backtest']].copy()
        
        print(f"  📊 Datos entrenamiento: {len(df_train)} velas")
        print(f"  📅 Periodo: {df_train.index[0].date()} a {df_train.index[-1].date()}")
        
        modelos_entrenados = 0
        
        for horizonte in TradingConfig.HORIZONTES[:2]:  # Solo 1h y 2h para empezar
            print(f"\n  🔄 Horizonte {horizonte}h...")
            
            try:
                df_prep, features = EtiquetadoDatos.preparar_dataset_ml(df_train, horizonte)
                etiqueta_col = f'etiqueta_{horizonte}h'
                
                # Verificar distribución
                etiquetas = df_prep[etiqueta_col].dropna()
                if len(etiquetas) == 0:
                    print(f"    ⚠️ No hay etiquetas válidas")
                    continue
                
                dist = etiquetas.value_counts()
                print(f"    Distribución: LONG={dist.get(1, 0)}, SHORT={dist.get(0, 0)}")
                
                # Seleccionar features importantes
                if len(features) > 10:
                    # Calcular correlación con etiquetas
                    correlations = []
                    for feat in features:
                        corr = df_prep[feat].corr(df_prep[etiqueta_col])
                        if not pd.isna(corr):
                            correlations.append((feat, abs(corr)))
                    
                    if correlations:
                        correlations.sort(key=lambda x: x[1], reverse=True)
                        features = [feat for feat, _ in correlations[:10]]
                
                modelo = ModeloPrediccion(horizonte, self.ticker)
                if modelo.entrenar_walk_forward(df_prep, features, etiqueta_col):
                    self.modelos[horizonte] = modelo
                    modelos_entrenados += 1
                    
            except Exception as e:
                print(f"    ❌ Error entrenando horizonte {horizonte}h: {e}")
                continue
        
        print(f"\n  ✅ Modelos entrenados: {modelos_entrenados}/{len(TradingConfig.HORIZONTES[:2])}")
        return modelos_entrenados > 0
    
    def ejecutar_backtest(self):
        """Ejecuta backtest con los modelos entrenados"""
        print(f"\n🔬 BACKTESTING - {self.ticker}")
        print("-" * 80)
        
        if not self.modelos:
            print("  ❌ No hay modelos disponibles")
            return False
        
        # Preparar datos completos
        df_completo, _ = EtiquetadoDatos.preparar_dataset_ml(
            self.df_historico,
            1
        )
        
        backtester = Backtester(df_completo, self.modelos, self.ticker)
        resultado = backtester.ejecutar(self.fechas['inicio_backtest'])
        
        if resultado is None:
            return False
        
        metricas, df_ops = resultado
        self.metricas_backtest = metricas
        
        # Mostrar resultados
        print(f"\n  📊 RESULTADOS:")
        print(f"    Operaciones: {metricas['n_operaciones']}")
        print(f"    Win rate: {metricas['tasa_exito']:.2%}")
        print(f"    Retorno total: {metricas['retorno_total']:.2%}")
        print(f"    Profit Factor: {metricas['profit_factor']:.2f}")
        print(f"    Sharpe Ratio: {metricas['sharpe_ratio']:.2f}")
        print(f"    Max Drawdown: {metricas['max_drawdown']:.2%}")
        
        if len(df_ops) > 0:
            long_ops = df_ops[df_ops['direccion'] == 'LONG']
            short_ops = df_ops[df_ops['direccion'] == 'SHORT']
            
            if len(long_ops) > 0:
                win_rate_long = (long_ops['retorno'] > 0).sum() / len(long_ops)
                print(f"    LONG: {len(long_ops)} ops, Win rate: {win_rate_long:.1%}")
            
            if len(short_ops) > 0:
                win_rate_short = (short_ops['retorno'] > 0).sum() / len(short_ops)
                print(f"    SHORT: {len(short_ops)} ops, Win rate: {win_rate_short:.1%}")
        
        return True
    
    def es_viable(self):
        """Evalúa si el sistema es viable"""
        if self.metricas_backtest is None:
            return False, 0
        
        m = self.metricas_backtest
        
        criterios = []
        criterios.append(('Operaciones >= 5', m['n_operaciones'] >= 5))
        criterios.append(('Win rate > 45%', m['tasa_exito'] > 0.45))
        criterios.append(('Profit factor > 1.2', m['profit_factor'] > 1.2))
        criterios.append(('Retorno total > 0%', m['retorno_total'] > 0))
        criterios.append(('Equity final > 1.0', m['equity_final'] > 1.0))
        criterios.append(('Max DD < 20%', abs(m['max_drawdown']) < 0.20))
        
        criterios_cumplidos = sum([c[1] for c in criterios])
        viable = criterios_cumplidos >= 4
        
        print(f"\n  📋 EVALUACIÓN:")
        for nombre, resultado in criterios:
            print(f"    {'✅' if resultado else '❌'} {nombre}")
        
        return viable, criterios_cumplidos
    
    def analizar_tiempo_real(self):
        """Analiza condiciones actuales"""
        if not self.modelos:
            return None
        
        try:
            # Descargar últimos 7 días
            fecha_inicio = datetime.now(TradingConfig.TIMEZONE) - timedelta(days=7)
            
            df_reciente = YahooDataDownloader.descargar_con_reintentos(
                self.ticker,
                fecha_inicio,
                datetime.now(TradingConfig.TIMEZONE),
                TradingConfig.INTERVALO
            )
            
            if df_reciente.empty:
                return None
            
            # Calcular features
            df_reciente = IndicadoresTecnicos.calcular_features(df_reciente)
            
            # Obtener predicciones
            predicciones = {}
            for horizonte, modelo in self.modelos.items():
                pred = modelo.predecir(df_reciente)
                if pred:
                    predicciones[horizonte] = pred
            
            if not predicciones:
                return None
            
            # Promediar
            probs_positivas = [p['probabilidad_positiva'] for p in predicciones.values()]
            prob_promedio = np.mean(probs_positivas)
            confianza_promedio = np.mean([p['confianza'] for p in predicciones.values()])
            
            if confianza_promedio < TradingConfig.UMBRAL_CONFIANZA_MIN:
                return None
            
            if prob_promedio > 0.5:
                señal = "LONG"
                prob_real = prob_promedio
            else:
                señal = "SHORT"
                prob_real = 1 - prob_promedio
            
            if prob_real < TradingConfig.UMBRAL_PROBABILIDAD_MIN:
                return None
            
            # Obtener datos actuales
            ultima_vela = df_reciente.iloc[-1]
            precio = ultima_vela['Close']
            rsi = ultima_vela.get('RSI_14', 50)
            
            # Filtros
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
            
            # Fuerza de señal
            fuerza = "DÉBIL"
            if prob_real > 0.6:
                fuerza = "FUERTE"
            elif prob_real > 0.55:
                fuerza = "MEDIA"
            
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
                'fuerza': fuerza,
                'stop_loss': sl,
                'take_profit': tp,
                'ratio_rr': ratio_rr,
                'rsi': rsi,
                'estado_rsi': estado_rsi,
                'tendencia': tendencia,
                'n_modelos': len(predicciones)
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
        
        if self.metricas_backtest:
            path_metricas = path_ticker / "metricas_backtest.json"
            with open(path_metricas, 'w') as f:
                json.dump(self.metricas_backtest, f, indent=2)
        
        print(f"  💾 Modelos guardados en {path_ticker}")
        return True


# ============================================
# UTILIDADES (sin cambios)
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
        print("  ⚠️ Telegram no configurado")
        return
    
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, data={"chat_id": chat_id, "text": mensaje}, timeout=10)
        
        if r.status_code == 200:
            print(f"  📨 Telegram enviado")
        else:
            print(f"  ⚠️ Error Telegram: {r.status_code}")
    except Exception as e:
        print(f"  ⚠️ Error enviando Telegram: {e}")


# ============================================
# MAIN
# ============================================

def main():
    """Sistema principal corregido para límites de Yahoo"""
    print("🚀 SISTEMA DE TRADING - VERSIÓN CORREGIDA")
    print("=" * 80)
    print("✅ Respeta límites de Yahoo Finance (730 días para 1h)")
    print("✅ Descarga inteligente por partes")
    print("✅ Configuración realista")
    print("=" * 80)
    
    fechas = TradingConfig.get_fechas()
    print(f"\n📅 Configuración:")
    print(f"  Fecha actual: {fechas['actual'].strftime('%Y-%m-%d %H:%M')}")
    print(f"  Inicio entrenamiento: {fechas['inicio_entrenamiento'].strftime('%Y-%m-%d')}")
    print(f"  Inicio backtest: {fechas['inicio_backtest'].strftime('%Y-%m-%d')}")
    print(f"  Límite Yahoo: {fechas['fecha_max_retroceso'].strftime('%Y-%m-%d')}")
    print(f"  Días totales: {(fechas['actual'] - fechas['inicio_entrenamiento']).days}")
    
    # Crear directorio para modelos
    TradingConfig.MODELOS_DIR.mkdir(exist_ok=True)
    
    resultados_globales = {}
    
    for ticker in TradingConfig.ACTIVOS:
        sistema = SistemaTradingTicker(ticker)
        
        # 1. Descargar datos
        print(f"\n{'='*80}")
        print(f"📊 PROCESANDO {ticker}")
        print(f"{'='*80}")
        
        if not sistema.descargar_datos():
            print(f"  ⏭️ Saltando {ticker}...")
            continue
        
        # 2. Entrenar modelos
        if not sistema.entrenar_modelos():
            print(f"  ⚠️ No se pudieron entrenar modelos para {ticker}")
            continue
        
        # 3. Backtest
        if not sistema.ejecutar_backtest():
            print(f"  ⚠️ Backtest falló para {ticker}")
            continue
        
        # 4. Evaluar viabilidad
        viable, criterios = sistema.es_viable()
        
        print(f"\n  {'✅ VIABLE' if viable else '❌ NO VIABLE'}")
        print(f"  Criterios cumplidos: {criterios}/6")
        
        señal_actual = None
        
        # 5. Análisis tiempo real si es viable
        if viable:
            print(f"\n  🔍 Analizando condiciones actuales...")
            señal_actual = sistema.analizar_tiempo_real()
            
            if señal_actual:
                print(f"\n  🚨 SEÑAL DETECTADA: {señal_actual['señal']} ({señal_actual['fuerza']})")
                print(f"    Probabilidad: {señal_actual['probabilidad']:.2%}")
                print(f"    Precio: ${señal_actual['precio']:,.2f}")
                print(f"    RSI: {señal_actual['rsi']:.0f} ({señal_actual['estado_rsi']})")
                print(f"    R:R = {señal_actual['ratio_rr']:.2f}")
                
                # Verificar si es señal nueva
                ultima = cargar_ultima_senal()
                es_nueva = True
                
                if ultima:
                    if (ultima.get("ticker") == ticker and 
                        ultima.get("señal") == señal_actual["señal"]):
                        try:
                            fecha_ultima = datetime.fromisoformat(ultima["fecha"])
                            if datetime.now(TradingConfig.TIMEZONE) - fecha_ultima < timedelta(hours=6):
                                es_nueva = False
                                print("  🔁 Señal repetida (< 6 horas)")
                        except:
                            pass
                
                if es_nueva:
                    # Enviar Telegram
                    emoji = "📈" if señal_actual['señal'] == "LONG" else "📉"
                    mensaje = (
                        f"{emoji} {ticker} - {señal_actual['señal']} ({señal_actual['fuerza']})\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"📊 Probabilidad: {señal_actual['probabilidad']:.1%}\n"
                        f"💰 Precio: ${señal_actual['precio']:,.2f}\n"
                        f"🛑 Stop Loss: ${señal_actual['stop_loss']:,.2f}\n"
                        f"🎯 Take Profit: ${señal_actual['take_profit']:,.2f}\n"
                        f"⚖️ Ratio R:R: {señal_actual['ratio_rr']:.2f}\n"
                        f"📈 RSI: {señal_actual['rsi']:.0f} ({señal_actual['estado_rsi']})\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"⏰ {señal_actual['fecha'].strftime('%Y-%m-%d %H:%M')}"
                    )
                    
                    enviar_telegram(mensaje)
                    
                    # Guardar señal
                    guardar_ultima_senal({
                        "ticker": ticker,
                        "señal": señal_actual["señal"],
                        "fecha": señal_actual["fecha"].isoformat(),
                        "precio": señal_actual["precio"],
                        "probabilidad": señal_actual["probabilidad"]
                    })
            else:
                print("  ℹ️ No hay señal en este momento")
        
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
            print(f"      Operaciones: {m['n_operaciones']}")
            print(f"      Win rate: {m['tasa_exito']:.1%}")
            print(f"      Retorno: {m['retorno_total']:.2%}")
            print(f"      Profit Factor: {m['profit_factor']:.2f}")
    
    if con_senal:
        print(f"\n  🚨 SEÑALES ACTIVAS:")
        for ticker in con_senal:
            s = resultados_globales[ticker]['señal_actual']
            emoji = "📈" if s['señal'] == "LONG" else "📉"
            print(f"    {emoji} {ticker}: {s['señal']} @ ${s['precio']:,.2f}")
    
    print(f"\n{'='*80}")
    print("✅ Proceso completado")
    print(f"{'='*80}\n")
    
    return resultados_globales


if __name__ == "__main__":
    main()
