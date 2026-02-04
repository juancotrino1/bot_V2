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
# CONFIGURACIÓN OPTIMIZADA
# ============================================

class TradingConfig:
    """Configuración mejorada con parámetros más conservadores"""
    
    TIMEZONE = pytz.timezone('America/Bogota')
    
    # Datos
    INTERVALO = "1h"
    DIAS_ENTRENAMIENTO = 180
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
    
    # Un solo horizonte corto
    HORIZONTES = [4]  # 4 horas
    
    # 🎯 GESTIÓN DE RIESGO MEJORADA (% fijo, no ATR)
    STOP_LOSS_PCT = 0.015    # 1.5%
    TAKE_PROFIT_PCT = 0.030  # 3.0%
    RATIO_MINIMO_RR = 1.8
    MAX_RIESGO_POR_OPERACION = 0.01
    
    # Validación
    N_FOLDS_WF = 5
    MIN_MUESTRAS_ENTRENAMIENTO = 1000
    MIN_MUESTRAS_CLASE = 50
    
    # 🔥 FILTROS CRÍTICOS
    UMBRAL_PROBABILIDAD_MIN = 0.72
    UMBRAL_CONFIANZA_MIN = 0.70
    
    # 🎯 ESTRATEGIA: MOMENTUM (no mean reversion puro)
    # Solo operamos cuando el modelo predice Y la tendencia es clara
    RSI_OVERSOLD = 35      # RSI < 35 = sobreventa (apoyo para LONG)
    RSI_OVERBOUGHT = 65    # RSI > 65 = sobrecompra (apoyo para SHORT)
    
    # Mean Reversion como FILTRO DE CONFIRMACIÓN (no señal principal)
    Z_EXTREME_THRESHOLD = 2.5  # Solo bloquear en extremos muy fuertes
    
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
# INDICADORES SIMPLIFICADOS
# ============================================

class IndicadoresTecnicos:
    """Features mínimas y robustas"""
    
    @staticmethod
    def calcular_rsi(precios, periodo=14):
        delta = precios.diff()
        ganancia = delta.where(delta > 0, 0).rolling(window=periodo).mean()
        perdida = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
        perdida = perdida.replace(0, 1e-10)
        rs = ganancia / perdida
        return (100 - (100 / (1 + rs))).fillna(50)
    
    @staticmethod
    def calcular_atr(df, periodo=14):
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
        """Solo features probadas y sin look-ahead"""
        df = df.copy()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df.get('Volume', pd.Series(1, index=df.index))
        
        # 1. Retornos
        df['retorno_1h'] = close.pct_change(1)
        df['retorno_4h'] = close.pct_change(4)
        df['retorno_12h'] = close.pct_change(12)
        
        # 2. Volatilidad
        df['volatilidad_24h'] = df['retorno_1h'].rolling(24).std()
        
        # 3. RSI
        df['RSI'] = IndicadoresTecnicos.calcular_rsi(close, 14)
        
        # 4. Medias móviles
        df['SMA_12'] = close.rolling(12).mean()
        df['SMA_24'] = close.rolling(24).mean()
        df['dist_sma_12'] = (close - df['SMA_12']) / close
        
        # 5. ATR normalizado
        df['ATR'] = IndicadoresTecnicos.calcular_atr(df, 14)
        df['ATR_pct'] = df['ATR'] / close
        
        # 6. Volumen relativo
        df['volumen_rel'] = volume / volume.rolling(24).mean()
        
        # 7. Rango del precio
        df['rango_hl_pct'] = (high - low) / close
        
        # 8. Mean Reversion Z-score (para filtro de extremos)
        df['ret_log'] = np.log(close / close.shift(1))
        window = 72
        df['mu'] = df['ret_log'].rolling(window).mean()
        df['sigma'] = df['ret_log'].rolling(window).std()
        df['sigma'] = df['sigma'].replace(0, np.nan)
        df['z_score'] = (df['ret_log'] - df['mu']) / df['sigma']
        df['z_score'] = df['z_score'].fillna(0)
        
        # 9. Tendencia binaria
        df['tendencia'] = (df['SMA_12'] > df['SMA_24']).astype(int)
        
        return df


# ============================================
# ETIQUETADO MEJORADO
# ============================================

class EtiquetadoDatos:
    
    @staticmethod
    def crear_etiquetas_direccion(df, horizonte, umbral_movimiento=0.008):
        """
        ✅ VERIFICADO: 
        - retorno_futuro > 0.008 (0.8%) → etiqueta = 1 → LONG
        - retorno_futuro < -0.008 → etiqueta = 0 → SHORT
        """
        retorno_futuro = df['Close'].shift(-horizonte) / df['Close'] - 1
        
        etiqueta = pd.Series(np.nan, index=df.index)
        etiqueta[retorno_futuro > umbral_movimiento] = 1   # LONG
        etiqueta[retorno_futuro < -umbral_movimiento] = 0  # SHORT
        
        return etiqueta, retorno_futuro
    
    @staticmethod
    def preparar_dataset_ml(df, horizonte):
        df = IndicadoresTecnicos.calcular_features(df)
        etiqueta, retorno_futuro = EtiquetadoDatos.crear_etiquetas_direccion(df, horizonte)
        
        df[f'etiqueta_{horizonte}h'] = etiqueta
        df[f'retorno_futuro_{horizonte}h'] = retorno_futuro
        
        # Features REDUCIDAS
        features = [
            'RSI',
            'volatilidad_24h',
            'dist_sma_12',
            'ATR_pct',
            'tendencia',
            'retorno_1h',
            'retorno_4h',
            'retorno_12h',
            'z_score',
            'volumen_rel',
            'rango_hl_pct'
        ]
        
        features_disponibles = [f for f in features if f in df.columns]
        
        return df, features_disponibles


# ============================================
# MODELO SIMPLIFICADO
# ============================================

class ModeloPrediccion:
    
    def __init__(self, horizonte, ticker):
        self.horizonte = horizonte
        self.ticker = ticker
        self.modelo = None
        self.scaler = None
        self.features = None
        self.metricas_validacion = {}
    
    def entrenar_walk_forward(self, df, features, etiqueta_col):
        """Walk-forward con gap temporal"""
        df_valido = df.dropna(subset=[etiqueta_col] + features).copy()
        
        if len(df_valido) < TradingConfig.MIN_MUESTRAS_ENTRENAMIENTO:
            print(f"    ⚠️ Datos insuficientes: {len(df_valido)}")
            return False
        
        X = df_valido[features]
        y = df_valido[etiqueta_col]
        
        if y.sum() < TradingConfig.MIN_MUESTRAS_CLASE:
            print(f"    ⚠️ Clases desbalanceadas")
            return False
        
        # Purged K-Fold
        tscv = TimeSeriesSplit(n_splits=TradingConfig.N_FOLDS_WF, gap=24)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            modelo = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                min_samples_split=50,
                min_samples_leaf=20,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            modelo.fit(X_train_scaled, y_train)
            
            y_pred = modelo.predict(X_val_scaled)
            y_proba = modelo.predict_proba(X_val_scaled)[:, 1]
            
            # Solo contar predicciones de alta confianza
            alta_confianza = np.max(modelo.predict_proba(X_val_scaled), axis=1) > 0.65
            
            if alta_confianza.sum() > 0:
                acc = accuracy_score(y_val[alta_confianza], y_pred[alta_confianza])
                prec = precision_score(y_val[alta_confianza], y_pred[alta_confianza], zero_division=0)
                rec = recall_score(y_val[alta_confianza], y_pred[alta_confianza], zero_division=0)
            else:
                acc = prec = rec = 0
            
            scores.append({'accuracy': acc, 'precision': prec, 'recall': rec})
        
        self.metricas_validacion = {
            'accuracy': np.mean([s['accuracy'] for s in scores]),
            'precision': np.mean([s['precision'] for s in scores]),
            'recall': np.mean([s['recall'] for s in scores]),
            'n_folds': len(scores)
        }
        
        # Solo aceptar modelos con accuracy > 55%
        if self.metricas_validacion['accuracy'] < 0.55:
            print(f"      ❌ Accuracy muy baja: {self.metricas_validacion['accuracy']:.2%}")
            return False
        
        print(f"      ✅ Acc: {self.metricas_validacion['accuracy']:.2%}, "
              f"Prec: {self.metricas_validacion['precision']:.2%}, "
              f"Rec: {self.metricas_validacion['recall']:.2%}")
        
        # Entrenar modelo final
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.modelo = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            min_samples_split=50,
            min_samples_leaf=20,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.modelo.fit(X_scaled, y)
        self.features = features
        
        return True
    
    def predecir(self, df_actual):
        """
        ✅ VERIFICADO:
        - probabilidad_positiva = P(etiqueta=1) = P(LONG)
        - probabilidad_negativa = P(etiqueta=0) = P(SHORT)
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
    
    def simular_operacion(self, idx, señal_long, prob, rsi, z_score):
        """SL/TP basados en % FIJO"""
        precio_entrada = self.df.loc[idx, 'Close']
        
        direccion = 'LONG' if señal_long else 'SHORT'
        
        # Niveles fijos en %
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
        
        # Simular hasta 12 horas
        idx_pos = self.df.index.get_loc(idx)
        max_ventana = min(12, len(self.df) - idx_pos - 1)
        
        if max_ventana < 4:
            return None
        
        precios_futuros = self.df.iloc[idx_pos:idx_pos + max_ventana + 1]['Close'].values
        
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
            else:
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
            'z_score': z_score,
            'resultado': resultado,
            'retorno': retorno,
            'velas_hasta_cierre': velas_hasta_cierre
        }
    
    def ejecutar(self, fecha_inicio):
        """
        ✅ LÓGICA CORREGIDA:
        - prob_promedio > 0.5 → modelo predice LONG (etiqueta=1)
        - prob_promedio < 0.5 → modelo predice SHORT (etiqueta=0)
        """
        df_backtest = self.df[self.df.index >= fecha_inicio].copy()
        
        if len(df_backtest) < 100:
            print(f"  ⚠️ Datos insuficientes")
            return None
        
        print(f"  📊 Backtesting: {df_backtest.index[0]} a {df_backtest.index[-1]}")
        
        for idx in df_backtest.index[:-12]:
            predicciones = {}
            
            for horizonte, modelo in self.modelos.items():
                pred = modelo.predecir(df_backtest.loc[:idx])
                if pred:
                    predicciones[horizonte] = pred
            
            if not predicciones:
                continue
            
            probs_positivas = [p['probabilidad_positiva'] for p in predicciones.values()]
            prob_promedio = np.mean(probs_positivas)
            confianza_promedio = np.mean([p['confianza'] for p in predicciones.values()])
            
            # FILTROS DE CONFIANZA
            if confianza_promedio < TradingConfig.UMBRAL_CONFIANZA_MIN:
                continue
            
            if prob_promedio > 0.5 and prob_promedio < TradingConfig.UMBRAL_PROBABILIDAD_MIN:
                continue
            
            if prob_promedio < 0.5 and (1 - prob_promedio) < TradingConfig.UMBRAL_PROBABILIDAD_MIN:
                continue
            
            # ✅ DECIDIR SEÑAL (CORREGIDO)
            rsi = df_backtest.loc[idx, 'RSI']
            z_score = df_backtest.loc[idx, 'z_score']
            
            if prob_promedio > 0.5:
                # Modelo predice LONG
                señal_long = True
                
                # 🎯 FILTROS DE CONFIRMACIÓN (no inversión)
                # Rechazar si RSI extremadamente overbought (> 80)
                if rsi > 80:
                    continue
                
                # Rechazar si z-score extremadamente alto (momentum contra nosotros)
                if z_score > TradingConfig.Z_EXTREME_THRESHOLD:
                    continue
                    
            else:
                # Modelo predice SHORT
                señal_long = False
                
                # Rechazar si RSI extremadamente oversold (< 20)
                if rsi < 20:
                    continue
                
                # Rechazar si z-score extremadamente bajo
                if z_score < -TradingConfig.Z_EXTREME_THRESHOLD:
                    continue
            
            operacion = self.simular_operacion(
                idx,
                señal_long,
                prob_promedio,
                rsi,
                z_score
            )
            
            if operacion:
                self.operaciones.append(operacion)
        
        if not self.operaciones:
            print(f"  ⚠️ No se generaron operaciones")
            return None
        
        return self.calcular_metricas()
    
    def calcular_metricas(self):
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
        equity_curve = (1 + retornos).cumprod()
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown.min()


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
                print(f"  ❌ No hay datos")
                return False
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            
            self.df_historico = df
            print(f"  ✅ Descargado: {len(df)} velas")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False
    
    def entrenar_modelos(self):
        print(f"\n🎯 ENTRENANDO MODELOS - {self.ticker}")
        print("-" * 80)
        
        if self.df_historico is None:
            return False
        
        df_train = self.df_historico[self.df_historico.index < self.fechas['inicio_backtest']].copy()
        
        print(f"  📊 Datos entrenamiento: {len(df_train)} velas")
        
        modelos_entrenados = 0
        
        for horizonte in TradingConfig.HORIZONTES:
            print(f"\n  🔄 Horizonte {horizonte}h...")
            
            df_prep, features = EtiquetadoDatos.preparar_dataset_ml(df_train, horizonte)
            etiqueta_col = f'etiqueta_{horizonte}h'
            
            modelo = ModeloPrediccion(horizonte, self.ticker)
            if modelo.entrenar_walk_forward(df_prep, features, etiqueta_col):
                self.modelos[horizonte] = modelo
                modelos_entrenados += 1
        
        print(f"\n  ✅ Modelos entrenados: {modelos_entrenados}/{len(TradingConfig.HORIZONTES)}")
        
        return modelos_entrenados > 0
    
    def ejecutar_backtest(self):
        print(f"\n🔬 BACKTESTING - {self.ticker}")
        print("-" * 80)
        
        if not self.modelos:
            return False
        
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
        
        print(f"\n  📊 RESULTADOS:")
        print(f"    Operaciones: {metricas['n_operaciones']}")
        print(f"    Win rate: {metricas['tasa_exito']:.2%}")
        print(f"    Hit TP: {metricas['hit_tp_rate']:.2%}")
        print(f"    Hit SL: {metricas['hit_sl_rate']:.2%}")
        print(f"    Retorno total: {metricas['retorno_total']:.2%}")
        print(f"    Retorno promedio: {metricas['retorno_promedio']:.2%}")
        print(f"    Profit Factor: {metricas['profit_factor']:.2f}")
        print(f"    Max DD: {metricas['max_drawdown']:.2%}")
        print(f"    Sharpe: {metricas['sharpe_ratio']:.2f}")
        
        return True
    
    def es_viable(self):
        if self.metricas_backtest is None:
            return False, 0
        
        m = self.metricas_backtest
        criterios = []
        
        criterios.append(m['tasa_exito'] > 0.55)
        criterios.append(m['retorno_total'] > 0.05)
        criterios.append(m['profit_factor'] > 2.0)
        criterios.append(abs(m['max_drawdown']) < 0.15)
        criterios.append(m['n_operaciones'] >= 15)
        criterios.append(m['sharpe_ratio'] > 1.0)
        
        criterios_cumplidos = sum(criterios)
        viable = criterios_cumplidos >= 5
        
        return viable, criterios_cumplidos
    
    def analizar_tiempo_real(self):
        """
        ✅ LÓGICA VERIFICADA:
        - prob_promedio > 0.5 → LONG
        - prob_promedio < 0.5 → SHORT
        """
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
            
            # Predicciones
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
            
            # FILTROS
            if confianza_promedio < TradingConfig.UMBRAL_CONFIANZA_MIN:
                return None
            
            # ✅ DECIDIR SEÑAL (CORREGIDO)
            señal = "LONG" if prob_promedio > 0.5 else "SHORT"
            prob_real = prob_promedio if señal == "LONG" else 1 - prob_promedio
            
            if prob_real < TradingConfig.UMBRAL_PROBABILIDAD_MIN:
                return None
            
            # DATOS ACTUALES
            ultima_vela = df_reciente.iloc[-1]
            precio = ultima_vela['Close']
            rsi = ultima_vela.get('RSI', 50)
            z_score = ultima_vela.get('z_score', 0)
            
            if pd.isna(z_score) or np.isinf(z_score):
                z_score = 0
            
            # 🎯 FILTROS DE CONFIRMACIÓN
            if señal == "LONG":
                # No comprar si RSI > 80 (extremo)
                if rsi > 80:
                    return None
                # No comprar si z > 2.5 (muy overbought)
                if z_score > TradingConfig.Z_EXTREME_THRESHOLD:
                    return None
            else:  # SHORT
                # No vender si RSI < 20 (extremo)
                if rsi < 20:
                    return None
                # No vender si z < -2.5 (muy oversold)
                if z_score < -TradingConfig.Z_EXTREME_THRESHOLD:
                    return None
            
            # NIVELES
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
            if rsi < TradingConfig.RSI_OVERSOLD:
                estado_rsi = "OVERSOLD"
            elif rsi > TradingConfig.RSI_OVERBOUGHT:
                estado_rsi = "OVERBOUGHT"
            
            estado_z = "NEUTRO"
            if z_score < -2.0:
                estado_z = "OVERSOLD"
            elif z_score > 2.0:
                estado_z = "OVERBOUGHT"
            
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
                'tendencia': 'ALCISTA' if ultima_vela.get('tendencia', 0) == 1 else 'BAJISTA',
                'z_score': float(z_score),
                'estado_z': estado_z,
            }
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return None
    
    def guardar_modelos(self):
        if not self.modelos:
            return False
        
        path_ticker = TradingConfig.MODELOS_DIR / self.ticker
        path_ticker.mkdir(parents=True, exist_ok=True)
        
        for horizonte, modelo in self.modelos.items():
            path_modelo = path_ticker / f"modelo_{horizonte}h.pkl"
            modelo.guardar(path_modelo)
        
        print(f"  💾 Modelos guardados")
        return True


# ============================================
# FUNCIONES AUXILIARES
# ============================================

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
    
    if not token or not chat_id:
        print("⚠️ Telegram no configurado")
        return
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, data={"chat_id": chat_id, "text": mensaje})
    
    print(f"📨 Telegram: {r.status_code}")


# ============================================
# MAIN
# ============================================

def main():
    print("🚀 SISTEMA DE TRADING - LÓGICA VERIFICADA")
    print("=" * 80)
    print("✅ ETIQUETADO: retorno_futuro > 0.8% → LONG (1)")
    print("✅ PREDICCIÓN: prob_positiva > 0.5 → señal LONG")
    print("✅ FILTROS: RSI y Z-score como confirmación (no inversión)")
    print("=" * 80)
    
    fechas = TradingConfig.get_fechas()
    print(f"\n📅 Configuración:")
    print(f"  Actual: {fechas['actual'].date()}")
    print(f"  Backtest: {fechas['inicio_backtest'].date()}")
    print(f"  SL: {TradingConfig.STOP_LOSS_PCT:.1%}, TP: {TradingConfig.TAKE_PROFIT_PCT:.1%}")
    
    resultados_globales = {}
    
    for ticker in TradingConfig.ACTIVOS:
        sistema = SistemaTradingTicker(ticker)
        
        if not sistema.descargar_datos():
            continue
        
        if not sistema.entrenar_modelos():
            continue
        
        if not sistema.ejecutar_backtest():
            continue
        
        viable, criterios = sistema.es_viable()
        
        print(f"\n{'='*80}")
        print(f"📊 EVALUACIÓN - {ticker}")
        print(f"{'='*80}")
        print(f"  Criterios: {criterios}/6")
        print(f"  Viable: {'✅ SÍ' if viable else '❌ NO'}")
        
        señal_actual = None
        
        if viable:
            try:
                señal_actual = sistema.analizar_tiempo_real()
                
                if señal_actual:
                    print(f"\n  🚨 SEÑAL: {señal_actual['señal']}")
                    print(f"    Prob: {señal_actual['probabilidad']:.2%}")
                    print(f"    Conf: {señal_actual['confianza']:.2%}")
                    print(f"    RSI: {señal_actual['rsi']:.0f} ({señal_actual['estado_rsi']})")
                    print(f"    Z: {señal_actual['z_score']:.2f} ({señal_actual['estado_z']})")
                    print(f"    ${señal_actual['precio']:,.2f} → SL ${señal_actual['stop_loss']:,.2f} / TP ${señal_actual['take_profit']:,.2f}")
                    
                    ultima = cargar_ultima_senal()
                    if ultima and ultima["ticker"] == ticker and ultima["señal"] == señal_actual["señal"]:
                        print("  🔁 Repetida")
                    else:
                        enviar_telegram(
                            f"🚨 {ticker} - {señal_actual['señal']}\n"
                            f"📊 Prob: {señal_actual['probabilidad']:.1%} | Conf: {señal_actual['confianza']:.1%}\n\n"
                            f"💰 ${señal_actual['precio']:.2f}\n"
                            f"🛑 SL: ${señal_actual['stop_loss']:.2f} (-{TradingConfig.STOP_LOSS_PCT:.1%})\n"
                            f"🎯 TP: ${señal_actual['take_profit']:.2f} (+{TradingConfig.TAKE_PROFIT_PCT:.1%})\n"
                            f"⚖️ R:R {señal_actual['ratio_rr']:.2f}\n\n"
                            f"📊 RSI: {señal_actual['rsi']:.0f} ({señal_actual['estado_rsi']})\n"
                            f"📈 Tendencia: {señal_actual['tendencia']}\n"
                            f"🔄 Z-score: {señal_actual['z_score']:.2f} ({señal_actual['estado_z']})"
                        )
                        
                        guardar_ultima_senal({
                            "ticker": ticker,
                            "señal": señal_actual["señal"],
                            "fecha": str(señal_actual["fecha"])
                        })
            
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        if viable:
            sistema.guardar_modelos()
        
        resultados_globales[ticker] = {
            'viable': viable,
            'criterios': criterios,
            'metricas': sistema.metricas_backtest,
            'señal_actual': señal_actual
        }
    
    print(f"\n{'='*80}")
    print("📊 RESUMEN")
    print(f"{'='*80}")
    
    viables = [t for t, r in resultados_globales.items() if r['viable']]
    
    print(f"\n  Procesados: {len(resultados_globales)}")
    print(f"  Viables: {len(viables)}")
    
    if viables:
        print(f"\n  ✅ VIABLES:")
        for ticker in viables:
            r = resultados_globales[ticker]
            m = r['metricas']
            print(f"    {ticker}: {m['retorno_total']:.2%} | Win {m['tasa_exito']:.0%} | PF {m['profit_factor']:.1f}")
    
    return resultados_globales


if __name__ == "__main__":
    main()
