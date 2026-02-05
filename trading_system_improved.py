"""
Trading Cuantitativo - Sistema Robustecido
Versión: 3.1 - Corregido para GitHub Actions
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
import os
import sys
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Tuple
import talib
from scipy import stats
import joblib

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class Config:
    """Configuración optimizada"""
    # Assets principales de crypto
    SYMBOLS = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD", "XRP-USD"]
    
    # Timeframes (1h para mejor data)
    INTERVAL = "1h"
    TRAIN_DAYS = 180  # 6 meses
    TEST_DAYS = 30    # 1 mes
    
    # Trading parameters
    COMMISSION = 0.001
    STOP_LOSS_PCT = 0.02
    TAKE_PROFIT_PCT = 0.04
    
    # Model parameters
    MIN_DATA_POINTS = 100
    CONFIDENCE_THRESHOLD = 0.55

class DataHandler:
    """Manejo robusto de datos"""
    
    @staticmethod
    def get_data(symbol: str, days: int = 180) -> Optional[pd.DataFrame]:
        """Obtiene datos con validación"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            logger.info(f"Descargando {symbol} desde {start_date.date()}")
            
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=Config.INTERVAL,
                progress=False,
                auto_adjust=True
            )
            
            if df.empty or len(df) < Config.MIN_DATA_POINTS:
                logger.warning(f"Datos insuficientes para {symbol}")
                return None
            
            # Renombrar columnas
            df.columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Validar datos
            if df.isnull().sum().sum() > 0:
                df = df.ffill().bfill()
            
            logger.info(f"✅ {symbol}: {len(df)} velas, Precio: ${df['close'].iloc[-1]:.2f}")
            return df
            
        except Exception as e:
            logger.error(f"Error descargando {symbol}: {e}")
            return None

class FeatureEngineer:
    """Ingeniería de características simplificada"""
    
    @staticmethod
    def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
        """Añade features básicas pero efectivas"""
        df = df.copy()
        
        # Precios
        close = df['close']
        
        # 1. RETORNOS
        df['returns_1h'] = close.pct_change(1)
        df['returns_4h'] = close.pct_change(4)
        df['returns_12h'] = close.pct_change(12)
        df['returns_24h'] = close.pct_change(24)
        
        # 2. MEDIAS MÓVILES (evitar muchas)
        df['sma_20'] = close.rolling(20).mean()
        df['sma_50'] = close.rolling(50).mean()
        df['ema_12'] = close.ewm(span=12, adjust=False).mean()
        df['ema_26'] = close.ewm(span=26, adjust=False).mean()
        
        # 3. RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 4. MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 5. BOLLINGER BANDS
        df['bb_middle'] = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # 6. ATR (Volatilidad)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - close.shift()).abs()
        low_close = (df['low'] - close.shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr'] / close
        
        # 7. VOLUMEN
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # 8. TENDENCIA
        df['trend'] = (df['sma_20'] > df['sma_50']).astype(int)
        
        # 9. MOMENTUM
        df['momentum_12h'] = close / close.shift(12) - 1
        
        # 10. SOPORTE/RESISTENCIA simple
        df['high_24h'] = df['high'].rolling(24).max()
        df['low_24h'] = df['low'].rolling(24).min()
        df['position_range'] = (close - df['low_24h']) / (df['high_24h'] - df['low_24h'])
        
        # Eliminar NaN iniciales
        df = df.dropna()
        
        logger.info(f"Features creadas: {len(df.columns)} columnas, {len(df)} filas válidas")
        return df

class LabelGenerator:
    """Generación de etiquetas robusta"""
    
    @staticmethod
    def create_labels(df: pd.DataFrame, horizon_hours: int = 4) -> pd.Series:
        """
        Crea etiquetas binarias basadas en movimiento futuro
        1 = precio sube más del 1% en horizon_hours
        0 = precio baja más del 1% en horizon_hours
        NaN = movimiento insuficiente (no operar)
        """
        # Precio futuro
        future_price = df['close'].shift(-horizon_hours)
        
        # Retorno futuro
        future_return = (future_price / df['close'] - 1)
        
        # Umbral dinámico basado en volatilidad
        volatility = df['returns_1h'].rolling(24).std().fillna(0.01)
        threshold = volatility * 1.5  # 1.5x la volatilidad
        threshold = threshold.clip(lower=0.005, upper=0.02)  # Entre 0.5% y 2%
        
        # Crear etiquetas
        labels = pd.Series(np.nan, index=df.index)
        labels[future_return > threshold] = 1   # COMPRAR
        labels[future_return < -threshold] = 0  # VENDER
        
        # Estadísticas
        n_buy = (labels == 1).sum()
        n_sell = (labels == 0).sum()
        total = n_buy + n_sell
        
        if total > 0:
            logger.info(f"Etiquetas: BUY={n_buy} ({n_buy/total:.1%}), SELL={n_sell} ({n_sell/total:.1%})")
        
        return labels

class TradingModel:
    """Modelo de ML simplificado pero efectivo"""
    
    def __init__(self):
        self.model = None
        self.features = None
        self.scaler = None
        
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Selecciona las mejores features"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Features base
        base_features = [
            'returns_1h', 'returns_4h', 'returns_12h', 'returns_24h',
            'rsi', 'macd', 'macd_hist', 'bb_position', 'atr_pct',
            'volume_ratio', 'trend', 'momentum_12h', 'position_range'
        ]
        
        # Filtrar features disponibles
        available = [f for f in base_features if f in X.columns]
        
        # Usar RandomForest para importancia
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X[available].fillna(0), y)
        
        # Seleccionar top features
        importance = pd.Series(rf.feature_importances_, index=available)
        selected = importance.nlargest(10).index.tolist()
        
        logger.info(f"Top 5 features: {selected[:5]}")
        return selected
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Entrena modelo XGBoost"""
        from xgboost import XGBClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Seleccionar features
        self.features = self.select_features(X_train, y_train)
        
        if len(self.features) < 5:
            logger.warning("Features insuficientes")
            return False
        
        # Preparar datos
        X = X_train[self.features].fillna(0)
        y = y_train.values
        
        # Escalar
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Entrenar modelo (hiperparámetros optimizados)
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        self.model.fit(X_scaled, y)
        
        # Validación simple
        train_pred = self.model.predict(X_scaled)
        accuracy = (train_pred == y).mean()
        
        logger.info(f"Modelo entrenado - Accuracy: {accuracy:.2%}")
        return accuracy > 0.52
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predicciones"""
        if self.model is None or self.features is None:
            return np.zeros(len(X)), np.zeros(len(X))
        
        # Preparar datos
        X_data = X[self.features].fillna(0)
        X_scaled = self.scaler.transform(X_data)
        
        # Predecir
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities

class SignalGenerator:
    """Genera señales de trading con filtros"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def generate(self, df: pd.DataFrame, predictions: np.ndarray, 
                 probabilities: np.ndarray) -> pd.DataFrame:
        """Genera señales con múltiples filtros"""
        signals = pd.DataFrame(index=df.index)
        signals['prediction'] = predictions
        signals['probability'] = probabilities
        
        # 1. Señal base
        signals['base_signal'] = 0
        signals.loc[signals['prediction'] == 1, 'base_signal'] = 1
        signals.loc[signals['prediction'] == 0, 'base_signal'] = -1
        
        # 2. Filtro de confianza
        signals['signal'] = 0
        high_conf = signals['probability'] > self.config.CONFIDENCE_THRESHOLD
        signals.loc[(signals['base_signal'] == 1) & high_conf, 'signal'] = 1
        signals.loc[(signals['base_signal'] == -1) & high_conf, 'signal'] = -1
        
        # 3. Filtro RSI (evitar extremos)
        if 'rsi' in df.columns:
            overbought = df['rsi'] > 70
            oversold = df['rsi'] < 30
            signals.loc[(signals['signal'] == 1) & overbought, 'signal'] = 0
            signals.loc[(signals['signal'] == -1) & oversold, 'signal'] = 0
        
        # 4. Filtro de tendencia
        if 'trend' in df.columns:
            # En tendencia alcista, evitar vender
            signals.loc[(signals['signal'] == -1) & (df['trend'] == 1), 'signal'] = 0
            # En tendencia bajista, evitar comprar
            signals.loc[(signals['signal'] == 1) & (df['trend'] == 0), 'signal'] = 0
        
        # 5. Eliminar señales consecutivas iguales
        signals['prev_signal'] = signals['signal'].shift(1)
        same_signal = signals['signal'] == signals['prev_signal']
        signals.loc[same_signal, 'signal'] = 0
        
        # Estadísticas
        buy_signals = (signals['signal'] == 1).sum()
        sell_signals = (signals['signal'] == -1).sum()
        
        logger.info(f"Señales generadas: {buy_signals} BUY, {sell_signals} SELL")
        
        return signals[['signal', 'probability']]

class Backtester:
    """Backtesting rápido y realista"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def run(self, df: pd.DataFrame, signals: pd.DataFrame) -> Dict:
        """Ejecuta backtesting simple pero efectivo"""
        try:
            # Inicializar
            cash = 10000
            position = 0
            trades = []
            equity = [cash]
            
            for i in range(1, len(df)):
                current_price = df['close'].iloc[i]
                signal = signals['signal'].iloc[i]
                
                # Cerrar posición si hay señal contraria o stop/take
                if position != 0:
                    entry_price = trades[-1]['entry_price']
                    
                    # Calcular P&L
                    if position > 0:  # Long
                        pl_pct = (current_price - entry_price) / entry_price
                        stop_loss = entry_price * (1 - self.config.STOP_LOSS_PCT)
                        take_profit = entry_price * (1 + self.config.TAKE_PROFIT_PCT)
                    else:  # Short
                        pl_pct = (entry_price - current_price) / entry_price
                        stop_loss = entry_price * (1 + self.config.STOP_LOSS_PCT)
                        take_profit = entry_price * (1 - self.config.TAKE_PROFIT_PCT)
                    
                    # Verificar cierres
                    close_trade = False
                    close_reason = ""
                    
                    if position > 0:
                        if current_price <= stop_loss:
                            close_trade = True
                            close_reason = "SL"
                        elif current_price >= take_profit:
                            close_trade = True
                            close_reason = "TP"
                        elif signal == -1:
                            close_trade = True
                            close_reason = "SIGNAL"
                    else:
                        if current_price >= stop_loss:
                            close_trade = True
                            close_reason = "SL"
                        elif current_price <= take_profit:
                            close_trade = True
                            close_reason = "TP"
                        elif signal == 1:
                            close_trade = True
                            close_reason = "SIGNAL"
                    
                    if close_trade:
                        # Aplicar comisión
                        commission = abs(position * current_price) * self.config.COMMISSION
                        
                        # Actualizar cash
                        cash += position * current_price - commission
                        position = 0
                        
                        # Registrar trade
                        trades[-1].update({
                            'exit_price': current_price,
                            'exit_time': df.index[i],
                            'pnl_pct': pl_pct,
                            'commission': commission,
                            'reason': close_reason
                        })
                
                # Abrir nueva posición
                if position == 0 and signal != 0:
                    # Tamaño de posición (10% del capital)
                    position_size = cash * 0.1 / current_price
                    
                    if signal == 1:  # BUY
                        position = position_size
                        trade_type = "LONG"
                    else:  # SELL
                        position = -position_size
                        trade_type = "SHORT"
                    
                    # Comisión de entrada
                    commission = abs(position * current_price) * self.config.COMMISSION
                    cash -= commission
                    
                    # Registrar trade
                    trades.append({
                        'entry_time': df.index[i],
                        'entry_price': current_price,
                        'type': trade_type,
                        'position': position,
                        'commission_entry': commission,
                        'signal_strength': signals['probability'].iloc[i]
                    })
                
                # Calcular equity
                current_equity = cash + (position * current_price if position != 0 else 0)
                equity.append(current_equity)
            
            # Cerrar posición final si queda abierta
            if position != 0 and trades:
                last_price = df['close'].iloc[-1]
                entry_price = trades[-1]['entry_price']
                
                if position > 0:
                    pl_pct = (last_price - entry_price) / entry_price
                else:
                    pl_pct = (entry_price - last_price) / entry_price
                
                commission = abs(position * last_price) * self.config.COMMISSION
                trades[-1].update({
                    'exit_price': last_price,
                    'exit_time': df.index[-1],
                    'pnl_pct': pl_pct,
                    'commission_exit': commission,
                    'reason': 'END'
                })
            
            # Calcular métricas
            equity_series = pd.Series(equity, index=df.index[:len(equity)])
            returns = equity_series.pct_change().dropna()
            
            if len(trades) > 0:
                winning_trades = [t for t in trades if 'pnl_pct' in t and t['pnl_pct'] > 0]
                losing_trades = [t for t in trades if 'pnl_pct' in t and t['pnl_pct'] <= 0]
                
                win_rate = len(winning_trades) / len(trades)
                avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
                avg_loss = np.mean([t['pnl_pct'] for t in losing_trades]) if losing_trades else 0
                
                total_return = (equity_series.iloc[-1] / 10000 - 1) * 100
                
                # Sharpe Ratio (asumiendo 0% risk-free rate)
                sharpe = returns.mean() / returns.std() * np.sqrt(365*24) if returns.std() > 0 else 0
                
                # Max Drawdown
                cum_returns = (1 + returns).cumprod()
                running_max = cum_returns.cummax()
                drawdown = (cum_returns / running_max - 1)
                max_drawdown = drawdown.min()
                
                # Profit Factor
                gross_profit = sum(t['pnl_pct'] for t in winning_trades if 'pnl_pct' in t)
                gross_loss = abs(sum(t['pnl_pct'] for t in losing_trades if 'pnl_pct' in t))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
                
                results = {
                    'total_trades': len(trades),
                    'winning_trades': len(winning_trades),
                    'losing_trades': len(losing_trades),
                    'win_rate': win_rate,
                    'avg_win_pct': avg_win * 100,
                    'avg_loss_pct': avg_loss * 100,
                    'profit_factor': profit_factor,
                    'total_return_pct': total_return,
                    'sharpe_ratio': sharpe,
                    'max_drawdown_pct': max_drawdown * 100,
                    'final_equity': equity_series.iloc[-1]
                }
            else:
                results = {
                    'total_trades': 0,
                    'total_return_pct': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown_pct': 0,
                    'final_equity': 10000
                }
            
            return {
                'metrics': results,
                'trades': trades,
                'equity_curve': equity_series
            }
            
        except Exception as e:
            logger.error(f"Error en backtesting: {e}")
            return None

class TradingSystem:
    """Sistema principal"""
    
    def __init__(self):
        self.config = Config()
        self.data_handler = DataHandler()
        self.feature_engineer = FeatureEngineer()
        self.label_generator = LabelGenerator()
        self.model = TradingModel()
        self.signal_generator = SignalGenerator(self.config)
        self.backtester = Backtester(self.config)
        
    def process_symbol(self, symbol: str) -> Optional[Dict]:
        """Procesa un símbolo completo"""
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"PROCESANDO: {symbol}")
            logger.info(f"{'='*60}")
            
            # 1. Obtener datos
            df = self.data_handler.get_data(symbol, self.config.TRAIN_DAYS + self.config.TEST_DAYS)
            if df is None or len(df) < 200:
                logger.warning(f"Datos insuficientes para {symbol}")
                return None
            
            # 2. Crear features
            df_features = self.feature_engineer.add_basic_features(df)
            if len(df_features) < 100:
                logger.warning(f"Features insuficientes después de limpieza")
                return None
            
            # 3. Split train/test (70/30)
            split_idx = int(len(df_features) * 0.7)
            train_data = df_features.iloc[:split_idx].copy()
            test_data = df_features.iloc[split_idx:].copy()
            
            logger.info(f"Train: {len(train_data)} | Test: {len(test_data)}")
            
            # 4. Generar labels para training
            labels = self.label_generator.create_labels(train_data)
            
            # Filtrar datos con labels válidas
            valid_idx = labels.dropna().index
            X_train = train_data.loc[valid_idx]
            y_train = labels.loc[valid_idx]
            
            if len(X_train) < 50:
                logger.warning(f"Datos de entrenamiento insuficientes")
                return None
            
            # 5. Entrenar modelo
            logger.info("Entrenando modelo...")
            if not self.model.train(X_train, y_train):
                logger.warning(f"Modelo no pudo ser entrenado para {symbol}")
                return None
            
            # 6. Predecir en test
            X_test = test_data
            predictions, probabilities = self.model.predict(X_test)
            
            # 7. Generar señales
            signals = self.signal_generator.generate(X_test, predictions, probabilities)
            
            # 8. Backtest
            backtest_results = self.backtester.run(X_test, signals)
            
            if backtest_results is None:
                return None
            
            # 9. Preparar resultados
            results = {
                'symbol': symbol,
                'data_points': len(df),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'signals': len(signals[signals['signal'] != 0]),
                'trades': backtest_results['metrics']['total_trades'],
                'win_rate': backtest_results['metrics']['win_rate'],
                'total_return': backtest_results['metrics']['total_return_pct'],
                'sharpe': backtest_results['metrics']['sharpe_ratio'],
                'max_dd': backtest_results['metrics']['max_drawdown_pct'],
                'profit_factor': backtest_results['metrics'].get('profit_factor', 0)
            }
            
            # Mostrar resultados
            logger.info(f"\n📊 RESULTADOS {symbol}:")
            logger.info(f"  Operaciones: {results['trades']}")
            logger.info(f"  Win Rate: {results['win_rate']:.1%}")
            logger.info(f"  Retorno Total: {results['total_return']:.2f}%")
            logger.info(f"  Sharpe Ratio: {results['sharpe']:.2f}")
            logger.info(f"  Max Drawdown: {results['max_dd']:.2f}%")
            logger.info(f"  Profit Factor: {results['profit_factor']:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error procesando {symbol}: {e}")
            return None

def main():
    """Función principal"""
    logger.info("🚀 SISTEMA DE TRADING CUANTITATIVO v3.1")
    logger.info("=" * 60)
    
    # Crear sistema
    system = TradingSystem()
    
    # Procesar símbolos
    all_results = {}
    
    for symbol in system.config.SYMBOLS[:4]:  # Procesar solo 4 para velocidad
        try:
            result = system.process_symbol(symbol)
            if result:
                all_results[symbol] = result
        except Exception as e:
            logger.error(f"Error con {symbol}: {e}")
            continue
    
    # Resumen final
    logger.info("\n" + "=" * 60)
    logger.info("📈 RESUMEN FINAL")
    logger.info("=" * 60)
    
    if not all_results:
        logger.warning("No se obtuvieron resultados válidos")
        return None
    
    # Calcular estadísticas
    total_trades = sum(r['trades'] for r in all_results.values())
    avg_return = np.mean([r['total_return'] for r in all_results.values()])
    avg_sharpe = np.mean([r['sharpe'] for r in all_results.values()])
    avg_win_rate = np.mean([r['win_rate'] for r in all_results.values()])
    
    logger.info(f"Símbolos procesados: {len(all_results)}")
    logger.info(f"Operaciones totales: {total_trades}")
    logger.info(f"Retorno promedio: {avg_return:.2f}%")
    logger.info(f"Sharpe promedio: {avg_sharpe:.2f}")
    logger.info(f"Win Rate promedio: {avg_win_rate:.1%}")
    
    # Mejor y peor
    if all_results:
        best = max(all_results.items(), key=lambda x: x[1]['total_return'])
        worst = min(all_results.items(), key=lambda x: x[1]['total_return'])
        
        logger.info(f"\n🏆 MEJOR: {best[0]} - {best[1]['total_return']:.2f}%")
        logger.info(f"📉 PEOR: {worst[0]} - {worst[1]['total_return']:.2f}%")
    
    # Guardar resultados
    output_file = 'trading_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"\n💾 Resultados guardados en {output_file}")
    logger.info("\n✅ PROCESO COMPLETADO")
    
    return all_results

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⏹️ Proceso interrumpido")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Error fatal: {e}")
        sys.exit(1)
