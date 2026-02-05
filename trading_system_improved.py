"""
Trading Cuantitativo - Sistema Multi-Modelo
Versión: 3.0 - GitHub Actions Optimizado
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
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from scipy import stats
import ta  # Technical Analysis library
import joblib
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import vectorbt as vbt  # Optimizado para backtesting rápido

warnings.filterwarnings('ignore')

# Configuración de logging para GitHub Actions
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuración optimizada para GitHub Actions"""
    # Assets
    SYMBOLS: List[str] = field(default_factory=lambda: [
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD",
        "XRP-USD", "ADA-USD", "AVAX-USD", "DOT-USD"
    ])
    
    # Timeframes
    INTERVAL: str = "1h"
    TRAIN_DAYS: int = 180  # 6 meses para entrenamiento rápido
    TEST_DAYS: int = 30    # 1 mes para test
    
    # Modelos a usar
    MODELS: List[str] = field(default_factory=lambda: [
        "xgboost", "lightgbm", "random_forest", "gradient_boosting"
    ])
    
    # Hiperparámetros (reducidos para velocidad)
    N_SPLITS: int = 3
    CV_FOLDS: int = 2
    
    # Trading
    COMMISSION: float = 0.001  # 0.1%
    SLIPPAGE: float = 0.0005   # 0.05%
    STOP_LOSS: float = 0.02    # 2%
    TAKE_PROFIT: float = 0.04  # 4%
    
    # Señales
    CONFIDENCE_THRESHOLD: float = 0.55
    MIN_SIGNAL_STRENGTH: float = 0.6
    
    # GitHub Actions
    MAX_SYMBOLS: int = 6  # Máximo de symbols a procesar
    TIMEOUT_PER_SYMBOL: int = 300  # 5 minutos por symbol
    
class DataFetcher:
    """Manejador de datos optimizado"""
    
    @staticmethod
    def fetch_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
        """Descarga datos con caché para GitHub Actions"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Convertir a string para yfinance
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            logger.info(f"Descargando {symbol} ({interval}) desde {start_str}")
            
            df = yf.download(
                symbol,
                start=start_str,
                end=end_str,
                interval=interval,
                progress=False,
                threads=False
            )
            
            if df.empty:
                logger.warning(f"No se obtuvieron datos para {symbol}")
                return None
            
            # Limpieza básica
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.dropna()
            
            logger.info(f"✅ {symbol}: {len(df)} velas descargadas")
            return df
            
        except Exception as e:
            logger.error(f"Error descargando {symbol}: {e}")
            return None

class FeatureEngineer:
    """Ingeniería de características avanzada"""
    
    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """Crea features técnicas usando múltiples indicadores"""
        df = df.copy()
        
        # Precios y retornos
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Momentum indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['stoch_k'] = ta.momentum.StochasticOscillator(
            df['high'], df['low'], df['close'], window=14, smooth_window=3
        ).stoch()
        df['stoch_d'] = ta.momentum.StochasticOscillator(
            df['high'], df['low'], df['close'], window=14, smooth_window=3
        ).stoch_signal()
        df['macd'] = ta.trend.MACD(df['close']).macd()
        df['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
        df['macd_diff'] = ta.trend.MACD(df['close']).macd_diff()
        
        # Trend indicators
        df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        
        # Volatility indicators
        df['bb_upper'] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
        df['bb_lower'] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # Volume indicators
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Support/Resistance
        df['pivot_point'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        df['r1'] = 2 * df['pivot_point'] - df['low'].shift(1)
        df['s1'] = 2 * df['pivot_point'] - df['high'].shift(1)
        
        # Price action
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        df['close_open_pct'] = (df['close'] - df['open']) / df['open']
        
        # Statistical features
        for window in [6, 12, 24]:
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()
            df[f'skew_{window}'] = df['returns'].rolling(window).skew()
            df[f'kurtosis_{window}'] = df['returns'].rolling(window).kurt()
        
        # Lag features
        for lag in [1, 2, 3, 4, 6]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # Target encoding-like features
        df['close_sma_ratio'] = df['close'] / df['sma_20']
        df['high_52w'] = df['high'].rolling(24*7*52).max()  # 52 semanas en horas
        df['low_52w'] = df['low'].rolling(24*7*52).min()
        df['position_52w'] = (df['close'] - df['low_52w']) / (df['high_52w'] - df['low_52w'])
        
        # Interaction features
        df['rsi_volume'] = df['rsi'] * df['volume_ratio']
        df['macd_volatility'] = df['macd'] * df['volatility_12']
        
        # Clean NaN values
        df = df.dropna()
        
        logger.info(f"Features creadas: {len(df.columns)} columnas")
        return df

class LabelGenerator:
    """Generación de labels para ML"""
    
    @staticmethod
    def triple_barrier_labels(
        df: pd.DataFrame, 
        horizon: int = 24,
        upper_barrier: float = 0.04,
        lower_barrier: float = -0.02
    ) -> pd.Series:
        """
        Método triple barrier labeling
        Returns: 1 (compra), 0 (venta), -1 (mantener)
        """
        prices = df['close'].values
        labels = np.zeros(len(prices))
        
        for i in range(len(prices) - horizon):
            current_price = prices[i]
            future_prices = prices[i+1:i+horizon+1]
            
            # Calcular retornos máximos y mínimos
            max_return = np.max(future_prices) / current_price - 1
            min_return = np.min(future_prices) / current_price - 1
            
            # Aplicar barreras
            if max_return >= upper_barrier:
                # Tocó barrera superior primero -> señal de compra
                labels[i] = 1
            elif min_return <= lower_barrier:
                # Tocó barrera inferior primero -> señal de venta
                labels[i] = 0
            else:
                # No tocó ninguna barrera -> mantener
                labels[i] = -1
        
        # Reemplazar -1 con NaN para eliminar esas muestras
        labels = pd.Series(labels, index=df.index)
        labels = labels.replace(-1, np.nan)
        
        return labels
    
    @staticmethod
    def volatility_adjusted_labels(
        df: pd.DataFrame,
        horizon: int = 24,
        volatility_lookback: int = 24
    ) -> pd.Series:
        """Labels ajustados por volatilidad"""
        returns = df['close'].pct_change(horizon).shift(-horizon)
        volatility = df['close'].pct_change().rolling(volatility_lookback).std()
        
        # Umbral dinámico basado en volatilidad
        threshold = volatility * 1.5
        
        labels = pd.Series(np.nan, index=df.index)
        labels[returns > threshold] = 1  # LONG
        labels[returns < -threshold] = 0  # SHORT
        
        return labels

class ModelFactory:
    """Factory para crear diferentes modelos de ML"""
    
    @staticmethod
    def get_model(model_type: str):
        """Retorna modelo configurado"""
        models = {
            'xgboost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'svm': SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            'logistic': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            ),
            'catboost': CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=False
            )
        }
        return models.get(model_type, models['xgboost'])

class EnsembleModel:
    """Ensemble de múltiples modelos"""
    
    def __init__(self, config: Config):
        self.config = config
        self.models = {}
        self.scaler = RobustScaler()
        self.feature_columns = None
        
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Selección de features usando importancia de Random Forest"""
        from sklearn.feature_selection import SelectFromModel
        
        selector = RandomForestClassifier(n_estimators=50, random_state=42)
        selector.fit(X, y)
        
        # Seleccionar features con importancia > percentil 75
        importances = selector.feature_importances_
        threshold = np.percentile(importances, 75)
        
        selected = X.columns[importances > threshold].tolist()
        logger.info(f"Seleccionadas {len(selected)}/{len(X.columns)} features")
        
        return selected
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Entrena ensemble de modelos"""
        # Seleccionar features
        self.feature_columns = self.select_features(X_train, y_train)
        X_train_selected = X_train[self.feature_columns]
        
        # Escalar datos
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        
        # Entrenar cada modelo
        for model_name in self.config.MODELS[:3]:  # Máximo 3 modelos para velocidad
            try:
                logger.info(f"Entrenando {model_name}...")
                model = ModelFactory.get_model(model_name)
                
                # Entrenamiento con cross-validation reducida
                model.fit(X_train_scaled, y_train)
                self.models[model_name] = model
                logger.info(f"✅ {model_name} entrenado")
                
            except Exception as e:
                logger.warning(f"Error entrenando {model_name}: {e}")
        
        # Crear ensemble por votación
        if len(self.models) >= 2:
            estimators = [(name, model) for name, model in self.models.items()]
            self.ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft',
                n_jobs=-1
            )
            self.ensemble.fit(X_train_scaled, y_train)
            logger.info(f"Ensemble creado con {len(self.models)} modelos")
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predicción con ensemble"""
        if not self.models:
            return np.zeros(len(X)), np.zeros(len(X))
        
        X_selected = X[self.feature_columns]
        X_scaled = self.scaler.transform(X_selected)
        
        # Predicción del ensemble
        if hasattr(self, 'ensemble'):
            probabilities = self.ensemble.predict_proba(X_scaled)
            predictions = self.ensemble.predict(X_scaled)
        else:
            # Promedio de probabilidades de todos los modelos
            all_probs = []
            for model in self.models.values():
                probs = model.predict_proba(X_scaled)
                all_probs.append(probs)
            
            probabilities = np.mean(all_probs, axis=0)
            predictions = np.argmax(probabilities, axis=1)
        
        return predictions, probabilities[:, 1]  # Probabilidad clase positiva

class TradingSignals:
    """Generador de señales de trading"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def generate_signals(
        self, 
        df: pd.DataFrame, 
        predictions: np.ndarray,
        probabilities: np.ndarray
    ) -> pd.DataFrame:
        """Genera señales de trading con filtros"""
        signals = pd.DataFrame(index=df.index)
        signals['prediction'] = predictions
        signals['probability'] = probabilities
        
        # Aplicar filtros
        signals['signal'] = 0  # 0: no operar, 1: comprar, -1: vender
        
        # Filtro de confianza
        high_confidence = signals['probability'] > self.config.CONFIDENCE_THRESHOLD
        
        # Señal de compra
        buy_signal = (signals['prediction'] == 1) & high_confidence
        signals.loc[buy_signal, 'signal'] = 1
        
        # Señal de venta
        sell_signal = (signals['prediction'] == 0) & high_confidence
        signals.loc[sell_signal, 'signal'] = -1
        
        # Filtro RSI (evitar sobrecompra/sobreventa)
        if 'rsi' in df.columns:
            overbought = df['rsi'] > 70
            oversold = df['rsi'] < 30
            
            # Ajustar señales
            signals.loc[(signals['signal'] == 1) & overbought, 'signal'] = 0
            signals.loc[(signals['signal'] == -1) & oversold, 'signal'] = 0
        
        # Filtro de tendencia (EMA)
        if 'ema_12' in df.columns and 'ema_26' in df.columns:
            uptrend = df['ema_12'] > df['ema_26']
            downtrend = df['ema_12'] < df['ema_26']
            
            # Preferir operaciones en dirección de la tendencia
            signals.loc[(signals['signal'] == 1) & downtrend, 'probability'] *= 0.7
            signals.loc[(signals['signal'] == -1) & uptrend, 'probability'] *= 0.7
        
        # Eliminar señales consecutivas del mismo tipo
        signals['signal_changed'] = signals['signal'].diff().fillna(0) != 0
        signals.loc[~signals['signal_changed'], 'signal'] = 0
        
        logger.info(f"Señales generadas: {len(signals[signals['signal'] != 0])} operaciones")
        
        return signals[['signal', 'probability']]

class FastBacktester:
    """Backtesting rápido usando vectorbt"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def run(
        self, 
        df: pd.DataFrame, 
        signals: pd.DataFrame
    ) -> Dict:
        """Ejecuta backtesting con vectorbt"""
        try:
            # Precios
            close = df['close']
            
            # Señales
            entries = signals['signal'] == 1
            exits = signals['signal'] == -1
            
            # Crear portfolio
            pf = vbt.Portfolio.from_signals(
                close=close,
                entries=entries,
                exits=exits,
                fees=self.config.COMMISSION,
                freq='1h'
            )
            
            # Métricas
            stats = pf.stats()
            
            # Calcular métricas adicionales
            returns = pf.returns()
            sharpe = pf.sharpe_ratio()
            max_dd = pf.max_drawdown()
            
            results = {
                'total_return': stats['Total Return [%]'],
                'sharpe': sharpe,
                'max_drawdown': max_dd,
                'win_rate': stats['Win Rate [%]'],
                'profit_factor': stats['Profit Factor'],
                'total_trades': stats['Total Trades'],
                'avg_trade': stats['Avg Trade [%]']
            }
            
            # Equity curve
            equity = pf.value()
            
            return {
                'metrics': results,
                'equity_curve': equity,
                'portfolio': pf
            }
            
        except Exception as e:
            logger.error(f"Error en backtesting: {e}")
            return None

class TradingSystem:
    """Sistema principal de trading"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_fetcher = DataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.label_generator = LabelGenerator()
        self.ensemble_model = EnsembleModel(config)
        self.signal_generator = TradingSignals(config)
        self.backtester = FastBacktester(config)
        
    def process_symbol(self, symbol: str) -> Optional[Dict]:
        """Procesa un símbolo completo"""
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Procesando: {symbol}")
            logger.info(f"{'='*60}")
            
            # 1. Obtener datos
            df = self.data_fetcher.fetch_data(
                symbol, 
                self.config.INTERVAL, 
                self.config.TRAIN_DAYS + self.config.TEST_DAYS
            )
            
            if df is None or len(df) < 100:
                logger.warning(f"Datos insuficientes para {symbol}")
                return None
            
            # 2. Crear features
            df_features = self.feature_engineer.create_features(df)
            
            # 3. Split train/test
            split_idx = int(len(df_features) * 0.7)
            train_data = df_features.iloc[:split_idx]
            test_data = df_features.iloc[split_idx:]
            
            logger.info(f"Train: {len(train_data)} | Test: {len(test_data)}")
            
            # 4. Generar labels para training
            labels = self.label_generator.volatility_adjusted_labels(
                train_data, 
                horizon=24
            )
            
            # Preparar datos para ML
            train_data_clean = train_data.dropna()
            labels_clean = labels.dropna()
            
            # Alinear índices
            common_idx = train_data_clean.index.intersection(labels_clean.index)
            X_train = train_data_clean.loc[common_idx]
            y_train = labels_clean.loc[common_idx]
            
            if len(X_train) < 100:
                logger.warning(f"Datos de entrenamiento insuficientes")
                return None
            
            # 5. Entrenar modelo
            self.ensemble_model.train(X_train, y_train)
            
            # 6. Predecir en test
            X_test = test_data
            predictions, probabilities = self.ensemble_model.predict(X_test)
            
            # 7. Generar señales
            signals = self.signal_generator.generate_signals(
                X_test, predictions, probabilities
            )
            
            # 8. Backtest
            backtest_results = self.backtester.run(X_test, signals)
            
            if backtest_results is None:
                return None
            
            # 9. Resultados
            results = {
                'symbol': symbol,
                'data_points': len(df),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'signals_generated': len(signals[signals['signal'] != 0]),
                'backtest': backtest_results['metrics'],
                'model_info': {
                    'models_trained': list(self.ensemble_model.models.keys()),
                    'features_used': len(self.ensemble_model.feature_columns)
                }
            }
            
            logger.info(f"\n📊 Resultados {symbol}:")
            for key, value in results['backtest'].items():
                logger.info(f"  {key}: {value}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error procesando {symbol}: {e}")
            return None

def main():
    """Función principal"""
    logger.info("🚀 Iniciando Sistema de Trading Cuantitativo")
    logger.info("=" * 60)
    
    # Configuración
    config = Config()
    
    # Limitar symbols para GitHub Actions
    symbols = config.SYMBOLS[:config.MAX_SYMBOLS]
    
    # Sistema
    system = TradingSystem(config)
    
    # Procesar cada symbol
    all_results = {}
    
    for symbol in symbols:
        try:
            result = system.process_symbol(symbol)
            if result:
                all_results[symbol] = result
        except Exception as e:
            logger.error(f"Error crítico con {symbol}: {e}")
            continue
    
    # Resumen final
    logger.info("\n" + "=" * 60)
    logger.info("📈 RESUMEN FINAL")
    logger.info("=" * 60)
    
    if not all_results:
        logger.warning("No se obtuvieron resultados")
        return
    
    # Estadísticas generales
    total_trades = sum(r['backtest']['total_trades'] for r in all_results.values())
    avg_return = np.mean([r['backtest']['total_return'] for r in all_results.values()])
    avg_sharpe = np.mean([r['backtest']['sharpe'] for r in all_results.values()])
    
    logger.info(f"Símbolos procesados: {len(all_results)}/{len(symbols)}")
    logger.info(f"Operaciones totales: {total_trades}")
    logger.info(f"Retorno promedio: {avg_return:.2f}%")
    logger.info(f"Sharpe promedio: {avg_sharpe:.2f}")
    
    # Mejores símbolos
    if all_results:
        best_symbol = max(all_results.items(), key=lambda x: x[1]['backtest']['total_return'])
        worst_symbol = min(all_results.items(), key=lambda x: x[1]['backtest']['total_return'])
        
        logger.info(f"\n🏆 Mejor: {best_symbol[0]} ({best_symbol[1]['backtest']['total_return']:.2f}%)")
        logger.info(f"📉 Peor: {worst_symbol[0]} ({worst_symbol[1]['backtest']['total_return']:.2f}%)")
    
    # Guardar resultados en GitHub Actions
    if 'GITHUB_ACTIONS' in os.environ:
        output_file = 'trading_results.json'
        with open(output_file, 'w') as f:
            # Convertir a tipos nativos de Python
            def convert(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                return obj
            
            json.dump(all_results, f, default=convert, indent=2)
        logger.info(f"\n💾 Resultados guardados en {output_file}")
    
    logger.info("\n✅ Proceso completado exitosamente")
    
    return all_results

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⏹️ Proceso interrumpido")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Error fatal: {e}", exc_info=True)
        sys.exit(1)
