"""Simple long/flat backtest of any strategy signal, one contract at a time.

A *strategy* is any object exposing ``name``, ``label``, ``rule``,
``min_periods`` and ``position(close) -> pd.Series`` (a long/flat 0/1 series).
The concrete strategies live here; the signals they wrap live in
``indicators``.

P&L is measured in **price points** (``position × close.diff()``) rather than
percentage returns. A calendar spread, crack or butterfly can be negative and
cross zero, so percentage returns are meaningless for them — points work
uniformly for every contract family (outrights included).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from crudewatch.infra.constants import FAMILY_LABELS
from backtesting.backtest.indicators import (
    adx_signal,
    aroon_signal,
    awesome_signal,
    bollinger_breakout_signal,
    bollinger_signal,
    cci_divergence_signal,
    cci_reversion_signal,
    cci_signal,
    cmo_signal,
    coppock_signal,
    crossover_signal,
    dema_cross_signal,
    donchian_signal,
    fisher_signal,
    gmma_signal,
    hma_signal,
    ichimoku_signal,
    kama_signal,
    keltner_breakout_signal,
    keltner_reversion_signal,
    kst_signal,
    linreg_signal,
    macd_divergence_signal,
    macd_signal,
    momentum_signal,
    regime_switch_signal,
    roc_signal,
    rsi_bollinger_signal,
    rsi_divergence_signal,
    rsi_signal,
    rsi_trend_signal,
    sma_trend_signal,
    stc_signal,
    stoch_divergence_signal,
    stoch_reversion_signal,
    stoch_rsi_signal,
    stochastic_signal,
    supertrend_signal,
    williams_reversion_signal,
    tema_cross_signal,
    triple_ma_signal,
    trix_signal,
    tsi_signal,
    williams_r_signal,
    zlema_cross_signal,
    zscore_signal,
)

TRADING_DAYS = 252

# Per-contract metric columns, in report order.
METRIC_COLUMNS = [
    "n_obs", "n_trades", "win_rate", "pnl_total", "pnl_annualized",
    "sharpe", "max_drawdown", "profit_factor", "expectancy",
    "avg_win", "avg_loss", "exposure",
]


class Strategy(Protocol):
    """A long/flat trading rule. See the concrete dataclasses below."""

    @property
    def name(self) -> str: ...
    @property
    def label(self) -> str: ...
    @property
    def rule(self) -> str: ...
    @property
    def entry_rule(self) -> str: ...
    @property
    def exit_rule(self) -> str: ...
    @property
    def min_periods(self) -> int: ...
    def position(self, close: pd.Series) -> pd.Series: ...


@dataclass(frozen=True)
class MaCrossover:
    """Fast/slow moving-average crossover (SMA or EMA)."""
    fast: int
    slow: int
    kind: str = "ema"

    @property
    def name(self) -> str:
        return f"{self.kind.upper()}_{self.fast}_{self.slow}"

    @property
    def label(self) -> str:
        return f"{self.kind.upper()} {self.fast}/{self.slow}"

    @property
    def rule(self) -> str:
        return (
            f"Largo mientras la media {self.kind.upper()} r\u00e1pida ({self.fast}) "
            f"est\u00e1 por encima de la lenta ({self.slow}); plano en caso contrario."
        )

    @property
    def entry_rule(self) -> str:
        return (
            f"Se abre un largo cuando la media {self.kind.upper()}({self.fast}) "
            f"cruza por encima de la {self.kind.upper()}({self.slow})."
        )

    @property
    def exit_rule(self) -> str:
        return (
            f"Se cierra (a plano) cuando la {self.kind.upper()}({self.fast}) "
            f"vuelve a cruzar por debajo de la {self.kind.upper()}({self.slow})."
        )

    @property
    def min_periods(self) -> int:
        return self.slow

    def position(self, close: pd.Series) -> pd.Series:
        return crossover_signal(close, self.fast, self.slow, self.kind)


@dataclass(frozen=True)
class Rsi:
    """RSI oversold-entry / overbought-exit."""
    period: int = 14
    low: float = 30.0
    high: float = 70.0

    @property
    def name(self) -> str:
        return f"RSI_{self.period}"

    @property
    def label(self) -> str:
        return f"RSI {self.period}"

    @property
    def rule(self) -> str:
        return (
            f"Largo cuando el RSI({self.period}) cruza al alza {self.low:.0f} "
            f"(sobreventa); plano cuando cruza al alza {self.high:.0f} (sobrecompra)."
        )

    @property
    def entry_rule(self) -> str:
        return (
            f"Se abre un largo cuando el RSI({self.period}) cruza al alza el nivel "
            f"{self.low:.0f}, es decir sale de zona de sobreventa (rebote)."
        )

    @property
    def exit_rule(self) -> str:
        return (
            f"Se cierra (a plano) cuando el RSI({self.period}) cruza al alza el nivel "
            f"{self.high:.0f}, es decir entra en zona de sobrecompra."
        )

    @property
    def min_periods(self) -> int:
        return self.period + 1

    def position(self, close: pd.Series) -> pd.Series:
        return rsi_signal(close, self.period, self.low, self.high)


@dataclass(frozen=True)
class Macd:
    """MACD line vs. signal line."""
    fast: int = 12
    slow: int = 26
    signal: int = 9

    @property
    def name(self) -> str:
        return f"MACD_{self.fast}_{self.slow}_{self.signal}"

    @property
    def label(self) -> str:
        return f"MACD {self.fast}/{self.slow}/{self.signal}"

    @property
    def rule(self) -> str:
        return (
            f"Largo cuando la l\u00ednea MACD ({self.fast}/{self.slow}) est\u00e1 por "
            f"encima de su se\u00f1al ({self.signal}); plano en caso contrario."
        )

    @property
    def entry_rule(self) -> str:
        return (
            f"Se abre un largo cuando la l\u00ednea MACD ({self.fast}\u2013{self.slow}) "
            f"cruza por encima de su se\u00f1al (EMA {self.signal} del MACD): el "
            f"histograma pasa a positivo."
        )

    @property
    def exit_rule(self) -> str:
        return (
            f"Se cierra (a plano) cuando la l\u00ednea MACD cruza por debajo de su "
            f"se\u00f1al ({self.signal}): el histograma pasa a negativo."
        )

    @property
    def min_periods(self) -> int:
        return self.slow + self.signal

    def position(self, close: pd.Series) -> pd.Series:
        return macd_signal(close, self.fast, self.slow, self.signal)


@dataclass(frozen=True)
class RsiDivergence:
    """Regular price-vs-RSI divergence."""
    period: int = 14
    lookback: int = 20

    @property
    def name(self) -> str:
        return f"RSI_DIV_{self.period}_{self.lookback}"

    @property
    def label(self) -> str:
        return f"RSI divergence {self.period}/{self.lookback}"

    @property
    def rule(self) -> str:
        return (
            f"Largo tras divergencia alcista (precio por debajo de hace "
            f"{self.lookback} barras pero el RSI({self.period}) por encima); "
            f"plano tras divergencia bajista."
        )

    @property
    def entry_rule(self) -> str:
        return (
            f"Se abre un largo al detectar divergencia alcista: el precio est\u00e1 por "
            f"debajo de su nivel de hace {self.lookback} barras, pero el RSI({self.period}) "
            f"est\u00e1 por encima del suyo (el precio cae pero el momentum mejora)."
        )

    @property
    def exit_rule(self) -> str:
        return (
            f"Se cierra (a plano) al detectar divergencia bajista: el precio por encima "
            f"de hace {self.lookback} barras pero el RSI({self.period}) por debajo."
        )

    @property
    def min_periods(self) -> int:
        return self.period + self.lookback

    def position(self, close: pd.Series) -> pd.Series:
        return rsi_divergence_signal(close, self.period, self.lookback)


@dataclass(frozen=True)
class MacdDivergence:
    """Regular price-vs-MACD-line divergence."""
    fast: int = 12
    slow: int = 26
    signal: int = 9
    lookback: int = 20

    @property
    def name(self) -> str:
        return f"MACD_DIV_{self.fast}_{self.slow}_{self.signal}_{self.lookback}"

    @property
    def label(self) -> str:
        return f"MACD divergence {self.fast}/{self.slow}/{self.signal} (lb {self.lookback})"

    @property
    def rule(self) -> str:
        return (
            f"Largo tras divergencia alcista (precio por debajo de hace "
            f"{self.lookback} barras pero la l\u00ednea MACD por encima); "
            f"plano tras divergencia bajista."
        )

    @property
    def entry_rule(self) -> str:
        return (
            f"Se abre un largo al detectar divergencia alcista: el precio est\u00e1 por "
            f"debajo de su nivel de hace {self.lookback} barras, pero la l\u00ednea MACD "
            f"({self.fast}\u2013{self.slow}) est\u00e1 por encima de la suya."
        )

    @property
    def exit_rule(self) -> str:
        return (
            f"Se cierra (a plano) al detectar divergencia bajista: el precio por encima "
            f"de hace {self.lookback} barras pero la l\u00ednea MACD por debajo."
        )

    @property
    def min_periods(self) -> int:
        return self.slow + self.signal + self.lookback

    def position(self, close: pd.Series) -> pd.Series:
        return macd_divergence_signal(close, self.fast, self.slow, self.signal, self.lookback)


@dataclass(frozen=True)
class SignalStrategy:
    """A generic strategy: metadata plus a ``close -> position`` signal function.

    Used for the trend/momentum indicators, whose only difference is the signal
    they wrap and the text that describes them.
    """
    name: str
    label: str
    rule: str
    entry_rule: str
    exit_rule: str
    min_periods: int
    fn: Callable[[pd.Series], pd.Series]

    def position(self, close: pd.Series) -> pd.Series:
        return self.fn(close)


# Backwards-compatible alias: the original moving-average "Spec".
Spec = MaCrossover

# Moving-average crossovers (one report each).
DEFAULT_SPECS: list[Strategy] = [
    MaCrossover(9, 21, "ema"),
    MaCrossover(5, 13, "ema"),
    MaCrossover(12, 26, "ema"),
    MaCrossover(20, 50, "sma"),
]

# Trend-following and momentum indicators (close-based; run on every family).
TREND_MOMENTUM: list[Strategy] = [
    SignalStrategy(
        "DONCHIAN_20_10", "Donchian 20/10",
        "Ruptura de canal: largo al superar el m\u00e1ximo de 20 barras, plano al perder el m\u00ednimo de 10.",
        "Se abre un largo cuando el cierre supera el m\u00e1ximo de las 20 barras anteriores (ruptura al alza).",
        "Se cierra (a plano) cuando el cierre cae por debajo del m\u00ednimo de las 10 barras anteriores.",
        20, partial(donchian_signal, entry=20, exit_len=10),
    ),
    SignalStrategy(
        "SMA_TREND_50", "SMA trend 50",
        "Filtro de tendencia: largo mientras el precio est\u00e1 por encima de su SMA(50).",
        "Se abre un largo cuando el cierre cruza por encima de la SMA(50).",
        "Se cierra (a plano) cuando el cierre cruza por debajo de la SMA(50).",
        50, partial(sma_trend_signal, window=50),
    ),
    SignalStrategy(
        "TRIPLE_EMA_8_21_55", "Triple EMA 8/21/55",
        "Alineaci\u00f3n de medias: largo cuando EMA(8) > EMA(21) > EMA(55).",
        "Se abre un largo cuando las tres EMAs se apilan al alza: EMA(8) > EMA(21) > EMA(55).",
        "Se cierra (a plano) cuando se rompe la alineaci\u00f3n (alguna media r\u00e1pida cae por debajo de una m\u00e1s lenta).",
        55, partial(triple_ma_signal, fast=8, mid=21, slow=55, kind="ema"),
    ),
    SignalStrategy(
        "LINREG_SLOPE_20", "Pendiente regresi\u00f3n 20",
        "Largo mientras la pendiente de la regresi\u00f3n lineal de 20 barras es positiva.",
        "Se abre un largo cuando la pendiente de la recta de regresi\u00f3n (20 barras) pasa a positiva.",
        "Se cierra (a plano) cuando esa pendiente pasa a negativa (la tendencia se gira a la baja).",
        20, partial(linreg_signal, window=20),
    ),
    SignalStrategy(
        "AROON_25", "Aroon 25",
        "Largo mientras Aroon-Up (barras desde el m\u00e1ximo) supera a Aroon-Down (barras desde el m\u00ednimo).",
        "Se abre un largo cuando Aroon-Up cruza por encima de Aroon-Down (m\u00e1ximos m\u00e1s recientes que los m\u00ednimos).",
        "Se cierra (a plano) cuando Aroon-Down supera a Aroon-Up.",
        26, partial(aroon_signal, window=25),
    ),
    SignalStrategy(
        "TRIX_15_9", "TRIX 15/9",
        "Largo mientras la l\u00ednea TRIX (EMA triple, 15) est\u00e1 por encima de su se\u00f1al (9).",
        "Se abre un largo cuando la l\u00ednea TRIX cruza por encima de su se\u00f1al (EMA 9 del TRIX).",
        "Se cierra (a plano) cuando la l\u00ednea TRIX cruza por debajo de su se\u00f1al.",
        45, partial(trix_signal, window=15, signal=9),
    ),
    SignalStrategy(
        "MOMENTUM_20", "Momentum 20",
        "Momentum absoluto: largo mientras el precio es mayor que hace 20 barras.",
        "Se abre un largo cuando el precio supera su nivel de hace 20 barras (momentum positivo).",
        "Se cierra (a plano) cuando el precio cae por debajo de su nivel de hace 20 barras.",
        20, partial(momentum_signal, window=20),
    ),
    SignalStrategy(
        "STOCH_14_3", "Estoc\u00e1stico 14/3",
        "Estoc\u00e1stico sobre cierre: largo mientras %K est\u00e1 por encima de su media %D.",
        "Se abre un largo cuando %K cruza por encima de %D (momentum al alza).",
        "Se cierra (a plano) cuando %K cruza por debajo de %D.",
        17, partial(stochastic_signal, k=14, d=3),
    ),
    SignalStrategy(
        "CCI_20", "CCI 20",
        "Commodity Channel Index sobre cierre: largo mientras el CCI(20) es positivo.",
        "Se abre un largo cuando el CCI(20) cruza por encima de 0.",
        "Se cierra (a plano) cuando el CCI(20) cruza por debajo de 0.",
        20, partial(cci_signal, window=20),
    ),
    SignalStrategy(
        "AWESOME_5_34", "Awesome Oscillator 5/34",
        "Largo mientras el oscilador (SMA5 - SMA34 del cierre) es positivo.",
        "Se abre un largo cuando el Awesome Oscillator (SMA5 - SMA34) cruza por encima de 0.",
        "Se cierra (a plano) cuando el Awesome Oscillator cruza por debajo de 0.",
        34, partial(awesome_signal, fast=5, slow=34),
    ),
]

# Extra trend / momentum / mean-reversion indicators.
EXTRA_INDICATORS: list[Strategy] = [
    SignalStrategy(
        "KAMA_10_2_30", "KAMA 10/2/30",
        "Media adaptativa de Kaufman: se acelera en tendencia y se frena en el ruido. Largo mientras el precio est\u00e1 por encima de la KAMA.",
        "Se abre un largo cuando el cierre cruza por encima de la KAMA (media adaptativa).",
        "Se cierra (a plano) cuando el cierre cruza por debajo de la KAMA.",
        11, partial(kama_signal, er_window=10, fast=2, slow=30),
    ),
    SignalStrategy(
        "DEMA_12_26", "DEMA 12/26",
        "Cruce de dobles-EMA (menos retardo que la EMA normal): largo si DEMA(12) > DEMA(26).",
        "Se abre un largo cuando la DEMA(12) cruza por encima de la DEMA(26).",
        "Se cierra (a plano) cuando la DEMA(12) cruza por debajo de la DEMA(26).",
        26, partial(dema_cross_signal, fast=12, slow=26),
    ),
    SignalStrategy(
        "HMA_20", "Hull MA 20",
        "Media de Hull (r\u00e1pida y suave): largo mientras el precio est\u00e1 por encima de la HMA(20).",
        "Se abre un largo cuando el cierre cruza por encima de la HMA(20).",
        "Se cierra (a plano) cuando el cierre cruza por debajo de la HMA(20).",
        25, partial(hma_signal, window=20),
    ),
    SignalStrategy(
        "COPPOCK", "Coppock",
        "Momentum de largo plazo (suma de dos look-backs suavizada): largo mientras la curva de Coppock es positiva.",
        "Se abre un largo cuando la curva de Coppock cruza por encima de 0.",
        "Se cierra (a plano) cuando la curva de Coppock cruza por debajo de 0.",
        24, partial(coppock_signal, roc_long=14, roc_short=11, smooth=10),
    ),
    SignalStrategy(
        "TSI_25_13_7", "TSI 25/13/7",
        "True Strength Index (momentum doblemente suavizado): largo mientras el TSI est\u00e1 por encima de su se\u00f1al.",
        "Se abre un largo cuando la l\u00ednea TSI cruza por encima de su se\u00f1al (EMA 7).",
        "Se cierra (a plano) cuando la l\u00ednea TSI cruza por debajo de su se\u00f1al.",
        45, partial(tsi_signal, long_window=25, short_window=13, signal=7),
    ),
    SignalStrategy(
        "CMO_20", "CMO 20",
        "Chande Momentum Oscillator: largo mientras el CMO(20) es positivo.",
        "Se abre un largo cuando el CMO(20) cruza por encima de 0.",
        "Se cierra (a plano) cuando el CMO(20) cruza por debajo de 0.",
        21, partial(cmo_signal, window=20),
    ),
    SignalStrategy(
        "BOLLINGER_20_2", "Bollinger 20/2 (reversi\u00f3n)",
        "Reversi\u00f3n a la media: largo al cerrar por debajo de la banda inferior (media - 2\u03c3); plano al volver a la media.",
        "Se abre un largo cuando el cierre cae por debajo de la banda inferior de Bollinger (media - 2\u03c3).",
        "Se cierra (a plano) cuando el cierre vuelve a su media m\u00f3vil de 20.",
        20, partial(bollinger_signal, window=20, num_std=2.0),
    ),
    SignalStrategy(
        "ZSCORE_20", "Z-score 20 (reversi\u00f3n)",
        "Reversi\u00f3n a la media: largo cuando el z-score cae por debajo de -1,5; plano al volver a la media (z\u22650).",
        "Se abre un largo cuando el z-score de 20 barras cae por debajo de -1,5 (precio por debajo de su media).",
        "Se cierra (a plano) cuando el z-score vuelve a 0 (precio de nuevo en su media).",
        20, partial(zscore_signal, window=20, entry=1.5, exit_z=0.0),
    ),
]

# Indicators widely used on WTI / energy futures (close-based versions so they
# run on every family, including the synthetic spreads with no high/low).
ENERGY_INDICATORS: list[Strategy] = [
    SignalStrategy(
        "ADX_14", "ADX/DMI 14",
        "Fuerza y direcci\u00f3n de tendencia (ADX/DMI): opera solo con tendencia fuerte (ADX>25).",
        "Se abre un largo cuando +DI est\u00e1 por encima de -DI y el ADX(14) supera 25 (tendencia alcista con fuerza).",
        "Se cierra (a plano) cuando +DI cae por debajo de -DI o el ADX cae por debajo de 25 (tendencia d\u00e9bil).",
        28, partial(adx_signal, window=14, threshold=25.0),
    ),
    SignalStrategy(
        "SUPERTREND_10_3", "Supertrend 10/3",
        "Supertrend (ATR close-to-close): bandera de tendencia binaria; largo mientras la tendencia es alcista.",
        "Se abre un largo cuando el precio cruza por encima de la banda Supertrend y la bandera pasa a alcista.",
        "Se cierra (a plano) cuando el precio cruza por debajo de la banda Supertrend (bandera bajista).",
        15, partial(supertrend_signal, window=10, mult=3.0),
    ),
    SignalStrategy(
        "KELTNER_20_2", "Keltner breakout 20/2",
        "Ruptura del canal de Keltner (EMA20 \u00b1 2\u00b7ATR): largo al superar la banda superior, plano al volver a la media.",
        "Se abre un largo cuando el cierre supera la banda superior de Keltner (EMA20 + 2\u00b7ATR).",
        "Se cierra (a plano) cuando el cierre vuelve por debajo de la l\u00ednea media (EMA20).",
        22, partial(keltner_breakout_signal, window=20, mult=2.0, atr_window=10),
    ),
    SignalStrategy(
        "BOLL_BREAKOUT_20_2", "Bollinger breakout 20/2",
        "Ruptura de Bollinger (momentum): largo al superar la banda superior (media + 2\u03c3), plano al volver a la media.",
        "Se abre un largo cuando el cierre supera la banda superior de Bollinger (media + 2\u03c3).",
        "Se cierra (a plano) cuando el cierre vuelve por debajo de su media m\u00f3vil de 20.",
        20, partial(bollinger_breakout_signal, window=20, num_std=2.0),
    ),
    SignalStrategy(
        "TEMA_12_26", "TEMA 12/26",
        "Cruce de triples-EMA (retardo m\u00ednimo): largo si TEMA(12) > TEMA(26).",
        "Se abre un largo cuando la TEMA(12) cruza por encima de la TEMA(26).",
        "Se cierra (a plano) cuando la TEMA(12) cruza por debajo de la TEMA(26).",
        30, partial(tema_cross_signal, fast=12, slow=26),
    ),
    SignalStrategy(
        "ZLEMA_9_21", "ZLEMA 9/21",
        "Cruce de EMAs de retardo cero: largo si ZLEMA(9) > ZLEMA(21).",
        "Se abre un largo cuando la ZLEMA(9) cruza por encima de la ZLEMA(21).",
        "Se cierra (a plano) cuando la ZLEMA(9) cruza por debajo de la ZLEMA(21).",
        32, partial(zlema_cross_signal, fast=9, slow=21),
    ),
    SignalStrategy(
        "ICHIMOKU_9_26", "Ichimoku TK 9/26",
        "Cruce Tenkan/Kijun (Ichimoku, close-based): largo si la l\u00ednea de conversi\u00f3n (9) est\u00e1 sobre la base (26).",
        "Se abre un largo cuando la Tenkan(9) cruza por encima de la Kijun(26).",
        "Se cierra (a plano) cuando la Tenkan(9) cruza por debajo de la Kijun(26).",
        26, partial(ichimoku_signal, tenkan=9, kijun=26),
    ),
    SignalStrategy(
        "GMMA", "Guppy MMA",
        "Guppy: largo cuando el grupo de EMAs cortas (3-15) est\u00e1 en media por encima del grupo de largas (30-60).",
        "Se abre un largo cuando la media del grupo corto de EMAs cruza por encima de la media del grupo largo.",
        "Se cierra (a plano) cuando la media del grupo corto cae por debajo de la del grupo largo.",
        60, gmma_signal,
    ),
    SignalStrategy(
        "STC_23_50_10", "Schaff Trend Cycle",
        "Schaff Trend Cycle (MACD + doble estoc\u00e1stico): largo mientras el STC est\u00e1 por encima de 50.",
        "Se abre un largo cuando el Schaff Trend Cycle cruza por encima de 50.",
        "Se cierra (a plano) cuando el Schaff Trend Cycle cruza por debajo de 50.",
        70, partial(stc_signal, fast=23, slow=50, cycle=10),
    ),
    SignalStrategy(
        "STOCH_RSI_14", "Stochastic RSI 14",
        "Estoc\u00e1stico del RSI: largo mientras %K est\u00e1 por encima de %D.",
        "Se abre un largo cuando %K del Stochastic RSI cruza por encima de %D.",
        "Se cierra (a plano) cuando %K cruza por debajo de %D.",
        35, partial(stoch_rsi_signal, rsi_period=14, stoch=14, k=3, d=3),
    ),
    SignalStrategy(
        "WILLIAMS_R_14", "Williams %R 14",
        "Williams %R (close-based): largo mientras %R est\u00e1 en la mitad superior del rango (por encima de -50).",
        "Se abre un largo cuando el %R(14) cruza por encima de -50 (precio en la parte alta del rango).",
        "Se cierra (a plano) cuando el %R(14) cae por debajo de -50.",
        14, partial(williams_r_signal, window=14),
    ),
    SignalStrategy(
        "FISHER_9", "Fisher Transform 9",
        "Fisher Transform: largo mientras la l\u00ednea Fisher est\u00e1 por encima de su valor previo (gatillo).",
        "Se abre un largo cuando la l\u00ednea Fisher cruza por encima de su gatillo (valor de la barra anterior).",
        "Se cierra (a plano) cuando la l\u00ednea Fisher cruza por debajo de su gatillo.",
        11, partial(fisher_signal, window=9),
    ),
    SignalStrategy(
        "ROC_12", "ROC 12",
        "Rate of Change (absoluto): largo mientras el precio es mayor que hace 12 barras.",
        "Se abre un largo cuando el precio supera su nivel de hace 12 barras (ROC positivo).",
        "Se cierra (a plano) cuando el precio cae por debajo de su nivel de hace 12 barras.",
        12, partial(roc_signal, window=12),
    ),
    SignalStrategy(
        "RSI_TREND_50", "RSI trend >50",
        "RSI como filtro de tendencia: largo mientras el RSI(14) est\u00e1 por encima de 50.",
        "Se abre un largo cuando el RSI(14) cruza por encima de 50 (momentum a favor del alza).",
        "Se cierra (a plano) cuando el RSI(14) cruza por debajo de 50.",
        15, partial(rsi_trend_signal, period=14, level=50.0),
    ),
    SignalStrategy(
        "KST", "KST",
        "Know Sure Thing (momentum compuesto de cuatro look-backs): largo mientras el KST est\u00e1 sobre su se\u00f1al.",
        "Se abre un largo cuando la l\u00ednea KST cruza por encima de su se\u00f1al (media de 9).",
        "Se cierra (a plano) cuando la l\u00ednea KST cruza por debajo de su se\u00f1al.",
        45, partial(kst_signal, signal=9),
    ),
]

# Mean-reversion / divergence "family": variants and alternatives of the rules
# that scored best (RSI, Bollinger, Z-score and divergences).
REVERSION_FAMILY: list[Strategy] = [
    # -- RSI variants (distinct periods/thresholds) --------------------------
    Rsi(2, low=5.0, high=50.0),
    Rsi(7, low=25.0, high=70.0),
    Rsi(21, low=35.0, high=65.0),
    SignalStrategy(
        "RSI_14_20_80", "RSI 14 (bandas 20/80)",
        "RSI de bandas anchas (el crudo aguanta mucho en sobrecompra): largo saliendo de 20, plano en 80.",
        "Se abre un largo cuando el RSI(14) cruza al alza el nivel 20 (rebote desde sobreventa profunda).",
        "Se cierra (a plano) cuando el RSI(14) cruza al alza el nivel 80 (sobrecompra extrema).",
        15, partial(rsi_signal, period=14, low=20.0, high=80.0),
    ),
    # -- Bollinger reversion variants ---------------------------------------
    SignalStrategy(
        "BOLLINGER_50_2", "Bollinger 50/2 (reversi\u00f3n)",
        "Bollinger de base 50 (\u00f3ptimo citado para crudo): largo bajo la banda inferior, plano en la media.",
        "Se abre un largo cuando el cierre cae por debajo de la banda inferior (media50 \u2212 2\u03c3).",
        "Se cierra (a plano) cuando el cierre vuelve por encima de su media m\u00f3vil de 50.",
        50, partial(bollinger_signal, window=50, num_std=2.0),
    ),
    SignalStrategy(
        "BOLLINGER_10_1_5", "Bollinger 10/1.5 (reversi\u00f3n r\u00e1pida)",
        "Bollinger corto y estrecho: capta reversiones r\u00e1pidas; largo bajo la banda inferior, plano en la media.",
        "Se abre un largo cuando el cierre cae por debajo de la banda inferior (media10 \u2212 1.5\u03c3).",
        "Se cierra (a plano) cuando el cierre vuelve por encima de su media m\u00f3vil de 10.",
        10, partial(bollinger_signal, window=10, num_std=1.5),
    ),
    SignalStrategy(
        "BOLLINGER_20_2_5", "Bollinger 20/2.5 (reversi\u00f3n extrema)",
        "Bollinger de bandas m\u00e1s anchas: solo entra en extremos; largo bajo la banda inferior, plano en la media.",
        "Se abre un largo cuando el cierre cae por debajo de la banda inferior (media20 \u2212 2.5\u03c3).",
        "Se cierra (a plano) cuando el cierre vuelve por encima de su media m\u00f3vil de 20.",
        20, partial(bollinger_signal, window=20, num_std=2.5),
    ),
    # -- Z-score reversion variants -----------------------------------------
    SignalStrategy(
        "ZSCORE_10", "Z-score 10 (reversi\u00f3n r\u00e1pida)",
        "Z-score de ventana corta: largo cuando el z cae por debajo de \u22122.0, plano al volver a 0.",
        "Se abre un largo cuando el z-score de 10 barras cae por debajo de \u22122.0 (precio muy por debajo de su media).",
        "Se cierra (a plano) cuando el z-score vuelve a 0 (precio de nuevo en su media).",
        10, partial(zscore_signal, window=10, entry=2.0, exit_z=0.0),
    ),
    SignalStrategy(
        "ZSCORE_50", "Z-score 50 (reversi\u00f3n lenta)",
        "Z-score de ventana larga: largo cuando el z cae por debajo de \u22121.5, plano al volver a 0.",
        "Se abre un largo cuando el z-score de 50 barras cae por debajo de \u22121.5.",
        "Se cierra (a plano) cuando el z-score vuelve a 0.",
        50, partial(zscore_signal, window=50, entry=1.5, exit_z=0.0),
    ),
    SignalStrategy(
        "ZSCORE_20_DEEP", "Z-score 20 (extremo 2.5)",
        "Z-score que solo entra en extremos: largo por debajo de \u22122.5, plano al recuperar \u22120.5.",
        "Se abre un largo cuando el z-score de 20 barras cae por debajo de \u22122.5 (desviaci\u00f3n extrema).",
        "Se cierra (a plano) cuando el z-score recupera hasta \u22120.5.",
        20, partial(zscore_signal, window=20, entry=2.5, exit_z=0.5),
    ),
    # -- Divergence variants -------------------------------------------------
    RsiDivergence(14, 40),
    RsiDivergence(7, 20),
    MacdDivergence(5, 35, 5, 20),
    SignalStrategy(
        "CCI_DIV_20_20", "CCI divergence 20/20",
        "Divergencia precio vs CCI: largo tras divergencia alcista, plano tras bajista.",
        "Se abre un largo cuando el precio est\u00e1 por debajo de hace 20 barras pero el CCI(20) est\u00e1 por encima.",
        "Se cierra (a plano) cuando el precio est\u00e1 por encima de hace 20 barras pero el CCI(20) por debajo.",
        40, partial(cci_divergence_signal, window=20, lookback=20),
    ),
    SignalStrategy(
        "STOCH_DIV_14_20", "Stochastic divergence 14/20",
        "Divergencia precio vs %K estoc\u00e1stico: largo tras divergencia alcista, plano tras bajista.",
        "Se abre un largo cuando el precio est\u00e1 por debajo de hace 20 barras pero el %K(14) est\u00e1 por encima.",
        "Se cierra (a plano) cuando el precio est\u00e1 por encima de hace 20 barras pero el %K(14) por debajo.",
        34, partial(stoch_divergence_signal, k=14, lookback=20),
    ),
    # -- New mean-reversion oscillators -------------------------------------
    SignalStrategy(
        "STOCH_REV_14_3", "Stochastic reversi\u00f3n 14/3",
        "Estoc\u00e1stico de reversi\u00f3n: largo al rebotar de sobreventa (20), plano al llegar a sobrecompra (80).",
        "Se abre un largo cuando el %D estoc\u00e1stico cruza al alza el nivel 20 (sale de sobreventa).",
        "Se cierra (a plano) cuando el %D cruza al alza el nivel 80 (entra en sobrecompra).",
        18, partial(stoch_reversion_signal, k=14, d=3, low=20.0, high=80.0),
    ),
    SignalStrategy(
        "CCI_REV_20", "CCI reversi\u00f3n 20 (\u00b1100)",
        "CCI de reversi\u00f3n: largo al rebotar de \u2212100, plano al alcanzar +100.",
        "Se abre un largo cuando el CCI(20) cruza al alza el nivel \u2212100 (rebote desde sobreventa).",
        "Se cierra (a plano) cuando el CCI(20) cruza al alza el nivel +100 (sobrecompra).",
        20, partial(cci_reversion_signal, window=20, low=-100.0, high=100.0),
    ),
    SignalStrategy(
        "WILLIAMS_REV_14", "Williams %R reversi\u00f3n 14",
        "Williams %R de reversi\u00f3n: largo al rebotar de \u221280, plano al llegar a \u221220.",
        "Se abre un largo cuando el %R(14) cruza al alza el nivel \u221280 (sale de sobreventa).",
        "Se cierra (a plano) cuando el %R(14) cruza al alza el nivel \u221220 (sobrecompra).",
        14, partial(williams_reversion_signal, window=14, low=-80.0, high=-20.0),
    ),
    SignalStrategy(
        "KELTNER_REV_20_2", "Keltner reversi\u00f3n 20/2",
        "Keltner de reversi\u00f3n (ATR close-to-close): largo bajo la banda inferior, plano al volver a la media.",
        "Se abre un largo cuando el cierre cae por debajo de la banda inferior de Keltner (EMA20 \u2212 2\u00b7ATR).",
        "Se cierra (a plano) cuando el cierre vuelve por encima de la l\u00ednea media (EMA20).",
        22, partial(keltner_reversion_signal, window=20, mult=2.0, atr_window=10),
    ),
    SignalStrategy(
        "RSI_BOLL_CONFLUENCE", "RSI + Bollinger (confluencia)",
        "Reversi\u00f3n por confluencia: largo solo si el precio est\u00e1 bajo la banda inferior de Bollinger y adem\u00e1s el RSI est\u00e1 en sobreventa.",
        "Se abre un largo cuando el cierre cae por debajo de la banda inferior (media20 \u2212 2\u03c3) y el RSI(14) est\u00e1 por debajo de 35.",
        "Se cierra (a plano) cuando el cierre vuelve por encima de su media m\u00f3vil de 20.",
        20, partial(rsi_bollinger_signal, window=20, num_std=2.0, rsi_period=14, rsi_low=35.0),
    ),
]

# Regime-switching meta-strategy (trend vs. mean-reversion via Efficiency Ratio).
REGIME: list[Strategy] = [
    SignalStrategy(
        "REGIME_SWITCH", "Cambio de r\u00e9gimen (ER)",
        "Detecta el r\u00e9gimen con el Efficiency Ratio (ER, 20) y cambia de t\u00e1ctica: "
        "tendencial si ER\u22650.35, rango si ER<0.35.",
        "En r\u00e9gimen tendencial (ER\u22650.35) se abre largo si la EMA(20) est\u00e1 por encima de la EMA(50). "
        "En r\u00e9gimen de rango (ER<0.35) se abre largo cuando el precio est\u00e1 por debajo de su media de 20 "
        "(z<0), apostando a la reversi\u00f3n.",
        "Se cierra (a plano) cuando la condici\u00f3n del r\u00e9gimen activo deja de cumplirse: en tendencia, "
        "EMA(20)<EMA(50); en rango, el precio vuelve por encima de su media (z\u22650).",
        50, partial(regime_switch_signal, er_window=20, er_threshold=0.35,
                    trend_fast=20, trend_slow=50, z_window=20, kind="ema"),
    ),
]

# Classic momentum / divergence oscillators.
CORE: list[Strategy] = [
    Rsi(14),
    Macd(12, 26, 9),
    RsiDivergence(14, 20),
    MacdDivergence(12, 26, 9, 20),
]

# Every indicator to test by default — one report per entry.
DEFAULT_STRATEGIES: list[Strategy] = [
    *DEFAULT_SPECS,
    *CORE,
    *TREND_MOMENTUM,
    *EXTRA_INDICATORS,
    *ENERGY_INDICATORS,
    *REVERSION_FAMILY,
    *REGIME,
]


def equity_curve(series: pd.DataFrame, strategy: Strategy) -> pd.DataFrame:
    """Return ``date`` and cumulative P&L (points) for one contract."""
    s = series.sort_values("date")
    close = s["close"].reset_index(drop=True)
    pos = strategy.position(close)
    pnl = (pos * close.diff()).fillna(0.0)
    return pd.DataFrame({"date": s["date"].to_numpy(), "equity": pnl.cumsum().to_numpy()})


def backtest_contract(series: pd.DataFrame, strategy: Strategy) -> dict[str, float]:
    """Backtest one contract's close series; return a dict of metrics.

    ``series`` must have ``date`` and ``close`` columns for a single contract.
    """
    s = series.sort_values("date")
    close = s["close"].reset_index(drop=True)
    pos = strategy.position(close)

    pnl = (pos * close.diff()).fillna(0.0)      # per-bar P&L in points
    equity = pnl.cumsum()

    # A trade is a maximal run of long bars; its P&L is the sum over that run.
    in_market = pos > 0
    entries = in_market & ~in_market.shift(1, fill_value=False)
    trade_id = entries.cumsum().where(in_market)
    trade_pnl = pnl.groupby(trade_id).sum()
    trade_pnl = trade_pnl[trade_pnl.index.notna()]

    wins = trade_pnl[trade_pnl > 0]
    losses = trade_pnl[trade_pnl < 0]
    n_trades = int(len(trade_pnl))
    n_obs = int(len(close))

    gross_loss = float(losses.sum())
    pnl_std = float(pnl.std(ddof=0))

    return {
        "n_obs": n_obs,
        "n_trades": n_trades,
        "win_rate": float(len(wins) / n_trades) if n_trades else np.nan,
        "pnl_total": float(equity.iloc[-1]) if n_obs else np.nan,
        "pnl_annualized": float(equity.iloc[-1] * TRADING_DAYS / n_obs) if n_obs else np.nan,
        "sharpe": float(np.sqrt(TRADING_DAYS) * pnl.mean() / pnl_std) if pnl_std > 0 else np.nan,
        "max_drawdown": float((equity - equity.cummax()).min()) if n_obs else np.nan,
        "profit_factor": float(wins.sum() / abs(gross_loss)) if gross_loss < 0 else np.inf if len(wins) else np.nan,
        "expectancy": float(trade_pnl.mean()) if n_trades else np.nan,
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "exposure": float(in_market.mean()) if n_obs else np.nan,
    }


def run_frame(frame: pd.DataFrame, strategy: Strategy) -> pd.DataFrame:
    """Backtest every contract in one family; one metrics row per contract.

    Contracts with fewer observations than the strategy warmup (no full signal)
    are skipped. Rows are sorted by total P&L, best first.
    """
    rows: list[dict[str, float]] = []
    for contract, group in frame.groupby("contract"):
        if int(group["close"].notna().sum()) <= strategy.min_periods:
            continue
        metrics = backtest_contract(group, strategy)
        metrics["contract"] = contract
        rows.append(metrics)

    if not rows:
        return pd.DataFrame(columns=["contract", *METRIC_COLUMNS])

    df = pd.DataFrame(rows)[["contract", *METRIC_COLUMNS]]
    return df.sort_values("pnl_total", ascending=False).reset_index(drop=True)
