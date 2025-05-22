import logging
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ta
import yfinance as yf
from matplotlib.colors import LinearSegmentedColormap

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SETTINGS: Dict[str, float] = {
    "min_history_days": 252 * 2,
    "pullback_threshold": 7.0,
    "support_cluster_pct": 3.0,
    "hist_bins": 50,
    "overlap_pct": 0.7,
    "near_zone_pct": 5.0,
}

# Simplified color scheme - darker colors for stronger signals
BUY_COLORS = {
    "High": "#004d00",    # Dark green
    "Medium": "#00b300",  # Medium green
    "Low": "#99ff99",     # Light green
}

SELL_COLORS = {
    "High": "#990000",    # Dark red
    "Medium": "#ff3333",  # Medium red
    "Low": "#ffb3b3",     # Light red
}


@dataclass
class Zone:
    lower: float
    upper: float
    confidence: str  
    description: str
    zone_type: str
    exact_value: Optional[float] = None
    activity_pct: Optional[float] = None
    additional_info: Optional[str] = None


@dataclass
class CorrectionContext:
    market_correction: bool
    market_correction_magnitude: float
    stock_drawdown: float
    daily_oversold: bool
    weekly_oversold: bool
    current_price: float


@dataclass
class AnalysisResult:
    correction_context: CorrectionContext
    buying_zones: List[Zone]
    selling_zones: List[Zone]
    confidence_score: int
    entry_strategy: str
    reasoning: str
    timestamp: str


class DataLoader:
    def __init__(self, ticker: str, period: str, benchmark: str):
        self.ticker = ticker
        self.period = period
        self.benchmark = benchmark

    @staticmethod
    @lru_cache(maxsize=32)
    def _download(symbol: str, period: str) -> pd.DataFrame:
        df = yf.Ticker(symbol).history(period=period)
        if df.empty:
            raise ValueError(f"No data for {symbol}")
        return df

    def fetch_ticker(self) -> pd.DataFrame:
        df = self._download(self.ticker, self.period)[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df

    def fetch_benchmark(self, index_like: pd.DatetimeIndex) -> pd.DataFrame:
        try:
            bmk = self._download(self.benchmark, self.period)[["Close"]].rename(columns={"Close": "Benchmark_Close"})
        except ValueError:
            bmk = pd.DataFrame(index=index_like, columns=["Benchmark_Close"])
        return bmk.reindex(index_like).ffill().bfill()


class IndicatorCalculator:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.weekly_df: pd.DataFrame = pd.DataFrame()
        self.monthly_df: pd.DataFrame = pd.DataFrame()
        self._run()

    def _run(self):
        self.df["SMA_50"] = ta.trend.sma_indicator(self.df["Close"], 50)
        self.df["SMA_100"] = ta.trend.sma_indicator(self.df["Close"], 100)
        self.df["SMA_200"] = ta.trend.sma_indicator(self.df["Close"], 200)
        if len(self.df) >= 300:
            self.df["SMA_300"] = ta.trend.sma_indicator(self.df["Close"], 300)
        self.df["RSI"] = ta.momentum.RSIIndicator(self.df["Close"]).rsi()
        macd = ta.trend.MACD(self.df["Close"])
        self.df["MACD"] = macd.macd()
        self.df["MACD_signal"] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(self.df["Close"])
        self.df["BB_upper"] = bb.bollinger_hband()
        self.df["BB_lower"] = bb.bollinger_lband()
        self.df["BB_mid"] = bb.bollinger_mavg()
        self.df["Volume_SMA"] = self.df["Volume"].rolling(20).mean()
        atr = ta.volatility.AverageTrueRange(self.df["High"], self.df["Low"], self.df["Close"], 14)
        self.df["ATR"] = atr.average_true_range()
        self.df["ATR_pct"] = self.df["ATR"] / self.df["Close"] * 100
        if len(self.df) >= 10:
            self.weekly_df = self.df.resample("W").agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
            if len(self.weekly_df) >= 10:
                self.weekly_df["RSI"] = ta.momentum.RSIIndicator(self.weekly_df["Close"]).rsi()
        if len(self.df) >= 30:
            self.monthly_df = self.df.resample("ME").agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
            if len(self.monthly_df) >= 10:
                self.monthly_df["RSI"] = ta.momentum.RSIIndicator(self.monthly_df["Close"]).rsi()


class ValueTechnicalAnalyzer:
    def __init__(self, ticker: str, period: str = "5y", benchmark: Optional[str] = None):
        self.ticker = ticker
        self.period = period
        self.benchmark = benchmark or ("^NSEI" if ticker.endswith(".NS") else "^GSPC")
        self.loader = DataLoader(ticker, period, self.benchmark)
        self.df = self.loader.fetch_ticker()
        if len(self.df) < SETTINGS["min_history_days"]:
            raise ValueError("Insufficient history for analysis")
        self.indicators = IndicatorCalculator(self.df)
        self.df = self.indicators.df
        self.benchmark_df = self.loader.fetch_benchmark(self.df.index)
        self._market_regime()
        self._value_zones()

    def _market_regime(self):
        merged = self.df[["Close"]].join(self.benchmark_df, how="left")
        merged["Benchmark_SMA_200"] = ta.trend.sma_indicator(merged["Benchmark_Close"], 200)
        merged["Market_Regime"] = np.where(
            merged["Benchmark_Close"] > merged["Benchmark_SMA_200"], "Bull", "Bear")
        merged["Benchmark_3M_Return"] = merged["Benchmark_Close"].pct_change(63) * 100
        merged["Benchmark_Correlation"] = merged["Close"].rolling(60).corr(merged["Benchmark_Close"])
        merged["Relative_Strength"] = merged["Close"].pct_change(63) / merged["Benchmark_Close"].pct_change(63)
        self.df = self.df.join(merged[["Market_Regime", "Benchmark_3M_Return", "Benchmark_Correlation", "Relative_Strength"]])

    def _value_zones(self):
        self.df["Rolling_High"] = self.df["Close"].rolling(60).max()
        self.df["Pullback_Pct"] = (self.df["Rolling_High"] - self.df["Low"]) / self.df["Rolling_High"] * 100
        
        lows = self.df["Low"].values
        hist, bins = np.histogram(lows, SETTINGS["hist_bins"])
        total_samples = len(self.df)
        activity_pcts = [count / total_samples * 100 for count in hist]
        
        freq_mask = hist > len(self.df) * SETTINGS["support_cluster_pct"] / 100
        clusters = [
            Zone(
                bins[i], 
                bins[i + 1], 
                "Medium" if activity_pcts[i] > 4.5 else "Low", 
                "Price level where buyers stepped in previously", 
                "Support Cluster",
                None,
                activity_pcts[i]
            )
            for i, keep in enumerate(freq_mask) if keep
        ]
        self.support_clusters = sorted(clusters, key=lambda z: z.lower)
        
        move_idx_label = self.df["High"].rolling(500).apply(np.ptp).idxmax()
        
        if pd.isna(move_idx_label):
            self.major_fibs = None
            return

        move_pos  = self.df.index.get_loc(move_idx_label)
        start_pos = max(0, move_pos - 120)

        low_price  = self.df["Low"].iloc[start_pos]
        high_price = self.df["High"].iloc[move_pos]
        rng = high_price - low_price
        
        price_change_pct = (high_price - low_price) / low_price * 100
        
        self.major_fibs = {
            "level_382": {
                "price": high_price - 0.382 * rng,
                "desc": f"Technical support based on past price moves"
            },
            "level_50": {
                "price": high_price - 0.5 * rng,
                "desc": f"Technical support based on past price moves"
            },
            "level_618": {
                "price": high_price - 0.618 * rng,
                "desc": f"Technical support based on past price moves"
            },
            "ext_1414": {
                "price": high_price + 0.414 * rng,
                "desc": "Technical resistance level"
            },
            "ext_1618": {
                "price": high_price + 0.618 * rng,
                "desc": "Technical resistance level"
            }
        }

    def analyse(self, fair_value: Optional[float] = None) -> AnalysisResult:
        latest = self.df.iloc[-1]
        current_price = latest["Close"]
        stock_high = self.df["Close"].rolling(60).max().iloc[-1]
        stock_draw = (stock_high - current_price) / stock_high * 100
        daily_oversold = latest["RSI"] < 40 if not pd.isna(latest["RSI"]) else False
        weekly_oversold = False
        if not self.indicators.weekly_df.empty and "RSI" in self.indicators.weekly_df.columns:
            weekly_oversold = self.indicators.weekly_df["RSI"].iloc[-1] < 40
        bench_high = self.benchmark_df["Benchmark_Close"].rolling(60).max().iloc[-1]
        bench_curr = self.benchmark_df["Benchmark_Close"].iloc[-1]
        market_corr_mag = (bench_high - bench_curr) / bench_high * 100
        market_corr = market_corr_mag > SETTINGS["pullback_threshold"]
        
        buying_zones: List[Zone] = []
        
        if "SMA_50" in latest and not pd.isna(latest["SMA_50"]):
            ma50_val = latest["SMA_50"]
            buying_zones.append(Zone(
                ma50_val * 0.97, 
                ma50_val * 1.01, 
                "Medium", 
                "50-day Moving Average support", 
                "Moving Average",
                ma50_val
            ))
            
        if "SMA_100" in latest and not pd.isna(latest["SMA_100"]):
            ma100_val = latest["SMA_100"]
            buying_zones.append(Zone(
                ma100_val * 0.97, 
                ma100_val * 1.01, 
                "Medium", 
                "100-day Moving Average support", 
                "Moving Average",
                ma100_val
            ))
            
        if "SMA_200" in latest and not pd.isna(latest["SMA_200"]):
            ma200_val = latest["SMA_200"]
            buying_zones.append(Zone(
                ma200_val * 0.97, 
                ma200_val * 1.01, 
                "Medium", 
                "200-day Moving Average support (strong historical reliability)", 
                "Moving Average",
                ma200_val
            ))
            
        bb_lower = latest.get("BB_lower")
        if not pd.isna(bb_lower) and bb_lower < current_price:
            buying_zones.append(Zone(
                bb_lower * 0.97, 
                bb_lower * 1.03, 
                "Medium", 
                "Statistical support level", 
                "Bollinger Band",
                bb_lower
            ))
            
        for cluster in self.support_clusters:
            if cluster.upper < current_price * 0.7 or cluster.lower > current_price:
                continue
            buying_zones.append(cluster)
            
        if self.major_fibs:
            for name, data in self.major_fibs.items():
                price = data["price"]
                if name.startswith("level_") and price < current_price and price > current_price * 0.7:
                    buying_zones.append(Zone(
                        price * 0.98, 
                        price * 1.02, 
                        "Medium", 
                        data["desc"], 
                        "Fibonacci",
                        price
                    ))
                    
        if fair_value:
            buying_zones.append(Zone(
                fair_value * 0.9, 
                fair_value, 
                "Medium", 
                "Fair value based on company fundamentals", 
                "Fundamental",
                fair_value
            ))
            
        buying_zones = self._dedupe(buying_zones)
        
        if buying_zones:
            sorted_zones = sorted(buying_zones, key=lambda z: z.lower)
            zone_count = len(sorted_zones)
            if zone_count >= 3:
                third = zone_count // 3
                for i in range(third):
                    sorted_zones[i].confidence = "High"
                for i in range(third, 2*third):
                    sorted_zones[i].confidence = "Medium"
                for i in range(2*third, zone_count):
                    sorted_zones[i].confidence = "Low"
            elif zone_count == 2:
                sorted_zones[0].confidence = "High"
                sorted_zones[1].confidence = "Medium"
            elif zone_count == 1:
                sorted_zones[0].confidence = "Medium"
        
        selling_zones = self._selling_zones(current_price)
        
        score = self._score(buying_zones, current_price, stock_draw, daily_oversold, weekly_oversold)
        strategy = self._entry_text(score)
        context = CorrectionContext(market_corr, round(market_corr_mag, 1), round(stock_draw, 1), daily_oversold, weekly_oversold, round(current_price, 2))
        reasoning = self._human_reasoning(buying_zones, context, score)
        
        return AnalysisResult(context, buying_zones, selling_zones, score, strategy, reasoning, datetime.utcnow().isoformat())

    def _dedupe(self, zones: List[Zone]) -> List[Zone]:
        out: List[Zone] = []
        conf_lvl = {"Low": 1, "Medium": 2, "High": 3}
        for z in sorted(zones, key=lambda x: x.lower):
            overlap = False
            for existing in out:
                lower = max(z.lower, existing.lower)
                upper = min(z.upper, existing.upper)
                if lower <= upper:
                    if (upper - lower) / (z.upper - z.lower) >= SETTINGS["overlap_pct"]:
                        overlap = True
                        if conf_lvl[z.confidence] > conf_lvl[existing.confidence]:
                            out.remove(existing)
                            out.append(z)
                        break
            if not overlap:
                out.append(z)
        return sorted(out, key=lambda x: x.lower)

    def _selling_zones(self, current_price: float) -> List[Zone]:
        res_levels: List[Zone] = []
        
        all_time_high = self.df["High"].max()
        if all_time_high > current_price * 1.03:
            res_levels.append(Zone(
                all_time_high * 0.98, 
                all_time_high * 1.02, 
                "Medium", 
                "Previous highest price (strong resistance)", 
                "All-Time High",
                all_time_high
            ))
            
        bb_upper = self.df["BB_upper"].iloc[-1]
        if not pd.isna(bb_upper) and bb_upper > current_price * 1.05:
            res_levels.append(Zone(
                bb_upper * 0.98, 
                bb_upper * 1.02, 
                "Medium", 
                "Statistical resistance level", 
                "Bollinger",
                bb_upper
            ))
            
        if self.major_fibs:
            for name, data in self.major_fibs.items():
                price = data["price"]
                if name.startswith("ext_") and price > current_price * 1.1:
                    res_levels.append(Zone(
                        price * 0.98, 
                        price * 1.02, 
                        "Medium", 
                        data["desc"], 
                        "Fibonacci",
                        price
                    ))
                    
        round_number_base = 1000
        while round_number_base <= current_price * 1.2:
            round_number_base += 1000
            
        if round_number_base > current_price * 1.1:
            res_levels.append(Zone(
                round_number_base - 40, 
                round_number_base + 40, 
                "Medium", 
                f"Psychological resistance at {round_number_base}", 
                "Psychological",
                round_number_base
            ))
            
        sorted_selling = sorted(res_levels, key=lambda z: z.lower)
        sell_count = len(sorted_selling)
        
        if sell_count >= 3:
            third = sell_count // 3
            for i in range(sell_count - third, sell_count):
                sorted_selling[i].confidence = "High"
            for i in range(sell_count - 2*third, sell_count - third):
                sorted_selling[i].confidence = "Medium"
            for i in range(sell_count - 2*third):
                sorted_selling[i].confidence = "Low"
        elif sell_count == 2:
            sorted_selling[1].confidence = "High"
            sorted_selling[0].confidence = "Medium"
        elif sell_count == 1:
            sorted_selling[0].confidence = "High"
            
        return sorted_selling

    def _score(self, zones: List[Zone], price: float, draw: float, d_ov: bool, w_ov: bool) -> int:
        score = 50
        if d_ov:
            score += 10
        if w_ov:
            score += 15
        if draw > 20:
            score += 20
        elif draw > 15:
            score += 15
        elif draw > 10:
            score += 10
        elif draw > 5:
            score += 5
        near = any(z.lower <= price <= z.upper for z in zones)
        if near:
            top_conf = max(zones, key=lambda z: {"Low": 1, "Medium": 2, "High": 3}[z.confidence])
            tiers = {"Low": 5, "Medium": 15, "High": 25}
            score += tiers[top_conf.confidence]
        else:
            dist = min(((price - z.upper) / price * 100) for z in zones if z.upper < price) if zones else 100
            if dist < SETTINGS["near_zone_pct"]:
                score += 5
        return min(score, 100)

    @staticmethod
    def _entry_text(score: int) -> str:
        if score >= 80:
            return "Strong Buy - Consider investing 70-100% of your budget now"
        if score >= 60:
            return "Buy - Consider investing 50-70% of your budget now"
        if score >= 40:
            return "Start Position - Consider investing 30-50% of your budget now"
        return "Wait - Consider investing less than 30% of your budget now"

    @staticmethod
    def _human_reasoning(zones: List[Zone], ctx: CorrectionContext, score: int) -> str:
        parts: List[str] = []
        if ctx.market_correction:
            parts.append(f"The overall market is down {ctx.market_correction_magnitude}% from recent highs, and this stock is down {ctx.stock_drawdown}% from its recent peak.")
        else:
            parts.append(f"This stock is down {ctx.stock_drawdown}% from its recent peak, while the broader market hasn't shown a significant correction.")
        
        if ctx.daily_oversold or ctx.weekly_oversold:
            frame = []
            if ctx.daily_oversold:
                frame.append("daily")
            if ctx.weekly_oversold:
                frame.append("weekly")
            timeframe = " and ".join(frame)
            parts.append(f"Technical indicators suggest the stock is oversold on a {timeframe} basis, which often precedes a bounce.")
        
        in_zone = [z for z in zones if z.lower <= ctx.current_price <= z.upper]
        if in_zone:
            confidence_text = "high" if in_zone[0].confidence == "High" else ("moderate" if in_zone[0].confidence == "Medium" else "low")
            parts.append(f"The current price is in a {confidence_text} confidence buying zone: {in_zone[0].description}.")
        else:
            below = [z for z in zones if z.upper < ctx.current_price]
            if below:
                close = min(below, key=lambda z: ctx.current_price - z.upper)
                pct = (ctx.current_price - close.upper) / ctx.current_price * 100
                parts.append(f"The nearest buying zone is about {pct:.1f}% below the current price ({close.lower:.2f}-{close.upper:.2f}).")
        
        if score >= 80:
            parts.append("The technical setup is very favorable for buying now. The combination of price, momentum, and support levels suggests a strong opportunity.")
        elif score >= 60:
            parts.append("The setup is favorable for starting to buy now, but consider keeping some funds ready for potential further dips.")
        elif score >= 40:
            parts.append("It's reasonable to start a small position now and add more shares if the price drops further.")
        else:
            parts.append("It may be better to wait for more favorable price levels before making a significant investment.")
        
        return " ".join(parts)


class ResultPresenter:
    def __init__(self):
        self.buy_cmap = LinearSegmentedColormap.from_list("BuyGradient", 
                                                    [(0, BUY_COLORS["Low"]), 
                                                     (0.5, BUY_COLORS["Medium"]), 
                                                     (1, BUY_COLORS["High"])], N=100)
        
        self.sell_cmap = LinearSegmentedColormap.from_list("SellGradient", 
                                                     [(0, SELL_COLORS["Low"]), 
                                                      (0.5, SELL_COLORS["Medium"]), 
                                                      (1, SELL_COLORS["High"])], N=100)

    def create_user_friendly_plot(self, ticker: str, res: AnalysisResult, df: pd.DataFrame):
        fig = plt.figure(figsize=(12, 10))  
        fig.suptitle(f"{ticker} - Buy & Sell Zones Analysis", fontsize=16, weight='bold')
        
        ax1 = fig.add_subplot(1, 1, 1) 
        recent = df.iloc[-126:]  
        
        if not recent.empty and "Close" in recent:
            ax1.plot(recent.index, recent["Close"], label="Price", color='black', linewidth=2)
        else:
            ax1.plot([0, 1], [res.correction_context.current_price * 0.9, res.correction_context.current_price], 
                    color='black', linewidth=2, label="Price")
        
        ax1.axhline(res.correction_context.current_price, color='blue', linestyle='-', 
                linewidth=1.5, label="Current Price")
        
        buy_zones = sorted(res.buying_zones, key=lambda x: x.lower)
        sell_zones = sorted(res.selling_zones, key=lambda x: x.lower)
        
        conf_values = {"High": 0.9, "Medium": 0.6, "Low": 0.3} 
        buy_handles = []
        buy_labels = []
        
        for z in buy_zones:
            color_val = conf_values[z.confidence]
            color = self.buy_cmap(color_val)
            
            span = ax1.axhspan(z.lower, z.upper, color=color, alpha=0.4)
            
            if z.confidence not in buy_labels:
                buy_handles.append(span)
                buy_labels.append(z.confidence)
        
        sell_handles = []
        sell_labels = []
        for z in sell_zones:
            color_val = conf_values[z.confidence]
            color = self.sell_cmap(color_val)
            
            span = ax1.axhspan(z.lower, z.upper, color=color, alpha=0.4)
            
            if z.confidence not in sell_labels:
                sell_handles.append(span)
                sell_labels.append(z.confidence)
        
        buy_patches = []
        for label in ["High", "Medium", "Low"]:
            if label in buy_labels:
                color = self.buy_cmap(conf_values[label])
                buy_patches.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.4))
        
        sell_patches = []
        for label in ["High", "Medium", "Low"]:
            if label in sell_labels:
                color = self.sell_cmap(conf_values[label])
                sell_patches.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.4))
        
        line_patch = plt.Line2D([0], [0], color='black', linewidth=2)
        current_patch = plt.Line2D([0], [0], color='blue', linewidth=1.5)
        
        all_handles = []
        all_labels = []
        
        for i, label in enumerate(["High", "Medium", "Low"]):
            if label in buy_labels:
                all_handles.append(buy_patches[buy_labels.index(label)])
                all_labels.append(f"Buy - {label}")
        
        for i, label in enumerate(["High", "Medium", "Low"]):
            if label in sell_labels:
                all_handles.append(sell_patches[sell_labels.index(label)])
                all_labels.append(f"Sell - {label}")
        
        all_handles.extend([line_patch, current_patch])
        all_labels.extend(["Price", "Current Price"])
        
        ax1.legend(all_handles, all_labels, loc='upper right', fontsize=10)
        
        ax1.text(0.02, 0.92, "Darker Green = Better Buy Opportunity (Lower Price)\nDarker Red = Better Sell Opportunity (Higher Price)",
                transform=ax1.transAxes, fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        # Grid and labels
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Price Chart with Buy & Sell Zones", fontsize=14)
        ax1.set_ylabel("Price", fontsize=12)
        
        ax1.annotate(f"Current: {res.correction_context.current_price:.2f}", 
                    xy=(0.02, 0.05), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))
        
        ymin, ymax = ax1.get_ylim()
        price_range = ymax - ymin
        
        ax1.set_ylim(ymin - price_range * 0.1, ymax + price_range * 0.1)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        return fig


def generate_recommendation_text(res: AnalysisResult) -> str:
    recommendation_text = f"RECOMMENDATION: {res.entry_strategy}\n"
    recommendation_text += f"Confidence Score: {res.confidence_score}/100\n\n"
    recommendation_text += "WHY THIS RECOMMENDATION:\n"
    recommendation_text += f"{res.reasoning}\n\n"
    
    recommendation_text += "BUY ZONES (from strongest to weakest):\n"
    for confidence in ["High", "Medium", "Low"]:
        zones = [z for z in res.buying_zones if z.confidence == confidence]
        if zones:
            recommendation_text += f"{confidence} Confidence:\n"
            for z in sorted(zones, key=lambda x: x.lower):
                recommendation_text += f"• {z.lower:.2f} - {z.upper:.2f}: {z.description}\n"
    
    recommendation_text += "\nSELL ZONES (from strongest to weakest):\n"
    for confidence in ["High", "Medium", "Low"]:
        zones = [z for z in res.selling_zones if z.confidence == confidence]
        if zones:
            recommendation_text += f"{confidence} Confidence:\n"
            for z in sorted(zones, key=lambda x: x.lower):
                recommendation_text += f"• {z.lower:.2f} - {z.upper:.2f}: {z.description}\n"
    
    return recommendation_text

def analyze_stock(ticker: str, fair_value: Optional[float] = None):
    analyzer = ValueTechnicalAnalyzer(ticker)
    res = analyzer.analyse(fair_value)
    
    presenter = ResultPresenter()
    fig = presenter.create_user_friendly_plot(ticker, res, analyzer.df)
    recommendation_text = generate_recommendation_text(res)
    
    logger.info("Analysis complete for %s", ticker)
    return res, fig, recommendation_text
