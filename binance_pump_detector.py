import asyncio
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
import requests
from collections import deque
import statistics

class BinancePumpDetector:
    def __init__(self, api_key: str, api_secret: str, telegram_bot_token: str, telegram_chat_id: str):
        self.client = Client(api_key, api_secret)
        self.telegram_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id

        # Historical data storage
        self.price_history = {}
        self.volume_history = {}
        self.trade_count_history = {}
        self.order_book_history = {}

        # Detection parameters
        self.min_volume_spike = 3.0  # 3x volume increase
        self.min_price_increase = 0.02  # 2% price increase
        self.lookback_period = 300  # 5 minutes
        self.scan_interval = 5  # seconds

        # Blacklist for already pumped coins
        self.recent_pumps = deque(maxlen=50)
        self.cooldown_period = 3600  # 1 hour cooldown

    def send_telegram_alert(self, message: str):
        """Send alert to Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            requests.post(url, data=data, timeout=10)
        except Exception as e:
            print(f"Telegram error: {e}")

    def get_top_volume_pairs(self, limit: int = 100) -> List[str]:
        """Get top trading pairs by volume"""
        try:
            tickers = self.client.get_ticker()
            usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT') and
                         not any(x in t['symbol'] for x in ['UP', 'DOWN', 'BULL', 'BEAR'])]

            # Sort by quote volume
            sorted_pairs = sorted(usdt_pairs,
                                key=lambda x: float(x['quoteVolume']),
                                reverse=True)

            return [p['symbol'] for p in sorted_pairs[:limit]]
        except Exception as e:
            print(f"Error getting pairs: {e}")
            return []

    def calculate_volume_profile(self, symbol: str) -> Dict:
        """Calculate volume profile metrics"""
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_1MINUTE,
                limit=60
            )

            volumes = [float(k[5]) for k in klines]
            prices = [float(k[4]) for k in klines]

            recent_volume = sum(volumes[-5:])
            avg_volume = sum(volumes[:-5]) / len(volumes[:-5]) if len(volumes) > 5 else 1

            volume_spike = recent_volume / avg_volume if avg_volume > 0 else 0

            return {
                'volume_spike': volume_spike,
                'recent_volume': recent_volume,
                'avg_volume': avg_volume,
                'current_price': prices[-1],
                'price_change_5m': (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
            }
        except Exception as e:
            return {}

    def analyze_order_book(self, symbol: str) -> Dict:
        """Analyze order book for buy pressure"""
        try:
            depth = self.client.get_order_book(symbol=symbol, limit=100)

            bids = depth['bids']
            asks = depth['asks']

            bid_volume = sum([float(b[1]) for b in bids])
            ask_volume = sum([float(a[1]) for a in asks])

            bid_value = sum([float(b[0]) * float(b[1]) for b in bids])
            ask_value = sum([float(a[0]) * float(a[1]) for a in asks])

            buy_pressure = bid_volume / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0.5
            value_ratio = bid_value / ask_value if ask_value > 0 else 1

            # Analyze bid/ask spread concentration
            top_5_bid_volume = sum([float(b[1]) for b in bids[:5]])
            top_5_ask_volume = sum([float(a[1]) for a in asks[:5]])

            bid_concentration = top_5_bid_volume / bid_volume if bid_volume > 0 else 0

            return {
                'buy_pressure': buy_pressure,
                'value_ratio': value_ratio,
                'bid_concentration': bid_concentration,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume
            }
        except Exception as e:
            return {}

    def calculate_momentum_indicators(self, symbol: str) -> Dict:
        """Calculate multiple momentum indicators"""
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_1MINUTE,
                limit=100
            )

            closes = np.array([float(k[4]) for k in klines])
            highs = np.array([float(k[2]) for k in klines])
            lows = np.array([float(k[3]) for k in klines])
            volumes = np.array([float(k[5]) for k in klines])

            # Price momentum
            momentum_5 = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
            momentum_15 = (closes[-1] - closes[-15]) / closes[-15] if len(closes) >= 15 else 0

            # Acceleration (second derivative)
            if len(closes) >= 10:
                recent_momentum = (closes[-1] - closes[-5]) / closes[-5]
                past_momentum = (closes[-5] - closes[-10]) / closes[-10]
                acceleration = recent_momentum - past_momentum
            else:
                acceleration = 0

            # Volume-weighted momentum
            if len(volumes) >= 10:
                recent_vol_momentum = np.sum(volumes[-5:]) / np.mean(volumes[-15:-5])
            else:
                recent_vol_momentum = 1

            # Volatility (price range)
            if len(highs) >= 20:
                recent_volatility = np.std(closes[-10:]) / np.mean(closes[-10:])
                past_volatility = np.std(closes[-20:-10]) / np.mean(closes[-20:-10])
                volatility_spike = recent_volatility / past_volatility if past_volatility > 0 else 1
            else:
                volatility_spike = 1

            # Price consistency (are we going up consistently?)
            if len(closes) >= 5:
                up_candles = sum([1 for i in range(-5, 0) if closes[i] > closes[i-1]])
                consistency = up_candles / 5
            else:
                consistency = 0

            return {
                'momentum_5m': momentum_5,
                'momentum_15m': momentum_15,
                'acceleration': acceleration,
                'volume_momentum': recent_vol_momentum,
                'volatility_spike': volatility_spike,
                'price_consistency': consistency
            }
        except Exception as e:
            return {}

    def analyze_trade_flow(self, symbol: str) -> Dict:
        """Analyze recent trades for buy/sell pressure"""
        try:
            trades = self.client.get_recent_trades(symbol=symbol, limit=500)

            current_time = int(time.time() * 1000)
            recent_trades = [t for t in trades if current_time - t['time'] < 60000]  # Last 1 minute

            if not recent_trades:
                return {}

            # Classify trades as buy or sell based on whether taker was buyer
            buy_volume = sum([float(t['qty']) for t in recent_trades if t['isBuyerMaker'] == False])
            sell_volume = sum([float(t['qty']) for t in recent_trades if t['isBuyerMaker'] == True])

            total_volume = buy_volume + sell_volume
            buy_ratio = buy_volume / total_volume if total_volume > 0 else 0.5

            # Large trade detection
            trade_sizes = [float(t['qty']) * float(t['price']) for t in recent_trades]
            avg_trade_size = statistics.mean(trade_sizes) if trade_sizes else 0
            large_trades = [t for t in trade_sizes if t > avg_trade_size * 3]

            return {
                'buy_ratio': buy_ratio,
                'trade_count': len(recent_trades),
                'large_trade_count': len(large_trades),
                'avg_trade_size': avg_trade_size
            }
        except Exception as e:
            return {}

    def calculate_pump_score(self, symbol: str) -> Tuple[float, Dict]:
        """Calculate comprehensive pump probability score"""

        # Get all metrics
        volume_data = self.calculate_volume_profile(symbol)
        order_book_data = self.analyze_order_book(symbol)
        momentum_data = self.calculate_momentum_indicators(symbol)
        trade_flow_data = self.analyze_trade_flow(symbol)

        if not all([volume_data, order_book_data, momentum_data, trade_flow_data]):
            return 0, {}

        # Scoring system with weights
        score = 0
        max_score = 100
        details = {}

        # Volume spike (25 points)
        volume_spike = volume_data.get('volume_spike', 0)
        if volume_spike >= 5:
            score += 25
        elif volume_spike >= 3:
            score += 15
        elif volume_spike >= 2:
            score += 8
        details['volume_spike'] = volume_spike

        # Price momentum (20 points)
        momentum_5 = momentum_data.get('momentum_5m', 0)
        if momentum_5 >= 0.03:  # 3% gain
            score += 20
        elif momentum_5 >= 0.02:
            score += 12
        elif momentum_5 >= 0.01:
            score += 6
        details['price_change'] = momentum_5

        # Acceleration (15 points)
        acceleration = momentum_data.get('acceleration', 0)
        if acceleration > 0.02:
            score += 15
        elif acceleration > 0.01:
            score += 8
        details['acceleration'] = acceleration

        # Order book buy pressure (15 points)
        buy_pressure = order_book_data.get('buy_pressure', 0.5)
        if buy_pressure >= 0.65:
            score += 15
        elif buy_pressure >= 0.55:
            score += 8
        details['buy_pressure'] = buy_pressure

        # Trade flow (10 points)
        buy_ratio = trade_flow_data.get('buy_ratio', 0.5)
        if buy_ratio >= 0.7:
            score += 10
        elif buy_ratio >= 0.6:
            score += 5
        details['buy_ratio'] = buy_ratio

        # Volume momentum (10 points)
        vol_momentum = momentum_data.get('volume_momentum', 1)
        if vol_momentum >= 3:
            score += 10
        elif vol_momentum >= 2:
            score += 5
        details['volume_momentum'] = vol_momentum

        # Price consistency (5 points)
        consistency = momentum_data.get('price_consistency', 0)
        score += consistency * 5
        details['consistency'] = consistency

        # Penalties
        # Penalize if already in recent pumps
        if symbol in self.recent_pumps:
            score *= 0.3
            details['penalty'] = 'recent_pump'

        # Penalize low volume coins
        if volume_data.get('recent_volume', 0) < 100000:  # Less than 100k USDT volume
            score *= 0.5
            details['penalty'] = details.get('penalty', '') + '_low_volume'

        details['final_score'] = score
        details['current_price'] = volume_data.get('current_price', 0)

        return score, details

    async def scan_market(self):
        """Main scanning loop"""
        print("Starting Binance Pump Detector...")
        self.send_telegram_alert("🚀 <b>Binance Pump Detector Started</b>\n\nMonitoring market for early pump signals...")

        while True:
            try:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Scanning market...")

                # Get top volume pairs
                symbols = self.get_top_volume_pairs(limit=80)

                pump_candidates = []

                for symbol in symbols:
                    try:
                        score, details = self.calculate_pump_score(symbol)

                        # High threshold for alerts
                        if score >= 60:
                            pump_candidates.append({
                                'symbol': symbol,
                                'score': score,
                                'details': details
                            })
                            print(f"🔥 {symbol}: Score {score:.1f}")

                        # Small delay to avoid rate limits
                        await asyncio.sleep(0.1)

                    except BinanceAPIException as e:
                        if e.code == -1003:  # Rate limit
                            await asyncio.sleep(2)
                        continue
                    except Exception as e:
                        continue

                # Sort by score and send alerts
                pump_candidates.sort(key=lambda x: x['score'], reverse=True)

                for candidate in pump_candidates[:3]:  # Top 3
                    symbol = candidate['symbol']
                    score = candidate['score']
                    details = candidate['details']

                    # Add to recent pumps to avoid duplicate alerts
                    if symbol not in self.recent_pumps:
                        self.recent_pumps.append(symbol)

                        message = f"""
🚨 <b>PUMP ALERT</b> 🚨

<b>Coin:</b> {symbol}
<b>Score:</b> {score:.1f}/100

📊 <b>Metrics:</b>
• Volume Spike: {details.get('volume_spike', 0):.2f}x
• Price Change: {details.get('price_change', 0)*100:.2f}%
• Acceleration: {details.get('acceleration', 0)*100:.2f}%
• Buy Pressure: {details.get('buy_pressure', 0)*100:.1f}%
• Buy Ratio: {details.get('buy_ratio', 0)*100:.1f}%
• Vol Momentum: {details.get('volume_momentum', 0):.2f}x

💰 <b>Current Price:</b> ${details.get('current_price', 0)}

⚡ <b>Action:</b> Consider entering position
⚠️ <b>Risk:</b> High volatility - use stop loss
"""
                        self.send_telegram_alert(message)
                        print(f"Alert sent for {symbol}")

                print(f"Scan complete. Found {len(pump_candidates)} potential pumps.")

                # Wait before next scan
                await asyncio.sleep(self.scan_interval)

            except Exception as e:
                print(f"Error in scan loop: {e}")
                await asyncio.sleep(10)

    def run(self):
        """Start the detector"""
        asyncio.run(self.scan_market())


if __name__ == "__main__":
    # Configuration
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

    if not all([BINANCE_API_KEY, BINANCE_API_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
        print("Error: Missing required environment variables")
        print("Please set: BINANCE_API_KEY, BINANCE_API_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID")
        exit(1)

    detector = BinancePumpDetector(
        api_key=BINANCE_API_KEY,
        api_secret=BINANCE_API_SECRET,
        telegram_bot_token=TELEGRAM_BOT_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID
    )

    detector.run()
