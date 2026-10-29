import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Optional, Tuple, Dict, Any

class ForexTradingEnv(gym.Env):
    """
    Simple Forex trading environment for EURUSD (or any pair).
    Actions: 0 = hold, 1 = long, 2 = short
    Reward: log return of equity (with spread and commission)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 64,
        spread_pips: float = 1.2,
        commission: float = 0.00003,
        initial_balance: float = 10000.0,
        lot_size: float = 100000.0,
        max_position: float = 1.0,
    ):
        super().__init__()

        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("DataFrame must contain open, high, low, close")

        self.df = df.reset_index(drop=True).copy()
        if 'volume' not in self.df.columns:
            self.df['volume'] = 0

        # Indicators
        self.df['rsi'] = ta.rsi(self.df['close'], length=14)
        macd = ta.macd(self.df['close'])
        self.df['macd'] = macd['MACD_12_26_9']
        self.df['macd_s'] = macd['MACDs_12_26_9']
        self.df['atr'] = ta.atr(self.df['high'], self.df['low'], self.df['close'], length=14)
        bb = ta.bbands(self.df['close'], length=20)
        self.df['bb_upper'] = bb.iloc[:, 0]
        self.df['bb_middle'] = bb.iloc[:, 1]
        self.df['bb_lower'] = bb.iloc[:, 2]

        self.df = self.df.dropna().reset_index(drop=True)

        if len(self.df) < window_size + 10:
            raise ValueError("Not enough data after indicators")

        self.window_size = window_size
        self.spread = spread_pips * 0.0001
        self.commission = commission
        self.initial_balance = initial_balance
        self.lot_size = lot_size
        self.max_position = max_position

        self.action_space = spaces.Discrete(3)

        # Observation: flattened window + position + balance_norm + equity_norm
        n_features = 8  # close, rsi, macd, macd_s, atr, bb_upper, bb_middle, bb_lower
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size * n_features + 3,), dtype=np.float32
        )

        self.current_step = 0
        self.balance = initial_balance
        self.equity = initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.max_equity = initial_balance

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.max_equity = self.initial_balance
        return self._get_observation(), self._get_info()

    def step(self, action: int):
        current_price = self.df.iloc[self.current_step]['close']
        prev_equity = self.equity

        # Close opposite position
        if (action == 1 and self.position < 0) or (action == 2 and self.position > 0):
            self._close_position(current_price)

        # Open new
        if action == 1 and self.position == 0:
            self._open_position(current_price, 1.0)
        elif action == 2 and self.position == 0:
            self._open_position(current_price, -1.0)

        # Mark to market
        if self.position != 0:
            profit = self.position * (current_price - self.entry_price) * self.lot_size
            self.equity = self.balance + profit
        else:
            self.equity = self.balance

        reward = np.log(max(self.equity, 1e-8) / max(prev_equity, 1e-8))

        # Soft drawdown penalty
        drawdown = (self.max_equity - self.equity) / max(self.max_equity, 1e-8)
        if drawdown > 0.05:
            reward -= drawdown * 0.3

        self.max_equity = max(self.max_equity, self.equity)

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        return self._get_observation(), float(reward), terminated, truncated, self._get_info()

    def _open_position(self, price: float, direction: float):
        cost = self.commission * self.lot_size * self.max_position
        self.balance -= cost
        self.position = direction * self.max_position
        self.entry_price = price + (self.spread / 2 if direction > 0 else -self.spread / 2)

    def _close_position(self, price: float):
        if self.position == 0:
            return
        profit = self.position * (price - self.entry_price) * self.lot_size
        cost = self.commission * self.lot_size * abs(self.position)
        self.balance += profit - cost
        self.position = 0.0
        self.entry_price = 0.0

    def _get_observation(self) -> np.ndarray:
        start = self.current_step - self.window_size
        end = self.current_step
        window = self.df.iloc[start:end]

        features = window[['close', 'rsi', 'macd', 'macd_s', 'atr',
                           'bb_upper', 'bb_middle', 'bb_lower']].values.flatten()

        pos = self.position
        bal_norm = (self.balance - self.initial_balance) / self.initial_balance
        eq_norm = (self.equity - self.initial_balance) / self.initial_balance

        return np.concatenate([features, [pos, bal_norm, eq_norm]]).astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "balance": self.balance,
            "equity": self.equity,
            "position": self.position,
            "step": self.current_step,
        }

    def render(self):
        print(f"Step {self.current_step} | Equity {self.equity:.2f} | Pos {self.position:.1f}")
