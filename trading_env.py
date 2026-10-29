import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class ForexTradingEnv(gym.Env):
    """Gym environment for Forex trading on price bar data (EURUSD etc.).

    Action space:
      0 = hold (do nothing)
      1 = buy  (long )
      2 = sell (short)

    Observation space:
      - window_size last bars with features [open, high, low, close, volume, indicators...]

    Reward:
      - log(return) (per-step) minus slippage fee and spread cost.
      - encourages high-risk-adjusted returns.

    This environment is written with maximum commentary for learning.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int = 64,
        commission: float = 0.00004,
        spread: float = 0.00001,
        initial_balance: float = 10000.0,
        position_size: float = 1.0,
        reward_scaling: float = 1.0,
        normalize: bool = True,
        use_instrument_features: bool = True,
        **kwargs,
    ):
        super().__init__()

        assert isinstance(data, pd.DataFrame), "data must be pandas DataFrame"
        assert "close" in data.columns, "data must contain close price"

        self.raw_data = data.copy().reset_index(drop=True)

        # Calculate technical indicators if not already present
        if use_instrument_features:
            self._add_indicators(self.raw_data)

        # Normalize features with rolling z-score in window or global min-max
        self.normalize = normalize

        # Save parameters
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.net_worth = initial_balance
        self.max_net_worth = initial_balance
        self.commission = commission
        self.spread = spread
        self.position_size = position_size
        self.reward_scaling = reward_scaling

        # Internal state indices
        self.current_step = window_size

        # Actions: hold, buy, sell
        self.action_space = spaces.Discrete(3)

        # observation space shape: window_size x n_features
        self.feature_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "rsi_14",
            "macd",
            "macd_signal",
            "atr_14",
            "log_return",
        ]

        self.n_features = len(self.feature_columns)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.n_features),
            dtype=np.float32,
        )

        # Position state: 0 neutral, +1 long, -1 short
        self.position = 0
        self.entry_price = 0.0
        self.total_steps = len(self.raw_data)

        # Account metrics
        self.trade_count = 0
        self.last_reward = 0.0

    def _add_indicators(self, df: pd.DataFrame):
        """Add built-in indicators to DataFrame (RSI, MACD, ATR, log return)."""
        # RSI 14 (simple) using close price
        delta = df["close"].diff(1)
        gain = np.maximum(delta, 0)
        loss = -np.minimum(delta, 0)

        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()

        rs = avg_gain / (avg_loss + 1e-9)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD and signal line
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

        # ATR 14
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = true_range.rolling(window=14, min_periods=14).mean()

        # log returns
        df["log_return"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)

        # Fill nan values
        df.fillna(method="bfill", inplace=True)

    def _get_observation(self):
        """Return observation (window_size rows x n_features cols) normalized if requested."""
        window = self.raw_data.loc[self.current_step - self.window_size : self.current_step - 1, self.feature_columns].values.astype(np.float32)

        if self.normalize:
            # simple z-score per feature, robust in online use
            mu = window.mean(axis=0)
            sigma = window.std(axis=0) + 1e-9
            window = (window - mu) / sigma

        return window

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.position = 0
        self.entry_price = 0.0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trade_count = 0
        self.last_reward = 0.0

        obs = self._get_observation()
        info = {
            "balance": self.balance,
            "net_worth": self.net_worth,
            "position": self.position,
            "step": self.current_step,
        }
        return obs, info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"

        done = False
        current_price = self.raw_data.loc[self.current_step, "close"]

        reward = 0.0

        # 执行 action 与 position 转换逻辑
        if action == 1:  # buy
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
                self.trade_count += 1
            elif self.position == -1:
                self._close_position(current_price)
                self.position = 1
                self.entry_price = current_price
                self.trade_count += 1

        elif action == 2:  # sell
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
                self.trade_count += 1
            elif self.position == 1:
                self._close_position(current_price)
                self.position = -1
                self.entry_price = current_price
                self.trade_count += 1

        # holding (action==0) or after switch we compute unrealized PnL anyway
        reward = self._compute_reward(current_price)

        self.net_worth = self.balance + self._unrealized_pnl(current_price)
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        self.current_step += 1

        if self.net_worth <= 0:
            done = True
            reward -= 5.0  # heavy penalty for ruin

        if self.current_step >= self.total_steps:
            done = True
            # Force close
            if self.position != 0:
                reward += self._close_position(current_price)

        self.last_reward = reward

        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "balance": self.balance,
            "net_worth": self.net_worth,
            "position": self.position,
            "entry_price": self.entry_price,
            "trade_count": self.trade_count,
        }

        return obs, reward * self.reward_scaling, done, False, info

    def _unrealized_pnl(self, price):
        if self.position == 0:
            return 0.0

        direction = 1 if self.position == 1 else -1
        return self.position_size * (price - self.entry_price) * direction

    def _close_position(self, price):
        # Compute profit/loss at close of position and deduct trade costs
        if self.position == 0:
            return 0.0

        direction = 1 if self.position == 1 else -1
        price_diff = (price - self.entry_price) * direction
        profit = self.position_size * price_diff

        cost = self.commission * self.position_size + self.spread * self.position_size

        self.balance += profit - cost
        self.position = 0
        self.entry_price = 0.0

        return profit - cost

    def _compute_reward(self, current_price):
        """Compute step reward given current position and price movements."""
        if self.position == 0:
            return 0.0

        # unrealized PnL per step as log-return
        direction = 1 if self.position == 1 else -1
        # Use log returns to stabilize, avoid huge values
        price_ratio = (current_price + 1e-9) / (self.entry_price + 1e-9)
        log_ret = np.log(price_ratio) * direction

        # commission and spread are subtracted at each trade on-close; also small per-step penalty
        cost = 0.0
        # Encourage fewer trades by small holding penalty
        cost -= 1e-5

        return log_ret - cost

    def render(self, mode="human"):
        print(
            f"Step: {self.current_step}, Balance: {self.balance:.2f}, "
            f"Net worth: {self.net_worth:.2f}, Position: {self.position}, "
            f"Entry: {self.entry_price:.5f}, Last reward: {self.last_reward:.6f}"
        )

    def close(self):
        pass
