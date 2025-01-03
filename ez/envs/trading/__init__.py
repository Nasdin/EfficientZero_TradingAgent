import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FeatureExtractor:
    """
    Example modular feature extractor.
    Define your custom logic (like timescales, indicators, flattening).
    """
    def __init__(self, obs_dim=50):
        self.obs_dim = obs_dim

    def build_observation(self, 
                          raw_data, 
                          position_size,
                          position_avg_price,
                          equity):
        """
        raw_data: a dict or row with all your timeframes, indicators, etc.
        position_size, position_avg_price, equity: current position state

        Return a flat numpy array (float32) of length self.obs_dim.
        """
        # Example placeholders
        mid_price = raw_data.get("mid_price", 100.0)
        ohlc_1m = raw_data.get("ohlc_1m", [100, 101, 99, 100])
        ohlc_5m = raw_data.get("ohlc_5m", [100, 102, 98, 101])
        volume_1m = raw_data.get("volume_1m", 1000)
        vol_indicator = raw_data.get("vol_indicator", 0.2)

        obs_array = np.array([
            mid_price,
            *ohlc_1m,
            *ohlc_5m,
            volume_1m,
            vol_indicator,
            position_size,
            position_avg_price,
            equity
        ], dtype=np.float32)

        # Pad or trim to desired dimension
        if len(obs_array) < self.obs_dim:
            pad_size = self.obs_dim - len(obs_array)
            obs_array = np.concatenate([obs_array, np.zeros(pad_size, dtype=np.float32)])
        elif len(obs_array) > self.obs_dim:
            obs_array = obs_array[:self.obs_dim]
        return obs_array


class TradingEnv(gym.Env):
    """
    Discrete-action RL trading environment that can train on a DataFrame (historical)
    or trade live by accepting incremental data updates, and uses
    an external FeatureExtractor for observation building.
    """
    def __init__(self,
                 df=None,
                 train_mode=True,
                 max_steps=1000,
                 initial_balance=100000,
                 margin_limit=5.0,
                 stop_out_threshold=50000,
                 obs_factory=None):
        super().__init__()

        # Actions:
        # 0 = buy 1, 1 = sell 1, 2 = buy 5, 3 = sell 5,
        # 4 = close all, 5 = close half, 
        # 6 = stop & reverse equal, 7 = stop & reverse half,
        # 8 = do nothing
        self.action_space = spaces.Discrete(9)

        # If no factory is supplied, default to FeatureExtractor with obs_dim=50
        if obs_factory is None:
            obs_factory = FeatureExtractor(obs_dim=50)
        self.obs_factory = obs_factory

        # We'll ask the factory for shape info
        obs_dim = self.obs_factory.obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.df = df
        self.train_mode = train_mode
        self.max_steps = max_steps
        self.initial_balance = initial_balance
        self.margin_limit = margin_limit
        self.stop_out_threshold = stop_out_threshold

        # Internal state
        self.current_step = 0
        self.done = False
        self.equity = self.initial_balance
        self.position_size = 0
        self.position_avg_price = 0.0
        self.prices = None  # current row's data
        self.live_data_buffer = None  # for real-time updates

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.done = False
        self.equity = self.initial_balance
        self.position_size = 0
        self.position_avg_price = 0.0

        if self.train_mode and self.df is not None:
            self.prices = self.df.iloc[self.current_step]
        else:
            self.prices = self.live_data_buffer or {}

        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # Execute action
        self._execute_action(action)

        # Advance data
        self.current_step += 1
        if self.train_mode and self.df is not None:
            if self.current_step < len(self.df):
                self.prices = self.df.iloc[self.current_step]
            else:
                self.done = True

        # Default reward is 0
        reward = 0.0

        # Margin breach
        margin_used = abs(self.position_size)
        if margin_used > self.margin_limit:
            reward -= 1.0
            self.done = True

        # Stop-out
        if self.equity < self.stop_out_threshold:
            reward -= 1.0
            self.done = True

        # Time-based termination
        if self.current_step >= self.max_steps:
            self.done = True

        # Final reward at end-of-episode = realized + unrealized PnL
        if self.done:
            reward += self._calculate_pnl()

        obs = self._get_observation()
        return obs, reward, self.done, False, {}

    def update_live_data(self, new_data):
        """
        In live mode, call this to supply the next row of data before step().
        """
        self.live_data_buffer = new_data

    def _execute_action(self, action):
        mid_price = self.prices.get("mid_price", 100.0)
        if action == 0:    # Buy 1
            self._adjust_position(1, mid_price)
        elif action == 1:  # Sell 1
            self._adjust_position(-1, mid_price)
        elif action == 2:  # Buy 5
            self._adjust_position(5, mid_price)
        elif action == 3:  # Sell 5
            self._adjust_position(-5, mid_price)
        elif action == 4:  # Close all
            self._close_position(mid_price)
        elif action == 5:  # Close half
            self._close_partial(mid_price, fraction=0.5)
        elif action == 6:  # Stop/reverse equal
            self._reverse_position(mid_price, fraction=1.0)
        elif action == 7:  # Stop/reverse half
            self._reverse_position(mid_price, fraction=0.5)
        elif action == 8:  # Do nothing
            pass
        else:
            raise ValueError(f"Invalid action: {action}")

    def _close_position(self, price):
        if self.position_size != 0:
            realized_pnl = self._realized_pnl(price, self.position_size)
            self.equity += realized_pnl
            self.position_size = 0
            self.position_avg_price = 0.0

    def _close_partial(self, price, fraction=0.5):
        if self.position_size != 0:
            close_qty = int(abs(self.position_size) * fraction)
            close_qty *= np.sign(self.position_size)
            realized_pnl = self._realized_pnl(price, close_qty)
            self.equity += realized_pnl
            self.position_size -= close_qty
            if self.position_size == 0:
                self.position_avg_price = 0.0

    def _reverse_position(self, price, fraction=1.0):
        old_size = self.position_size
        partial_size = int(abs(old_size) * fraction) * np.sign(old_size)
        realized_pnl = self._realized_pnl(price, partial_size)
        self.equity += realized_pnl
        self.position_size -= partial_size
        if self.position_size == 0:
            self.position_avg_price = 0.0

        new_size = -partial_size
        if new_size != 0:
            self._adjust_position(new_size, price)

    def _adjust_position(self, quantity, price):
        if self.position_size == 0:
            # Opening new position from flat
            self.position_size = quantity
            self.position_avg_price = price
        else:
            # Same direction => adjust weighted average price
            if np.sign(self.position_size) == np.sign(quantity):
                total_qty = self.position_size + quantity
                new_avg = (
                    self.position_avg_price * self.position_size +
                    price * quantity
                ) / total_qty
                self.position_avg_price = new_avg
                self.position_size = total_qty
            else:
                # Partial or full offset
                if abs(quantity) > abs(self.position_size):
                    leftover = quantity + self.position_size
                    realized_pnl = self._realized_pnl(price, -self.position_size)
                    self.equity += realized_pnl
                    self.position_size = leftover
                    self.position_avg_price = price
                else:
                    realized_pnl = self._realized_pnl(price, quantity)
                    self.equity += realized_pnl
                    self.position_size += quantity
                    if self.position_size == 0:
                        self.position_avg_price = 0.0

    def _realized_pnl(self, current_price, qty):
        avg_px = self.position_avg_price
        direction = np.sign(self.position_size)
        return (current_price - avg_px) * (-qty if direction < 0 else qty)

    def _calculate_pnl(self):
        if self.position_size == 0:
            return self.equity - self.initial_balance
        # Unrealized
        mid_price = self.prices.get("mid_price", 100.0)
        unrealized = (mid_price - self.position_avg_price) * self.position_size
        return (self.equity - self.initial_balance) + unrealized

    def _get_observation(self):
        return self.obs_factory.build_observation(
            raw_data=self.prices,
            position_size=self.position_size,
            position_avg_price=self.position_avg_price,
            equity=self.equity
        )
