import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FairWarehouseEnv(gym.Env):
    def __init__(self, real_demand, forecast_demand, prices, snaps, events,
                 h, S, p, initial_inv, max_capacity=500, lead_time=2):
        super(FairWarehouseEnv, self).__init__()
        self.real_demand = real_demand
        self.forecast_demand = forecast_demand
        self.prices = prices
        self.snaps = snaps
        self.events = events
        self.max_steps = len(real_demand) - 1
        self.h, self.S, self.p = h, S, p
        self.initial_inv = initial_inv
        self.max_capacity = max_capacity
        self.lead_time = lead_time

        self.action_space = spaces.Discrete(int(max_capacity) + 1)
        obs_dim = 6 + self.lead_time + 1
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.inv_rl = self.initial_inv
        self.last_action = 0.0
        self.pipeline_rl = [0] * self.lead_time
        return self._get_obs(), {}

    def _get_obs(self):
        next_forecast = self.forecast_demand[self.current_step] if self.current_step < len(self.forecast_demand) else 0.0
        next_price = self.prices[self.current_step] if self.current_step < len(self.prices) else 0.0
        next_snap = self.snaps[self.current_step] if self.current_step < len(self.snaps) else 0.0
        next_event = self.events[self.current_step] if self.current_step < len(self.events) else 0.0
        day_of_week = self.current_step % 7
        obs = [self.inv_rl, next_forecast, self.last_action, next_price, next_snap, next_event, day_of_week] + self.pipeline_rl
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        actual_demand = self.real_demand[self.current_step]
        received_rl = self.pipeline_rl.pop(0) if self.lead_time > 0 else 0
        self.inv_rl = min(self.inv_rl + received_rl, self.max_capacity)

        order_rl = int(action)
        self.last_action = float(order_rl)
        setup_cost_rl = self.S if order_rl > 0 else 0

        if self.lead_time > 0:
            self.pipeline_rl.append(order_rl)
        else:
            self.inv_rl = min(self.inv_rl + order_rl, self.max_capacity)

        shortage_rl = max(0, actual_demand - self.inv_rl)
        self.inv_rl = max(0, self.inv_rl - actual_demand)

        holding_cost_rl = self.inv_rl * self.h
        penalty_cost_rl = shortage_rl * self.p
        total_cost = setup_cost_rl + holding_cost_rl + penalty_cost_rl

        reward = -total_cost / 10.0
        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        info = {'Cost': total_cost, 'inv_rl': self.inv_rl, 'shortage': shortage_rl}
        return self._get_obs(), float(reward), terminated, False, info


class MultiItemWarehouseEnv(gym.Env):
    def __init__(self, demand_dict, forecast_dict, price_dict, num_items=3,
                 h_daily=0.001, fixed_setup_cost=0.5, per_item_setup_cost=0.01,
                 penalty_cost=5.0, max_capacity=500, lead_time=2,
                 eoq_params=None):
        super(MultiItemWarehouseEnv, self).__init__()
        self.num_items = num_items
        self.demand = demand_dict
        self.forecast = forecast_dict
        self.price = price_dict
        self.max_steps = len(self.demand[0]) - 1
        self.h = h_daily
        self.S_fixed = fixed_setup_cost
        self.S_item = per_item_setup_cost
        self.p = penalty_cost
        self.max_capacity = max_capacity
        self.lead_time = lead_time
        self.eoq_params = eoq_params

        self.action_space = spaces.MultiDiscrete([self.max_capacity + 1] * self.num_items)
        obs_dim = self.num_items * (4 + self.lead_time)
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.inv_rl = np.zeros(self.num_items)
        self.inv_eoq = np.zeros(self.num_items)
        for i in range(self.num_items):
            self.inv_rl[i] = self.eoq_params[i]['Q']
            self.inv_eoq[i] = self.eoq_params[i]['Q']

        self.last_action = np.zeros(self.num_items)
        self.pipeline_rl = [np.zeros(self.num_items) for _ in range(self.lead_time)]
        self.pipeline_eoq = [np.zeros(self.num_items) for _ in range(self.lead_time)]
        return self._get_obs(), {}

    def _get_obs(self):
        obs = []
        for i in range(self.num_items):
            next_forecast = self.forecast[i][self.current_step] if self.current_step < len(self.forecast[i]) else 0.0
            next_price = self.price[i][self.current_step] if self.current_step < len(self.price[i]) else 0.0
            item_obs = [self.inv_rl[i], next_forecast, self.last_action[i], next_price]
            for step_pipe in self.pipeline_rl:
                item_obs.append(step_pipe[i])
            obs.extend(item_obs)
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        if self.lead_time > 0:
            received_rl = self.pipeline_rl.pop(0)
            received_eoq = self.pipeline_eoq.pop(0)
        else:
            received_rl = np.zeros(self.num_items)
            received_eoq = np.zeros(self.num_items)

        self.inv_rl = np.minimum(self.inv_rl + received_rl, self.max_capacity)
        self.inv_eoq = np.minimum(self.inv_eoq + received_eoq, self.max_capacity)

        order_rl = np.array(action, dtype=np.float32)
        self.last_action = order_rl

        rl_items_ordered = np.sum(order_rl > 0)
        setup_cost_rl = 0.0
        if rl_items_ordered > 0:
            setup_cost_rl = self.S_fixed + (rl_items_ordered * self.S_item)

        if self.lead_time > 0:
            self.pipeline_rl.append(order_rl)
        else:
            self.inv_rl = np.minimum(self.inv_rl + order_rl, self.max_capacity)

        order_eoq = np.zeros(self.num_items)
        for i in range(self.num_items):
            pipeline_sum = sum([p[i] for p in self.pipeline_eoq]) if self.lead_time > 0 else 0
            inventory_position = self.inv_eoq[i] + pipeline_sum
            if inventory_position <= self.eoq_params[i]['ROP']:
                order_eoq[i] = self.eoq_params[i]['Q']

        eoq_items_ordered = np.sum(order_eoq > 0)
        setup_cost_eoq = 0.0
        if eoq_items_ordered > 0:
            setup_cost_eoq = self.S_fixed + (eoq_items_ordered * self.S_item)

        if self.lead_time > 0:
            self.pipeline_eoq.append(order_eoq)
        else:
            self.inv_eoq = np.minimum(self.inv_eoq + order_eoq, self.max_capacity)

        holding_cost_rl, penalty_cost_rl = 0.0, 0.0
        holding_cost_eoq, penalty_cost_eoq = 0.0, 0.0
        shortages_rl = np.zeros(self.num_items)
        shortages_eoq = np.zeros(self.num_items)

        for i in range(self.num_items):
            actual_demand = self.demand[i][self.current_step]
            shortage_rl = max(0, actual_demand - self.inv_rl[i])
            self.inv_rl[i] = max(0, self.inv_rl[i] - actual_demand)
            holding_cost_rl += self.inv_rl[i] * self.h
            penalty_cost_rl += shortage_rl * self.p
            shortages_rl[i] = shortage_rl

            shortage_eoq = max(0, actual_demand - self.inv_eoq[i])
            self.inv_eoq[i] = max(0, self.inv_eoq[i] - actual_demand)
            holding_cost_eoq += self.inv_eoq[i] * self.h
            penalty_cost_eoq += shortage_eoq * self.p
            shortages_eoq[i] = shortage_eoq

        total_cost_rl = setup_cost_rl + holding_cost_rl + penalty_cost_rl
        total_cost_eoq = setup_cost_eoq + holding_cost_eoq + penalty_cost_eoq

        reward = (total_cost_eoq - total_cost_rl) / 10.0
        if np.sum(shortages_rl) > 0:
            reward -= 2.0

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        info = {'C_rl': total_cost_rl, 'C_eoq': total_cost_eoq}
        return self._get_obs(), float(reward), terminated, False, info
