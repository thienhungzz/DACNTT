import os
import importlib
from stable_baselines3 import PPO

# Import module bằng importlib
rl_env = importlib.import_module("03_rl_environments")
FairWarehouseEnv = rl_env.FairWarehouseEnv
MultiItemWarehouseEnv = rl_env.MultiItemWarehouseEnv

def train_ppo_single(env_kwargs, total_timesteps=150000, model_dir="../MoHinh/"):
    print("\n--- Training PPO Single-Item Model ---")
    env = FairWarehouseEnv(**env_kwargs)
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=total_timesteps)
    
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "PPO_Final.zip")
    model.save(model_path)
    return model

def train_ppo_multi(env_kwargs, total_timesteps=500000, model_dir="../MoHinh/"):
    print("\n--- Training PPO Multi-Item (Joint) Model ---")
    env = MultiItemWarehouseEnv(**env_kwargs)
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=total_timesteps)
    
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "PPO_MultiItem_Final.zip")
    model.save(model_path)
    return model