import gymnasium as gym
import panda_gym
from sb3_contrib import TQC
from huggingface_sb3 import load_from_hub
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import time

def main():
    gym.pprint_registry()
    chk = load_from_hub(repo_id="BanUrsus/tqc-PandaPickAndPlace-v3", filename="tqc-PandaPickAndPlace-v3.zip")
    stats = load_from_hub(repo_id="BanUrsus/tqc-PandaPickAndPlace-v3", filename="vec_normalize.pkl")
   
    env = gym.make('PandaPickAndPlace-v3', render_mode="human", autoreset=False)
    
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(stats, env)
    env.training = False
    env.norm_reward = False

    model = TQC.load(chk,env) 

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, dones, info = env.step(action)
        #env.render()
        print(info)
        done = dones[0]
        time.sleep(1)
        
if __name__ == "__main__":
    main()
