import os
import time

import gymnasium as gym
import numpy as np
import panda_gym  # NOTE: keep the import, it registers the environment for the gym library
from agent import Agent
from huggingface_sb3 import load_from_hub
from panda_gym.utils import distance
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def main():
    if not os.path.exists("../results"):
        os.mkdir("../results")

    chk = load_from_hub(
        repo_id="BanUrsus/tqc-PandaPickAndPlace-v3",
        filename="tqc-PandaPickAndPlace-v3.zip",
    )
    stats = load_from_hub(
        repo_id="BanUrsus/tqc-PandaPickAndPlace-v3",
        filename="vec_normalize.pkl",
    )

    env = gym.make("PandaPickAndPlace-v3", autoreset=False)

    sim = env.sim
    robot = env.robot
    task = env.task

    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(stats, env)
    env.training = False
    env.norm_reward = False

    model = TQC.load(chk, env)

    a = Agent("kuka", model, {"robot": robot, "task": task, "sim": sim})

    return

    (
        action_parameters,
        pred_type_mapping,
        agent_model_actions,
        abstract_model_actions,
        objects,
        old_types,
        init_state,
        domain_name,
    ) = agent.agent_model.generate_ds()

    abstract_predicates = {}
    types = modify_types(old_types)

    pp = pprint.PrettyPrinter(indent=2)
    abstract_model = Model(abstract_predicates, abstract_model_actions)

    # comment to include static predicates
    abs_preds_test, abs_actions_test, _ = agent.agent_model.bootstrap_model()
    abstract_model.predicates = abs_preds_test
    abstract_model.actions = abs_actions_test

    if not check_results:
        iaa_main = AgentInterrogation(
            agent,
            abstract_model,
            objects,
            domain_name,
            abstract_predicates,
            pred_type_mapping,
            action_parameters,
            types,
            load_old_q=True,
        )

        query_count, running_time, data_dict, pal_tuple_count, valid_models = (
            iaa_main.agent_interrogation_algo()
        )
        agent.agent_model.show_actions()


if __name__ == "__main__":
    main()
