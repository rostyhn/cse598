import copy
import pickle

import numpy as np
import pybullet
from config import *
from huggingface_sb3 import load_from_hub
from kuka_state import AbstractKukaState, KukaState
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from utils.helpers import invert_dictionary, state_to_set

from gvg_agents.Search import search
from gvg_agents.sims.gvg_translator import Translator


def saved_plan(function):
    def _saved_plan(self, state1, state2, algo, full_trace=False):
        pkey = (
            str("||".join(sorted(state_to_set(state1.state))))
            + "|||"
            + str("||".join(sorted(state_to_set(state2.state))))
        )
        # doing is None causes an error...
        if self.saved_plans.get(pkey) != None:
            return self.saved_plans.get(pkey)
        a, b = function(self, state1, state2, algo, full_trace)
        self.saved_plans[pkey] = [a, b]
        with open(self.plan_history_file, "wb") as f:
            pickle.dump(self.saved_plans, f)
        return a, b

    return _saved_plan


class KukaTranslator(Translator):
    def __init__(self, model, env, ground_actions=False, files_dir=""):
        super().__init__(AbstractKukaState)
        self.files = files_dir
        self.high_actions = {}
        self.random_states = []
        self.ground_actions = ground_actions
        self.saved_plans = {}
        self.plan_history_file = f"{files_dir}/plans"
        self.environment = env
        self.model = model

    def update_high_actions(self, actions):
        # just to make sure this is called atleast once
        self.high_actions.update(actions)

    def get_next_state(self, state, action):
        """
        might not be needed, or might have to generate a random action... might make sense
        given state and action, apply action virtually and get resulting state
        assume only legal actions applied, including no effect
        """
        self.reset_sim_to_state(state)
        robot = self.environment["robot"]
        sim = self.environment["sim"]

        obs, reward, done, info = self.model.env.step(np.array([0, 0, 0, 0]))

        action, _states = self.model.predict(obs, deterministic=True)

        robot.set_action(action[0])
        sim.step()

        return KukaState(self.environment)

    def get_observation(self):
        robot = self.environment["robot"]
        task = self.environment["task"]

        robot_obs = robot.get_obs().astype(np.float32)  # robot state
        task_obs = task.get_obs().astype(
            np.float32
        )  # object position, velocity, etc...
        observation = np.concatenate([robot_obs, task_obs])
        achieved_goal = task.get_achieved_goal().astype(np.float32)

        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": task.get_goal().astype(np.float32),
        }

    def get_successor(self, state):
        self.reset_sim_to_state(state)
        robot = self.environment["robot"]
        sim = self.environment["sim"]
        obs, reward, done, info = self.model.env.step(np.array([0, 0, 0, 0]))

        action, _states = self.model.predict(obs, deterministic=True)

        robot.set_action(action[0])
        sim.step()

        return action, KukaState(self.environment)

    def is_goal_state(self, current_state, goal_state):
        # all orientations should be corrent goal state
        return current_state.state["finished"]

    # https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.2ye70wns7io3

    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/kukaGymEnv.py

    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/kuka.py

    def reset_sim_to_state(self, state):

        # might need to include more info here, joint angles etc

        sim = self.environment["sim"]
        robot = self.environment["robot"]

        sim.goal = state.state["goal_position"]
        sim.object_position = state.state["block_position"]
        jv = [j[0] for j in state.state["joint_values"]]
        # reset sim to match state
        sim.set_base_pose(
            "target",
            state.state["goal_position"],
            np.array([0.0, 0.0, 0.0, 1.0]),
        )
        sim.set_base_pose(
            "object",
            state.state["block_position"],
            np.array([0.0, 0.0, 0.0, 1.0]),
        )
        robot.set_joint_angles(jv)

    # @saved_plan no need to load in from pickle
    def plan_to_state(
        self, state1, state2, algo="custom-astar", full_trace=False
    ):
        """
        orientation is not considered for goal check, this is done since
        we need to plan only to abstract states which do not differ by orientation
        """
        state1_ = copy.deepcopy(state1)
        state2_ = copy.deepcopy(state2)
        total_nodes_expanded = []
        action_list = []

        # a bit confusing, but the model.env is the env that the model can
        # actually generate actions for
        env = self.model.env
        self.reset_sim_to_state(state1_)
        if algo == "human":
            done = False
            obs, reward, dones, info = env.step([[0, 0, 0, 0]])
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, dones, info = env.step(action)
                done = dones[0]
                action_list.append(action)

        else:
            action_list, total_nodes_expanded = search(
                state1_, state2_, self, algo
            )

        return action_list, total_nodes_expanded

    def execute_from_ID(self, abs_state, abs_action):
        try:
            abs_before, abs_after = self.high_actions[abs_action]
            # checking if just state equality is enough
            # need to be equal?
            if abs_before.state == abs_state.state:
                return True, abs_after
            else:
                return False, abs_state
        except KeyError:
            print("Unknown Action ID!")

    def validate_state(self, ostate):
        """
        Given ABSTRACT STATE, validate it
        assuming cell positioning is correct already, those are not to be learnt anyway
        """
        return True

    # convert low-level state into abstract high-level one
    def abstract_state(self, low_state):
        self.reset_sim_to_state(low_state)
        return AbstractKukaState(self.environment)

    def generate_ds(self):
        """
        assume the actions are assigned
        """
        abstract_model = {}
        action_parameters = {}
        types = {}
        objects = {}
        predTypeMapping = {}
        agent_model = {}
        for action, states in self.high_actions.items():
            for state in states:
                gstate = self.get_ground_state(state)
                for pred in gstate.state:
                    predTypeMapping[pred] = []

        for action in self.high_actions:
            abstract_model[action] = {}
            action_parameters[action] = []
            agent_model[action] = {}
            for pred in predTypeMapping:
                agent_model[action][pred] = [Literal.ABS, Literal.ABS]

        return (
            action_parameters,
            predTypeMapping,
            agent_model,
            abstract_model,
            objects,
            types,
            None,
            "kuka",
        )

    def refine_abstract_state(self, abstract_state_):
        """
        Concretize an input abstract state
        """
        abstract_state = copy.deepcopy(abstract_state_)
        all_keys = [
            "at_0",
            "at_1",
            "at_2",
            "at_3",
            "monster_alive",
            "has_key",
            "escaped",
            "wall",
            "leftOf",
            "rightOf",
            "above",
            "below",
        ]
        refined_state = Zelda_State()
        # for obj in abstract_state.rev_objects:
        #     if abstract_state.rev_objects[obj] in ['location']:
        #         refined_state.rev_objects[obj]=abstract_state.rev_objects[obj]

        refined_state.rev_objects = abstract_state.rev_objects
        refined_state.state["wall"] = abstract_state.state.get("wall")
        refined_state.state["leftOf"] = abstract_state.state.get("leftOf")
        refined_state.state["rightOf"] = abstract_state.state.get("rightOf")
        refined_state.state["above"] = abstract_state.state.get("above")
        refined_state.state["below"] = abstract_state.state.get("below")
        refined_state.state["player_orientation"].append("NORTH")

        if abstract_state.state["escaped"] == None:
            refined_state.state["escaped"] = [False]
        else:
            refined_state.state["escaped"] = [True]

        if abstract_state.state["has_key"] == None:
            refined_state.state["has_key"] = [False]
        else:
            refined_state.state["has_key"] = [True]

        if abstract_state.state.get("at_0") != None:
            for pair in abstract_state.state["at_0"]:
                refined_state.state["player"].append(pair[1])
                # refined_state.rev_objects[pair[0]]='sprite'
        if abstract_state.state.get("at_1") != None:
            for pair in abstract_state.state["at_1"]:
                refined_state.state["key"].append(pair[1])
                # refined_state.rev_objects[pair[0]]='sprite'
        if abstract_state.state.get("at_3") != None:
            for pair in abstract_state.state["at_3"]:
                refined_state.state["door"].append(pair[1])
                # refined_state.rev_objects[pair[0]]='sprite'
        if abstract_state.state.get("at_2") != None:
            for pair in abstract_state.state["at_2"]:
                refined_state.state["monster"].append((pair[0], pair[1]))
                # refined_state.rev_objects[pair[0]]='sprite'

        if abstract_state.state["clear"] != None:
            for cell in abstract_state.state["clear"]:
                refined_state.state["clear"].append(cell[0])

        refined_state.grid_height = abstract_state.grid_height
        refined_state.grid_width = abstract_state.grid_width
        for k, v in refined_state.state.items():
            if (v) == None:
                print(str(k) + " is empty")
                refined_state.state[k] = []
        refined_state.objects = invert_dictionary(refined_state.rev_objects)
        return refined_state

    def get_relational_state(self, state):
        rstate = AbstractZeldaState()
        rstate.grid_height = 0
        rstate.grid_width = 0
        for p in state.state:
            pred = p.split("-")[0]
            params = p.replace(pred, "").split("-")[1:]
            if params != [""]:
                v = []
                for _p in params:
                    if len(_p) != 0:
                        v.append(_p)
                        if "cell" in _p:
                            x = int(_p.split("_")[1]) + 1
                            y = int(_p.split("_")[2]) + 1
                            rstate.rev_objects[_p] = "location"
                            rstate.grid_height = max(rstate.grid_height, y)
                            rstate.grid_width = max(rstate.grid_width, x)
                        else:
                            rstate.rev_objects[_p] = _p[:-1]
                rstate.state[pred].append(tuple(v))
            else:
                if rstate.state[pred] == None:
                    rstate.state[pred] = [()]
                else:
                    rstate.state[pred].append(tuple([]))
        temp_k = []
        for k, v in rstate.state.items():
            if len(v) == 0:
                temp_k.append(k)
        [rstate.state.pop(k_, None) for k_ in temp_k]
        return rstate

    # TODO: figure out what ground state is
    def get_ground_state(self, state):
        gstate = copy.deepcopy(state)
        """
        gstate.state = {}
        for k, v in state.state.items():
            if v != None:
                p = k
                for _v in v:
                    gstate.state[k + "-" + "-".join(list(_v))] = [()]
        gstate.objects = {}
        gstate.rev_objects = {}
        """
        return gstate

    def iaa_query(self, abs_state, plan):
        """
        state: abstract
        plan: hashed values corresponding to stored actions
        """
        if self.validate_state(abs_state):
            state = copy.deepcopy(abs_state)
            for i, action in enumerate(plan):
                """
                can check plan possibility here itself
                if subsequent states are not equal, can't execute
                """
                can_execute, abs_after = self.execute_from_ID(state, action)
                if can_execute:
                    state = abs_after
                else:
                    return False, i, abs_after  # check from sokoban code
            return True, len(plan), abs_after  # check from sokoban code
        else:
            return False, 0, abs_stateclass
