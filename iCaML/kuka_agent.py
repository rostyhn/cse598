import copy
import os
import pickle
import random
import time

import numpy as np
import pybullet
from kuka_translator import KukaTranslator

from iCaML.action import Action
from iCaML.kuka_state import AbstractKukaState, KukaState


class KukaAgent:
    def __init__(self, model, env, ground_actions=False):
        self.model = model
        self.environment = env
        self.ground_actions = ground_actions
        files_dir = os.path.abspath("../results/kuka/")
        if not os.path.exists(files_dir):
            os.mkdir(files_dir)

        self.translator = KukaTranslator(
            model, env, ground_actions=ground_actions, files_dir=files_dir
        )
        self.random_states_file = f"{files_dir}/random_states"
        self.traces_file = f"{files_dir}/test_trace"
        self.high_actions_dict = f"{files_dir}/high_actions_dict"
        self.high_traces = f"{files_dir}/high_traces"

        self.random_states = []
        self.avg_trace_length = 0.0
        self.num_traces = 1
        num_random_states = 5
        self.ground_to_relational_map = {}
        self.query_history_file = f"{files_dir}/queries"
        self.queries = {}
        self.saved_correct_states = {}
        self.saved_correct_states_file = f"{files_dir}/saved_correct_states"
        try:
            with open(self.query_history_file, "rb") as f:
                self.queries = pickle.load(f)
        except IOError:
            print("No old queries to load")

        temp_states = self.generate_random_states(
            n=num_random_states,
            algo="human",
            abstract=True,
            random=True,
            save_trace=True,
        )

        # now we generate high-level actions...
        self.load_actions()
        self.combine_actions()

        # if ground_actions:
        #    for state in temp_states:
        #       temp_gstate = self.translator.get_ground_state(state)
        #      self.random_states.append(temp_gstate)
        #     self.ground_to_relational_map[temp_gstate] = state
        # else:
        self.random_states = temp_states

        (
            self.action_parameters,
            self.predicates,
            _,
            _,
            self.objects,
            self.types,
            _,
            _,
        ) = self.generate_ds()
        """
        this just generates more states, not sure if needed
        n_extra = 0
        temp_states_extra = self.generate_random_states(n = n_extra, r = r,c = c,min_walls=min_walls,add_intermediate=True,abstract = True,random = True, save_trace = False)
        with open(files_dir+"temp_states_extra","wb") as f:
            pickle.dump(temp_states_extra,f)

        # not sure what grounding does, need to figure that out
        temp_states.extend(temp_states_extra)
        if ground_actions:
            for state in temp_states:
                temp_gstate = self.translator.get_ground_state(state)
                self.random_states.append(temp_gstate)
                self.ground_to_relational_map[temp_gstate] = state
        else:
            self.random_states = temp_states
        # if len(temp_states) == 0:
        #     self.random_states = self.generate_random_states(n = num_random_states, abstract = True,random = True, save_trace = True)
        # else:
        #     self.random_states = temp_states
        self.action_parameters, self.predicates, _, _, self.objects, self.types, _, _ = self.generate_ds()
        print("number of random states: "+str(len(self.random_states)))
        for st in self.random_states:
            temp_k = []
            for k,v in st.state.items():
                if v == None:
                    temp_k.append(k)
            [st.state.pop(k_,None) for k_ in temp_k]
        """

    def generate_ds(self):
        if len(self.translator.high_actions) != 0:
            return self.translator.generate_ds()
        else:
            print("Actions not stored yet")
            return False

    def load_actions(self):
        # generates high level actions
        file = self.traces_file
        with open(file, "rb") as f:
            test_trace = pickle.load(f)
        new_test_traces = []
        for i, run in enumerate(test_trace):
            sas_trace = []
            # first_state = run[0][0]
            # objects = first_state.rev_objects
            # monster_mapping = first_state.monster_mapping
            self.avg_trace_length += len(run)
            for j, (sa1, sa2) in enumerate(zip(run, run[1:])):
                state1, action1 = sa1
                state2, action2 = sa2
                sas_trace.append([state1, action1, state2])
                new_test_traces.append(sas_trace)

        high_level_traces = []
        high_level_actions = {}

        action_number = 0
        for trace in new_test_traces:
            abs_trace = []
            for s1, a, s2 in trace:
                abs_s1 = self.translator.abstract_state(s1)
                abs_s2 = self.translator.abstract_state(s2)
                if abs_s1 != abs_s2:
                    # create a new action
                    action_id = "a" + str(action_number)  # str(uuid.uuid1())
                    action_number += 1
                    high_level_actions[action_id] = [
                        abs_s1,
                        abs_s2,
                    ]
                    abs_trace.append((abs_s1, action_id, abs_s2))
            high_level_traces.append(abs_trace)
            self.avg_trace_length += len(abs_trace)

        self.num_traces = len(high_level_traces)
        self.avg_trace_length = 0.0
        self.avg_trace_length /= self.num_traces

        with open(self.high_actions_dict, "wb") as f:
            pickle.dump(high_level_actions, f)

        with open(self.high_traces, "wb") as f:
            pickle.dump(high_level_traces, f)

        self.translator.update_high_actions(high_level_actions)
        print(len(high_level_traces))
        print(len(high_level_actions))
        print("Saved High-level actions as traces")

    def combine_actions(self):
        if len(self.translator.high_actions) == 0:
            print("Actions dict not saved yet!")
            return False

        self.actions = {}
        self.action_objects = {}
        for action, s in self.translator.high_actions.items():
            temp_action = Action(action, s[0], s[1])
            # print(action, s[0].rev_objects, s[1].rev_objects)
            temp_action.assign_predicate_types()

            if temp_action not in self.action_objects.values():
                self.actions[action] = s
                self.action_objects[action] = temp_action
            else:
                print(f"Pruned {str(action)}")

    def show_actions(self, action=None):
        sim = self.environment["sim"]
        for k, v in self.actions.items():
            s1, s2 = v
            print(s1, s2)
            print(f"------action_name:{str(k)}--------")
            print("before")
            sim.restore_state(s1.pyBulletStateID)
            time.sleep(3)
            print("after")
            sim.restore_state(s2.pyBulletStateID)
            time.sleep(3)

    def generate_random_states(
        self,
        n=5,
        save=True,
        abstract=False,
        random=False,
        save_trace=False,
        add_intermediate=True,
        algo="custom_astar",
    ):

        old_random_states = []
        old_traces = []

        new_traces = []
        initial_random_states = []

        if add_intermediate:
            abs_random_states = []
            solved_random_states = []
            num_random_traces = 1
            add_rs = []
            # get intermediate states
            for i in range(n):
                # as discussed in kuka.py, the states generated by this aren't random enough
                s = self.get_random_state()
                st, actions = self.solve_game(s, _actions=True, algo="human")

                if st is not False:
                    solved_random_states.extend(st)
                    new_traces.append(list(zip(st, actions)))

                for _ in range(num_random_traces):
                    tr = self.get_random_trace(s)
                    for s_, a in tr:
                        if s_ not in initial_random_states:
                            add_rs.append(s_)
                    new_traces.append(tr)
            solved_random_states.extend(add_rs)
            # convert new random states to abstract form
            if abstract:
                for s in solved_random_states:
                    s_ = self.translator.abstract_state(s)
                    if (
                        s_ not in old_random_states
                        and s_ not in abs_random_states
                    ):
                        abs_random_states.append(s_)
                solved_random_states = abs_random_states

            # append to old pickled states
            if save_trace:
                old_traces.extend(new_traces)
                with open(self.traces_file, "wb") as f:
                    pickle.dump(old_traces, f)
            old_random_states.extend(solved_random_states)
            final_random = solved_random_states
        else:
            final_random = []
            if abstract:
                for s in initial_random_states:
                    s_ = self.translator.abstract_state(s)
                    if s_ not in old_random_states:
                        final_random.append(s_)
            old_random_states.extend(final_random)

        with open(self.random_states_file, "wb") as f:
            pickle.dump(old_random_states, f)
        print("finished final random")
        return final_random

    def get_random_state(self):
        # randomize blocks
        self.model.env.reset()
        # randomize joint vals
        s = KukaState(self.environment)

        # s.state["joint_values"] = [
        #    (robot.get_joint_angle(j) + random.uniform(-0.25, 0.25),)
        #    for j in robot.joint_indices
        # ]
        return s

    def get_solved_state(self, state):
        # low level solved state
        temp_state = KukaState(self.environment)
        temp_state.state["finished"] = np.array(True)
        temp_state.state["block_position"] = state.state["goal_position"]

        # does it matter what the positions of the joints are here?
        # if it does, we need to actually solve the env first before returning

        return temp_state

    def solve_game(self, state, _actions=False, algo="custom-astar"):
        states = []
        final_state = self.get_solved_state(state)
        actions, total_nodes_expanded = self.translator.plan_to_state(
            state, final_state, algo
        )

        states.append(state)
        cstate = copy.deepcopy(state)
        if actions is None:
            # print(plot_state(state))
            print("Unsolvable!")
            if _actions:
                return False, False
            else:
                return False
        # print(plot_state(state))
        print("Solved!")
        print(actions)
        # why not just drop this and return the states along the way?
        for a in actions:
            tstate = self.translator.get_next_state(cstate, a)
            cstate = tstate
            states.append(tstate)
        
        # hack to mark last state as finished to get around env
        states[-1].state["finished"] = True
        if not _actions:
            return states
        else:
            return states, actions

    def get_random_trace(self, state):
        max_len = 50
        trace = []
        for _ in range(max_len):
            action, succ = self.translator.get_successor(state)
            trace.append((succ, action))
            state = succ
            if succ.state["finished"]:
                break
        return trace

    # no clue what this does but it works...
    # might make sense to move to another file
    def bootstrap_model(self):
        abs_preds_test = {}
        abs_actions_test = {}
        pal_tuples_fixed = []
        fix_preds = ["leftOf", "rightOf", "above", "below"]

        for action, states in self.translator.high_actions.items():
            s_before = self.translator.get_ground_state(states[0])
            s_after = self.translator.get_ground_state(states[1])
            abs_actions_test[action] = {}
            for k, v in s_before.state.items():
                if k.split("-")[0] in fix_preds:
                    abs_preds_test[k] = 0
                    abs_actions_test[action][k] = [Literal.POS, Literal.ABS]
                    pal_tuples_fixed.append((action, k, Location.PRECOND))
            for k, v in s_after.state.items():
                if k.split("-")[0] in fix_preds:
                    abs_preds_test[k] = 0
                    if k in abs_actions_test[action]:
                        abs_actions_test[action][k][1] = Literal.ABS
                    else:
                        abs_actions_test[action][k] = [
                            Literal.ABS,
                            Literal.ABS,
                        ]
                    pal_tuples_fixed.append((action, k, Location.EFFECTS))

        return abs_preds_test, abs_actions_test, pal_tuples_fixed
