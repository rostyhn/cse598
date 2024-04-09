import copy
import os
import pickle

from kuka_translator import KukaTranslator

from iCaML.kuka_state import AbstractKukaState, KukaState
import random

class KukaAgent:
    def __init__(self, env, motors, ground_actions=False):
        self.environment = env
        self.motors = motors
        self.ground_actions = ground_actions
        files_dir = os.path.abspath("../results/kuka/")
        if not os.path.exists(files_dir):
            os.mkdir(files_dir)
        self.translator = KukaTranslator(
            env, motors, ground_actions=ground_actions, files_dir=files_dir
        )
        self.random_states_file = f"{files_dir}random_states"
        self.traces_file = f"{files_dir}test_trace"
        self.high_actions_dict = f"{files_dir}high_actions_dict"
        self.high_traces = f"{files_dir}high_traces"

        self.random_states = []
        self.avg_trace_length = 0.0
        self.num_traces = 0
        num_random_states = 1
        self.ground_to_relational_map = {}
        self.query_history_file = f"{files_dir}queries"
        self.queries = {}
        self.saved_correct_states = {}
        self.saved_correct_states_file = f"{files_dir}saved_correct_states"
        try:
            with open(self.query_history_file, "rb") as f:
                self.queries = pickle.load(f)
        except IOError:
            print("No old queries to load")

        try:
            with open(self.random_states_file, "rb") as f:
                temp_states = pickle.load(f)
        except IOError:
            temp_states = self.generate_random_states(
                n=num_random_states,
                algo="human",
                abstract=True,
                random=True,
                save_trace=True,
            )
        ###generate additional states for data
        """
        n_extra = 0
        self.load_actions()
        self.combine_actions()
        temp_states_extra = self.generate_random_states(n = n_extra, r = r,c = c,min_walls=min_walls,add_intermediate=True,abstract = True,random = True, save_trace = False)
        self.show_actions()
        with open(files_dir+"temp_states_extra","wb") as f:
            pickle.dump(temp_states_extra,f)
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

    # TODO: build random state generator
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
        try:
            with open(self.random_states_file, "rb") as f:
                old_random_states = pickle.load(f)
        except IOError:
            old_random_states = []

        if save_trace:
            try:
                with open(self.traces_file, "rb") as f:
                    old_traces = pickle.load(f)
            except IOError:
                old_traces = []
        else:
            old_traces = []

        new_traces = []
        initial_random_states = []

        if add_intermediate:
            abs_random_states = []
            solved_random_states = []
            num_random_traces = 0
            add_rs = []
            # get intermediate states
            for i in range(n):
                # environment.reset() always returns a random state
                self.environment.reset() 
                s = KukaState(self.environment._p.saveState())
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

        return final_random

    def get_solved_state(self, state):
        temp_state = AbstractKukaState() 
        temp_state.state["grasped"] = [True]
        return temp_state

    def solve_game(self, state, _actions=False, algo="custom-astar"):
        states = []
        self.environment._p.restoreState(state.state["stateID"])
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
            states.append(tstate)
        if not _actions:
            return states
        else:
            return states, actions


    # https://github.com/bulletphysics/bullet3/discussions/3814
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_robots/panda/batchsim3_grasp.py
    def get_random_trace(self, state):
        max_len = 50
        trace = []
        for _ in range(max_len):
            succ = self.translator.get_successor(state)
            choice = random.choice(list(succ.keys()))
            if state.state["escaped"][0]:
                trace.append((succ[choice][1], "ACTION_ESCAPE"))
                state = succ[choice][1]
                break
            else:
                trace.append((succ[choice][1], choice))
                state = succ[choice][1]
        return trace
