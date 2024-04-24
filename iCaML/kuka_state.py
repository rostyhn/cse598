import pickle
from collections import defaultdict

import numpy as np
from lattice import State
from panda_gym.utils import distance
from utils.helpers import state_to_set

# partitions on the table

q1 = np.array([0.15, 0.15, 0.0])
q2 = np.array([0.15, -0.15, 0.0])
q3 = np.array([-0.15, 0.15, 0.0])
q4 = np.array([-0.15, -0.15, 0.0])


class AbstractKukaState(State):
    def __init__(self, env):

        sim = env["sim"]
        robot = env["robot"]
        task = env["task"]

        ee = robot.get_ee_position()
        fw = robot.get_fingers_width()
        block = task.get_achieved_goal()
        goal = np.array(sim.get_base_position("target"))

        distance_to_block = distance(ee, block)

        self.state = defaultdict(None)
        is_succ = task.is_success(block, goal)
        is_grasping = (fw > 0.02 and fw < 0.04) and distance_to_block < 0.01
        is_closed = fw < 0.01
        tstate = {}

        # super weird, but if a predicate is true without parameters,
        # its just an empty tuple
        # and yes, all the values are lists

        if is_succ:
            tstate["finished"] = [()]
        if is_grasping:
            tstate["is_grasping"] = [()]
        if is_closed:
            tstate["is_closed"] = [()]

        # get nearest quadrant
        dq = [
            distance(ee, q1),
            distance(ee, q2),
            distance(ee, q3),
            distance(ee, q4),
        ]

        dq_rank = [0, 1, 2, 3]
        dq_rank.sort(key=lambda i: dq[i])

        quadrant = dq.index(min(dq)) + 1
        tstate[f"in_quad-{quadrant}"] = [()]

        # "effector_pos": ee,
        # "finger_width": fw,
        # "d_q1": [(distance(ee, q1),)],
        # "d_q2": [(distance(ee, q2),)],
        # "d_q3": [(distance(ee, q3),)],
        # "d_q4": [(distance(ee, q4),)],
        # "distance_to_block": [(distance(ee, block),)],

        # "block_to_goal": [(distance(block, goal),)],
        # you add this as a key iff its true
        # "is_grasping": [(fw > 0.02 and fw < 0.04,)],
        # "is_closed": [(fw < 0.01)],
        # "finished": (task.is_success(block, goal),),

        for k, v in tstate.items():
            self.state[k] = v
        self.rev_objects = {}  #
        self.objects = {}
        self.pyBulletStateID = sim.save_state()

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, abstract2):
        """
        TODO:
            Check for equivalency, not equality
            when dealing with float values this becomes important!
        """
        for pred in self.state:
            if pred in abstract2.state:
                if isinstance(self.state[pred], list):
                    if sorted(abstract2.state[pred]) != sorted(
                        self.state[pred]
                    ):
                        return False
                else:
                    if abstract2.state[pred] != self.state[pred]:
                        return False
            else:
                return False
        if self.rev_objects != abstract2.rev_objects:
            return False

        return True

    def __str__(self):
        s = ""
        for k, v in self.state.items():
            s += f"{k}: {str(v)}\n"
        return s


def invert_dictionary(idict):
    odict = defaultdict(list)
    for k, v in idict.items():
        odict[v].append(k)
    return odict


# these are low-level states
class KukaState:
    def __eq__(self, state2):
        try:
            for pred, vals in self.state.items():
                if isinstance(vals, list):
                    if sorted(vals) != sorted(state2.state[pred]):
                        return False
                else:
                    if state2.state[pred] != self.state[pred]:
                        return False
            if self.rev_objects != state2.rev_objects:
                return False
            return True
        except AttributeError:
            return False

    def __str__(self):
        s = ""
        for k, v in self.state.items():
            s += f"{k}: {str(v)}\n"
        return s

    def __hash__(self):
        return hash(str(self))

    def __init__(self, env):
        sim = env["sim"]
        task = env["task"]
        robot = env["robot"]

        block = task.get_achieved_goal()
        goal = np.array(sim.get_base_position("target"))

        self.state = {
            "finished": task.is_success(block, goal),
            "goal_position": goal,
            "block_position": block,
            # might need to make each joint angle an individual thing
            # again, list of tuples weirdness
            "joint_values": [
                (robot.get_joint_angle(j),) for j in robot.joint_indices
            ],
        }
        self.g_score = 0  # for search
        self.best_path = None  # for search
        self.rev_objects = {}


def save_query(function):
    def _save_query(self, query):
        qkey = (
            str("||".join(sorted(state_to_set(query["init_state"].state))))
            + "|||"
            + str("||".join(query["plan"]))
        )
        a, b, c = function(self, query)
        self.queries[qkey] = [a, b, c]
        with open(self.query_history_file, "wb") as f:
            pickle.dump(self.queries, f)
        return a, b, c

    return _save_query
