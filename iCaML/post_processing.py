# BOILING_WATER,RAW_PASTA,TOMATO,TUNA,PASTA_IN_PLACE,SAUCE_IN_PLACE,PASTA_DONE,PLAYER,\
#     KEY,DOOR,TOP_PIECE,MIDDLE_PIECE,BOTTOM_PIECE,BLOCK,HOLE,WALL,CLEAR,MONSTER,LOCK = range(19)
import copy
import subprocess
from collections import OrderedDict, defaultdict

# from .utils_sokoban import get_asset_path, render_from_layout
# from .utils import get_asset_path, render_from_layout
import matplotlib.pyplot as plt
import numpy as np
from config import *
from config import Literal
from utils.file_utils import FileUtils

# def lift_action():


class ExecuteSimplePlan:
    """
    This class executes a plan on a model sarting at an initial state, no precondition checks since
    model is directly used to plan

    :param targetModel: an instance of class Model on which plan is to be executed
    :type targetModel: object of class Model
    :param init_state: Initial state (list of predicates)
    :type init_state: list of strs
    :param rawPlan: list of actions
    :type rawPlan: list of strs
    """

    def __init__(self, actions, init_state, rawPlan):
        """
        This method creates a new instance of ExecutePlan.

        """

        self.init_state = []
        for p, v in init_state.items():
            for items in v:
                t_init_state = p
                for i in items:
                    t_init_state += "|" + i
                self.init_state.append(t_init_state)

        self.actions = actions
        self.plan = rawPlan

    def execute_plan(self):
        """
        This method calculates the state after a plan is executed.
        This only works for add delete lists in preconditions and effects.

        """

        actions = copy.deepcopy(self.actions)
        actions = {k.lower(): v for k, v in actions.items()}

        def applyAction(actions, state, p):
            plan_split_list = p.split("|")
            action_name = plan_split_list[0]
            action_params = plan_split_list[1:]

            actionPred_original = actions[action_name]

            actionPreds = {}
            for pred, v in actionPred_original.items():
                temp_pred = pred.split("|")[0]
                type_pos = pred.rstrip("|").split("|")[1:]
                for type_positions in type_pos:
                    temp_pred += "|" + action_params[int(type_positions)]
                if temp_pred in actionPreds.keys():
                    v1 = actionPreds[temp_pred]
                    v2 = v
                    if v1 == [Literal.ABS, Literal.ABS]:
                        actionPreds[temp_pred] = v2
                    elif v2 == [Literal.ABS, Literal.ABS]:
                        actionPreds[temp_pred] = v1
                    else:
                        return False, None
                else:
                    actionPreds[temp_pred] = v

            tempState = copy.deepcopy(state)

            for pred, val in actionPreds.items():
                t_value = copy.deepcopy(val)
                if t_value[1] == Literal.AN or t_value[1] == Literal.AP:
                    t_value[1] = Literal.ABS
                elif t_value[1] == Literal.NP:
                    t_value[1] = Literal.POS

                if t_value[1] == Literal.POS:
                    tempState.add(pred)
                elif t_value[1] == Literal.NEG:
                    # If it was absent in precondition, we can make it negative.
                    if pred in tempState:
                        tempState.remove(pred)
                    elif t_value[0] == Literal.ABS:
                        continue
                    else:
                        return False, None

            return True, tempState

        initialState = set(self.init_state)
        currState = copy.deepcopy(initialState)

        plan_index = 0
        for p in self.plan:
            is_ok, newState = applyAction(actions, currState, p)
            if is_ok == False:
                return False, None, None
            currState = copy.deepcopy(newState)
            plan_index += 1
        return True, currState, plan_index


def compare_dict(d1, d2):
    in_d1_only = defaultdict(list)
    in_d2_only = defaultdict(list)

    for pred, values in d1.items():
        if pred not in d2:
            in_d1_only[pred] = values
        else:
            for v in values:
                if v not in d2[pred]:
                    in_d1_only[pred].append(v)
    for pred, values in d2.items():
        if pred not in d1:
            in_d2_only[pred] = values
        else:
            for v in values:
                if v not in d1[pred]:
                    in_d2_only[pred].append(v)

    print("in first dict only:")
    print(in_d1_only)
    print("in second dict only:")
    print(in_d2_only)


# TOKEN_IMAGES = {
#     BOILING_WATER : plt.imread(get_asset_path('sokoban_stone.png')),
#     RAW_PASTA : plt.imread(get_asset_path('sokoban_stone_at_goal.png')),
#     TOMATO : plt.imread(get_asset_path('sokoban_goal.png')),
#     TUNA:plt.imread(get_asset_path('sokoban_stone.png')),
#     PASTA_IN_PLACE:plt.imread(get_asset_path('sokoban_stone.png')),
#     SAUCE_IN_PLACE:plt.imread(get_asset_path('sokoban_stone.png')),
#     PASTA_DONE:plt.imread(get_asset_path('sokoban_stone.png')),
#     PLAYER:plt.imread(get_asset_path('sokoban_player.png')),
#     KEY:plt.imread(get_asset_path('sokoban_stone.png')),
#     DOOR:plt.imread(get_asset_path('sokoban_stone.png')),
#     TOP_PIECE:plt.imread(get_asset_path('sokoban_stone.png')),
#     MIDDLE_PIECE:plt.imread(get_asset_path('sokoban_stone.png')),
#     BOTTOM_PIECE:plt.imread(get_asset_path('sokoban_stone.png')),
#     BLOCK:plt.imread(get_asset_path('sokoban_stone.png')),
#     HOLE:plt.imread(get_asset_path('sokoban_stone.png')),
#     WALL : plt.imread(get_asset_path('sokoban_wall.png')),
#     CLEAR : plt.imread(get_asset_path('sokoban_clear.png')),
#     MONSTER:plt.imread(get_asset_path('robot.png')),
#     LOCK: plt.imread(get_asset_path('robot.png'))
# }


zelda = ["wall", "player", "monster", "key", "door", "has_key"]
cookmepasta = [
    "wall",
    "player",
    "raw_pasta",
    "boiling_water",
    "tomato",
    "tuna",
    "pasta_in_place",
    "sauce_in_place",
    "pasta_done",
]
escape = ["wall", "player", "block", "hole", "door"]
snowman = [
    "bottom_piece",
    "middle_piece",
    "top_piece",
    "key",
    "goal",
    "wall",
    "has_key",
    "player",
    "lock",
]
IM_SCALE = 0.25


def apply_effect(state, add, dele):
    [state.pop(d, None) for d in dele]
    for a in add:
        state[a] = [()]
    return state


def can_apply_action(action, state, final_state):
    required_precondition = []
    negative_precondition = []
    add_effects = []
    del_effects = []
    for pred, vals in action.items():
        if vals[0] == Literal.POS:
            required_precondition.append(pred)
        if vals[0] == Literal.NEG:
            negative_precondition.append(pred)
        if vals[1] == Literal.POS:
            add_effects.append(pred)
        if vals[1] == Literal.NEG:
            del_effects.append(pred)
    s1 = set(required_precondition).difference(set(state.keys()))
    s2 = set(negative_precondition).intersection(set(state.keys()))
    print("needed in precondition but not present:")
    print(s1)
    print("should not be present in state:")
    print(s2)
    result = apply_effect(state, add_effects, del_effects)
    compare_dict(result, final_state)


def auto_name_action():

    return


def remove_extra_preconds(model, agent):

    for action, preds in model.actions.items():
        effects = []
        for action, preds in model.actions.items():
            modified_objs = agent.agent_model.action_objects[
                action
            ].modified_objects
            effects = []
            for pred in preds:
                objects = pred.split("-")[1:]
                if (
                    not (set(objects).issubset(set(modified_objs.keys())))
                    and len(objects) != 0
                ):
                    model.actions[action][pred][0] = Literal.ABS

    return model


def validate_model():
    return


def write_problem_to_file(
    f,
    init_state,
    final_state,
    domain_name,
    pred_type_mapping,
    action_parameters,
    objects=None,
):
    """
    This method creates files.

    :param fd: file descriptor of the pddl file in which problem will be written
    :type fd: file descriptor
    :param domain_name: domain name of the model
    :type domain_name: str

    :rtype: None

    """
    fd = open(f, "w")
    problemName = "problem0"
    fd.write("(define (problem " + problemName + ")\n")
    ####### Domain #######
    fd.write("  (:domain " + domain_name + ")\n")

    ####### Objects #######
    fd.write("  (:objects ")

    k = 0
    if objects == None:
        objects = {}
    for t, vals in objects.items():
        if len(vals) == 0:
            # This case happens with domains like logistics
            # Here physobj has no actual object
            continue
        if k > 0:
            for k in range(len("  (:objects ")):
                fd.write(" ")
        for v in vals:
            fd.write(v + str(" "))
        fd.write(" - " + t + " ")
        k += 1
    fd.write(")\n")

    fd.write("  (:init ")
    it = ""
    for pred, values in init_state.items():
        for v in values:
            for _ in range(len("  (:init")):
                it += " "
            it += "  ("
            it += pred
            for _v in v:
                it += "-" + _v
            it += " )\n"
    it += ")"
    fd.write(it)
    fd.write("  (:goal (and\n")
    it = "   \n"
    for pred, values in final_state.items():
        for v in values:
            for _ in range(len("  (:goal")):
                it += " "
            it += "  ("
            it += pred
            for _v in v:
                it += "-" + _v
            it += " )\n"
    fd.write(it)
    fd.write(")\n")
    fd.write(")\n")
    fd.write(")\n")
    fd.close()


def write_model_to_file(
    model, fd, domain_name, pred_type_mapping, action_parameters, objects=None
):
    """
    This method creates files.

    :param fd: file descriptor of the pddl file in which model will be written
    :type fd: file descriptor
    :param domain_name: domain name of the model
    :type domain_name: str
    :param pred_type_mapping:
    :type pred_type_mapping:
    :param action_parameters:
    :type action_parameters:
    :param objects:
    :type objects:

    :rtype: None

    """
    if objects is None:
        objects = dict()
    fd.write("(define (domain " + domain_name + ")\n")
    fd.write("(:requirements :strips :typing :equality)\n")

    # Typing
    fd.write("(:types")
    for t in objects.keys():
        fd.write(" " + t)
    fd.write(")\n")
    skip_preds = ["leftOf", "rightOf", "above", "below"]
    # Predicates
    fd.write("(:predicates ")
    count = 0
    preds_printed = []
    for key, value in model.predicates.items():
        params = ""
        cnt = 0
        pred_name = key.split("|")[0]
        if pred_name in preds_printed:
            continue
        else:
            preds_printed.append(pred_name)

        if pred_name.split("-")[-1] in ["1", "2"]:
            actual_pred_name_splits = pred_name.split("-")[0:-1]
            actual_pred_name = "_".join(actual_pred_name_splits)
        else:
            actual_pred_name = pred_name
        for val in pred_type_mapping[actual_pred_name]:
            params = params + " ?" + val[0] + str(cnt) + " - " + val
            cnt += 1

        if count > 0:
            fd.write("\n")
            for k in range(len("(:predicates ")):
                fd.write(" ")
        fd.write("(" + pred_name + params + ")")
        count += 1
    fd.write(")\n\n")

    # Actions
    for actionName, predicateDict in model.actions.items():
        head = "(:action " + actionName + "\n" + "  :parameters"
        fd.write(head)
        type_count = {}
        param_ordering = []
        for p in action_parameters[actionName]:
            if p not in type_count.keys():
                type_count[p] = 1
            else:
                type_count[p] = type_count[p] + 1
            param_ordering.append(p + str(type_count[p]))

        fd.write(" (")
        head = ""
        param_count = len(action_parameters[actionName])
        for i in range(param_count):
            if i > 0:
                for k in range(len("  :parameters (")):
                    head += " "
            head += (
                "?"
                + param_ordering[i]
                + " - "
                + action_parameters[actionName][i]
                + "\n"
            )
        for k in range(len("  :parameters ")):
            head += " "
        head += ")\n"
        fd.write(head)

        fd.write("  :precondition (and")
        equality_needed = False
        if param_count > 1:
            equality_needed = True

        if equality_needed:
            combs = combinations(list(range(0, param_count)), 2)
            for c in combs:
                fd.write("(not (= ")
                for j in range(2):
                    i = c[j]
                    fd.write("?" + param_ordering[i])
                    if j == 0:
                        fd.write(" ")
                    else:
                        fd.write(")) ")

        for predicate, value in predicateDict.items():
            if (
                predicate.split("-")[0] in skip_preds
            ):  # or 'is_' in predicate.split("-")[0]:
                continue
            pred_split = predicate.split("|")
            pred_name = pred_split[0]

            t_value = value[:]
            if t_value[0] != Literal.ABS:
                param = " ("
                if t_value[0] == Literal.NEG:
                    param += "not ("
                elif t_value[0] == Literal.AN:
                    param += "0/- ("
                elif t_value[0] == Literal.AP:
                    param += "0/+ ("
                elif t_value[0] == Literal.NP:
                    param += "+/- ("
                param += pred_name

                if len(pred_split) > 1:
                    pred_params = pred_split[1:]
                    for p in pred_params:
                        print(p)
                        param += " ?" + param_ordering[int(p)]
                param += ")\n       "
                if t_value[0] != Literal.ABS and t_value[0] != Literal.POS:
                    param += ")\n       "
                fd.write(param)
        fd.write(")\n       ")

        fd.write("  :effect (and")
        for predicate, value in predicateDict.items():
            if (
                predicate.split("-")[0] in skip_preds
                or "is_" in predicate.split("-")[0]
            ):
                continue
            pred_split = predicate.split("|")
            pred_name = pred_split[0]
            if value[1] != Literal.ABS:
                param = " ("
                if value[1] == Literal.NEG:
                    param += "not ("
                param += pred_name

                if len(pred_split) > 1:
                    pred_params = pred_split[1:]
                    for p in pred_params:
                        param += " ?" + param_ordering[int(p)]

                param += ")\n       "
                if value[1] == Literal.NEG:
                    param += ")\n       "
                fd.write(param)
        fd.write("))\n\n\n       ")
    fd.write(")\n\n       ")


def call_planner(domain_file, problem_file, result_file):
    """
    This method calls the planner.
    The planner can be either FF Planner (ff) or Madagascar (mg).
    It needs to be set in config.py in the root directory.

    :param domain_file: domain file (operator file) for the planner
    :type domain_file: str
    :param problem_file: problem file (fact file) for the planner
    :type problem_file: str
    :param result_file: result file to store output of the planner
    :type result_file: str

    :rtype: None

    """
    if PLANNER == "FF":
        param = "ff-planner"
        param += " -o " + domain_file
        param += " -f " + problem_file
        param += " > " + result_file

    elif PLANNER == "FD":
        param = FD_PATH + "fast-downward.py "
        param += " --plan-file ../" + FD_SAS_FILE
        param += " --alias seq-sat-lama-2011"
        param += " " + domain_file
        param += "  " + problem_file
        # param += " --search \"astar(lmcut(), verbosity=silent)\""

    else:
        print("Error: No planner provided")
        exit()
    p = subprocess.Popen([param], shell=True)
    p.wait()
    plan = FileUtils.get_plan_from_file(result_file)
    return plan
