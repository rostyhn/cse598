#!/usr/local/bin/python3
# encoding: utf-8

import copy
import importlib

#import cv2
from PIL import Image

from config import *
from query import ExecutePlan
from lattice import Model
#fromsim import StateHelper
from utils import state_to_set

class Agent:
    def __init__(self, domain, env, motors):
        if is_simulator_agent:
            self.agent_type = "simulator"
            from kuka_agent import KukaAgent 
            self.agent_model = KukaAgent(env, motors)

    def run_query(self, query, pal_tuple_dict, partial_check=False):
        """
        Added partial_Check so that is_simulator_agent can check partial init state for correctness.
        :param query:
        :param pal_tuple_dict:
        :param partial_check:
        :return:
        """
        if self.agent_type == "symbolic":
            plan = ExecutePlan(self.agent_model, query['init_state'].state, query['plan'])
            is_executable_agent, possible_state, failure_index = plan.execute_plan(pal_tuple_dict)
            return is_executable_agent, failure_index, possible_state

        # elif self.agent_type == "simulator":
        #     if self.agent_model.validate_state(query['init_state'], partial_check):
        #         temp_query = copy.deepcopy(query)
        #         temp_query['init_state'] = self.stateHelper.iaa_to_gym_state(copy.deepcopy(temp_query['init_state']))
        #         success, plan_length, state, renders = self.agent_model.run_query(temp_query, partial_check)
        #         if success:
        #             image_path = image_resource_dir + self.agent_model.pddl_domain_name + "-aia_image.png"
        #             Image.fromarray(renders[-1]).save(image_path)
        #             img = cv2.imread(image_path)
        #             detected_state = self.stateHelper.img_to_iaa_state(img, temp_query['init_state'])
        #             return success, plan_length, state_to_set(detected_state.state)
        #         else:
        #             return False, plan_length, state_to_set(query['init_state'].state)
        #     else:
        #         return False, -1, state_to_set(query['init_state'].state)
        
        elif self.agent_type == "simulator":
            if self.agent_model.ground_actions:
                init_state = self.agent_model.get_relational_state(query['init_state'])
                query['init_state'] = init_state
            if self.agent_model.validate_state(query['init_state']):
                temp_query = copy.deepcopy(query)
                success, plan_length, state = self.agent_model.run_query(temp_query)
                if success:
                    return success, plan_length, state_to_set(state.state)
                else:
                    return False, plan_length, state_to_set(query['init_state'].state)
            else:
                print("Invalid query init state!")
                return False, -1, state_to_set(query['init_state'].state)
    
    def get_correct_state(self,fixed_preds,cstate):
        temp_state = copy.deepcopy(cstate)
        return self.agent_model.get_correct_state(fixed_preds,temp_state)
    
    def fix_state(self,fixed_preds,cstate,removed,predTypeMapping):
        temp_state = copy.deepcopy(cstate)
        return self.agent_model.fix_state(fixed_preds,cstate,removed,predTypeMapping)
