import copy
import pickle

import numpy as np
from kuka_state import AbstractKukaState, KukaState
from tf_agents.environments.suite_gym import wrap_env
from utils.helpers import invert_dictionary, state_to_set

from gvg_agents.Search import search
from gvg_agents.sims.gvg_translator import Translator


def saved_plan(function):
    def _saved_plan(self,state1,state2,algo,full_trace=False):
        pkey = str("||".join(sorted(state_to_set(state1.state)))) + "|||" + str("||".join(sorted(state_to_set(state2.state))))
        if self.saved_plans.get(pkey)!=None:
            return self.saved_plans.get(pkey) 
        a,b = function(self,state1,state2,algo,full_trace)
        self.saved_plans[pkey] = [a,b]
        with open(self.plan_history_file,"wb") as f:
            pickle.dump(self.saved_plans,f)
        return a,b
    return _saved_plan

class KukaTranslator(Translator):
    def __init__(self, environment, motors, ground_actions=False, files_dir=""):
        super().__init__(AbstractKukaState)
        self.files = files_dir
        self.high_actions = {}
        self.random_states = []
        self.ground_actions = ground_actions
        self.saved_plans = {}
        self.plan_history_file = f"{files_dir}plans"
        self.environment = environment
        self.motors = motors

    def update_high_actions(self, actions):
        #just to make sure this is called atleast once
        self.high_actions.update(actions)
   
    def get_next_state(self, state, action):
        '''
            given state and action, apply action virtually and get resulting state
            assume only legal actions applied, including no effect
        '''
        self.environment._p.restoreState(state.state["stateID"])
        state, reward, done, info = self.environment.step2(action)
        s_id = self.environment._p.saveState()
        return KukaState(s_id)

    def get_successor(self, state):
        action_dict = {
        'ACTION_DOWN':[],
        'ACTION_RIGHT':[],
        'ACTION_LEFT':[],
        'ACTION_USE':[],
        }
        for action in action_dict:
            next_state = self.get_next_state(state, action)
            if next_state == state:
                action_dict[action] = [0,state]
            else:
                action_dict[action] = [1,next_state]
        return action_dict
    
    def is_goal_state(self,current_state,goal_state):
        #all orientations should be corrent goal state
        dcurrent_state = copy.deepcopy(current_state)
        dcurrent_state.state['player_orientation'] = None
        dgoal_state = copy.deepcopy(goal_state)
        dgoal_state.state['player_orientation'] = None
        if dcurrent_state==dgoal_state:
            return True
        else:
            return False

    # https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.2ye70wns7io3

    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/kukaGymEnv.py

    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/kuka.py

    from tf_agents.agents.dqn import dqn_agent
    from tf_agents.environments.suite_gym import wrap_env
    from tf_agents.utils import common
    from tf_agents.networks import sequential
    from tf_agents.specs import tensor_spec

    def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))

    @saved_plan
    def plan_to_state(self,state1,state2,algo="custom-astar",full_trace = False):
        '''
        orientation is not considered for goal check, this is done since
        we need to plan only to abstract states which do not differ by orientation
        '''
        state1_ = copy.deepcopy(state1)
        state2_ = copy.deepcopy(state2)
        total_nodes_expanded = []
        action_list = []
        tf_env = wrap_env(self.environment)
        print(type(tf_env))

        # gym_env = suite_gym.load(self.environment)
        # tf_env = tf_py_environment.TFPyEnvironment(gym_env)

        # build DQN agent here https://www.tensorflow.org/agents
        """
        TODO
        agent = dqn_agent.DqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=tf.Variable(0))
        etc...
        """
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)


        fc_layer_params = (100, 50)
        action_tensor_spec = tensor_spec.from_spec(tf_env.action_spec())
        num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1




        dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
        q_values_layer = tf.keras.layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2))

        q_net = sequential.Sequential(dense_layers + [q_values_layer])
        
        train_step_counter = tf.Variable(0)

        agent = dqn_agent.DqnAgent(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter)
        agent.initialize()



        print("Planning")
        if algo == "human":
            # figure out how to get the agent to solve the environment
            done = False
            obs = np.array(self.environment._observation)
            while not done:
                self.environment.render()

                action = agent.collect_policy.action(time_step)
                time_step = tf_env.step(action)

                obs, reward, done, info = self.environment.step(action)
                # save the actions the agent takes in action_list
                action_list.append(action)
        else:
            action_list, total_nodes_expanded = search(state1_,state2_,self,algo)
        return action_list, total_nodes_expanded

    def execute_from_ID(self,abs_state,abs_action):
        try:
            abs_before,abs_after = self.high_actions[abs_action]
            #checking if just state equality is enough
            #if abs_before.state == abs_state.state and abs_before.rev_objects == abs_state.objects:
            #if abs_before!=abs_state: #Might have to change this to a more sophisticated comparison
            if abs_before.state == abs_state.state:
                return True,abs_after
            else:
                return False,abs_state
        except KeyError:
            print("Unknown Action ID!")

    def validate_state(self, ostate):
        '''
            Given ABSTRACT STATE, validate it
            assuming cell positioning is correct already, those are not to be learnt anyway
        '''
        return True    
    
    # convert low-level state into abstract high-level one
    def abstract_state(self,low_state):
        abs_state = AbstractKukaState()
        return abs_state

    def generate_ds(self):
        '''
           assume the actions are assigned
        '''
        abstract_model = {}
        action_parameters = {}
        abstract_predicates = {}
        types = {}
        objects = {}
        predTypeMapping = {}
        agent_model ={}
        init_state = None
        for action,states in self.high_actions.items():
            for state in states:
                gstate = self.get_ground_state(state)
                for pred in gstate.state:
                    predTypeMapping[pred]=[]
        
        # for state in self.random_states:
        #     gstate = self.get_ground_state(self.abstract_state(state))
        #     for pred in gstate.state:
        #         if pred not in predTypeMapping:
        #             predTypeMapping[pred]=[]

        for action in self.high_actions:
            abstract_model[action] = {}
            action_parameters[action] = []
            agent_model[action]= {}
            for pred in predTypeMapping:
                agent_model[action][pred] = [Literal.ABS,Literal.ABS]

        return action_parameters, predTypeMapping, agent_model, abstract_model, objects, types , None, "cookmepasta_GVG"
        
    def refine_abstract_state(self,abstract_state_):
        '''
            Concretize an input abstract state
        '''
        abstract_state = copy.deepcopy(abstract_state_)
        all_keys = ['at_0','at_1','at_2','at_3','monster_alive','has_key','escaped','wall','leftOf','rightOf','above','below']
        refined_state = Zelda_State()        
        # for obj in abstract_state.rev_objects:
        #     if abstract_state.rev_objects[obj] in ['location']:
        #         refined_state.rev_objects[obj]=abstract_state.rev_objects[obj]
        
        refined_state.rev_objects = abstract_state.rev_objects
        refined_state.state['wall'] = abstract_state.state.get('wall')
        refined_state.state['leftOf'] = abstract_state.state.get('leftOf')
        refined_state.state['rightOf'] = abstract_state.state.get('rightOf')
        refined_state.state['above'] = abstract_state.state.get('above')
        refined_state.state['below'] = abstract_state.state.get('below')
        refined_state.state['player_orientation'].append('NORTH')
        
        if abstract_state.state['escaped'] == None:
            refined_state.state['escaped'] = [False]
        else:
            refined_state.state['escaped'] = [True]
        
        if abstract_state.state['has_key'] == None:
            refined_state.state['has_key'] = [False]
        else:
            refined_state.state['has_key'] = [True]

        if abstract_state.state.get('at_0')!=None:
            for pair in abstract_state.state['at_0']:
                refined_state.state['player'].append(pair[1])
                #refined_state.rev_objects[pair[0]]='sprite'
        if abstract_state.state.get('at_1')!=None:
            for pair in abstract_state.state['at_1']:
                refined_state.state['key'].append(pair[1])
                #refined_state.rev_objects[pair[0]]='sprite'
        if abstract_state.state.get('at_3')!=None:
            for pair in abstract_state.state['at_3']:
                refined_state.state['door'].append(pair[1])
                #refined_state.rev_objects[pair[0]]='sprite'
        if abstract_state.state.get('at_2')!=None:
            for pair in abstract_state.state['at_2']:
                refined_state.state["monster"].append((pair[0],pair[1]))
                #refined_state.rev_objects[pair[0]]='sprite'
        
        if abstract_state.state['clear']!=None:
            for cell in abstract_state.state['clear']:
                refined_state.state['clear'].append(cell[0])

        refined_state.grid_height = abstract_state.grid_height
        refined_state.grid_width = abstract_state.grid_width
        for k,v in refined_state.state.items():
            if(v) == None:
                print(str(k)+" is empty")
                refined_state.state[k] = []
        refined_state.objects = invert_dictionary(refined_state.rev_objects)
        return refined_state

    def get_relational_state(self,state):
        rstate = AbstractZeldaState()
        rstate.grid_height = 0
        rstate.grid_width = 0
        for p in state.state:
            pred = p.split('-')[0]
            params = p.replace(pred,'').split('-')[1:]
            if params!=['']:
                v = []
                for _p in params:
                    if len(_p)!=0:
                        v.append(_p) 
                        if 'cell' in _p:
                            x = int(_p.split('_')[1])+1
                            y = int(_p.split('_')[2])+1
                            rstate.rev_objects[_p] = 'location'
                            rstate.grid_height = max(rstate.grid_height,y)
                            rstate.grid_width = max(rstate.grid_width,x)
                        else:
                            rstate.rev_objects[_p] = _p[:-1]
                rstate.state[pred].append(tuple(v))
            else:
                if rstate.state[pred] == None:
                    rstate.state[pred]=[()]
                else:
                    rstate.state[pred].append(tuple([]))
        temp_k = []
        for k,v in rstate.state.items():
            if len(v) == 0:    
                temp_k.append(k)
        [rstate.state.pop(k_,None) for k_ in temp_k]
        return rstate
   
    def get_ground_state(self,state):
        gstate = AbstractZeldaState()
        gstate.state = {}
        for k,v in state.state.items():
            if v!=None:
                p = k
                for _v in v:
                    gstate.state[k+"-"+"-".join(list(_v))]=[()]
        gstate.objects = {}
        gstate.rev_objects = {}
        return gstate
    
    def iaa_query(self,abs_state,plan):
        '''
        state: abstract
        plan: hashed values corresponding to stored actions
        '''
        if self.validate_state(abs_state):            
            state  = copy.deepcopy(abs_state)
            for i,action in enumerate(plan):
                '''
                    can check plan possibility here itself
                    if subsequent states are not equal, can't execute
                '''
                can_execute,abs_after = self.execute_from_ID(state,action)
                if can_execute:
                    state = abs_after
                else:
                    return False,i,abs_after #check from sokoban code
            return True,len(plan),abs_after #check from sokoban code
        else:
            return False,0,abs_stateclass 