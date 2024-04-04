import copy
import pickle

from kuka_state import AbstractKukaState, KukaState
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

    def update_high_actions(self,actions):
        #just to make sure this is called atleast once
        self.high_actions.update(actions)

    # TODO: implement for translator
    def generate_random_state(self):
        random_state = KukaState()
        return random_state
   
    def get_next_state(self,state,action):
        '''
            given state and action, apply action virtually and get resulting state
            actions: up,down,right,left,use
            input: ZeldaState, Action name
            assume only legal actions applied, including no effect
        '''

        return next_state

    def get_successor(self,state):
        action_dict = {
        'ACTION_UP':[],
        'ACTION_DOWN':[],
        'ACTION_RIGHT':[],
        'ACTION_LEFT':[],
        'ACTION_USE':[],
        }
        for action in action_dict:
            next_state = self.get_next_state(state,action)
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

    @saved_plan
    def plan_to_state(self,state1,state2,algo="custom-astar",full_trace = False):
        '''
        orientation is not considered for goal check, this is done since
        we need to plan only to abstract states which do not differ by orientation
        '''
        state1_ = copy.deepcopy(state1)
        state2_ = copy.deepcopy(state2)
        #action_dict = self.get_successor(state1_)
        total_nodes_expanded = []
        action_list = []
        print("Planning")
        if algo == "human":
            done = False
            print("about to")
            while not done:
                action = []
                for motorId in self.motors:
                    action.append(self.environment._p.readUserDebugParameter(motorId))
                state, reward, done, info = self.environment.step2(action)
                obs = self.environment.getExtendedObservation()
                print(obs)
        else:
            action_list,total_nodes_expanded = search(state1_,state2_,self,algo)
        return action_list,total_nodes_expanded

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

    def validate_state(self,ostate):
        '''
            Given ABSTRACT STATE, validate it
            assuming cell positioning is correct already, those are not to be learnt anyway
        '''
        state = copy.deepcopy(ostate.state)
        rev_objs = copy.deepcopy(ostate.rev_objects)
        player_loc = None
        key_loc = None
        door_loc = None
        monster_loc = None
        occupied_cells = []
        
        necessary_keys = ['leftOf','rightOf','above','below']
        if len(set(tuple(necessary_keys)).difference(tuple(state.keys())))>0:
            return False

        for k,values in state.items():
            if 'at' in k:
                for v in values:
                    if v[0] == 'player0':
                        if player_loc!=None:
                            return False
                        player_loc = v[1]
                        occupied_cells.append(v[1])
                    elif v[0] == 'key0':
                        key_loc = v[1]
                        occupied_cells.append(v[1])
                    elif v[0] == 'door0':
                        door_loc = v[1]
                    else:
                        monster_loc = v[1]
                        occupied_cells.append(v[1])

        if state.get('wall')!=None:
            for v in state.get('wall'):
                occupied_cells.append(v[0])

        #at, wall, clear
        if len(occupied_cells)!=len(set(occupied_cells)):
            return False
        
        for k,v in rev_objs.items():
            if v == 'location':
                if state.get('clear')!=None:
                    if (k in occupied_cells and (k,) in state.get('clear')) or (k not in occupied_cells and (k,) not in state.get('clear')):
                        return False

        if player_loc!=None:
            #at player, is_player, next_to_monster
            player_x = int(player_loc.replace('cell_','').split('_')[0])
            player_y = int(player_loc.replace('cell_','').split('_')[1])
            if monster_loc!=None:
                monster_x = int(monster_loc.replace('cell_','').split('_')[0])
                monster_y = int(monster_loc.replace('cell_','').split('_')[1])
                if (abs(monster_x-player_x) + abs(monster_y-player_y) == 1) and state.get('next_to_monster') == None:
                    return False
            else:
                if state.get('next_to_monster')!=None:
                    return False
        else:
            if state.get('next_to_monster')!=None:
                    return False

        if key_loc!=None:
            #at key, is_key, has_key
            if state.get('has_key')!=None:
                return False
        
        if door_loc == None:
            if state.get('escaped')!=None:
                return False

        if monster_loc!= None:
            #at monster, is_monster, monster_alive, next_to_monster
            monster_x = int(monster_loc.replace('cell_','').split('_')[0])
            monster_y = int(monster_loc.replace('cell_','').split('_')[1])
            monster_name = 'monster_'+str(monster_x)+'_'+str(monster_y)
            if state.get('monster_alive')!=[(monster_name,)]:
                return False
            if state.get('next_to_monster')!=None:
                if player_loc!=None:
                    player_x = int(player_loc.replace('cell_','').split('_')[0])
                    player_y = int(player_loc.replace('cell_','').split('_')[1])
                    if(abs(monster_x-player_x) + abs(monster_y-player_y) != 1):
                        return False
                else:
                    return False
        else:
            if state.get('monster_alive')!=None:
                return False
            if state.get('next_to_monster')!=None:
                return False       
        
        if state.get('escaped')!=None:
            if state.get('monster_alive')!=None or state.get('has_key')==None or (player_loc!=door_loc and player_loc!=None):
                return False    
        else:
            if (door_loc == player_loc) and (door_loc!=None) and state.get('has_key')!=None and monster_loc == None:
                return False
        return True    
          
    def abstract_state(self,low_state):
        unary_predicates = ['has_key','escaped','monster_alive','next_to_monster']
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
