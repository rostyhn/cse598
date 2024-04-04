from lattice import State
from collections import defaultdict
import pickle
from utils.helpers import state_to_set

class AbstractKukaState(State):
    def __init__(self):
        self.grid_height = 4 #assign dynamically later
        self.grid_width = 4 #assign dynamically later
        self.state = defaultdict(None)
        tstate = {
            'grasping': [] 
        }
        for k,v in tstate.items():
            self.state[k]=v 
        self.rev_objects = {}#locations(cells),monster,player,door,key
        self.objects = {}
    
    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self,abstract2):
        '''
            TODO: 
                Check for equivalency, not equality
                But is that required?
        '''
        for pred in self.state:
            if pred in abstract2.state:
                if isinstance(self.state[pred],list):
                    if sorted(abstract2.state[pred])!=sorted(self.state[pred]):
                        return False    
                else:
                    if abstract2.state[pred]!=self.state[pred]:
                        return False
            else:
                return False
        if self.rev_objects!=abstract2.rev_objects:
            return False

        return True

    def __str__(self):
        s = ""
        for k,v in self.state.items():
            s += f"{k}: {str(v)}\n"
        return s
       
def invert_dictionary(idict):
        odict = defaultdict(list)
        for k,v in idict.items():
            odict[v].append(k)
        return odict

class KukaState:
    def __eq__(self,state2):
        try:
            for pred,vals in self.state.items():
                if isinstance(vals,list):
                    if sorted(vals)!=sorted(state2.state[pred]):
                        return False
                else:
                    if state2.state[pred]!=self.state[pred]:
                        return False
            if self.rev_objects != state2.rev_objects:
                return False
            return True
        except AttributeError:
            return False

    def __str__(self):
        s = ""
        for k,v in self.state.items():
            s += f"{k}: {str(v)}\n"
        return s         

    def __hash__(self):
        return hash(str(self)) 
    def __init__(self):
        self.objects = {}
        self.rev_objects = {}
        #self.monster_mapping=monster_mapping #key:monster name, value: original monster-location
        #self.trace_id=trace_id
        self.state = {
            'grasping': [False],
        }
        self.g_score = 0 #for search
        self.best_path = None #for search 
    
def save_query(function):
    def _save_query(self,query):
        qkey = str("||".join(sorted(state_to_set(query['init_state'].state)))) + "|||" + str("||".join(query['plan']))
        a,b,c = function(self,query)
        self.queries[qkey] = [a,b,c]
        with open(self.query_history_file,"wb") as f:
            pickle.dump(self.queries,f)
        return a,b,c
    return _save_query


