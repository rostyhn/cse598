import copy
from collections import defaultdict


class Action:
    def __init__(self, name, s1, s2):
        self.name = name
        self.header = None
        self.modified_predicates = defaultdict(list)
        self.modified_predicates_typed = defaultdict(list)
        self.state_before = s1
        self.state_after = s2
        self.static_preds = defaultdict(list)
        self.relavant_static_props = {}
        self.static_relations = defaultdict(list)
        self.static_relations_typed = defaultdict(list)
        self.modified_objects = defaultdict(None)
        self.modified_objects_typed = defaultdict(set)
        self.added_predicates = defaultdict(list)
        self.deleted_predicates = defaultdict(list)

    def __eq__(self, action):
        # if sorted(list(self.header.values())) == sorted(list(action.header.values())):
        #    return True
        # else:
        #    return False
        if (
            self.state_before == action.state_before
            and self.state_after == action.state_after
        ):
            return True
        return False

    def assign_predicate_types(self):
        s1_ = copy.deepcopy(self.state_before.state)
        s2_ = copy.deepcopy(self.state_after.state)
        for k, vals in s1_.items():
            if k not in s2_:  # pred has been deleted
                # zeroary predicate added:
                if vals == [()]:
                    self.modified_predicates_typed[k].append([()])
                    self.modified_predicates[k] = [()]
                    self.deleted_predicates[k] = [()]
                else:
                    for v in vals:
                        if len(v) != 0:
                            typed_v = []
                            [
                                typed_v.append(
                                    self.state_before.rev_objects[t_v]
                                )
                                for t_v in v
                            ]
                            self.modified_predicates_typed[k].append(
                                tuple(typed_v)
                            )
                            self.modified_predicates[k].append(v)
                            self.deleted_predicates[k].append(v)
            else:
                for v in vals:
                    if v not in s2_[k]:  ##CLEAN THIS UP!!!
                        self.modified_predicates[k].append(v)
                        self.deleted_predicates[k].append(v)
                        typed_v = []
                        for t_v in v:
                            typed_v.append(self.state_before.rev_objects[t_v])
                        self.modified_predicates_typed[k].append(
                            tuple(typed_v)
                        )
                    else:
                        self.static_preds[k].append(v)
            # if len(self.modified_predicates[k])==0:
            #     self.modified_predicates.pop(k)

        for k, vals in s2_.items():
            if k not in s1_:  # pred has been deleted
                # zeroary predicate added:
                if vals == [()]:
                    self.modified_predicates_typed[k] = [()]
                    self.modified_predicates[k] = [()]
                    self.added_predicates[k] = [()]
                else:
                    for v in vals:
                        if len(v) != 0:
                            typed_v = []
                            for t_v in v:
                                typed_v.append(
                                    self.state_after.rev_objects[t_v]
                                )
                            self.modified_predicates_typed[k].append(
                                tuple(typed_v)
                            )
                            self.modified_predicates[k].append(v)
                            self.added_predicates[k].append(v)
            else:
                # self.modified_predicates[k] = []
                for v in vals:
                    if v not in s1_[k]:  ##CLEAN THIS UP!!!
                        self.modified_predicates[k].append(v)
                        self.added_predicates[k].append(v)
                        typed_v = []
                        [
                            typed_v.append(self.state_before.rev_objects[t_v])
                            for t_v in v
                        ]
                        self.modified_predicates_typed[k].append(
                            tuple(typed_v)
                        )

        for k, vals in self.modified_predicates.items():
            if len(vals) != 0:
                for v in vals:
                    if isinstance(v, str):
                        self.modified_objects[v] = (
                            self.state_before.rev_objects[v]
                        )
                        self.modified_objects_typed[
                            self.state_before.rev_objects[v]
                        ].add(v)
                    else:
                        for r in v:
                            self.modified_objects[r] = (
                                self.state_before.rev_objects[r]
                            )
                            self.modified_objects_typed[
                                self.state_before.rev_objects[r]
                            ].add(r)
        self.header = tuple(sorted(self.modified_objects.values()))
        relavant_static = defaultdict(list)
        # combine these into previous loops
        for pred, vals in self.static_preds.items():
            for v in vals:
                if len(vals) == 0:
                    relavant_static[pred] = [()]
                elif isinstance(v, str):
                    if v in list(self.modified_objects.keys()):
                        relavant_static[pred].append([v])
                elif set(v).issubset(set(self.modified_objects.keys())):
                    relavant_static[pred].append(v)

        self.relavant_static_props = relavant_static
        for k, v in self.relavant_static_props.items():
            for _v in v:
                self.static_relations[tuple(_v)].append(k)
                typed_v = []
                [
                    typed_v.append(self.state_before.rev_objects[t_v])
                    for t_v in _v
                ]
                self.static_relations_typed[tuple(typed_v)].append(k)
