import os

from agent import Agent
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv


def main():
    if not os.path.exists("../results"):
        os.mkdir("../results")

    environment = KukaGymEnv(renders=True, isDiscrete=True, maxSteps=10000000)
    # TODO: can we get these vars to update with the agent?
    motorsIds = []
    # motorsIds.append(environment._p.addUserDebugParameter("posX",0.4,0.75,0.537))
    # motorsIds.append(environment._p.addUserDebugParameter("posY",-.22,.3,0.0))
    # motorsIds.append(environment._p.addUserDebugParameter("posZ",0.1,1,0.2))
    # motorsIds.append(environment._p.addUserDebugParameter("yaw",-3.14,3.14,0))
    # motorsIds.append(environment._p.addUserDebugParameter("fingerAngle",0,0.3,.3))

    dv = 0.01
    motorsIds.append(environment._p.addUserDebugParameter("posX", -dv, dv, 0))
    motorsIds.append(environment._p.addUserDebugParameter("posY", -dv, dv, 0))
    motorsIds.append(environment._p.addUserDebugParameter("posZ", -dv, dv, 0))
    motorsIds.append(environment._p.addUserDebugParameter("yaw", -dv, dv, 0))
    motorsIds.append(
        environment._p.addUserDebugParameter("fingerAngle", 0, 0.3, 0.3)
    )
    a = Agent("kuka", environment, motorsIds)

    """
    done = False
    while (not done):

      action = []
      for motorId in motorsIds:
        action.append(environment._p.readUserDebugParameter(motorId))

      state, reward, done, info = environment.step2(action)
      obs = environment.getExtendedObservation()
      print(obs, info)
    """
    return

    (
        action_parameters,
        pred_type_mapping,
        agent_model_actions,
        abstract_model_actions,
        objects,
        old_types,
        init_state,
        domain_name,
    ) = agent.agent_model.generate_ds()

    abstract_predicates = {}
    types = modify_types(old_types)

    pp = pprint.PrettyPrinter(indent=2)
    abstract_model = Model(abstract_predicates, abstract_model_actions)

    # comment to include static predicates
    abs_preds_test, abs_actions_test, _ = agent.agent_model.bootstrap_model()
    abstract_model.predicates = abs_preds_test
    abstract_model.actions = abs_actions_test

    if not check_results:
        iaa_main = AgentInterrogation(
            agent,
            abstract_model,
            objects,
            domain_name,
            abstract_predicates,
            pred_type_mapping,
            action_parameters,
            types,
            load_old_q=True,
        )

        query_count, running_time, data_dict, pal_tuple_count, valid_models = (
            iaa_main.agent_interrogation_algo()
        )
        agent.agent_model.show_actions()


if __name__ == "__main__":
    main()
