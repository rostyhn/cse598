import os
import tensorflow as tf
#import reverb

from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from tf_agents.environments import suite_gym, tf_py_environment, utils
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks.q_network import QNetwork
from tf_agents.policies import py_tf_eager_policy
from tf_agents.utils import common
from tf_agents.drivers import dynamic_episode_driver, dynamic_step_driver, py_driver
from tf_agents.replay_buffers import reverb_replay_buffer, tf_uniform_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.policies import policy_saver


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def main():
    if not os.path.exists("../weights"):
        os.mkdir("../weights")

    environment = KukaGymEnv(renders=True, isDiscrete=True, maxSteps=1000)
    
    py_env = suite_gym.wrap_env(environment, auto_reset=False)
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    
    motorsIds = []
    dv = 0.01
    motorsIds.append(environment._p.addUserDebugParameter("posX", -dv, dv, 0))
    motorsIds.append(environment._p.addUserDebugParameter("posY", -dv, dv, 0))
    motorsIds.append(environment._p.addUserDebugParameter("posZ", -dv, dv, 0))
    motorsIds.append(environment._p.addUserDebugParameter("yaw", -dv, dv, 0))
    motorsIds.append(
        environment._p.addUserDebugParameter("fingerAngle", 0, 0.3, 0.3)
    )
    
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)

    q_net = QNetwork(tf_env.observation_spec(), tf_env.action_spec())
    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter)
    agent.initialize()

    num_iterations = 20000
    collect_steps_per_iteration = 1
    replay_buffer_max_length = 100000
    
    #avg_return = compute_avg_return(tf_env, agent.policy)
    returns = [0]#avg_return]

    ts = tf_env.reset()

    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(
          agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(
        replay_buffer_signature)

    """
    table = reverb.Table(
        table_name,
        max_size=replay_buffer_max_length,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature)
    """

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=1000)

    replay_observer = [replay_buffer.add_batch]

    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            tf_env,
            agent.collect_policy,
            observers=replay_observer,
            num_episodes=1)
   
    # num episodes really
    training_steps = 1000
    for i in range(training_steps):
        cd_ret = collect_driver.run()
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=4,
            num_steps=2).prefetch(3)

        iterator = iter(dataset)
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience)
        print(train_loss)
        
        if i % 100 == 0:
            print('step = {0}: loss = {1}'.format(i, train_loss))

        #if i % 1000 == 0:
            #avg_return = compute_avg_return(environment, agent.policy)
            #print('step = {0}: Average Return = {1}'.format(i, avg_return))
            #returns.append(avg_return)
    
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)
    tf_policy_saver.save(os.path.abspath("./weights"))
    
    return


if __name__ == "__main__":
    main()
