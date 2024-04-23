import os
import tensorflow as tf
import reverb

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

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import suite_pybullet
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy, PolicySaver
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils

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

# https://arxiv.org/abs/1912.01588
# try heist with PPO?
# https://github.com/openai/procgen
def main():
    if not os.path.exists("../weights"):
        os.mkdir("../weights")
    #https://blog.otoro.net/2017/11/12/evolving-stable-strategies/
    #collect_env = suite_gym.wrap_env(KukaGymEnv(renders=True, isDiscrete=False, maxSteps=1000),auto_reset=False)
    #eval_env = suite_gym.wrap_env(KukaGymEnv(renders=True, isDiscrete=False, maxSteps=1000), auto_reset=False) 
    collect_env = suite_pybullet.load('MinitaurBallGymEnv-v0')
    eval_env = suite_pybullet.load('MinitaurBallGymEnv-v0')
    #py_env = suite_gym.wrap_env(environment, auto_reset=False)
    #tf_env = tf_py_environment.TFPyEnvironment(py_env)

    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)
    observation_spec, action_spec, time_step_spec = (
      spec_utils.get_tensor_specs(collect_env))
    
    num_iterations = 500000

    initial_collect_steps = 10000 # @param {type:"integer"}
    collect_steps_per_iteration = 1 # @param {type:"integer"}
    replay_buffer_capacity = 50000 # @param {type:"integer"}

    batch_size = 256 # @param {type:"integer"}

    critic_learning_rate = 3e-4 # @param {type:"number"}
    actor_learning_rate = 3e-4 # @param {type:"number"}
    alpha_learning_rate = 3e-4 # @param {type:"number"}
    target_update_tau = 0.005 # @param {type:"number"}
    target_update_period = 1 # @param {type:"number"}
    gamma = 0.99 # @param {type:"number"}
    reward_scale_factor = 1.0 # @param {type:"number"}

    actor_fc_layer_params = (256, 256)
    critic_joint_fc_layer_params = (256, 256)

    log_interval = 5000 # @param {type:"integer"}

    num_eval_episodes = 20 # @param {type:"integer"}
    eval_interval = 10000 # @param {type:"integer"}

    policy_save_interval = 5000 # @param {type:"integer"}
    
    with strategy.scope():
        critic_net = critic_network.CriticNetwork(
            (observation_spec, action_spec),
            observation_fc_layer_params=None,
            action_fc_layer_params=None,
            joint_fc_layer_params=critic_joint_fc_layer_params,
            kernel_initializer='glorot_uniform',
            last_kernel_initializer='glorot_uniform')

    with strategy.scope():
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec,
            action_spec,
            fc_layer_params=actor_fc_layer_params,
            continuous_projection_net=(
            tanh_normal_projection_network.TanhNormalProjectionNetwork))
    
    with strategy.scope():
      train_step = train_utils.create_train_step()

      tf_agent = sac_agent.SacAgent(
            time_step_spec,
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.keras.optimizers.Adam(
                learning_rate=actor_learning_rate),
            critic_optimizer=tf.keras.optimizers.Adam(
                learning_rate=critic_learning_rate),
            alpha_optimizer=tf.keras.optimizers.Adam(
                learning_rate=alpha_learning_rate),
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=tf.math.squared_difference,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            train_step_counter=train_step)

      tf_agent.initialize()

    rate_limiter=reverb.rate_limiters.SampleToInsertRatio(samples_per_insert=3.0, min_size_to_sample=3, error_buffer=3.0)

    table_name = 'uniform_table'
    table = reverb.Table(
        table_name,
        max_size=replay_buffer_capacity,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1))

    reverb_server = reverb.Server([table])

    reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
        tf_agent.collect_data_spec,
        sequence_length=2,
        table_name=table_name,
        local_server=reverb_server)
   
    dataset = reverb_replay.as_dataset(
      sample_batch_size=batch_size, num_steps=2).prefetch(50)
    experience_dataset_fn = lambda: dataset 
    tf_eval_policy = tf_agent.policy
    eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
    tf_eval_policy, use_tf_function=True)

    tf_collect_policy = tf_agent.collect_policy
    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
    tf_collect_policy, use_tf_function=True)

    random_policy = random_py_policy.RandomPyPolicy(
    collect_env.time_step_spec(), collect_env.action_spec())
   
    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
        reverb_replay.py_client,
        table_name,
        sequence_length=2,
        stride_length=1)

    initial_collect_actor = actor.Actor(
        collect_env,
        random_policy,
        train_step,
        steps_per_run=initial_collect_steps,
        observers=[rb_observer])
    initial_collect_actor.run()
    
    tempdir = "../weights/"

    env_step_metric = py_metrics.EnvironmentSteps()
    collect_actor = actor.Actor(
      collect_env,
      collect_policy,
      train_step,
      steps_per_run=1,
      metrics=actor.collect_metrics(10),
      summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
      observers=[rb_observer, env_step_metric])

    eval_actor = actor.Actor(
      eval_env,
      eval_policy,
      train_step,
      episodes_per_run=num_eval_episodes,
      metrics=actor.eval_metrics(num_eval_episodes),
      summary_dir=os.path.join(tempdir, 'eval'),
    )


    saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

# Triggers to save the agent's policy checkpoints.
    learning_triggers = [
        triggers.PolicySavedModelTrigger(
            saved_model_dir,
            tf_agent,
            train_step,
            interval=policy_save_interval),
        triggers.StepPerSecondLogTrigger(train_step, interval=1000),
    ]

    agent_learner = learner.Learner(
    tempdir,
    train_step,
    tf_agent,
    experience_dataset_fn,
    triggers=learning_triggers,
    strategy=strategy)
       
    def get_eval_metrics():
        eval_actor.run()
        results = {}
        for metric in eval_actor.metrics:
            results[metric.name] = metric.result()
        return results

    metrics = get_eval_metrics()

    def log_eval_metrics(step, metrics):
      eval_results = (', ').join(
          '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
      print('step = {0}: {1}'.format(step, eval_results))

    log_eval_metrics(0, metrics)
    # num episodes really

    tf_agent.train_step_counter.assign(0)
    
    avg_return = get_eval_metrics()["AverageReturn"]
    returns = [avg_return]

    for _ in range(num_iterations):
      # Training.
      collect_actor.run()
      loss_info = agent_learner.run(iterations=1)
      # Evaluating.
      step = agent_learner.train_step_numpy
    
      if eval_interval and step % eval_interval == 0:
        metrics = get_eval_metrics()
        log_eval_metrics(step, metrics)
        returns.append(metrics["AverageReturn"])

      if log_interval and step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))
   
    rb_observer.close()
    reverb_server.stop()

if __name__ == "__main__":
    main()
