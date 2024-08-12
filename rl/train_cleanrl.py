import argparse
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with CleanRL. ")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=4000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--capture_video", action="store_true", default=False, help="Capture video of the agent performance.")
parser.add_argument("--exp_name", type=str, default="cleanrl_test", help="Name of the experiment.")
parser.add_argument("--anneal_lr", action="store_true", default=False, help="Anneal the learning rate.")
parser.add_argument("--learning_rate", type=float, default=0.0026, help="Learning rate of the optimizer.")
parser.add_argument("--total_timesteps", type=int, default=3e7, help="Total timesteps of the experiments.")
parser.add_argument("--goal_task", type=str, default="rand", help="Goal task for the environment.")
parser.add_argument("--frame", type=str, default="root", help="Frame of the task.")

AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

args_cli.enable_cameras = True
args_cli.headless = True # make false to see the simulation

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import envs

from omni.isaac.lab_tasks.utils import parse_env_cfg


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Isaac-AerialManipulator-Hover-v0"
    """the id of the environment"""
    total_timesteps: int = 30000000
    """total timesteps of the experiments"""
    learning_rate: float = 0.003
    """the learning rate of the optimizer"""
    num_envs: int = 1024
    """the number of parallel game environments"""
    num_steps: int = 16
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 2
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 2
    """coefficient of the value function"""
    max_grad_norm: float = 1
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    reward_scaler: float = 1
    """the scale factor applied to the reward during training"""
    record_video_step_frequency: int = 512*200
    """the frequency at which to record the videos"""

    # Miscellaneous
    save_interval: int = 500000
    video_length: int = 1000

    save_model: bool = True

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    task: str = "Isaac-AerialManipulator-Hover-v0"
    """the name of the task"""
    video: bool = True
    disable_fabric: bool = False
    goal_task: str = "rand"
    frame: str = "root"

class RecordEpisodeStatisticsTorch(gym.Wrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.device = device
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.returned_episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.returned_episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        return observations

    def step(self, action):
        observations, rewards, dones, truncateds, infos = super().step(action)
        self.episode_returns += rewards
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - dones.long()
        self.episode_lengths *= 1 - dones.long()
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            truncateds,
            infos,
        )


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        num_obs = np.array(envs.single_observation_space.shape[1]).prod()
        num_acts = np.array(envs.single_action_space.shape[1]).prod()

        self.critic = nn.Sequential(
            layer_init(nn.Linear(num_obs, 256)),
            # nn.Tanh(),
            nn.ELU(),
            layer_init(nn.Linear(256, 256)),
            # nn.Tanh(),
            nn.ELU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(num_obs, 256)),
            # nn.Tanh(),
            nn.ELU(),
            layer_init(nn.Linear(256, 256)),
            # nn.Tanh(),
            nn.ELU(),
            layer_init(nn.Linear(256, num_acts), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, num_acts))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        assert torch.all(action_std >= 0), f"std: {action_std} \n logstd: {action_logstd}"
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class ExtractObsWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return obs["policy"]
    
def main():
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    run_name = f"{args.exp_name}_1"

    # If run exists, increment run number
    run_number = 1
    while os.path.exists(f"runs/{run_name}"):
        run_number += 1
        run_name = f"{args.exp_name}_{run_number}"
    
    
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=True, num_envs=args.num_envs, use_fabric=True
    )

    # Any environment specific configuration goes here such as camera placement
    env_cfg.goal_task = args.goal_task
    env_cfg.task_body = args.frame

    # create environment
    envs = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    video_kwargs = {
        "video_folder": f"runs/{run_name}",
        "step_trigger": lambda step: step % args_cli.video_interval == 0,
        # "episode_trigger": lambda episode: (episode % args.save_interval) == 0,
        "video_length": args.video_length,
        "name_prefix": "training_video"
    }
    envs = gym.wrappers.RecordVideo(envs, **video_kwargs)
    envs = ExtractObsWrapper(envs)
    envs = RecordEpisodeStatisticsTorch(envs, device)
    if args.capture_video:
        envs.is_vector_env = True
        print(f"record_video_step_frequency={args.record_video_step_frequency}")
        envs = gym.wrappers.RecordVideo(
            envs,
            f"videos/{run_name}",
            step_trigger=lambda step: step % args.record_video_step_frequency == 0,
            video_length=1000,  # for each video record up to 10 seconds
        )
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space

    envs.single_action_space_shape = (np.array(envs.single_action_space.shape[1]).prod(),)
    envs.single_observation_space_shape = (np.array(envs.single_observation_space.shape[1]).prod(),)

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space_shape, dtype=torch.float).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space_shape, dtype=torch.float).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    advantages = torch.zeros_like(rewards, dtype=torch.float).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    save_steps = 0
    save_iterations = 0
    start_time = time.time()
    next_obs, info = envs.reset()
    next_done = torch.zeros(args.num_envs, dtype=torch.float).to(device)

    max_episode_return = -np.inf


    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            save_steps += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards[step], next_done, next_truncated, info = envs.step(action)
            if 0 <= step <= 2:
                for idx, d in enumerate(next_done):
                    if d or next_truncated[idx]:
                        episodic_return = info["r"][idx].item()
                        print(f"global_step={global_step}, episodic_return={episodic_return}")
                        writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                        writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)
                        writer.add_scalar("charts/episodic_summed_vel_reward", info["log"]["Episode Reward/endeffector_lin_vel"].item(), global_step)
                        writer.add_scalar("charts/episodic_summed_ang_vel_reward", info["log"]["Episode Reward/endeffector_ang_vel"].item(), global_step)
                        writer.add_scalar("charts/episodic_summed_pos_reward", info["log"]["Episode Reward/endeffector_pos_distance"].item(), global_step)
                        writer.add_scalar("charts/episodic_summed_ori_reward", info["log"]["Episode Reward/endeffector_ori_error"].item(), global_step)
                        writer.add_scalar("charts/episodic_summed_joint_vel_reward", info["log"]["Episode Reward/joint_vel"].item(), global_step)
                        writer.add_scalar("charts/episode_final_distance_to_goal", info["log"]["Metrics/Final Distance to Goal"], global_step)
                        writer.add_scalar("charts/episode_final_orientation_error", info["log"]["Metrics/Final Orientation Error to Goal"], global_step)

                        # save model on best episodic return
                        if episodic_return > max_episode_return:
                            max_episode_return = episodic_return
                            torch.save(agent.state_dict(), f"runs/{run_name}/best_cleanrl_model.pt")
                            
                        if "consecutive_successes" in info:  # ShadowHand and AllegroHand metric
                            writer.add_scalar(
                                "charts/consecutive_successes", info["consecutive_successes"].item(), global_step
                            )
                        break
            
            # Save the model every 500k global steps but since there's many environments it could step over 500k
            if save_steps >= args.save_interval:
                save_steps = 0
                model_save_number = (save_iterations+1) * args.save_interval
                torch.save(agent.state_dict(), f"runs/{run_name}/model_{model_save_number}_steps.pt")
                save_iterations += 1


        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done.long()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(args.batch_size, device=device)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                # newlogprob = newlogprob.sum(1)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)



    if args.save_model:
        model_path = f"runs/{run_name}/model_final.pt"
        torch.save(agent.state_dict(), model_path)

    # envs.close()
    writer.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()