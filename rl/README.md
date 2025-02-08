# Training Policies
The script `train_rslrl.py` is used to train a policy with PPO implemened in the RSL-RL library. 
- The task must be specified with `--task {TASK_NAME}`
- `--num_envs` environments will be simulated in parallel
- The experiment name must be specified with `--experiment_name {EXP_NAME}` which is where the model checkpoints will be saved (`./logs/rsl_rl/EXP_NAME/RUN_NAME`). 
- The run name is the timestamp of when this training code was executed, appended with any string that is useful for differentiating runs `--run_name {RUN_NAME}`
- Hydra can be used to configure some parameters, such as the total number of model updates `agent.max_iterations=750`, the number of steps per env per update `agent.num_steps_per_env=64`, the policy control mode `env.control_mode="CTBR"`, reward shaping terms `env.yaw_error_reward_scale=-2.0` etc. 

Example training script for CTBR (works out of the box in sim2real):
```bash
python train_rslrl.py --task Isaac-Crazyflie-Hover-v0 --num_envs 4096 agent.num_steps_per_env=64 --experiment_name test_ctbr env.control_mode="CTBR" 
```


Example training script for SLURM is provided in `run_job.bash`. Partition name, QOS, etc. may need to be changed. 

# Evaluating Policies
The script `eval_rslrl.py` is used to evaluate pre-trained policies and baseline GC implementations. 
- The task must be specified with `--task {TASK_NAME}`
- The script will load a policy specified by `--experiment_name {EXP_NAME} --load_run {RUN_NAME}`, and will load the last checkpoint in that folder by default unless `agent.load_checkpoint={MODEL_PATH}` is specified.
- `--num_envs` environments will be simulated in parallel, for a total of 500 steps. 
- This script will execute the evaluation in a headless mode always, but to render videos `--video` can be set to save a video. `--follow_robot {ROBOT_NUMBER}` can be set to change the video to follow a particular robot, which is useful for a "zoomed-in" video instead of all robots. 
- The evaluation states are saved in the same folder as the model, and can be parsed for metrics downstream.
- To run the baseline GC implementation, use `--baseline` and this will not execute the RL policy, but the experiment name and run name must still be specified. 

Example: Evaluating RL Policy
```bash
python eval_rslrl.py --video --task Isaac-Crazyflie-Hover-v0 --num_envs 100 --experiment_name test_ctbr --load_run 2025-02-08_15-44-09 --follow_robot 0
```

Example: Evaluating RL Policy with specific checkpoint
```bash
python eval_rslrl.py --video --task Isaac-Crazyflie-Hover-v0 --num_envs 100 --experiment_name test_ctbr --load_run 2025-02-08_15-44-09 --follow_robot 0 agent.load_checkpoint=model_200.pt
```

Example: Evaluating Baseline
```bash
python eval_rslrl.py --video --task Isaac-Crazyflie-Hover-v0 --num_envs 100 --experiment_name test_ctbr --load_run 2025-02-08_15-44-09 --follow_robot 0 --baseline 
```

# Hydra Configuration
Hydra is used to configure any experiments for both training and evaluation. This is done for dynamic configuration and makes it really easy to run numerous expeirments of any sort of parameters. After training, a list of parameters is printed to screen and the exact configurations are also saved in `./logs/rsl_rl/EXP_NAME/RUN_NAME/params/agent.yaml` and `./logs/rsl_rl/EXP_NAME/RUN_NAME/params/env.yaml`. The agent configuration is used to modify PPO-related parameters, such as the number of steps per environment `agent.num_steps_per_env=64` for example, or when loading a specific checkpoint for evaluation `agent.load_checkpoint=model_200.pt`. The environment configuration is more involved, and all available parameters can be seen in the environment file specifically.
