LOGS=(
    "/home/balfaro/AerialManipulation/rl/logs/rsl_rl/2DOF-Trajectory/2026-04-09_17-11-03_full-fast-lissaajous-train/eval_traj_track_50Hz__eval_trajectory_fast_lissaajous_5000_envs_eval_full_states.pt"
    "/home/balfaro/AerialManipulation/rl/logs/rsl_rl/2DOF-Trajectory/2026-04-09_17-11-03_full-fast-lissaajous-train/eval_traj_track_50Hz__eval_trajectory_hover_5000_envs_eval_full_states.pt"    
)   
LOGS2=(
    "/home/balfaro/AerialManipulation/rl/logs/rsl_rl/2DOF-Trajectory/2026-06-30_12-54-41_ctbr-traj-continuous-joint-representation/eval_traj_track_50Hz__eval_trajectory_fast_lissaajous_5000_envs_eval_full_states.pt"
    "/home/balfaro/AerialManipulation/rl/logs/rsl_rl/2DOF-Trajectory/2026-06-30_12-54-41_ctbr-traj-continuous-joint-representation/eval_traj_track_50Hz__eval_trajectory_hover_5000_envs_eval_full_states.pt"
)
LOGS3=(
    "/home/balfaro/AerialManipulation/rl/logs/rsl_rl/2DOF-Trajectory/2026-06-30_14-39-01_ctbr-hover-continuous-joint-representation/eval_traj_track_50Hz__eval_trajectory_fast_lissaajous_5000_envs_eval_full_states.pt"
    "/home/balfaro/AerialManipulation/rl/logs/rsl_rl/2DOF-Trajectory/2026-06-30_14-39-01_ctbr-hover-continuous-joint-representation/eval_traj_track_50Hz__eval_trajectory_hover_5000_envs_eval_full_states.pt"
)
NAMES=(
    "Whole Body RL"
    "Whole Body RL"
)
NAMES2=(
    "CTBR + Joint Angles (Trajectory Tracking Policy)"
    "CTBR + Joint Angles (Trajectory Tracking Policy)"
)
NAMES3=(
    "CTBR + Joint Angles (Hover Policy)"
    "CTBR + Joint Angles (Hover Policy)"
)
OUTPUTS=(
    "Trajectory Tracking 3 part study"
    "Hover 3 part study"
)
for i in ${!LOGS[@]}; do
    LOG=${LOGS[$i]}
    NAME=${NAMES[$i]}
    LOG2=${LOGS2[$i]}
    NAME2=${NAMES2[$i]}
    LOG3=${LOGS3[$i]}
    NAME3=${NAMES3[$i]}
    OUTPUT=${OUTPUTS[$i]}
    echo "Processing $LOG with name $NAME"
    python compare_controllers_cli.py -c1 $LOG -c1n "$NAME" -c1s "$NAME" --output "$OUTPUT" \
    -c2 $LOG2 -c2n "$NAME2" -c2s "$NAME2" -c3 $LOG3 -c3n "$NAME3" -c3s "$NAME3"
    #  --vel-analysis --vel-bins-lin 3 --vel-bins-ang 4
done

echo "Processing relative error plot for ${LOGS[0]} vs ${LOGS2[0]}"
python compare_controllers_cli.py -c1 ${LOGS[0]} -c1n "${NAMES[0]}" -c1s "${NAMES[0]}" --output "Trajectory Tracking Relative Error" \
-c2 ${LOGS2[0]} -c2n "${NAMES2[0]}" -c2s "${NAMES2[0]}" \
--vel-analysis --vel-bins-lin 3 --vel-bins-ang 4

