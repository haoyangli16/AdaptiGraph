from functools import partial
from planning.forward_dynamics import dynamics
from planning.plan_utils import sample_action_seq, clip_actions, optimize_action_mppi
from planning.real_world.planner import Planner


def configure_planner(
    task_config,
    action_lower_lim,
    action_upper_lim,
    model,
    device,
    ppm_optimizer,
    running_cost_func,
):
    planner_config = {
        "action_dim": len(action_lower_lim),
        "model_rollout_fn": partial(
            dynamics, model=model, device=device, ppm_optimizer=ppm_optimizer
        ),
        "evaluate_traj_fn": running_cost_func,
        "sampling_action_seq_fn": partial(
            sample_action_seq,
            action_lower_lim=action_lower_lim,
            action_upper_lim=action_upper_lim,
            n_sample=min(task_config["n_sample"], task_config["n_sample_chunk"]),
            device=device,
            noise_level=task_config["noise_level"],
            push_length=task_config["push_length"],
        ),
        "clip_action_seq_fn": partial(
            clip_actions,
            action_lower_lim=action_lower_lim,
            action_upper_lim=action_upper_lim,
        ),
        "optimize_action_mppi_fn": partial(
            optimize_action_mppi,
            reward_weight=task_config["reward_weight"],
            action_lower_lim=action_lower_lim,
            action_upper_lim=action_upper_lim,
            push_length=task_config["push_length"],
        ),
        "n_sample": min(task_config["n_sample"], task_config["n_sample_chunk"]),
        "n_look_ahead": task_config["n_look_ahead"],
        "n_update_iter": 10,
        "reward_weight": task_config["reward_weight"],
        "action_lower_lim": action_lower_lim,
        "action_upper_lim": action_upper_lim,
        "planner_type": "MPPI",
        "device": device,
        "verbose": False,
        "noise_level": task_config["noise_level"],
        "rollout_best": True,
    }

    planner = Planner(planner_config)
    planner.total_chunks = int(task_config["n_sample"] / task_config["n_sample_chunk"])

    return planner, planner_config
