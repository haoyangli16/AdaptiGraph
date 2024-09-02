import argparse
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import time
import cv2
import math
import os
import yaml
import glob
import copy
from functools import partial
import sys
import os
import sapienpd
from sapien import Pose

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planning.real_world.planner import Planner
from planning.forward_dynamics import dynamics
from planning.plan_utils import (
    visualize_img,
    clip_actions,
    optimize_action_mppi,
    sample_action_seq,
)
from planning.physics_param_optimizer import PhysicsParamOnlineOptimizer
from planning.losses import (
    chamfer,
    box_loss,
    rope_penalty,
    cloth_penalty,
    granular_penalty,
)

from dynamics.gnn.model import DynamicsPredictor
from dynamics.utils import set_seed
import sapien as sapien
from sapien.utils import Viewer

import cv2
import numpy as np
import torch
import sapien

from unisoft.env.scene import (
    setup_pd_components,
    simulate_step,
    SimplePickClothEnv,
)


def running_cost(
    state, action, state_cur, error_func, penalty_func, bbox, **kwargs
):  # tabletop coordinates
    # state: (bsz, n_look_forward, max_nobj, 3)
    # action: (bsz, n_look_forward, action_dim)
    # target_state: numpy.ndarray (n_target, 3)
    # state_cur: (max_nobj, 3)
    bsz = state.shape[0]
    n_look_forward = state.shape[1]

    state_flat = state.reshape(bsz * n_look_forward, state.shape[2], state.shape[3])
    error = error_func(state_flat).reshape(bsz, n_look_forward)
    error_weight = 2.0 / (error.max().item() + 1e-6)

    collision_penalty = penalty_func(state, action, state_cur)

    xmax = state.max(dim=2).values[:, :, 0]  # (bsz, n_look_forward)
    xmin = state.min(dim=2).values[:, :, 0]  # (bsz, n_look_forward)
    zmax = state.max(dim=2).values[:, :, 2]  # (bsz, n_look_forward)
    zmin = state.min(dim=2).values[:, :, 2]  # (bsz, n_look_forward)
    box_penalty = torch.stack(
        [
            torch.maximum(xmin - bbox[0, 0], torch.zeros_like(xmin)),
            torch.maximum(bbox[0, 1] - xmax, torch.zeros_like(xmax)),
            torch.maximum(zmin - bbox[1, 0], torch.zeros_like(zmin)),
            torch.maximum(bbox[1, 1] - zmax, torch.zeros_like(zmax)),
        ],
        dim=-1,
    )  # (bsz, n_look_forward, 4)
    box_penalty = (
        torch.exp(-box_penalty * 100.0).max(dim=-1).values
    )  # (bsz, n_look_forward)

    reward = (
        -error_weight * error[:, -1]
        - 5.0 * collision_penalty.mean(dim=1)
        - 5.0 * box_penalty.mean(dim=1)
    )  # (bsz,)

    print(f"min error {error[:, -1].min().item()}, max reward {reward.max().item()}")

    out = {
        "reward_seqs": reward,
    }
    return out


def visualize_sim(
    env,
    state_cur,
    res,
    target_state=None,
    target_box=None,
    save_dir=None,
    postfix="",
    task_config=None,
):
    # Render the current state
    rgba = env.get_camera_image()
    rgba = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    # Draw predicted trajectory
    state_pred = res["best_model_output"]["state_seqs"][0].detach().cpu().numpy()
    for i in range(1, len(state_pred)):
        start = state_pred[i - 1].mean(axis=0)
        end = state_pred[i].mean(axis=0)
        start_2d = env.world_to_pixel(start)
        end_2d = env.world_to_pixel(end)
        cv2.line(
            rgba,
            tuple(start_2d.astype(int)),
            tuple(end_2d.astype(int)),
            (0, 255, 0),
            2,
        )

    # Draw target state or box if available
    if target_state is not None:
        pass
        # for point in target_state:
        #     point_2d = env.world_to_pixel(point)
        #     cv2.circle(rgba, tuple(point_2d.astype(int)), 3, (255, 0, 0), -1)
    elif target_box is not None:
        corners = np.array(
            [
                [target_box[0, 0], target_box[1, 0], 0],
                [target_box[0, 1], target_box[1, 0], 0],
                [target_box[0, 1], target_box[1, 1], 0],
                [target_box[0, 0], target_box[1, 1], 0],
            ]
        )
        for i in range(4):
            start = env.world_to_pixel(corners[i])
            end = env.world_to_pixel(corners[(i + 1) % 4])
            cv2.line(
                rgba,
                tuple(start.astype(int)),
                tuple(end.astype(int)),
                (255, 0, 0),
                2,
            )

    if save_dir is not None:
        cv2.imwrite(f"{save_dir}/rgba_{postfix}.png", rgba)

    return rgba


def get_state_cur(env, device, fps_radius=0.01, sample_num=1000):
    cloth_points = env.get_cloth_points(sample_num)
    state_cur = torch.tensor(cloth_points, dtype=torch.float32, device=device)

    # Get RGB image in simulator
    # rgba = env.get_camera_image()
    # Get camera intrinsics and extrinsics
    # intr = env.get_camera_intrinsic_matrix()
    # extr = env.get_camera_extrinsic_matrix()

    return state_cur, cloth_points


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--task_config", type=str)
    arg_parser.add_argument("--resume", action="store_true")
    arg_parser.add_argument("--seed", type=int, default=43)
    arg_parser.add_argument("--use_ppo", action="store_true")
    args = arg_parser.parse_args()

    base_path = os.path.dirname(os.path.abspath(__file__))
    set_seed(args.seed)

    with open(args.task_config, "r") as f:
        task_config = yaml.load(f, Loader=yaml.CLoader)["task_config"]
    config_path = task_config["config"]
    epoch = task_config["epoch"]
    material = task_config["material"]
    gripper_enable = task_config["gripper_enable"]

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    train_config = config["train_config"]
    dataset_config = config["dataset_config"]
    model_config = config["model_config"]
    material_config = config["material_config"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the simulation environment
    env = SimplePickClothEnv(
        render_mode="human",
        control_mode="pd_joint_pos",
        robot_uids="panda",
        cloth_size=task_config["cloth_size"],
        cloth_resolution=task_config["cloth_resolution"],
    )
    env.reset()
    if env.robot_uids == "panda":
        env.agent.robot.set_root_pose(Pose([-0.34, 0.01, 0.77], [1, 0, 0, 0]))
    elif env.robot_uids == "xarm7":
        env.agent.robot.set_root_pose(Pose([-0.36, 0.01, 0.77], [1, 0, 0, 0]))
    rigid_comp, rigid_xarm_comp, xarm_link, pd_comp = setup_pd_components(env)

    sapienpd.scene_update_render(env.scene.sub_scenes[0])
    env.scene.update_render()

    action_lower_lim = torch.tensor(
        task_config["action_lower_lim"], dtype=torch.float32, device=device
    )
    action_upper_lim = torch.tensor(
        task_config["action_upper_lim"], dtype=torch.float32, device=device
    )

    run_name = dataset_config["data_name"]
    save_dir = os.path.join(base_path, f"dump/vis/planning-{run_name}-model_{epoch}")
    if (
        not args.resume
        and os.path.exists(save_dir)
        and len(glob.glob(os.path.join(save_dir, "*.npz"))) > 0
    ):
        print("save dir already exists")
        return
    os.makedirs(save_dir, exist_ok=True)
    if args.resume:
        print("resume")
        n_resume = len(glob.glob(os.path.join(save_dir, "interaction_*.npz")))
    else:
        n_resume = 0
    print("starting from iteration {}".format(n_resume))
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = os.path.join(
        train_config["out_dir"],
        dataset_config["data_name"],
        "checkpoints",
        "model_{}.pth".format(epoch),
    )

    model = DynamicsPredictor(model_config, material_config, dataset_config, device)
    model.to(device)

    model.eval()
    model.load_state_dict(torch.load(checkpoint_dir, map_location="cpu"))

    push_length = task_config["push_length"]
    sim_real_ratio = task_config["sim_real_ratio"]

    # Set up target, criteria, and penalty as before
    # target
    if task_config["target_type"] == "pcd":
        pcd = o3d.io.read_point_cloud(os.path.join(base_path, task_config["target"]))
        target_state = np.array(pcd.points) * sim_real_ratio
        target_state = target_state[:, [0, 2, 1]].copy()  # (x, y, z) -> (x, z, y)
        target_state[:, 1] *= -1  # (x, z, y) -> (x, -z, y)
        target_state = torch.tensor(target_state, dtype=torch.float32, device=device)
        target_box = None
        criteria = partial(chamfer, y=target_state[None])
    elif task_config["target_type"] == "box":
        target_box = np.array(
            [
                [float(task_config["target"][0]), float(task_config["target"][1])],
                [float(task_config["target"][2]), float(task_config["target"][3])],
            ]
        )  # (x_min, x_max), (z_min, z_max)
        target_box = target_box * sim_real_ratio
        target_box = torch.tensor(target_box, dtype=torch.float32, device=device)
        target_state = None
        criteria = partial(box_loss, target=target_box)
    else:
        raise NotImplementedError(
            f"target type {task_config['target_type']} not implemented"
        )

    # penalty
    if task_config["penalty_type"] == "rope":
        penalty = partial(rope_penalty, sim_real_ratio=sim_real_ratio)
    elif task_config["penalty_type"] == "cloth":
        penalty = partial(cloth_penalty, sim_real_ratio=sim_real_ratio)
    elif task_config["penalty_type"] == "granular":
        penalty = partial(granular_penalty, sim_real_ratio=sim_real_ratio)
    else:
        raise NotImplementedError(
            f"penalty type {task_config['penalty_type']} not implemented"
        )
    # bounding box penalty
    bbox_2d = np.array(
        [
            [float(task_config["bbox"][0]), float(task_config["bbox"][1])],
            [float(task_config["bbox"][2]), float(task_config["bbox"][3])],
        ]
    )  # (x_min, x_max), (z_min, z_max)
    bbox_2d = bbox_2d * sim_real_ratio
    running_cost_func = partial(
        running_cost, error_func=criteria, penalty_func=penalty, bbox=bbox_2d
    )

    n_actions = task_config["n_actions"]
    n_look_ahead = task_config["n_look_ahead"]
    n_sample = task_config["n_sample"]
    n_sample_chunk = task_config["n_sample_chunk"]

    n_chunk = np.ceil(n_sample / n_sample_chunk).astype(int)
    noise_level = task_config["noise_level"]
    reward_weight = task_config["reward_weight"]

    ppm_optimizer = PhysicsParamOnlineOptimizer(
        task_config, model, material, device, save_dir
    )

    # Set up planner_config as before
    noise_level = task_config["noise_level"]
    reward_weight = task_config["reward_weight"]
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
            n_sample=min(n_sample, n_sample_chunk),
            device=device,
            noise_level=noise_level,
            push_length=push_length,
        ),
        "clip_action_seq_fn": partial(
            clip_actions,
            action_lower_lim=action_lower_lim,
            action_upper_lim=action_upper_lim,
        ),
        "optimize_action_mppi_fn": partial(
            optimize_action_mppi,
            reward_weight=reward_weight,
            action_lower_lim=action_lower_lim,
            action_upper_lim=action_upper_lim,
            push_length=push_length,
        ),
        "n_sample": min(n_sample, n_sample_chunk),
        "n_look_ahead": n_look_ahead,
        "n_update_iter": 1,
        "reward_weight": reward_weight,
        "action_lower_lim": action_lower_lim,
        "action_upper_lim": action_upper_lim,
        "planner_type": "MPPI",
        "device": device,
        "verbose": False,
        "noise_level": noise_level,
        "rollout_best": True,
    }

    # print the planner config
    print("planner config:")
    for k, v in planner_config.items():
        print(f"{k}: {v}")

    planner = Planner(planner_config)
    planner.total_chunks = n_chunk
    act_seq = (
        torch.rand(
            (planner_config["n_look_ahead"], action_upper_lim.shape[0]), device=device
        )
        * (action_upper_lim - action_lower_lim)
        + action_lower_lim
    )

    res_act_seq = torch.zeros((n_actions, action_upper_lim.shape[0]), device=device)

    if n_resume > 0:
        interaction_list = sorted(
            glob.glob(os.path.join(save_dir, "interaction_*.npz"))
        )
        for i in range(n_resume):
            res = np.load(interaction_list[i])
            act_save = res["act"]
            state_init_save = res["state_init"]
            state_pred_save = res["state_pred"]
            state_real_save = res["state_real"]
            res_act_seq[i] = torch.tensor(act_save, dtype=torch.float32, device=device)

    error_seq = []
    n_steps = n_actions
    for i in range(n_resume, n_actions):
        time1 = time.time()
        # get state
        state_cur, obj_pcd = get_state_cur(
            env,
            device,
            fps_radius=ppm_optimizer.fps_radius,
            sample_num=task_config["n_sample"],
        )
        if i == 0:
            error = criteria(torch.from_numpy(obj_pcd)[None].to(device)).item()
            print("error", error)
            error_seq.append(error)

        # NOTE(haoyang): get action
        res_all = []
        for ci in range(n_chunk):
            planner.chunk_id = ci
            breakpoint()
            res = planner.trajectory_optimization(state_cur, act_seq)
            for k, v in res.items():
                res[k] = v.detach().clone() if isinstance(v, torch.Tensor) else v
            res_all.append(res)
        res = planner.merge_res(res_all)

        # vis
        visualize_sim(
            env,
            state_cur,
            res,
            target_state=target_state,
            target_box=target_box,
            save_dir=save_dir,
            postfix=f"{i}_0",
            task_config=task_config,
        )
        # get the action
        # (x_start, z_start, x_end, z_end) -> But you have to convert into sapien / robot frame
        action = res["act_seq"][0].detach().cpu().numpy()

        eef_pos = Pose(p=np.array([action[0], 0.8, action[1]]))
        env.eef_point.set_pose(eef_pos)
        print(f"action: {action} \n")

        # make sure the env can be stepped
        action_fake = env.action_space.sample()
        env.step(action_fake)
        # env.render()
        simulate_step(env, action_fake, rigid_comp, rigid_xarm_comp, xarm_link, pd_comp)
        sapienpd.scene_update_render(env.scene.sub_scenes[0])
        env.scene.update_render()

        # Update action
        res_act_seq[i] = res["act_seq"][0].detach().clone()
        act_seq = torch.cat(
            [
                res["act_seq"][1:],
                torch.rand((1, action_upper_lim.shape[0]), device=device)
                * (action_upper_lim - action_lower_lim)
                + action_lower_lim,
            ],
            dim=0,
        )
        n_look_ahead = min(n_actions - i, planner_config["n_look_ahead"])
        act_seq = act_seq[:n_look_ahead]  # sliding window
        planner.n_look_ahead = n_look_ahead

        # Save results and visualize after step
        # ...

        print(f"final action sequence {res_act_seq}")
        print(f"final error sequence {error_seq}")

        with open(os.path.join(save_dir, "stats.txt"), "w") as f:
            f.write(f"final action sequence {res_act_seq}\n")
            f.write(f"final error sequence {error_seq}\n")

    # Make video with cv2 (you may need to modify this for the simulator)
    # ...


if __name__ == "__main__":
    with torch.no_grad():
        main()
