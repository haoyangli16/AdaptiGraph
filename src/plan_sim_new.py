import argparse
import glob
import os
import time
from functools import partial

import numpy as np
import open3d as o3d
import torch
from config.config_loader import load_configs, process_configs
from dynamics.gnn.model import DynamicsPredictor
from dynamics.utils import set_seed
from sim.environment_setup import initialize_environment
import sapienpd
from planning.losses import (
    box_loss,
    chamfer,
    cloth_penalty,
    granular_penalty,
    rope_penalty,
)
from planning.physics_param_optimizer import PhysicsParamOnlineOptimizer
from planning.utils.obs import get_state_cur, visualize_sim
from planning.utils.planner_setup import configure_planner
from planning.utils.reward import running_cost
from planning.real_world.utils import visualize_o3d
from unisoft.env.scene import (
    setup_pd_components,
    simulate_step,
    SimplePickClothEnv,
)


def setup_target_and_criteria(task_config, sim_real_ratio, device):
    if task_config["target_type"] == "pcd":
        pcd = o3d.io.read_point_cloud(
            os.path.join(os.path.dirname(__file__), task_config["target"])
        )
        target_state = np.array(pcd.points) * sim_real_ratio

        # NOTE: turn (x, y, z) to (x, -z, y)
        # NOTE(haoyang): be attention that get observation from the sapien simulator should be (x, -z, y)
        target_state = target_state[:, [0, 2, 1]].copy()
        target_state[:, 1] *= -1

        target_state = torch.tensor(target_state, dtype=torch.float32, device=device)
        target_box = None
        criteria = partial(chamfer, y=target_state[None])
    elif task_config["target_type"] == "box":
        target_box = (
            np.array(
                [
                    [float(task_config["target"][0]), float(task_config["target"][1])],
                    [float(task_config["target"][2]), float(task_config["target"][3])],
                ]
            )
            * sim_real_ratio
        )
        target_box = torch.tensor(target_box, dtype=torch.float32, device=device)
        target_state = None
        criteria = partial(box_loss, target=target_box)
    else:
        raise NotImplementedError(
            f"target type {task_config['target_type']} not implemented"
        )

    return target_state, target_box, criteria


def setup_penalty(task_config, sim_real_ratio):
    if task_config["penalty_type"] == "rope":
        return partial(rope_penalty, sim_real_ratio=sim_real_ratio)
    elif task_config["penalty_type"] == "cloth":
        return partial(cloth_penalty, sim_real_ratio=sim_real_ratio)
    elif task_config["penalty_type"] == "granular":
        return partial(granular_penalty, sim_real_ratio=sim_real_ratio)
    else:
        raise NotImplementedError(
            f"penalty type {task_config['penalty_type']} not implemented"
        )


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--task_config", type=str)
    arg_parser.add_argument("--resume", action="store_true")
    arg_parser.add_argument("--seed", type=int, default=43)
    args = arg_parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    task_config, config = load_configs(args.task_config)
    (
        train_config,
        dataset_config,
        model_config,
        material_config,
        action_lower_lim,
        action_upper_lim,
    ) = process_configs(task_config, config, device)

    env, rigid_comp, rigid_xarm_comp, xarm_link, pd_comp = initialize_environment(
        task_config, warm_up_step=20, visualize=True
    )

    save_dir = os.path.join(
        os.path.dirname(__file__),
        f"dump/vis/planning-{dataset_config['data_name']}-model_{task_config['epoch']}",
    )
    os.makedirs(save_dir, exist_ok=True)
    n_resume = (
        len(glob.glob(os.path.join(save_dir, "interaction_*.npz")))
        if args.resume
        else 0
    )

    model = DynamicsPredictor(model_config, material_config, dataset_config, device)
    model.to(device)
    model.eval()
    model.load_state_dict(
        torch.load(
            os.path.join(
                train_config["out_dir"],
                dataset_config["data_name"],
                "checkpoints",
                f"model_{task_config['epoch']}.pth",
            ),
            map_location="cpu",
        )
    )

    target_state, target_box, criteria = setup_target_and_criteria(
        task_config, task_config["sim_real_ratio"], device
    )
    penalty = setup_penalty(task_config, task_config["sim_real_ratio"])

    bbox_2d = (
        np.array(
            [
                [float(task_config["bbox"][0]), float(task_config["bbox"][1])],
                [float(task_config["bbox"][2]), float(task_config["bbox"][3])],
            ]
        )
        * task_config["sim_real_ratio"]
    )
    running_cost_func = partial(
        running_cost, error_func=criteria, penalty_func=penalty, bbox=bbox_2d
    )

    ppm_optimizer = PhysicsParamOnlineOptimizer(
        task_config, model, task_config["material"], device, save_dir
    )
    planner, planner_config = configure_planner(
        task_config,
        action_lower_lim,
        action_upper_lim,
        model,
        device,
        ppm_optimizer,
        running_cost_func,
    )

    act_seq = (
        torch.rand(
            (planner_config["n_look_ahead"], action_upper_lim.shape[0]), device=device
        )
        * (action_upper_lim - action_lower_lim)
        + action_lower_lim
    )
    res_act_seq = torch.zeros(
        (task_config["n_actions"], action_upper_lim.shape[0]), device=device
    )

    error_seq = []
    for i in range(n_resume, task_config["n_actions"]):
        # ---- get the current state ---- #
        state_cur, obj_pcd = get_state_cur(
            env,
            device,
            fps_radius=ppm_optimizer.fps_radius,
            if_sample=True,
            sample_num=task_config["n_sample"],
        )
        # NOTE(haoyang): change the obj_pcd to (x, -z, y) to align with the adaptigraph
        obj_pcd = obj_pcd[:, [0, 2, 1]].copy()
        obj_pcd[:, 1] *= -1

        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_pcd)

        # ---- compute the error ---- #'
        if i == 0:
            # visualize the obj_pcd
            print(f"obj_pcd shape: {torch.from_numpy(obj_pcd).shape}")
            error = criteria(torch.from_numpy(obj_pcd)[None].to(device)).item()
            print("error", error)
            error_seq.append(error)

        # ---- compute the action sequence (predict the next action) ---- #
        res_all = [
            planner.trajectory_optimization(state_cur, act_seq)
            for _ in range(planner.total_chunks)
        ]
        res = planner.merge_res(res_all)
        # get the action (x_start, z_start, x_end, z_end)
        action = res["act_seq"][0].detach().cpu().numpy()

        # convert to the sapien action
        sapien_action = convert_to_sapien_action(action, obj_pcd)

        # ---- step the simulation ---- #
        env.create_eef_sphere(
            name=f"eef_point_{sapien_action['start_index']}", color=[0.0, 1.0, 0.0, 1.0]
        )
        env.set_pd_delta_position(
            [sapien_action["start_index"]],
            0.05 * sapien_action["displacement"][None],
        )
        print(f"action: {0.05 * sapien_action['displacement']}")

        action_fake = env.action_space.sample()
        simulate_step(env, action_fake, rigid_comp, rigid_xarm_comp, xarm_link, pd_comp)
        # sapienpd.scene_update_render(env.scene.sub_scenes[0])
        # env.scene.update_render()
        env.render()
        sapienpd.scene_update_render(env.scene.sub_scenes[0])

        # ---- visualize the simulation ---- #
        # visualize_sim(
        #     env,
        #     state_cur,
        #     res,
        #     target_state=target_state,
        #     target_box=target_box,
        #     save_dir=save_dir,
        #     postfix=f"{i}_0",
        #     task_config=task_config,
        # )
        # visualize_pointcloud_and_original_action(obj_pcd, action)
        # visualize_pointcloud_and_action(obj_pcd, action, sapien_action, scale=0.05)
        # ---- update the action sequence ---- #
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
        n_look_ahead = min(task_config["n_actions"] - i, planner_config["n_look_ahead"])
        act_seq = act_seq[:n_look_ahead]
        planner.n_look_ahead = n_look_ahead

        # ---- Save results and visualize after step (implement as needed) ---- #

    print(f"final action sequence {res_act_seq}")
    print(f"final error sequence {error_seq}")

    with open(os.path.join(save_dir, "stats.txt"), "w") as f:
        f.write(f"final action sequence {res_act_seq}\n")
        f.write(f"final error sequence {error_seq}\n")

    # Make video with cv2 (implement as needed)


import numpy as np


def convert_to_sapien_action(action, obj_pcd, scale=1.0):
    # The action is (x_start, z_start, x_end, z_end)
    x_start, z_start, x_end, z_end = action

    # 1. Align the start point to the closest point on the cloth
    # start_point = np.array([x_start, 0, -z_start])  # Convert to (x, y, -z)
    # get the average y value of the obj_pcd
    avg_y = np.mean(obj_pcd[:, 1])
    start_point = np.array([x_start, avg_y, z_start])

    distances = np.linalg.norm(obj_pcd - start_point, axis=1)
    closest_point_index = np.argmin(distances)
    aligned_start_point = obj_pcd[closest_point_index]

    # 2. Determine the index of the start point on the cloth
    start_index = int(closest_point_index)

    # 3. Scale the action sequence (if necessary)
    # Assuming no scaling is needed, but you can add it if required
    scaled_end_point = scale * np.array([x_end, avg_y, z_end])

    # Calculate the displacement vector
    displacement = scaled_end_point - aligned_start_point
    # turn the displacement to (x, y, z)
    displacement = np.array([displacement[0], -displacement[2], displacement[1]])

    # normalize the displacement
    displacement = displacement / np.linalg.norm(displacement)

    # 5. Return the Sapien action and start point index
    sapien_action = {"start_index": start_index, "displacement": displacement}

    return sapien_action


def visualize_pointcloud_and_action(obj_pcd, action, sapien_action, scale=0.01):
    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj_pcd)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray color for cloth points

    # Create start point
    start_point = (
        sapien_action["displacement"] * 0 + obj_pcd[sapien_action["start_index"]]
    )
    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    start_sphere.translate(start_point)
    start_sphere.paint_uniform_color([1, 0, 0])  # Red color for start point

    # Create end point
    end_point = (
        sapien_action["displacement"] * scale + start_point
    )  # Scale displacement for visibility
    end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    end_sphere.translate(end_point)
    end_sphere.paint_uniform_color([0, 1, 0])  # Green color for end point

    # Create action vector
    action_vector = o3d.geometry.LineSet()
    action_vector.points = o3d.utility.Vector3dVector([start_point, end_point])
    action_vector.lines = o3d.utility.Vector2iVector([[0, 1]])
    action_vector.colors = o3d.utility.Vector3dVector(
        [[0, 0, 1]]
    )  # Blue color for action vector

    # Visualize
    o3d.visualization.draw_geometries([pcd, start_sphere, end_sphere, action_vector])


def visualize_pointcloud_and_original_action(obj_pcd, action):
    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj_pcd)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray color for cloth points

    # Extract action components
    x_start, z_start, x_end, z_end = action

    # Calculate average y-value from the point cloud
    avg_y = np.mean(obj_pcd[:, 1])

    # Create start point
    start_point = np.array([x_start, avg_y, z_start])
    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    start_sphere.translate(start_point)
    start_sphere.paint_uniform_color([1, 0, 0])  # Red color for start point

    # Create end point
    end_point = np.array([x_end, avg_y, z_end])
    end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    end_sphere.translate(end_point)
    end_sphere.paint_uniform_color([0, 1, 0])  # Green color for end point

    # Create action vector
    action_vector = o3d.geometry.LineSet()
    action_vector.points = o3d.utility.Vector3dVector([start_point, end_point])
    action_vector.lines = o3d.utility.Vector2iVector([[0, 1]])
    action_vector.colors = o3d.utility.Vector3dVector(
        [[0, 0, 1]]
    )  # Blue color for action vector

    # Visualize
    o3d.visualization.draw_geometries([pcd, start_sphere, end_sphere, action_vector])


if __name__ == "__main__":
    main()
