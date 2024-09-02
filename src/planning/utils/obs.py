import cv2
import numpy as np
import torch
from unisoft.env.scene.pick_cloth import SimplePickClothEnv


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
    print(f"state_pred: {state_pred}")

    for i in range(1, len(state_pred)):
        start = state_pred[i - 1].mean(axis=0)
        end = state_pred[i].mean(axis=0)
        # TODO: check if the world_to_pixel is correct,
        # 1. the start and end are 3d or 2d?
        # 2. the start and end are in the world coordinate
        # 3. the start and end are in the sensor camera (the same camera with the rgba)
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
        for point in target_state:
            point_2d = env.world_to_pixel(point)
            cv2.circle(rgba, tuple(point_2d.astype(int)), 3, (255, 0, 0), -1)
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


def get_state_cur(
    env: SimplePickClothEnv,
    device,
    fps_radius=0.01,
    if_sample=False,
    sample_num=1000,
):
    cloth_points = env.get_cloth_points(if_sample, sample_num)
    state_cur = torch.tensor(cloth_points, dtype=torch.float32, device=device)

    # Get RGB image in simulator
    # rgba = env.get_camera_image()
    # Get camera intrinsics and extrinsics
    # intr = env.get_camera_intrinsic_matrix()
    # extr = env.get_camera_extrinsic_matrix()

    return state_cur, cloth_points
