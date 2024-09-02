from unisoft.env.scene import (
    setup_pd_components,
    simulate_step,
    SimplePickClothEnv,
)
from sapien import Pose
import sapienpd
import numpy as np


def initialize_environment(task_config, warm_up_step=10, visualize=False):
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

    for _ in range(warm_up_step):
        simulate_step(
            env,
            np.zeros_like(env.action_space.sample()),
            rigid_comp,
            rigid_xarm_comp,
            xarm_link,
            pd_comp,
        )
        sapienpd.scene_update_render(env.scene.sub_scenes[0])
        if visualize:
            env.render()
        else:
            env.scene.update_render()
    return env, rigid_comp, rigid_xarm_comp, xarm_link, pd_comp
