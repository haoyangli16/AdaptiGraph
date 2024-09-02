import torch


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
