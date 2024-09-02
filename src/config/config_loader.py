import yaml
import torch


def load_configs(task_config_path):
    with open(task_config_path, "r") as f:
        task_config = yaml.load(f, Loader=yaml.CLoader)["task_config"]

    with open(task_config["config"], "r") as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    return task_config, config


def process_configs(task_config, config, device):
    train_config = config["train_config"]
    dataset_config = config["dataset_config"]
    model_config = config["model_config"]
    material_config = config["material_config"]

    action_lower_lim = torch.tensor(
        task_config["action_lower_lim"], dtype=torch.float32, device=device
    )
    action_upper_lim = torch.tensor(
        task_config["action_upper_lim"], dtype=torch.float32, device=device
    )

    return (
        train_config,
        dataset_config,
        model_config,
        material_config,
        action_lower_lim,
        action_upper_lim,
    )
