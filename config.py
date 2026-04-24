import argparse
import torch
import warnings


class Config:
    """
    🔥 全局统一配置类
    所有参数都在这里，修改、查找、维护超级方便
    """
    # ============== 基础设置 ==============
    model: str = "CROSS"
    gpu: int = 0
    data_root: str = "dataset"
    exp_type: str = "ad"
    seed: int = 42

    # ============== 数据集 ==============
    ad_dataset: str = "DHFR"
    ood_dataset: str = "AIDS+DHFR"

    # ============== 模型结构 ==============
    rw_dim: int = 16
    dg_dim: int = 16
    num_layer: int = 5
    hidden_dim: int = 32
    lr: int = 0.001
    temperature: int = 0.7
    dropout: float = 0.1
    eps: float = 0.001
    scalar: float = 20
    k: int = 10
    num_heads: int = 5
    pooling: str = 'mean'
    readout: str = 'concat'

    # ============== 训练超参 ==============
    batch_size: int = 128
    batch_size_test: int = 9999
    lr: float = 0.001
    num_epoch: int = 300
    num_trial: int = 5
    eval_freq: int = 5
    alpha: float = 1.0
    beta: float = 1.0
    n_train: int = 10


def parse_args() -> Config:
    """解析命令行参数，映射到 Config"""
    parser = argparse.ArgumentParser(description="Clean Research Pipeline")

    for key, value in vars(Config).items():
        if not key.startswith("__") and not callable(getattr(Config, key)):
            parser.add_argument(f"--{key}", default=value, type=type(value))

    args = parser.parse_args()
    config = Config()

    for key, value in args.__dict__.items():
        setattr(config, key, value)

    return config


def setup_env(config: Config):
    """初始化环境：种子、GPU、警告过滤"""
    warnings.filterwarnings("ignore")
    torch.manual_seed(config.seed)

    # GPU 设置
    if config.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(config.gpu)
        device = torch.device(f"cuda:{config.gpu}")
    else:
        device = torch.device("cpu")

    config.device = device
