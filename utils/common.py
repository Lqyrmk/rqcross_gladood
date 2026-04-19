import os
import random
import numpy as np
import torch
import pandas as pd


def set_seed(seed: int = 3407):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_results_to_csv(result_dict, model_name: str):
    results_dir = "benchmark/results"
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"{model_name}.csv")

    df = pd.DataFrame([result_dict])
    if os.path.exists(save_path):
        df.to_csv(save_path, mode='a', header=False, index=False)
    else:
        df.to_csv(save_path, mode='w', header=True, index=False)

    print(f"✅ 结果已保存 → {save_path}")