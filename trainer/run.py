import statistics
from tqdm import tqdm
from utils.common import set_seed, save_results_to_csv
from utils.metrics import ood_auc, ood_aupr, fpr95
from utils.data_utils import get_dataset
from models.detector.cross_detector import CrossDetector

def run_single_trial(config, trial_idx: int):
    config.trial_idx = trial_idx
    set_seed(config.seed + trial_idx)

    datasets = get_dataset(config)
    train, val, test, loader, loader_val, loader_test, meta = datasets

    # meta info
    config.max_nodes_num = meta["max_nodes_num"]
    config.dataset_num_features = meta["num_feat"]
    config.n_train = meta["num_train"]
    config.n_edge_feat = meta.get("num_edge_feat", 0)

    # init detector
    detector = CrossDetector(config)
    print(f"🔹 Trial {trial_idx} ...")

    # train & predict
    detector.fit(dataloader=loader, dataloader_val=loader_val)

    score, y_true = detector.predict(dataloader=loader_test)

    # metrics
    auc = ood_auc(y_true, score)
    ap = ood_aupr(y_true, score)
    rec = fpr95(y_true, score)

    print(f"✅ AUROC: {auc:.4f} | AUPRC: {ap:.4f} | FPR95: {rec:.4f}\n")
    return auc, ap, rec


def run_experiment(config):
    auc_list, ap_list, fpr_list = [], [], []

    for i in tqdm(range(config.num_trial), desc="Running Trials"):
        auc, ap, fpr = run_single_trial(config, i)
        auc_list.append(auc)
        ap_list.append(ap)
        fpr_list.append(fpr)

    # 汇总结果
    def mean(arr):
        return sum(arr) / len(arr)

    def var(arr):
        return statistics.variance(arr) if len(arr) > 1 else 0.0

    result = {
        "Dataset": config.ad_dataset if config.exp_type == "ad" else config.ood_dataset,
        "AUROC": f"{mean(auc_list)*100:.2f}%",
        "AUROC_Var": f"{var(auc_list)*100:.2f}%",
        "AUPRC": f"{mean(ap_list)*100:.2f}%",
        "AUPRC_Var": f"{var(ap_list)*100:.2f}%",
        "FPR95": f"{mean(fpr_list)*100:.2f}%",
        "FPR95_Var": f"{var(fpr_list)*100:.2f}%",
    }

    save_results_to_csv(result, config.model)

    return {"AUROC": auc_list, "AUPRC": ap_list, "FPR95": fpr_list}