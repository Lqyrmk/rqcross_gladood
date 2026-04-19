import os

def get_model_save_path(config):
    base = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(base, "..", "model_save")

    exp_type = config.exp_type
    model = config.model

    if exp_type == 'oodd':
        path = os.path.join(base, model, exp_type, config.ood_dataset)
    elif config.ad_dataset.startswith('Tox21'):
        path = os.path.join(base, model, exp_type + 'Tox21', config.ad_dataset)
    else:
        path = os.path.join(base, model, exp_type, config.ad_dataset)

    os.makedirs(path, exist_ok=True)
    return path

def clear_directory(path):
    if not os.path.exists(path):
        return
    for f in os.listdir(path):
        fp = os.path.join(path, f)
        if os.path.isfile(fp):
            os.remove(fp)