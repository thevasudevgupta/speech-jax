import yaml


def read_yaml(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data
