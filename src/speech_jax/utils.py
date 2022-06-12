from typing import Any, Dict

import yaml

# TODO: look into how we can freeze this class?
# class DictStore:
#     def __init__(self, dictionary: Dict[str, Any]):
#         for k, v in dictionary.items():
#             if isinstance(v, dict):
#                 v = DictStore(v)
#             setattr(self, k, v)

#         self._dictionary = dictionary

#     def __repr__(self) -> str:
#         return str(self._dictionary)

#     def to_dict(self) -> Dict[str, Any]:
#         return self._dictionary


def read_yaml(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data


# def configs_from_yaml(path):
#     return DictStore(read_yaml(path))
