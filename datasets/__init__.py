import os
import os.path as osp
import importlib
import inspect
from .base_dataset import BaseDataset

def create_datasets(cfg, split_types, log):

    filenames = os.listdir(osp.dirname(__file__))
    filenames = filter(lambda x: x.endswith('.py') and x!='__init__.py', filenames)
    type2dataset = dict()
    for filename in filenames:
        module = importlib.import_module('datasets.%s' % filename[:-3])
        clsmembers = inspect.getmembers(module, inspect.isclass)
        for clsmember in clsmembers:
            is_dataset = False
            for base_cls in inspect.getmro(clsmember[1]):
                if base_cls is BaseDataset or isinstance(base_cls, BaseDataset):
                    is_dataset = True
                    break
            if is_dataset:
                type2dataset[clsmember[0]] = clsmember[1]


    datasets = []
    s_types = [split_types] if type(split_types) == str else split_types

    for split_type in s_types:
        dataset = type2dataset[cfg.dataset_type](split_type, cfg, log)
        datasets.append(dataset.get_dataset())

    return datasets[0] if type(split_types) == str else datasets

    