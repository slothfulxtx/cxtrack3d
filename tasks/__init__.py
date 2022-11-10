import os
import os.path as osp
import importlib
import inspect
from .base_task import BaseTask


def create_task(cfg, log):
    filenames = os.listdir(osp.dirname(__file__))
    filenames = filter(lambda x: x.endswith(
        '.py') and x != '__init__.py', filenames)
    type2task = dict()
    for filename in filenames:
        module = importlib.import_module('tasks.%s' % filename[:-3])
        clsmembers = inspect.getmembers(module, inspect.isclass)
        for clsmember in clsmembers:
            is_task = False
            for base_cls in inspect.getmro(clsmember[1]):
                if base_cls is BaseTask or isinstance(base_cls, BaseTask):
                    is_task = True
                    break
            if is_task:
                type2task[clsmember[0]] = clsmember[1]

    return type2task[cfg.task_type](cfg, log)
