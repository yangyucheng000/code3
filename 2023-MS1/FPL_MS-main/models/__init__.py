# 1

import os
import importlib

def get_all_models():
    """
    return : all the models in the models directory 
    (excluding those that have "__" and those that are not Python files)
    (without the .py extension)
    """
    return [model.split('.')[0] for model in os.listdir('models')
            if not model.find('__') > -1 and 'py' in model]

names = {}
for model in get_all_models():
    mod = importlib.import_module('models.' + model)
    class_name = {x.lower():x for x in mod.__dir__()}[model.replace('_', '')] 
    # print(class_name)
    names[model] = getattr(mod, class_name)

def get_model(nets_list,args, transform):
    
    return names[args.model](nets_list,args,transform)
