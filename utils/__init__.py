import os
from backalla_utils.misc import recursive_ls
import random


def get_label_from_path(path):
    filename = os.path.basename(path)
    name = filename[:4]
    return name

def get_dataset(data_path = "./data/"):
    all_images = recursive_ls(data_path,filter="*.png")    
    name_path_dict = {}
    for image_path in all_images:
        name = get_label_from_path(image_path)
        if name in name_path_dict:
            name_path_dict[name].append(image_path)
        else:
            name_path_dict[name] = [image_path]

    name_path_dict_multiple = {name:name_path_dict[name] for name in name_path_dict if len(name_path_dict[name])>1}
    name_data_dict = {}
    for name in name_path_dict_multiple:
        paths_list = name_path_dict_multiple[name]
        random.shuffle(paths_list)
        train_paths = paths_list[:len(paths_list)//2]
        valid_paths = paths_list[len(paths_list)//2:]
        name_data_dict[name] = {"train":train_paths,"valid":valid_paths}
    
    return name_data_dict


