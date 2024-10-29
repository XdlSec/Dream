import os
import numpy as np
import tensorflow as tf


def set_gpu_growing():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    return len(gpus)


def set_gpu_id(id):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(id)


def lst_to_str(lst):
    if len(lst) == 0:
        return ""
    elif len(lst) == 1:
        return f'[{str(lst[0])}]'
    elif len(lst) == 2:
        return f"[{lst[0]}] and [{lst[1]}]"
    else:
        return ", ".join(f'[{str(x)}]' for x in lst[:-1]) + f", and [{lst[-1]}]"
    

def set_random_seed(seed=0):
    # Set random seeds
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed) # after TF 2.7
    