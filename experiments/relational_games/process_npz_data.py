import matplotlib.pyplot as plt
import numpy as np
import torch


data_path = '../../data/relational_games/npz_files'

tasks = ('occurs', 'same', 'xoccurs', '1task_between', '1task_match_patt')
splits = ('hexos', 'pentos', 'stripes')

for task in tasks:
    for split in splits:
        fname = f'{data_path}/{task}_{split}.npz'
        with np.load(fname) as data:
            imgs = data['images'].astype('int32')
            labels = data['labels']
            spec = dict(shape=imgs.shape, labels=labels)

        with open(f'{data_path}/{task}_{split}.bin', "wb") as f:
            f.write(imgs.tobytes())

        torch.save(spec, f'{data_path}/{task}_{split}_spec.pt')