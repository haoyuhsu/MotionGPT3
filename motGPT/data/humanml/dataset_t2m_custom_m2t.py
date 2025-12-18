import rich
import random
import pickle
import os
import numpy as np
import codecs as cs
from torch.utils import data
from os.path import join as pjoin
from rich.progress import track
import json
import spacy
from tqdm import tqdm


class Text2MotionDatasetM2T(data.Dataset):
    def __init__(
        self,
        data_root, 
        split,
        mean,
        std,
        max_motion_length=196,
        min_motion_length=20,
        unit_length=4,
        fps=20,
        custom_dataset="humanml",
        **kwargs,
    ):
        self.data_root = '/work/hdd/bfyo/hhsu2/imu-humans/final_data_per_sequence'
        self.split = split
        self.mean = mean
        self.std = std
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length
        self.fps = fps
        self.custom_dataset = custom_dataset

        print('<----- Dataset settings: ----->')
        print(f'-->  max_motion_length: {self.max_motion_length}')
        print(f'-->  min_motion_length: {self.min_motion_length}')
        print(f'-->  unit_length: {self.unit_length}')
        print(f'-->  fps: {self.fps}')
        print(f'-->  custom_dataset: {self.custom_dataset}')

        instructions = '/projects/benk/hhsu2/imu-humans/related_works/MotionGPT3/prepare/instructions/template_instructions.json'

        self.motion_dir = pjoin(self.data_root, 'motion_data', split)
        self.splits_dir = pjoin(self.data_root, 'splits')

        # Get all ids in specific split
        t2m_txt = pjoin(self.splits_dir, f't2m_{split}.txt')
        t2m_f_list = [line.strip() for line in open(t2m_txt).readlines()]
        id_list = sorted(t2m_f_list)

        if custom_dataset == "humanml":
            target_datasets = [
                'MotionUnion/humanml',
                'Mirror_MotionUnion/humanml',
            ]
        elif custom_dataset == "lingo":
            target_datasets = [
                'LINGO',
            ]
        else:
            raise ValueError(f'Unknown custom_dataset: {custom_dataset}')

        self.id_list = [
            id.replace('/', '_') for id in id_list
            if any(id.startswith(ds_name) for ds_name in target_datasets)
        ]

        # TODO: fast debugging
        # self.id_list = self.id_list[:100]

        new_name_list = []
        data_dict = {}

        for i, name in tqdm(enumerate(self.id_list), desc=f'Loading {split} data'):

            with open(os.path.join(self.motion_dir, f'{name}.pkl'), 'rb') as f:
                data = pickle.load(f)

            motion = data['motion_263']
            texts = data['texts']

            if len(motion) < self.min_motion_length:  # skip short motions
                continue

            if len(motion) >= self.max_motion_length:  # truncate long motions
                motion = motion[:self.max_motion_length]

            data_dict[name] = {
                'motion': motion,
                'text': texts
            }
            new_name_list.append(name)

        self.data_dict = data_dict
        self.name_list = new_name_list
        self.instructions = json.load(open(instructions, 'r'))
        self.tasks = []
        # for task in self.instructions.keys():
        #     for subtask in self.instructions[task].keys():
        #         self.tasks.append(self.instructions[task][subtask])
        self.tasks.append(self.instructions["Motion-to-Text"]["caption"])   # only m2t task
        
        print(f'dataset {split} loaded, {len(self.data_dict)}')

    def __len__(self):
        return len(self.name_list) * len(self.tasks)

    def __getitem__(self, item):

        fname = self.name_list[item]
        data = self.data_dict[fname]

        motion = data['motion']  # Shape: [seq_len, feature_dim]
        text_list = data['text']  # List of text descriptions

        # Random caption selection
        caption = np.random.choice(text_list)
        
        all_captions = [str(text) for text in text_list]

        # Random crop
        coin = np.random.choice([False, False, True])
        m_length = motion.shape[0]
        if coin:
            # drop one token at the head or tail
            coin2 = np.random.choice([True, False])
            if coin2:
                m_length = (m_length // self.unit_length - 1) * self.unit_length
            else:
                m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        # Z Normalization
        motion = (motion - self.mean) / self.std

        # Task format for m2t
        tasks = {
            'class': 'm2t',
            'input': ['Describe the motion represented by <Motion_Placeholder> using plain English.'],
            'output': ['']
        }

        return caption, None, None, motion, m_length, None, None, None, None, all_captions, tasks, fname
