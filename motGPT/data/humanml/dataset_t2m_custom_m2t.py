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
        if custom_dataset == "humanml" or custom_dataset == "lingo":
            self.data_root = '/work/hdd/bfyo/hhsu2/imu-humans/final_data_per_sequence'
        elif custom_dataset == "parahome":
            self.data_root = '/scratch/bfyo/tcheng1/dataset_process/ParaHome'
        elif custom_dataset == "humoto":
            self.data_root = '/scratch/bfyo/tcheng1/dataset_process/humoto_data'
        else:
            raise ValueError(f'Unknown custom_dataset: {custom_dataset}')
        
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
        print(f'-->  data_root: {self.data_root}')

        instructions = '/projects/benk/hhsu2/imu-humans/related_works/MotionGPT3/prepare/instructions/template_instructions.json'

        self.motion_dir = pjoin(self.data_root, 'motion_data', split)

        if custom_dataset in ["humanml", "lingo"]:
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
            
            self.id_list = [
                id.replace('/', '_') for id in id_list
                if any(id.startswith(ds_name) for ds_name in target_datasets)
            ]
        elif custom_dataset in ["parahome", "humoto"]:
            # ParaHome and Humoto use all data in the split
            all_files = os.listdir(self.motion_dir)
            self.id_list = [f.replace('.pkl', '') for f in all_files if f.endswith('.pkl')]
            self.id_list = sorted(self.id_list)
        else:
            raise ValueError(f'Unknown custom_dataset: {custom_dataset}')

        # (TODO: for temp DEBUGGING)
        # self.id_list = self.id_list[:10]

        new_name_list = []
        data_dict = {}

        for i, name in tqdm(enumerate(self.id_list), desc=f'Loading {split} data'):

            with open(os.path.join(self.motion_dir, f'{name}.pkl'), 'rb') as f:
                data = pickle.load(f)

            motion = data['motion_263']

            if custom_dataset in ["humanml", "lingo", "humoto"]:
                texts = data['texts']

                if len(motion) < self.min_motion_length:  # skip short motions
                    continue

                if len(motion) >= self.max_motion_length:  # truncate long motions
                    motion = motion[:self.max_motion_length]

                if len(texts) == 0:  # skip no text motions
                    continue

                data_dict[name] = {
                    'motion': motion,
                    'text': texts
                }
                new_name_list.append(name)

            elif custom_dataset in ["parahome"]:
                # ParaHome: data['texts'] is a dict with frame ranges as keys
                #   Example: {'300 520': 'The person move cup from desk to table.', ...}
                texts_dict = data['texts']
                sorted_items = sorted(texts_dict.items(), key=lambda x: int(x[0].split()[0]))
                
                motion_length = motion.shape[0]
                num_chunks = (motion_length + self.max_motion_length - 1) // self.max_motion_length
                
                for chunk_idx in range(num_chunks):
                    chunk_start = chunk_idx * self.max_motion_length
                    chunk_end = min(chunk_start + self.max_motion_length, motion_length)

                    motion_chunk = motion[chunk_start:chunk_end]

                    # Skip if chunk is too short
                    if len(motion_chunk) < self.min_motion_length:
                        continue

                    agg_texts = []
                    for frame_range, text in sorted_items:
                        start_frame, end_frame = map(int, frame_range.split())

                        # calculate overlap region
                        overlap_start = max(start_frame, chunk_start)
                        overlap_end = min(end_frame, chunk_end)
                        overlap_length = max(0, overlap_end - overlap_start)

                        # Include text only if more than half of its range is within this chunk
                        annotation_length = end_frame - start_frame
                        if overlap_length >= annotation_length / 2:
                            agg_texts.append(text)

                    # Skip chunk if no texts exist within it
                    if not agg_texts:
                        continue

                    texts = [' '.join(agg_texts)]
                    chunk_name = f"{name}_{str(chunk_idx + 1).zfill(3)}"

                    data_dict[chunk_name] = {
                        'motion': motion_chunk,
                        'text': texts
                    }
                    new_name_list.append(chunk_name)
            else:
                raise ValueError(f'Unknown custom_dataset: {custom_dataset}')

        self.data_dict = data_dict
        self.name_list = new_name_list
        self.instructions = json.load(open(instructions, 'r'))
        self.tasks = []
        # for task in self.instructions.keys():
        #     for subtask in self.instructions[task].keys():
        #         self.tasks.append(self.instructions[task][subtask])
        self.tasks.append(self.instructions["Motion-to-Text"]["caption"])   # only m2t task
        
        print(f'dataset {split} loaded, {len(self.data_dict)} sequences')

    def __len__(self):
        return len(self.name_list) * len(self.tasks)

    def __getitem__(self, item):

        fname = self.name_list[item]
        data = self.data_dict[fname]

        motion = data['motion']  # Shape: [seq_len, feature_dim]
        text_list = data['text']  # List of text descriptions

        # Random caption selection
        caption = np.random.choice(text_list)

        # caption = text_list[0]  # always use the first caption for m2t (TODO: for temp DEBUGGING)
        
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
            'output': ['<Caption_Placeholder>']
        }

        return caption, None, None, motion, m_length, None, None, None, None, all_captions, tasks, fname
