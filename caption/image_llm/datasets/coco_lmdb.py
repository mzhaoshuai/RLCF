# coding=utf-8
import os
import lmdb
import json
import pickle
import torch
import numpy as np
from typing import Tuple
from transformers import AutoTokenizer


class COCOCLIPCapTrainDatasetLMDB(torch.utils.data.Dataset):

    def __init__(self, data_path: str,  prefix_length: int, config_dir: str = "gpt2",
                    normalize_prefix=False, use_image_embedding=True, annotations=None):
        """
        LAVIS train Format:
        [{"caption": "A woman eating fresh vegetables from a bowl.", "image": "val2014/COCO_val2014_000000328757.jpg", "image_id": "coco_328757"}]

        data_path: path to the lmdb file
        """
        self.tokenizer = AutoTokenizer.from_pretrained(config_dir)
        token_filename = os.path.basename(config_dir)

        self.data_path = data_path
        self.db_txn = None
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        self.use_image_embedding = use_image_embedding

        # load embeddings
        with open(annotations, 'r') as f:
            data = json.load(f)
        self.captions = [d["caption"] for d in data]
        self.image_ids = [d["image_id"] for d in data]
        self.image_subpath = [d["image"] for d in data]
        print("The dataset size is {}".format(len(self.captions)))

        token_filename = "{}_image_tokens_lmdb.pkl".format(token_filename) if use_image_embedding else \
                            "{}_text_tokens_lmdb.pkl".format(token_filename)
        llm_token_file = os.path.join(os.path.dirname(data_path), token_filename)

        if os.path.isfile(llm_token_file):
            # read token file
            print(f"[dataset] loading {token_filename}...")
            with open(llm_token_file, 'rb') as f:
                self.captions_tokens, self.max_seq_len = pickle.load(f)
            print(f"[dataset] loading {token_filename}... Done")
        else:
            # if not, create the token file
            print(f"[dataset] creating {token_filename}...")
            self.captions_tokens = []
            max_seq_len = 0
            for i, caption in enumerate(self.captions):
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64))
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])

            with open(llm_token_file, 'wb') as f:
                pickle.dump([self.captions_tokens, max_seq_len], f)
            print(f"[dataset] creating {token_filename}...Done")

        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens

        # mask is zero where we out of sequence
        mask = tokens.ge(0)
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask

        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        if self.db_txn is None:
            env = lmdb.open(self.data_path, subdir=os.path.isdir(self.data_path),
                                    readonly=True, lock=False,
                                    readahead=False, meminit=False,
                                    map_size=1<<41,)
            self.db_txn = env.begin(write=False)

        tokens, mask = self.pad_tokens(item)
        image_ids = self.image_ids[item]
        key = "image_" + image_ids if self.use_image_embedding else "text_" + image_ids
        buf = self.db_txn.get(key.encode())

        prefix = np.frombuffer(buf, dtype=np.float32).reshape(self.prefix_length, -1)
        prefix = torch.from_numpy(np.copy(prefix)).float()

        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)

        return {
                "tokens": tokens,
                "mask": mask,
                "prefix": prefix,
                }

    # def __getstate__(self):
    #     state = self.__dict__
    #     state["db_txn"] = None
    #     return state

    # def __setstate__(self, state):
    #     # https://github.com/pytorch/vision/issues/689
    #     self.__dict__ = state
    #     if self.data_path not in [None, 'None']:
    #         env = lmdb.open(self.data_path, subdir=os.path.isdir(self.data_path),
    #                                 readonly=True, lock=False,
    #                                 readahead=False, meminit=False,
    #                                 map_size=1<<41,)
    #         self.db_txn = env.begin(write=False)
