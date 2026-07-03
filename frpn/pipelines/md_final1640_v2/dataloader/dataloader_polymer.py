# -*- coding: utf-8 -*-
import pickle, glob, os, random, zlib

# Optional dependencies: only required for LMDB datasets.
try:
    import lmdb  # type: ignore
except Exception:  # pragma: no cover
    lmdb = None

try:
    import msgpack  # type: ignore
except Exception:  # pragma: no cover
    msgpack = None
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from typing import List, Dict, Any, Optional, Tuple
from argparse import Namespace

PAD_TOKEN_ID   = 0      # <- pad for src_token
PAD_FEAT_VAL   = 0      # <- pad for discrete features
PAD_SPD_VAL    = 511    # <- unreachable shortest-path  (already +1)
PAD_BIAS_VAL   = float('-inf')

# ===========================================
# 0. General Util: Deserialize LMDB to torch 
# ===========================================

def _to_tensor(obj):
    if isinstance(obj, dict):
        if obj.get("__tensor__", False):
            dtype_str = obj["dtype"]  # e.g., "torch.float32" or "float32"
            if isinstance(dtype_str, str) and dtype_str.startswith("torch."):
                dtype_str = dtype_str.split(".", 1)[1]  # e.g., "float32
            np_dtype = np.dtype(dtype_str)
            shape = obj["shape"]
            data_list = obj["data"]
            array = np.array(data_list, dtype=np_dtype).reshape(shape)
            return torch.from_numpy(array)
        else:
            return {k: _to_tensor(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [_to_tensor(v) for v in obj]

    else:
        return obj


def _deserialize(buf: bytes):
    if msgpack is None:
        raise RuntimeError("msgpack is required to read LMDB datasets; please install `msgpack`.")
    decompressed = zlib.decompress(buf)
    unpacked = msgpack.unpackb(decompressed, raw=False)
    return _to_tensor(unpacked)
    

def _open_lmdb_readonly(path: str):
    if lmdb is None:
        raise RuntimeError("lmdb is required to read LMDB datasets; please install `lmdb`.")
    is_dir = os.path.isdir(path)
    return lmdb.open(
        path,
        readonly=True,
        lock=False,
        subdir=is_dir,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    

def _lmdb_load_obj(buf: bytes):
    try:
        return pickle.loads(buf)
    except Exception:
        pass

    try:
        return _deserialize(buf)
    except Exception as e:
        raise RuntimeError("Unknown LMDB value format") from e

# ============================================================
# 1. Dataset Implementation ----------------------------------
# ============================================================

class _BaseDataset(Dataset):
    """
    Common logic for filtering by split and applying random 3D coordinate views.
    Shared by both Pickle and LMDB subclasses.
    """
    def __init__(self, fold: int, mode: str):
        self.fold    = fold
        self.mode    = mode
        self.augment = mode == "train"

    # ------- Interfaces that must be implemented by subclasses -------------
    def _get_raw_sample(self, idx: int) -> dict: ...
    def _build_id_list(self): ...

    # ------- Common logic -----------------------------------
    def __len__(self): 
        return len(self.ids)

    def __getitem__(self, idx):
        sample = self._get_raw_sample(self.ids[idx])
        s = sample.copy()
        coords = s.get("src_pos", None)
        if isinstance(coords, torch.Tensor) and coords.dim() == 3:
            k = coords.size(0)
            choice = random.randrange(k) if self.augment else 0
            s["src_pos"] = coords[choice]  # [N, 3]
        return s

# ---------------- PKL version -----------------------------
class PolymerPickleDataset(_BaseDataset):
    def __init__(self, pkl_path, fold: int, mode: str, eval_parent_only: bool = False):
        super().__init__(fold, mode)
        self.eval_parent_only = eval_parent_only
        header = pickle.load(open(pkl_path, "rb"))
        self.header = header
        self.samples = header["samples"]
        self.label_names = header.get("label_names", [])
        self.seg_smiles_vocab = header.get("seg_smiles_vocab", {})
        self.seg_smiles_vocab_size = header.get("seg_smiles_vocab_size", None)
        self.ids = self._build_id_list()

    def _build_id_list(self):
        if self.mode == "train":
            return [i for i, s in enumerate(self.samples) if s["split"] != f"fold{self.fold}"]
        elif self.mode == "val":
            ids = [i for i, s in enumerate(self.samples) if s["split"] == f"fold{self.fold}"]
            if self.eval_parent_only:
                ids = [i for i in ids if not self.samples[i].get("is_chain_aug", False)]
            return ids
        else:
            return list(range(len(self.samples)))

    def _get_raw_sample(self, idx):
        return self.samples[idx]


# ---------------- LMDB version ----------------------------
class PolymerLmDBDataset(_BaseDataset):
    def __init__(self, lmdb_path: str, fold: int, mode: str, eval_parent_only: bool = False):
        super().__init__(fold, mode)
        self.lmdb_path = lmdb_path
        self.eval_parent_only = eval_parent_only
        self.env = None  # type: Optional[lmdb.Environment]  # Delay opening to avoid dataloader fork issues

        # Read __len__
        with _open_lmdb_readonly(lmdb_path) as env:
            with env.begin() as txn:
                meta_buf = txn.get(b"__meta__")
                if meta_buf is not None:
                    meta = pickle.loads(meta_buf)
                    self._size = int(meta["num_samples"])
                    self.label_names = meta.get("label_names", [])
                else:
                    self._size = int(txn.get(b"__len__").decode())
                    self.label_names = []

        # Build split index
        if mode == "full":
            self.ids = list(range(self._size))
        else:
            self.ids = []
            with _open_lmdb_readonly(lmdb_path) as env:
                with env.begin() as txn:
                    for i in range(self._size):
                        buf = txn.get(f"sample_{i}".encode("utf-8"))
                        if buf is None:
                            buf = txn.get(str(i).encode("utf-8"))
                        if buf is None:
                            continue
        
                        sample = _lmdb_load_obj(bytes(buf))
                        sp = sample.get("split", "")
                        is_aug = sample.get("is_chain_aug", False)
        
                        if mode == "val" and self.eval_parent_only and is_aug:
                            continue
        
                        if (mode == "train" and sp != f"fold{fold}") or \
                           (mode == "val"   and sp == f"fold{fold}"):
                            self.ids.append(i)

    # Called separately by each worker to ensure env is local to their process
    def _require_env(self):
        if self.env is None:
            self.env = _open_lmdb_readonly(self.lmdb_path)

    def _get_raw_sample(self, idx: int) -> dict:
        self._require_env()
        with self.env.begin(buffers=True) as txn:
            buf = txn.get(f"sample_{idx}".encode("utf-8"))
            if buf is None:
                buf = txn.get(str(idx).encode("utf-8"))
            if buf is None:
                raise KeyError(f"LMDB missing key for idx={idx}")
        return _lmdb_load_obj(bytes(buf))

    def _build_id_list(self):  # Already handled in __init__
        raise NotImplementedError


# ---------------- Pretraining Logic Mixin ----------------
class PolymerPretrainDatasetMixin:
    MASK_IDX = 127
    MASK_TOKEN_PROB = 0.15
    NOISE_STD = 0.2
    DROP_PROB = 0.5

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        token = sample["src_token"].clone()
        N = token.size(0)
        mask_flag = torch.rand(N) < self.MASK_TOKEN_PROB
        target_token = token.clone()
        target_token[~mask_flag] = 0
        token[mask_flag] = self.MASK_IDX
        sample["src_token"] = token
        sample["target_token"] = target_token

        pos = sample["src_pos"].float()
        noise = torch.randn_like(pos) * self.NOISE_STD
        noised_pos = pos + noise

        c0 = noised_pos.mean(0, keepdim=True)
        c1 = pos.mean(0, keepdim=True)
        H = (noised_pos - c0).T @ (pos - c1)
        U, _, Vt = torch.linalg.svd(H)
        R = Vt.T @ U.T
        if torch.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        noised_pos = (noised_pos - c0) @ R + c1

        sample["target_pos"] = pos
        sample["src_pos"] = noised_pos
        sample["src_mask_cord"] = mask_flag.bool()

        if torch.rand(1) < self.DROP_PROB:
            sample["atom_feat"].fill_(1)
            sample["edge_feat"].fill_(1)
            sample["degree"].fill_(1)
            sample["shortest_path"].fill_(PAD_SPD_VAL)

        return sample


# Pickle source
class PolymerPretrainDataset(PolymerPretrainDatasetMixin, PolymerPickleDataset):
    pass

# LMDB source
class PolymerPretrainLmDBDataset(PolymerPretrainDatasetMixin, PolymerLmDBDataset):
    pass




# ---------- 2. Padding helpers ----------------------------------------
def pad_1d(samples: List[torch.Tensor], pad_len: int, pad_value=0):
    batch_size = len(samples)
    tensor = samples[0].new_full((batch_size, pad_len), pad_value)
    for i, x in enumerate(samples):
        tensor[i, : x.shape[0]] = x
    return tensor

def pad_1d_feat(samples: List[torch.Tensor], pad_len: int, pad_value=0):
    batch_size = len(samples)
    feat_size = samples[0].shape[-1]
    tensor = samples[0].new_full((batch_size, pad_len, feat_size), pad_value)
    for i, x in enumerate(samples):
        tensor[i, : x.shape[0]] = x
    return tensor

def pad_2d(samples: List[torch.Tensor], pad_len: int, pad_value=0):
    batch_size = len(samples)
    tensor = samples[0].new_full((batch_size, pad_len, pad_len), pad_value)
    for i, x in enumerate(samples):
        n, m = x.shape
        tensor[i, :n, :m] = x
    return tensor

def pad_2d_feat(samples: List[torch.Tensor], pad_len: int, pad_value=0):
    batch_size = len(samples)
    feat_size = samples[0].shape[-1]
    tensor = samples[0].new_full((batch_size, pad_len, pad_len, feat_size), pad_value)
    for i, x in enumerate(samples):
        n, m, _ = x.shape
        tensor[i, :n, :m] = x
    return tensor

def pad_base_mask(samples: List[torch.Tensor], pad_len: int):
    batch_size = len(samples)
    # note: official adds +1 inside
    tensor = samples[0].new_full((batch_size, pad_len, pad_len),
                                 float("-inf"))
    for i, b in enumerate(samples):
        n, m = b.shape
        # copy real block
        tensor[i, :n, :m] = b
        # allow padded *rows* to attend to real *cols*
        tensor[i, n:, :m] = 0
    return tensor


# ---------- 3. collate_fn ---------------------------------------------
def collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:

    # 0) Ensure each sample has atom_mask, 
    for s in samples:
        N = s["atom_feat"].shape[0]
        s["atom_mask"] = torch.ones(N, dtype=torch.long)

    # ---------- 1. Calculate padding length ----------
    special_T   = samples[0]['special_T']
    max_node    = max(s["atom_mask"].shape[0] for s in samples)
    token_dim   = (max_node + special_T + 3) // 4 * 4  # Multiple of 4: e.g. 12
    node_pad_len = token_dim - special_T               # e.g. 8

    # ---------- 2. Define padding functions ----------
    pad_fns = {
        "src_token"     : (pad_1d,       PAD_TOKEN_ID),
        "src_pos"       : (pad_1d_feat,  0.0),
        "atom_feat"     : (pad_1d_feat,  PAD_FEAT_VAL),
        "atom_mask"     : (pad_1d,       0),
        "edge_feat"     : (pad_2d_feat,  PAD_FEAT_VAL),
        "shortest_path" : (pad_2d,       PAD_FEAT_VAL),
        "degree"        : (pad_1d,       PAD_FEAT_VAL),
        #"pair_type"     : (pad_2d_feat,  PAD_FEAT_VAL),
        "segment_id"    : (pad_1d,       -1),
        "target_token"  : (pad_1d,       PAD_TOKEN_ID),
        "target_pos"    : (pad_1d_feat,  0.0),
        "src_mask_cord" : (pad_1d,       0),
    }

    batched: Dict[str, Any] = {}

    for key in samples[0].keys():
        vals = [s[key] for s in samples]

        # ---------- 3a. Keys that need padding based on atom count ----------
        if key in pad_fns:
            fn, pad_val = pad_fns[key]
            pad_len = node_pad_len
            batched[key] = fn(vals, pad_len, pad_val)
            
        elif key == "base_mask":                # The only exception
            fn = pad_base_mask
            pad_len = token_dim
            batched[key] = fn(vals, pad_len)
            
        elif key == "pair_type":
            if vals[0].dim() == 2:
                batched[key] = pad_2d(vals, node_pad_len, PAD_FEAT_VAL)
            elif vals[0].dim() == 3:
                batched[key] = pad_2d_feat(vals, node_pad_len, PAD_FEAT_VAL)
            else:
                raise ValueError(f"pair_type must be 2D or 3D, got {vals[0].shape}")
            continue

        # ---------- 3b. Fixed-length keys, directly stack ----------
        elif key in {
            "glob_feat", "glob_mask", "glob_valid_mask",
            "seg_feat", "seg_feat_mask", "seg_valid_mask",
        }:
            batched[key] = torch.stack(vals)

        # ---------- 3b.2 Stage-2 fixed-shape tensors ----------
        elif key in {
            "chain_glob_feat", "chain_glob_mask",
            "block_feat", "block_feat_mask",
            "seg_dop", "seg_block_id", "seg_smiles_id",
        }:
            batched[key] = torch.stack(vals)

        # ---------- 3c. Label ----------
        elif key == "label":
            batched[key] = torch.stack([
                torch.tensor(list(v.values()), dtype=torch.float32) for v in vals
            ])

        # ---------- 3d. Keep the rest as a list ----------
        else:
            batched[key] = vals

    return batched
    

# ------------4. Dataloader Builder----------------------------------------------------
def build_dataloader(
    pkl_path: str,
    args: Namespace,
    mode: str = "train",
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    is_train = mode == "train"
    is_pretrain = args.main_task == "pretrain"
    is_train_loader = (
        (mode == "train") or
        (is_pretrain and mode == "full")
    )
    is_eval_loader = not is_train_loader
    if args.main_task == "pretrain" and \
       os.path.abspath(pkl_path) == os.path.abspath(
           getattr(args, "pretrain_val_path", "")
       ):
        is_train_loader = False

    # Choose dataset class based on file type and task
    is_lmdb = os.path.isdir(pkl_path) or pkl_path.endswith(".lmdb")
    
    if is_lmdb:
        dataset_cls = PolymerPretrainLmDBDataset if is_pretrain else PolymerLmDBDataset
    else:
        dataset_cls = PolymerPretrainDataset if is_pretrain else PolymerPickleDataset


    eval_parent_only = getattr(args, "eval_parent_only", True) if mode == "val" and args.main_task != "pretrain" else False
    
    dataset = dataset_cls(
        pkl_path,
        fold=0 if is_pretrain else args.fold,
        mode="full" if is_pretrain else mode,
        eval_parent_only=eval_parent_only,
    )

    # Distributed sampler
    if getattr(args, "distributed", False):
        sampler = DistributedSampler(
            dataset,
            num_replicas=getattr(args, "world_size", 1),
            rank=getattr(args, "rank", 0),
            shuffle=is_train_loader,
            seed=getattr(args, "seed", 42),
            drop_last=is_train_loader,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = is_train_loader
    # Dataloader
    loader = DataLoader(
        dataset,
        batch_size=getattr(args, "batch_size", 32),
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=getattr(args, "num_workers", 4),
        pin_memory=getattr(args, "pin_memory", True),
        drop_last=is_train_loader,
        persistent_workers=getattr(args, "num_workers", 4) > 0,
    )

    return loader, sampler
