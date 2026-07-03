from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Set

import torch
import torch.nn as nn
from .utils_insym import (
    AtomFeaturePlus,
    EdgeFeaturePlus,
    SE3InvariantKernel,
    _build_attn_mask,
    MovementPredictionHead,
    MaskLMHead,
    build_padding_only_attn_mask,
    ChainTokenFeaturePlus,
    ChainEdgeFeaturePlus
)
from .encoder import EncoderBlock


class MultiMolModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args  # keep reference �C other modules rely on it

        # --------------------------------------------------------------
        # Stage-1: Embeddings & basic feature extractors
        # --------------------------------------------------------------
        d_model = getattr(args, "encoder_embed_dim", 768)
        self.embed_tokens = nn.Embedding(
            128, d_model, getattr(args, "padding_idx", 0)
        )

        self.atom_feature = AtomFeaturePlus(
            num_atom=getattr(args, "num_atom", 512),
            num_degree=getattr(args, "num_degree", 128),
            hidden_dim=d_model,
            wo_node=getattr(args, "wo_node", False),
            wo_atom_feat=getattr(args, "wo_atom_feat", None),
            num_glob_feat=getattr(args, "num_chain_glob_feat", 3),
            
        )
        self.edge_feature = EdgeFeaturePlus(
            pair_dim=getattr(args, "pair_embed_dim", 512),
            num_edge=getattr(args, "num_edge", 64),
            num_spatial=getattr(args, "num_spatial", 512),
            wo_spd=getattr(args, "wo_spd", False),
            wo_edge=getattr(args, "wo_edge", False),            
        )

        # --------------------------------------------------------------
        # Stage-1: Encoder (token-pair joint reasoning)
        # --------------------------------------------------------------
        self.encoder = EncoderBlock(
            num_encoder_layers=getattr(args, "encoder_layers", 12),
            embedding_dim=d_model,
            pair_dim=getattr(args, "pair_embed_dim", 512),
            pair_hidden_dim=getattr(args, "pair_hidden_dim", 64),
            ffn_embedding_dim=getattr(args, "encoder_ffn_embed_dim", 3072),
            num_attention_heads=getattr(args, "encoder_attention_heads", 48),
            dropout=getattr(args, "dropout", 0.1),
            attention_dropout=getattr(args, "attention_dropout", 0.1),
            activation_dropout=getattr(args, "activation_dropout", 0.1),
            activation_fn=getattr(args, "activation_fn", "gelu"),
            droppath_prob=getattr(args, "droppath_prob", 0.0),
            pair_dropout=getattr(args, "pair_dropout", 0.25),
            wo_triopm=getattr(args, "wo_triopm", False),
            wo_pair=getattr(args, "wo_pair", False),
        )

        # 3-D geometric bias (shared across layers)
        self.se3_invariant_kernel = SE3InvariantKernel(
            pair_dim=getattr(args, "pair_embed_dim", 512),
            num_pair=128*128,
            num_kernel=getattr(args, "num_kernel", 128),
            std_width=getattr(args, "gaussian_std_width", 1.0),
            start=getattr(args, "gaussian_mean_start", 0.0),
            stop=getattr(args, "gaussian_mean_stop", 9.0),
        )
        
        
        # --------------------------------------------------------------
        # Stage-2: chain-level modules
        # --------------------------------------------------------------    
        
        self.chain_token_feature = ChainTokenFeaturePlus(
            embed_dim=d_model,
            num_glob_feat=args.num_chain_glob_feat,    # e.g. temperature, coexistence
            num_block_feat=args.num_chain_block_feat,  # e.g. volume fraction
            max_chain_tokens=getattr(args, "max_chain_tokens", 1024),
            enable_anchor_symmetry_break=(not getattr(args, "disable_anchor_symmetry_break", False)),
        )
        
        # ---- chain-level edge / pair bias ----
        self.chain_edge_feature = ChainEdgeFeaturePlus(
            pair_dim=args.chain_pair_dim,       
            max_chain_dist=args.max_chain_dist,  
        )

    
        self.chain_encoder = EncoderBlock(
            num_encoder_layers=getattr(args, "chain_encoder_layers", 6),
            embedding_dim=d_model,
            pair_dim=args.chain_pair_dim,
            pair_hidden_dim=getattr(args, "chain_pair_hidden_dim", 64),
            ffn_embedding_dim=getattr(args, "chain_ffn_embed_dim", 4*d_model),
            num_attention_heads=args.chain_attention_heads,
            dropout=getattr(args, "dropout", 0.1),
            attention_dropout=getattr(args, "attention_dropout", 0.1),
            wo_triopm=True,
            wo_pair=False,
        )

        # Stage-2-only segment encoder (topology only; no monomer chemistry)
        self.stage2_only_block_embed = nn.Embedding(
            getattr(args, "num_block_types", 8), d_model
        )
        self.stage2_only_smiles_embed = nn.Embedding(
            getattr(args, "num_seg_smiles_types", 20000), d_model
        )
        self.stage2_only_seg_pos = nn.Embedding(
            getattr(args, "max_segments", 16), d_model
        )
        self.stage2_only_dop_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Stage-2-only: fixed chemistry via fingerprints (few trainable params)
        fp_nbits = int(getattr(args, "fp_nbits", 2048))
        self.fp_proj = nn.Sequential(
            nn.Linear(fp_nbits, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # fp_table[0] reserved for PAD/unknown smiles-id -> all zeros.
        self.register_buffer("fp_table", torch.zeros(1, fp_nbits), persistent=True)

        # --------------------------------------------------------------
        # Pretrain Heads
        # --------------------------------------------------------------
        self.lm_head = MaskLMHead(
            embed_dim=getattr(args, "encoder_embed_dim", 768),
            output_dim=128,
            weight=self.embed_tokens.weight,
        )
        
        self.movement_pred_head = MovementPredictionHead(
            getattr(args, "encoder_embed_dim", 768),
            getattr(args, "pair_embed_dim", 512),
            getattr(args, "encoder_attention_heads", 48),
        )

        # --------------------------------------------------------------
        # Downstream NN (dimension based on args)
        # --------------------------------------------------------------
        self.reg_head = nn.Sequential(
        nn.Linear(getattr(args, "encoder_embed_dim", 768), 128),
        nn.GELU(),
        nn.Linear(128, 32),
        nn.GELU(),
        nn.Linear(32, getattr(args, "num_tasks", 1))
        )

        self.equalize_active_params = bool(getattr(args, "equalize_active_params", False))
        self.reg_head_equalizer = None
        self.reg_head_equalizer_shallow = None
        self.reg_head_equalizer_reserve = None
        if self.equalize_active_params:
            main_task = getattr(args, "main_task", "finetune")
            if main_task == "finetune":
                # Match FRPN active params by adding stage-2 equivalent capacity.
                extra_params = (
                    self._count_module_params(self.chain_token_feature)
                    + self._count_module_params(self.chain_edge_feature)
                    + self._count_module_params(self.chain_encoder)
                )
                self._build_finetune_active_equalizer(
                    extra_params=extra_params,
                    in_dim=getattr(args, "encoder_embed_dim", 768),
                    out_dim=getattr(args, "num_tasks", 1),
                )
            elif main_task == "stage2_only":
                # Match FRPN active params by compensating omitted stage-1 compute.
                stage1_params = (
                    self._count_module_params(self.embed_tokens)
                    + self._count_module_params(self.atom_feature)
                    + self._count_module_params(self.edge_feature)
                    + self._count_module_params(self.encoder)
                    + self._count_module_params(self.se3_invariant_kernel)
                )
                stage2_repr_mode = getattr(args, "stage2_only_repr", "full")
                if stage2_repr_mode == "smiles_only":
                    stage2_only_private = self._count_module_params(self.stage2_only_smiles_embed)
                elif stage2_repr_mode == "fp_morgan2048":
                    stage2_only_private = self._count_module_params(self.fp_proj)
                else:
                    stage2_only_private = (
                        self._count_module_params(self.stage2_only_block_embed)
                        + self._count_module_params(self.stage2_only_smiles_embed)
                        + self._count_module_params(self.stage2_only_seg_pos)
                        + self._count_module_params(self.stage2_only_dop_proj)
                    )
                extra_params = max(0, stage1_params - stage2_only_private)
                self._build_finetune_active_equalizer(
                    extra_params=extra_params,
                    in_dim=getattr(args, "encoder_embed_dim", 768),
                    out_dim=getattr(args, "num_tasks", 1),
                )

    @staticmethod
    def _count_module_params(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())

    @staticmethod
    def _deep_equalizer_param_count(width: int, in_dim: int, out_dim: int) -> int:
        if width <= 0:
            return 0
        return (
            in_dim * width + width
            + width * width + width
            + width * out_dim + out_dim
        )

    @staticmethod
    def _shallow_equalizer_param_count(width: int, in_dim: int, out_dim: int) -> int:
        if width <= 0:
            return 0
        return in_dim * width + width + width * out_dim + out_dim

    def _build_finetune_active_equalizer(self, extra_params: int, in_dim: int, out_dim: int):
        if extra_params <= 0:
            return

        c = in_dim + out_dim + 2
        discr = c * c + 4 * (extra_params - out_dim)
        width = int(max(0, (math.sqrt(max(discr, 0)) - c) / 2))

        while width > 0 and self._deep_equalizer_param_count(width, in_dim, out_dim) > extra_params:
            width -= 1

        deep_params = self._deep_equalizer_param_count(width, in_dim, out_dim)
        remain = extra_params - deep_params

        shallow_unit = in_dim + out_dim + 1
        shallow_width = 0
        shallow_params = 0
        if remain > out_dim:
            shallow_width = max(0, (remain - out_dim) // shallow_unit)
            shallow_params = self._shallow_equalizer_param_count(shallow_width, in_dim, out_dim)
            while shallow_width > 0 and shallow_params > remain:
                shallow_width -= 1
                shallow_params = self._shallow_equalizer_param_count(shallow_width, in_dim, out_dim)

        reserve_params = remain - shallow_params

        if width > 0:
            self.reg_head_equalizer = nn.Sequential(
                nn.Linear(in_dim, width),
                nn.GELU(),
                nn.Linear(width, width),
                nn.GELU(),
                nn.Linear(width, out_dim),
            )

        if shallow_width > 0:
            self.reg_head_equalizer_shallow = nn.Sequential(
                nn.Linear(in_dim, shallow_width),
                nn.GELU(),
                nn.Linear(shallow_width, out_dim),
            )

        if reserve_params > 0:
            self.reg_head_equalizer_reserve = nn.Parameter(torch.zeros(reserve_params))

    def _apply_reg_head_equalizer(self, pred: torch.Tensor, rep: torch.Tensor) -> torch.Tensor:
        if self.reg_head_equalizer is not None:
            pred = pred + self.reg_head_equalizer(rep)
        if self.reg_head_equalizer_shallow is not None:
            pred = pred + self.reg_head_equalizer_shallow(rep)
        if self.reg_head_equalizer_reserve is not None:
            pred = pred + self.reg_head_equalizer_reserve.mean().expand_as(pred)
        return pred

    @staticmethod
    def _iter_unique_smiles_from_csv(csv_path: Path, smiles_cols: list[str]) -> Set[str]:
        unique_smiles: Set[str] = set()
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for c in smiles_cols:
                    smi = row.get(c, "")
                    if smi is None:
                        continue
                    smi = str(smi).strip()
                    if not smi or smi.lower() == "nan":
                        continue
                    unique_smiles.add(smi)
        return unique_smiles

    def _build_fp_table_from_csv(self) -> torch.Tensor:
        """Build Morgan fingerprint table aligned with preprocessing_polymer.py smiles_vocab.

        Returns:
            fp_table: FloatTensor [V, nBits], where fp_table[0] is all-zero.
        """
        fp_csv = Path(getattr(self.args, "fp_csv", "data/raw/bcdb.csv"))
        smiles_prefix = str(getattr(self.args, "fp_smiles_prefix", "SMILES"))
        radius = int(getattr(self.args, "fp_radius", 2))
        nbits = int(getattr(self.args, "fp_nbits", 2048))
        replace_star_fallback = bool(getattr(self.args, "fp_replace_star_fallback", True))

        if not fp_csv.exists():
            raise FileNotFoundError(f"fp_csv not found: {fp_csv}")

        with fp_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

        pat = re.compile(rf"{re.escape(smiles_prefix)}\d+")
        smiles_cols = [c for c in fieldnames if pat.fullmatch(c or "")]
        smiles_cols.sort(key=lambda x: int(x[len(smiles_prefix):]))
        if not smiles_cols:
            raise ValueError(f"No SMILES columns found in {fp_csv} with prefix={smiles_prefix!r}")

        unique_smiles = self._iter_unique_smiles_from_csv(fp_csv, smiles_cols)
        smiles_list = sorted(unique_smiles)

        # vocabulary: smi -> idx+1 (idx=0 reserved for PAD/unknown), consistent with preprocessing_polymer.py
        smiles_vocab = {smi: i + 1 for i, smi in enumerate(smiles_list)}
        vocab_size = len(smiles_vocab) + 1

        try:
            from rdkit import Chem, DataStructs  # type: ignore
            from rdkit.Chem import AllChem  # type: ignore
            import numpy as np  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "stage2_only_repr=fp_morgan2048 requires RDKit (+ numpy) in the runtime environment."
            ) from e

        fp_table = torch.zeros(vocab_size, nbits, dtype=torch.float32)
        failed: list[str] = []

        for smi, idx in smiles_vocab.items():
            mol = Chem.MolFromSmiles(smi)
            if mol is None and replace_star_fallback and "*" in smi:
                mol = Chem.MolFromSmiles(smi.replace("*", "C"))

            if mol is None:
                failed.append(smi)
                continue

            bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            arr = np.zeros((nbits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(bitvect, arr)
            fp_table[idx] = torch.from_numpy(arr.astype(np.float32))

        rank = int(getattr(self.args, "rank", 0))
        if rank == 0:
            print(
                f"[FP] Built Morgan fp_table from {fp_csv} cols={len(smiles_cols)} "
                f"unique_smiles={len(smiles_vocab)} vocab_size={vocab_size} "
                f"radius={radius} nbits={nbits} failed={len(failed)}"
            )
            if failed:
                preview = ", ".join(failed[:8])
                more = "" if len(failed) <= 8 else f" ... (+{len(failed) - 8} more)"
                print(f"[FP] RDKit parse failed SMILES (fp=0): {preview}{more}")

        # Align to args.num_seg_smiles_types if provided.
        expected = int(getattr(self.args, "num_seg_smiles_types", vocab_size))
        if expected != vocab_size:
            rank = int(getattr(self.args, "rank", 0))
            if rank == 0:
                print(f"[FP][WARN] fp_table rows={vocab_size} != num_seg_smiles_types={expected}; padding/truncating.")
            if expected > vocab_size:
                pad = torch.zeros(expected - vocab_size, nbits, dtype=fp_table.dtype)
                fp_table = torch.cat([fp_table, pad], dim=0)
            else:
                fp_table = fp_table[:expected]

        return fp_table

    def _ensure_fp_table(self, device: torch.device) -> None:
        # If loaded from checkpoint or already built in this process, only ensure device.
        if (
            hasattr(self, "fp_table")
            and torch.is_tensor(self.fp_table)
            and self.fp_table.dim() == 2
            and self.fp_table.size(0) > 1
        ):
            if self.fp_table.device != device:
                self.fp_table = self.fp_table.to(device)
            return

        fp_table = self._build_fp_table_from_csv()
        if fp_table.device != device:
            fp_table = fp_table.to(device)
        self.fp_table = fp_table

    def _build_stage2_only_seg_repr(self, batch):
        seg_dop = batch["seg_dop"].float()  # [B, S]
        seg_block_id = batch["seg_block_id"].long()  # [B, S]
        seg_valid = batch["seg_valid_mask"].float()  # [B, S]
        seg_smiles_id = batch.get("seg_smiles_id", None)
        if seg_smiles_id is None:
            seg_smiles_id = seg_block_id
        seg_smiles_id = seg_smiles_id.long()

        B, S = seg_dop.shape
        device = seg_dop.device
        pos_idx = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

        stage2_repr_mode = getattr(self.args, "stage2_only_repr", "full")

        if stage2_repr_mode == "fp_morgan2048":
            self._ensure_fp_table(device=device)
            idx = seg_smiles_id.clamp_min(0)
            if idx.max().item() >= self.fp_table.size(0):
                idx = idx.clamp_max(self.fp_table.size(0) - 1)
            fp_vec = self.fp_table[idx]  # [B, S, nbits]
            seg_repr = self.fp_proj(fp_vec) * seg_valid.unsqueeze(-1)
            return seg_repr

        smiles_vec = self.stage2_only_smiles_embed(seg_smiles_id.clamp_min(0))
        if stage2_repr_mode == "smiles_only":
            seg_repr = smiles_vec * seg_valid.unsqueeze(-1)
            return seg_repr

        dop_norm = seg_dop / (seg_dop.sum(dim=1, keepdim=True) + 1e-6)
        dop_vec = self.stage2_only_dop_proj(dop_norm.unsqueeze(-1))
        block_vec = self.stage2_only_block_embed(seg_block_id.clamp_min(0))
        pos_vec = self.stage2_only_seg_pos(pos_idx)
        seg_repr = (dop_vec + block_vec + smiles_vec + pos_vec) * seg_valid.unsqueeze(-1)
        return seg_repr

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, batch):
        """Args:
            batch (dict): see dataloader for full specification
        Returns:
            node_repr (Tensor): [B, T, D]
            pair_repr (Tensor): [B, T, T, D_p]
        """

        # ---------- Stage-2-only forward: topology path without monomer chemistry ----------
        if self.args.main_task == "stage2_only":
            seg_repr = self._build_stage2_only_seg_repr(batch)
            chain_tokens, chain_attn_mask = self.chain_token_feature(
                seg_repr=seg_repr,
                seg_dop=batch["seg_dop"],
                seg_block_id=batch["seg_block_id"],
                glob_feat=batch["chain_glob_feat"],
                glob_feat_mask=batch["chain_glob_mask"],
                block_feat=batch["block_feat"],
                block_feat_mask=batch["block_feat_mask"],
                num_heads=self.args.chain_attention_heads,
            )

            B, T, _ = chain_tokens.shape
            graph_attn_bias = chain_tokens.new_zeros(B, T, T, self.args.chain_pair_dim)
            graph_attn_bias = self.chain_edge_feature(batch, graph_attn_bias)

            chain_repr, _ = self.chain_encoder(
                chain_tokens,
                graph_attn_bias,
                atom_mask=None,
                pair_mask=None,
                attn_mask=chain_attn_mask,
            )

            chain_cls = chain_repr[:, 0, :]
            pred = self.reg_head(chain_cls)
            pred = self._apply_reg_head_equalizer(pred, chain_cls)
            return pred

        # ---------- unpack -------------------------------------------------
        atom_mask = batch["atom_mask"]  # [B, N]
        seg_id = batch["segment_id"]  # [B, N]
        pos = batch["src_pos"]  # [B, N, 3]
        pair_type = batch["pair_type"]  # [B, N, N]
        token_ids = batch["src_token"]  # [B, N]

        B, N = atom_mask.shape
        n_seg = batch["seg_feat"].size(1)
        special_len = 2 + n_seg  # CLS + GLOB + SEG
        total_len = special_len + N

        # ---------- node embedding -----------------------------------------
        token_feat = self.embed_tokens(token_ids)
        node_repr = self.atom_feature(batch, token_feat)  # [B, T, D]
        if node_repr.dtype == torch.float32 and torch.is_autocast_enabled():
            node_repr = node_repr.to(torch.get_autocast_gpu_dtype())

        # ---------- masks ---------------------------------------------------
        attn_mask = _build_attn_mask(
            batch["base_mask"],
            seg_id,
            atom_mask,
            batch["seg_valid_mask"],
            glob_valid=batch["glob_valid_mask"],
            num_heads=self.args.encoder_attention_heads,
        )
        
        if not self.args.wo_pair:
            attn_mask = _build_attn_mask(
                batch["base_mask"],
                seg_id,
                atom_mask,
                batch["seg_valid_mask"],
                glob_valid=batch["glob_valid_mask"],
                num_heads=self.args.encoder_attention_heads,
            )
        else:
            attn_mask = build_padding_only_attn_mask(
            atom_mask,
            batch["seg_valid_mask"],
            glob_valid=batch["glob_valid_mask"],
            num_heads=self.args.encoder_attention_heads,
        )

        # ---------- pair-wise bias -----------------------------------------
        pair_repr = node_repr.new_zeros(
            B, total_len, total_len, self.args.pair_embed_dim, dtype=node_repr.dtype
        )
        pair_repr = self.edge_feature(batch, pair_repr)

        # 3-D SE(3) bias �C inside each segment only
        if self.args.wo_geom_3d:
            pass
        else:
            delta_pos = pos.unsqueeze(2) - pos.unsqueeze(1)  # [B, N, N, 3]
            dist = delta_pos.norm(dim=-1)  # [B, N, N]
            geom_bias = self.se3_invariant_kernel(dist.detach(), pair_type.long())
            same_seg = (seg_id.unsqueeze(-1) == seg_id.unsqueeze(-2)).unsqueeze(-1)
            geom_bias.masked_fill_(~same_seg, 0.0)
            pair_repr[:, special_len:, special_len:, :].add_(geom_bias)
                
        # ---------- padding masks ------------------------------------------
        cls_mask  = atom_mask.new_ones(B, 1, dtype=torch.bool)
        glob_mask = batch["glob_valid_mask"].bool()
        seg_mask  = batch["seg_valid_mask"].bool()
        node_mask = torch.cat([cls_mask, glob_mask, seg_mask, atom_mask.bool()], dim=1)
        pair_mask = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)

        # ---------- encoder --------------------------------------------------
        node_repr, pair_repr = self.encoder(
            node_repr,
            pair_repr,
            atom_mask=node_mask,
            pair_mask=pair_mask,
            attn_mask=attn_mask,
        )
        
        # ---------- Downstream REG head --------------------------------------------------
        if self.args.main_task == "finetune":
            mol_rep = node_repr[:, 0, :]
            pred_val = self.reg_head(mol_rep)
            pred_val = self._apply_reg_head_equalizer(pred_val, mol_rep)
            return pred_val
            
        # ---------- Downstream chain-level forward --------------------------------------------------    
        elif self.args.main_task == "chain":

            n_seg = batch["seg_feat"].size(1)
            seg_start = 2                    # CLS + GLOB
            seg_end = 2 + n_seg
        
            seg_repr = node_repr[:, seg_start:seg_end, :]   # [B, S, D]

            chain_tokens, chain_attn_mask = self.chain_token_feature(
                seg_repr=seg_repr,
                seg_dop=batch["seg_dop"],
                seg_block_id=batch["seg_block_id"],
                glob_feat=batch["chain_glob_feat"],
                glob_feat_mask=batch["chain_glob_mask"],
                block_feat=batch["block_feat"],              # [B, 2, F]
                block_feat_mask=batch["block_feat_mask"],    # [B, 2, F]
                num_heads=self.args.chain_attention_heads,
            )
            # chain_tokens: [B, T, D]
            # chain_attn_mask: [B, H, T, T]
        
            B, T, _ = chain_tokens.shape
            graph_attn_bias = chain_tokens.new_zeros(
                B, T, T, self.args.chain_pair_dim
            )
        
            graph_attn_bias = self.chain_edge_feature(
                batch,
                graph_attn_bias
            )
            # shape: [B, T, T, pair_dim]
        

            chain_repr, _ = self.chain_encoder(
                chain_tokens,
                graph_attn_bias,
                atom_mask=None,          
                pair_mask=None,        
                attn_mask=chain_attn_mask,
            )
        
            chain_cls = chain_repr[:, 0, :]    # CLS token
            pred = self.reg_head(chain_cls)
        
            return pred

            
        # ---------- Pretrain: Masked-Atom Prediction & Coordinate Reconstruction ---------------
        elif self.args.main_task == "pretrain":
        
            # ---- 3.1 Masked Atom Prediction ----
            atom_repr = node_repr[:, special_len:, :]
            logits = self.lm_head(atom_repr)       # [B,N,V] �� predicted token logits for each atom        
            # ---- 3.2 Coordinate Reconstruction ----
            delta_pos = pos.unsqueeze(2) - pos.unsqueeze(1)  # [B, N, N, 3]
            atom_repr = node_repr[:, special_len:, :]        # [B, N, d]
            pair_repr_a = pair_repr[:, special_len:, special_len:, :]  # [B, N, N, d_p]
            attn_mask_a = attn_mask[:, :, special_len:, special_len:]  # [B, H, N, N]
            delta = self.movement_pred_head(
                atom_repr,                         # [B,N,d] �� node representations
                pair_repr_a,
                attn_mask_a,                         # [B,N,N,p] �� pairwise representations
                delta_pos.detach(),                   # Noisy input coordinates (��pos will be used inside the head)
            )                                      # Returns ��xyz, shape [B,N,3]        
            pred_pos = pos + delta    # [B,N,3] �� recovered positions        
            # ---- 3.3 Distance Prediction ----
            pred_dist = torch.cdist(pred_pos, pred_pos)        
            return logits, pred_pos, pred_dist               
