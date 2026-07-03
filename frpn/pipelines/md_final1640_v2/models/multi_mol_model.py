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
        self._fp_table_built = False

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
            wo_atom_feat=getattr(args, "wo_atom_feat", None)
            
        )

        # Keep pair representation width consistent across all stage-1 modules.
        pair_embed_dim = getattr(args, "pair_embed_dim", None)
        if pair_embed_dim is None:
            pair_embed_dim = getattr(args, "pair_dim", 512)

        self.edge_feature = EdgeFeaturePlus(
            pair_dim=int(pair_embed_dim),
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
            pair_dim=int(pair_embed_dim),
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
            pair_dim=int(pair_embed_dim),
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
            enable_topology_symmetry_break=not getattr(args, "disable_chain_symmetry_break", False),
        )

        # Stage-2-only path: initialize segment tokens from a SMILES-id dictionary embedding.
        # This keeps topology-level modeling while avoiding atom-level chemistry encoding.
        self.stage2_only_smiles_embed = nn.Embedding(
            getattr(args, "num_seg_smiles_types", 20000), d_model
        )

        fp_nbits = int(getattr(args, "fp_nbits", 2048))
        self.fp_proj = nn.Sequential(
            nn.Linear(fp_nbits, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.register_buffer("fp_table", torch.empty(0), persistent=True)
        
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
            int(pair_embed_dim),
            getattr(args, "encoder_attention_heads", 48),
        )

        # --------------------------------------------------------------
        # Downstream NN (dimension based on args)
        # --------------------------------------------------------------
        stage1_capacity_hidden = int(getattr(args, "stage1_capacity_mlp_hidden", 0) or 0)
        if stage1_capacity_hidden > 0:
            self.stage1_capacity_mlp = nn.Sequential(
                nn.Linear(getattr(args, "encoder_embed_dim", 768), stage1_capacity_hidden),
                nn.GELU(),
                nn.Linear(stage1_capacity_hidden, getattr(args, "encoder_embed_dim", 768)),
            )
        else:
            self.stage1_capacity_mlp = None

        self.reg_head = nn.Sequential(
        nn.Linear(getattr(args, "encoder_embed_dim", 768), 128),
        nn.GELU(),
        nn.Linear(128, 32),
        nn.GELU(),
        nn.Linear(32, getattr(args, "num_tasks", 1))
        )

    def _ensure_fp_table(self):
        if self._fp_table_built:
            return
        if getattr(self.args, "chain_only_repr", "smiles_embed") != "fp_morgan2048":
            return
        # If fp_table was restored from a checkpoint, reuse it without rebuilding.
        # This allows inference in environments without RDKit.
        if isinstance(getattr(self, "fp_table", None), torch.Tensor) and self.fp_table.numel() > 0:
            self._fp_table_built = True
            return

        import pickle
        import numpy as np

        pkl_path = getattr(self.args, "pkl_path", None)
        if not pkl_path:
            raise ValueError("fp_morgan2048 requires args.pkl_path to read seg_smiles_vocab")

        header = pickle.load(open(pkl_path, "rb"))
        seg_smiles_vocab = header.get("seg_smiles_vocab", None)
        seg_smiles_vocab_size = header.get("seg_smiles_vocab_size", None)
        if not isinstance(seg_smiles_vocab, dict) or not seg_smiles_vocab:
            raise ValueError("fp_morgan2048 requires pkl header['seg_smiles_vocab'] (string->id mapping)")

        if seg_smiles_vocab_size is None:
            seg_smiles_vocab_size = int(max(int(v) for v in seg_smiles_vocab.values()) + 1)
        seg_smiles_vocab_size = int(seg_smiles_vocab_size)

        # Build id->smiles table aligned with seg_smiles_id indices.
        id_to_smiles = [None] * seg_smiles_vocab_size
        for smi, idx in seg_smiles_vocab.items():
            try:
                iid = int(idx)
            except Exception:
                continue
            if 0 <= iid < seg_smiles_vocab_size:
                id_to_smiles[iid] = str(smi)

        fp_radius = int(getattr(self.args, "fp_radius", 2))
        fp_nbits = int(getattr(self.args, "fp_nbits", 2048))
        replace_star = bool(getattr(self.args, "fp_replace_star_fallback", True))

        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import AllChem

        RDLogger.DisableLog("rdApp.*")
        RDLogger.DisableLog("rdWarning.*")
        RDLogger.DisableLog("rdError.*")

        fp_table = np.zeros((seg_smiles_vocab_size, fp_nbits), dtype=np.float32)
        failed = []

        for iid in range(1, seg_smiles_vocab_size):
            smi = id_to_smiles[iid]
            if not smi:
                continue

            mol = Chem.MolFromSmiles(smi)
            if mol is None and replace_star:
                mol = Chem.MolFromSmiles(smi.replace("*", "C"))

            if mol is None:
                failed.append(smi)
                continue

            try:
                bv = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_nbits)
                arr = np.zeros((fp_nbits,), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(bv, arr)
                fp_table[iid] = arr.astype(np.float32, copy=False)
            except Exception:
                failed.append(smi)

        if failed:
            print(f"[fp_morgan2048] RDKit failed for {len(failed)} SMILES; set fp row to zeros.")
            uniq = sorted(set(failed))
            print(f"[fp_morgan2048] failed_smiles (unique={len(uniq)}): {uniq[:50]}")

        fp_tensor = torch.from_numpy(fp_table)
        self.fp_table = fp_tensor.to(device=self.embed_tokens.weight.device)
        self._fp_table_built = True

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

        # Chain-only path intentionally bypasses all stage-1 atom-level computation.
        if self.args.main_task == "chain_only":
            seg_valid_mask = batch["seg_valid_mask"]
            seg_smiles_id = batch.get("seg_smiles_id", None)
            if seg_smiles_id is None:
                # Backward-compatible fallback for old preprocessed files.
                seg_smiles_id = batch["seg_block_id"]

            seg_smiles_id = seg_smiles_id.long()
            if getattr(self.args, "chain_only_repr", "smiles_embed") == "fp_morgan2048":
                self._ensure_fp_table()
                fp_size = int(self.fp_table.size(0))
                if fp_size <= 0:
                    raise RuntimeError("fp_table is empty; failed to build fingerprints")
                seg_smiles_id = torch.remainder(seg_smiles_id, fp_size)
                seg_fp = self.fp_table[seg_smiles_id]
                seg_repr = self.fp_proj(seg_fp.to(dtype=self.embed_tokens.weight.dtype))
            else:
                vocab_size = int(self.stage2_only_smiles_embed.num_embeddings)
                seg_smiles_id = torch.remainder(seg_smiles_id, vocab_size)
                seg_repr = self.stage2_only_smiles_embed(seg_smiles_id)
            seg_repr = seg_repr * seg_valid_mask.unsqueeze(-1).float()
            seg_repr = seg_repr.to(dtype=self.embed_tokens.weight.dtype)

            chain_tokens, chain_attn_mask, chain_meta = self.chain_token_feature(
                seg_repr=seg_repr,
                seg_dop=batch["seg_dop"],
                seg_block_id=batch["seg_block_id"],
                glob_feat=batch["chain_glob_feat"],
                glob_feat_mask=batch["chain_glob_mask"],
                block_feat=batch["block_feat"],
                block_feat_mask=batch["block_feat_mask"],
                chain_node_seg_id=batch.get("chain_node_seg_id", None),
                chain_topo_dist=batch.get("chain_topo_dist", None),
                num_heads=self.args.chain_attention_heads,
            )

            B3, T, _ = chain_tokens.shape
            graph_attn_bias = chain_tokens.new_zeros(B3, T, T, self.args.chain_pair_dim)
            graph_attn_bias = self.chain_edge_feature(
                batch,
                graph_attn_bias,
                repeat_start=int(chain_meta["repeat_start"]),
                chain_len=chain_meta["chain_len"],
                topo_dist=chain_meta["topo_dist"],
            )

            chain_repr, _ = self.chain_encoder(
                chain_tokens,
                graph_attn_bias,
                atom_mask=None,
                pair_mask=None,
                attn_mask=chain_attn_mask,
            )

            chain_cls = chain_repr[:, 0, :]
            pred = self.reg_head(chain_cls)
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
        
        if not getattr(self.args, "wo_pair", False):
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
        pair_embed_dim = getattr(self.args, "pair_embed_dim", None)
        if pair_embed_dim is None:
            pair_embed_dim = getattr(self.args, "pair_dim", 512)

        pair_repr = node_repr.new_zeros(
            B, total_len, total_len, int(pair_embed_dim), dtype=node_repr.dtype
        )
        pair_repr = self.edge_feature(batch, pair_repr)

        # 3-D SE(3) bias �C inside each segment only
        if getattr(self.args, "wo_geom_3d", False):
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
            if self.stage1_capacity_mlp is not None:
                mol_rep = mol_rep + self.stage1_capacity_mlp(mol_rep)
            pred_val = self.reg_head(mol_rep)
            return pred_val
            
        # ---------- Downstream chain-level forward --------------------------------------------------    
        elif self.args.main_task == "chain":

            n_seg = batch["seg_feat"].size(1)
            seg_start = 2                    # CLS + GLOB
            seg_end = 2 + n_seg
        
            seg_repr = node_repr[:, seg_start:seg_end, :]   # [B, S, D]

            chain_tokens, chain_attn_mask, chain_meta = self.chain_token_feature(
                seg_repr=seg_repr,
                seg_dop=batch["seg_dop"],
                seg_block_id=batch["seg_block_id"],
                glob_feat=batch["chain_glob_feat"],
                glob_feat_mask=batch["chain_glob_mask"],
                block_feat=batch["block_feat"],              # [B, 2, F]
                block_feat_mask=batch["block_feat_mask"],    # [B, 2, F]
                chain_node_seg_id=batch.get("chain_node_seg_id", None),
                chain_topo_dist=batch.get("chain_topo_dist", None),
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
                graph_attn_bias,
                repeat_start=int(chain_meta["repeat_start"]),
                chain_len=chain_meta["chain_len"],
                topo_dist=chain_meta["topo_dist"],
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

        # ---------- Downstream chain-only forward (skip stage-1 monomer encoder) --------
        elif self.args.main_task == "chain_only":

            seg_valid_mask = batch["seg_valid_mask"]
            seg_smiles_id = batch.get("seg_smiles_id", None)
            if seg_smiles_id is None:
                # Backward-compatible fallback for old preprocessed files.
                seg_smiles_id = batch["seg_block_id"]

            seg_smiles_id = seg_smiles_id.long()
            vocab_size = int(self.stage2_only_smiles_embed.num_embeddings)
            seg_smiles_id = torch.remainder(seg_smiles_id, vocab_size)

            seg_repr = self.stage2_only_smiles_embed(seg_smiles_id)
            seg_repr = seg_repr * seg_valid_mask.unsqueeze(-1).float()
            seg_repr = seg_repr.to(dtype=self.embed_tokens.weight.dtype)

            chain_tokens, chain_attn_mask, chain_meta = self.chain_token_feature(
                seg_repr=seg_repr,
                seg_dop=batch["seg_dop"],
                seg_block_id=batch["seg_block_id"],
                glob_feat=batch["chain_glob_feat"],
                glob_feat_mask=batch["chain_glob_mask"],
                block_feat=batch["block_feat"],
                block_feat_mask=batch["block_feat_mask"],
                chain_node_seg_id=batch.get("chain_node_seg_id", None),
                chain_topo_dist=batch.get("chain_topo_dist", None),
                num_heads=self.args.chain_attention_heads,
            )

            B3, T, _ = chain_tokens.shape
            graph_attn_bias = chain_tokens.new_zeros(B3, T, T, self.args.chain_pair_dim)
            graph_attn_bias = self.chain_edge_feature(
                batch,
                graph_attn_bias,
                repeat_start=int(chain_meta["repeat_start"]),
                chain_len=chain_meta["chain_len"],
                topo_dist=chain_meta["topo_dist"],
            )

            chain_repr, _ = self.chain_encoder(
                chain_tokens,
                graph_attn_bias,
                atom_mask=None,
                pair_mask=None,
                attn_mask=chain_attn_mask,
            )

            chain_cls = chain_repr[:, 0, :]
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
