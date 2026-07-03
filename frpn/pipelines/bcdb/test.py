import torch
from models.multi_mol_model import MultiMolModel
import argparse, os

def parse_args():
    parser = argparse.ArgumentParser(description="DDP Trainer")
    
    # ---- Ablation Studys -------------------------
    parser.add_argument("--wo_geom_3d", action="store_true")
    parser.add_argument("--wo_triopm", action="store_true")
    parser.add_argument("--wo_edge", action="store_true")
    parser.add_argument("--wo_spd", action="store_true")
    parser.add_argument("--wo_pair", action="store_true")
    parser.add_argument("--wo_node", action="store_true")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--wo_atom_feat", type=int, nargs="*", default=None)

    # ---- Dataset / Path -----------------------------------
    parser.add_argument("--dataset_name",help="Name of the dataset")
    parser.add_argument("--non_kfold", default=False, action="store_true", help="Use separate train/test pkl instead of CV folds")
    parser.add_argument("--pkl_path", help="Path to the *.pkl file")
    parser.add_argument("--test_pkl_path", help="Path to the *_test.pkl file")
    parser.add_argument("--weight_path", default="/share/home/202320162823/unimacro/no_pretrain.pt")
    parser.add_argument("--pretrain_train_path", default="/public/home/202320162823/version2_pretrain/pretrain_data/pretrain_train.lmdb")
    parser.add_argument("--pretrain_val_path",default="/public/home/202320162823/version2_pretrain/pretrain_data/pretrain_val.lmdb")
    parser.add_argument("--fold", type=int, help="Cross-validation fold index")
    parser.add_argument("--results_root", default="results", help="Root directory to save results")

    # ---- Training Hyperparameters -------------------------
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--early_stop_patience", type=int, default=40)
    parser.add_argument("--val_every_steps", type=int, default=50)
    parser.add_argument("--min_delta_abs", type=float, default=1e-3)
    parser.add_argument("--min_delta_rel", type=float, default=1e-2)
    
    # ---- Chain-level model args ----
    parser.add_argument("--num_chain_glob_feat", type=int, default=3)   # T, Mn, coexistence
    parser.add_argument("--num_chain_block_feat", type=int, default=1)  # volume fraction
    
    parser.add_argument("--chain_pair_dim", type=int, default=32)
    parser.add_argument("--max_chain_dist", type=int, default=15000)
    
    parser.add_argument("--chain_encoder_layers", type=int, default=4)
    parser.add_argument("--chain_attention_heads", type=int, default=12)
    parser.add_argument("--chain_pair_hidden_dim", type=int, default=64)
    parser.add_argument("--chain_ffn_embed_dim", type=int, default=3072)
    
    # ---- Target RMSE Early Stop (for finetune/reg) -------
    parser.add_argument(
        "--stop_on_target_rmse",
        action="store_true",
        help="Stop early if val RMSE reaches target within a given epoch range (finetune/reg)."
    )
    parser.add_argument("--target_rmse", type=float, default=0.4)
    parser.add_argument("--target_rmse_max_epoch", type=int, default=100)

    # ---- System Settings ----------------------------------
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--pin_memory", action="store_true")

    # ---- Distributed Training -----------------------------
    parser.add_argument("--distributed", action="store_true", help="Enable DDP")
    parser.add_argument("--dist-backend", dest="dist_backend", default="nccl")
    parser.add_argument("--dist-url", default="env://")
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", 0)))
    parser.add_argument("--world_size", type=int, default=int(os.getenv("WORLD_SIZE", 1)))
    parser.add_argument("--rank", type=int, default=int(os.getenv("RANK", 0)))

    # ---- Model / Task -------------------------------------
    parser.add_argument("--main_task", choices=["pretrain", "finetune", "chain"], default="finetune")
    parser.add_argument("--task_type", choices=["reg", "cls"], default="reg")
    parser.add_argument("--num_tasks", type=int, default=1)
    parser.add_argument("--encoder_embed_dim", type=int, default=768)
    parser.add_argument("--padding_idx", type=int, default=0)
    parser.add_argument("--num_atom", type=int, default=512)
    parser.add_argument("--num_degree", type=int, default=128)
    parser.add_argument("--pair_embed_dim", type=int, default=512)
    parser.add_argument("--num_edge", type=int, default=64)
    parser.add_argument("--num_spatial", type=int, default=512)
    parser.add_argument("--encoder_layers", type=int, default=12)
    parser.add_argument("--pair_hidden_dim", type=int, default=64)
    parser.add_argument("--encoder_ffn_embed_dim", type=int, default=768)
    parser.add_argument("--encoder_attention_heads", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--activation_dropout", type=float, default=0.1)
    parser.add_argument("--activation_fn", type=str, default="gelu")
    parser.add_argument("--droppath_prob", type=float, default=0)
    parser.add_argument("--pair_dropout", type=float, default=0.1)
    parser.add_argument("--num_pair", type=int, default=512)
    parser.add_argument("--num_kernel", type=int, default=128)
    parser.add_argument("--gaussian_std_width", type=float, default=1.0)
    parser.add_argument("--gaussian_mean_start", type=float, default=0.0)
    parser.add_argument("--gaussian_mean_stop", type=float, default=9.0)

    return parser.parse_args()

def inspect_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_state" in ckpt:
        state = ckpt["model_state"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    print("\n=== Checkpoint keys ===")
    for k in sorted(state.keys()):
        print(k)
    print(f"\nTotal keys in checkpoint: {len(state)}")
args = parse_args()
model = MultiMolModel(args)
def inspect_model(model):
    print("\n=== Model state_dict keys ===")
    for k in sorted(model.state_dict().keys()):
        print(k)
    print(f"\nTotal keys in model: {len(model.state_dict())}")


inspect_checkpoint("/share/home/202320162823/unimacro/checkpoint_4400.pt")

inspect_model(model)