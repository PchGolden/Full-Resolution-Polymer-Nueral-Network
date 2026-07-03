import numpy as np
import torch
from typing import List, Tuple, Dict
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import GetDistanceMatrix
from rdkit import RDLogger

allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)) + ["misc"],
    "possible_chirality_list": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_TRIGONALBIPYRAMIDAL",
        "CHI_OCTAHEDRAL",
        "CHI_SQUAREPLANAR",
        "CHI_OTHER",
    ],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization_list": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring_list": [False, True],
    "possible_bond_type_list": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "possible_bond_stereo_list": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "possible_is_conjugated_list": [False, True],
}

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1
        
def atom_to_feature_vector(atom):
    return [
        safe_index(allowable_features["possible_atomic_num_list"], atom.GetAtomicNum()),
        allowable_features["possible_chirality_list"].index(str(atom.GetChiralTag())),
        safe_index(allowable_features["possible_degree_list"], atom.GetTotalDegree()),
        safe_index(
            allowable_features["possible_formal_charge_list"], atom.GetFormalCharge()
        ),
        safe_index(
            allowable_features["possible_numH_list"], atom.GetTotalNumHs()
        ),
        safe_index(
            allowable_features["possible_number_radical_e_list"],
            atom.GetNumRadicalElectrons(),
        ),
        safe_index(
            allowable_features["possible_hybridization_list"],
            str(atom.GetHybridization()),
        ),
        allowable_features["possible_is_aromatic_list"].index(atom.GetIsAromatic()),
        allowable_features["possible_is_in_ring_list"].index(atom.IsInRing()),
    ]
    
def bond_to_feature_vector(bond):
    bond_feature = [
        safe_index(
            allowable_features["possible_bond_type_list"], str(bond.GetBondType())
        ),
        allowable_features["possible_bond_stereo_list"].index(str(bond.GetStereo())),
        allowable_features["possible_is_conjugated_list"].index(bond.GetIsConjugated()),
    ]
    return bond_feature

def compute_shortest_path_rdkit(mol: Chem.Mol) -> np.ndarray:
    """
    Use RDKit's GetDistanceMatrix to compute topological distances (number of bonds)
    between all pairs of atoms in the molecule.
    Unconnected pairs are set to INF=510.
    """
    # get raw distance matrix (float64)
    d = GetDistanceMatrix(mol).astype(np.int32)
    n = d.shape[0]
    INF = 509
    # replace off-diagonal zeros (unreachable) with INF
    mask = (d == 0)
    # keep diagonal zeros
    for i in range(n):
        mask[i, i] = False
    d[mask] = INF
    return d


def embed_discrete_features(x: np.ndarray, sizes: List[int]) -> np.ndarray:
    """
    Convert multi-column discrete features into unique integer embeddings by offsetting.
    x: array of shape (..., F), sizes: list of length F, x[...,i] < sizes[i]
    """

    assert x.shape[-1] == len(sizes), "Feature dimension mismatch"
    out = x.copy().astype(np.int32)
    offset = 1
    for i, size in enumerate(sizes):
        assert np.all(out[..., i] < size), f"Feature value {out[...,i].max()} >= size {size}"
        out[..., i] += offset
        offset += size
    return out


def embed_discrete_features_tensor(x: torch.Tensor, sizes: List[int]) -> torch.Tensor:
    """
    Torch version of embed_discrete_features.
    """
    out = x.clone().long()
    offset = 1
    for i, size in enumerate(sizes):
        out[..., i] = out[..., i] + offset
        offset += size
    return out


def build_initial_graph(mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Chem.Mol, np.ndarray]:
    """
    Create a basic molecular graph from an RDKit Mol object.
    Uses heavy atoms only. Returns:
      node_attr: (N,8) int array of atom features in order
        [chirality, total_degree, formal_charge, total_H, radical_e, hybridization, is_aromatic, is_in_ring]
      edge_index: (2,E) int array of connectivity
      edge_attr: (E,3) int array of bond features
        [bond_type, bond_stereo, is_conjugated]
      proc_mol: RDKit Mol with explicit H added then removed (heavy-only)
    """
    # add explicit H then remove to ensure correct connectivity
    mol_h = AllChem.AddHs(mol, addCoords=False)
    mol_h = AllChem.RemoveAllHs(mol_h)
    # Replace atomic number 0 (e.g., "*") with 118 to avoid collision with padding_idx=0 in embeddings
    atomic_nums = np.array([atom.GetAtomicNum() for atom in mol_h.GetAtoms()], dtype=np.int32)
    atomic_nums[atomic_nums == 0] = 119
    # atom features
    node_feats: List[List[int]] = []
    for atom in mol_h.GetAtoms():
        node_feats.append(atom_to_feature_vector(atom))
    node_attr = np.array(node_feats, dtype=np.int32)
    # bond features
    edge_list: List[Tuple[int,int]] = []
    edge_feats: List[List[int]] = []
    for bond in mol_h.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        bf = bond_to_feature_vector(bond)
        
        edge_list.extend([(i, j), (j, i)])
        edge_feats.extend([bf, bf])
    if edge_list:
        edge_index = np.array(edge_list, dtype=np.int64).T
        edge_attr = np.array(edge_feats, dtype=np.int32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, 3), dtype=np.int32)
    return node_attr, edge_index, edge_attr, mol_h, atomic_nums


def build_graph_features(
    node_attr: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    mol: Chem.Mol,
    drop_feat: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Construct feature tensors from molecular graph data for neural models.
    Outputs dict with keys:
      atom_feat: (N,8) int64
      atom_mask: (N,) int64 all ones
      edge_feat: (N,N,3) int64
      shortest_path: (N,N) int64
      degree: (N,) int64
      pair_type: (N,N,2) int64
      attn_bias: (N+1,N+1) float32 zeros
    """
    N = node_attr.shape[0]
    
# ------------------------------------------------------------
# NOTE: Feature Embedding Index Offset Explanation
# ------------------------------------------------------------
# We offset all discrete features (atom_feat, edge_feat, degree, etc.)
# after converting them to embedding indices using `embed_discrete_features`.
# The final shift logic is as follows:
#
#   - All features use an initial offset of +1 during embedding (start from 1)
#   - Then we add an additional +1 (→ total +2) to ensure:
#       - index 0 is reserved for padding
#       - index 1 is reserved for dropped/masked features
#       - valid features start from index 2+
#
# This aligns with the UniMol official implementation, and ensures
# proper use of `padding_idx=0` and masking during training.
# ------------------------------------------------------------
# Example:
#   - drop_feat=True → feature[:] = 1  (explicitly masked)
#   - drop_feat=False → feature += 2   (offset to valid embedding space)
# ------------------------------------------------------------

    # embed atom features (skip no dims)
    atom_feat = embed_discrete_features(node_attr[:, 1:], [16]*8)
    # adjacency and degree
    adj = np.zeros((N, N), dtype=np.int32)
    adj[edge_index[0], edge_index[1]] = 1
    degree = adj.sum(axis=-1)
    # embed edge features
    ea = edge_attr
    if ea.ndim == 1:
        ea = ea[:, None]
    edge_feat = np.zeros((N, N, ea.shape[-1]), dtype=np.int32)
    ef_emb = embed_discrete_features(ea, [16, 16, 16]) + 1
    edge_feat[edge_index[0], edge_index[1]] = ef_emb
    # shortest path
    sp = compute_shortest_path_rdkit(mol)
    # drop or shift
    if drop_feat:
        atom_feat[...] = 1
        edge_feat[...] = 1
        degree[...] = 1
        sp[...] = 511
    else:
        atom_feat += 2
        edge_feat += 2
        degree += 2
        sp += 1
    # to tensors
    feat: Dict[str, torch.Tensor] = {}
    feat["atom_feat"] = torch.from_numpy(atom_feat).long()
    feat["atom_mask"] = torch.ones(N, dtype=torch.long)
    feat["edge_feat"] = torch.from_numpy(edge_feat).long()
    feat["shortest_path"] = torch.from_numpy(sp).long()
    feat["degree"] = torch.from_numpy(degree).long()
    # pair type
    z_idx = torch.from_numpy(node_attr[:, 0]).long()    
    pair = torch.stack(
        (z_idx.unsqueeze(1).expand(N, N),    # Z_i
         z_idx.unsqueeze(0).expand(N, N)),   # Z_j
        dim=-1                               # (N, N, 2)
    )
    feat["pair_type"] = embed_discrete_features_tensor(pair, [128, 128])
    feat["attn_bias"] = torch.zeros((N + 1, N + 1), dtype=torch.float32)
    return feat
