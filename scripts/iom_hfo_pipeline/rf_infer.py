"""
MATLAB Random Forest inference helpers for RF_CRDL_ASLR_Model.mat.

This module is intentionally narrow and fidelity-oriented:
- Reads the MATLAB cell/struct model saved in Data/RF_CRDL_ASLR_Model.mat
- Evaluates trees using nodeCutVar/nodeCutValue/childnode/nodelabel
- Returns MATLAB-style class labels (1/2) and vote tallies
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class MatlabRfTree:
    """One CART tree as stored by the MATLAB Random_Forests toolbox."""

    node_cut_var: np.ndarray  # 1-based feature index per node; 0 on leaves
    node_cut_value: np.ndarray  # split threshold per node
    childnode: np.ndarray  # 1-based index of left child; 0 on leaves
    nodelabel: np.ndarray  # class label on leaves (1/2), 0 on internal nodes
    method: str
    oobe: float


def _coerce_1d(name: str, x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"RF tree field '{name}' is empty.")
    return arr


def _coerce_method_str(x: object) -> str:
    """Normalize MATLAB char/array scalar to a plain Python string."""
    if isinstance(x, np.ndarray):
        flat = [str(v) for v in x.ravel().tolist()]
        return "".join(flat).strip()
    return str(x).strip()


def _extract_tree_structs(model_mat: np.ndarray, model_cell_index: int) -> list[object]:
    """
    Extract list of tree structs from MATLAB 'model' variable.

    Expected demo layout:
    - model is a cell array, typically shape (1, 1)
    - model{1} is a struct array of trees, shape (1, 100)
    """
    if not isinstance(model_mat, np.ndarray) or model_mat.dtype != object:
        raise ValueError(
            "Unsupported RF model format: expected MATLAB cell array (object ndarray)."
        )

    model_cells = model_mat.ravel()
    if model_cell_index < 0 or model_cell_index >= model_cells.size:
        raise ValueError(
            f"model_cell_index={model_cell_index} out of range for "
            f"{model_cells.size} model cell(s)."
        )

    tree_block = model_cells[model_cell_index]
    if isinstance(tree_block, np.ndarray):
        trees = [t for t in tree_block.ravel()]
    else:
        trees = [tree_block]

    if not trees:
        raise ValueError("RF model contains no trees.")
    return trees


def load_matlab_rf_model(
    path: Path | str,
    model_cell_index_1based: int = 1,
) -> tuple[list[MatlabRfTree], dict]:
    """
    Load MATLAB RF model and return parsed trees + metadata.

    Parameters
    ----------
    path:
        Path to RF_CRDL_ASLR_Model.mat
    model_cell_index_1based:
        MATLAB-style 1-based index for model cell selection (Demo_Run.m uses model{1}).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"RF model file not found: {p}")

    m = loadmat(str(p), struct_as_record=False, squeeze_me=False)
    if "model" not in m:
        raise ValueError(f"'model' variable not found in RF model file: {p}")

    model_cell_index = int(model_cell_index_1based) - 1
    tree_structs = _extract_tree_structs(m["model"], model_cell_index=model_cell_index)

    trees: list[MatlabRfTree] = []
    for i, t in enumerate(tree_structs):
        missing = [
            name
            for name in ("nodeCutVar", "nodeCutValue", "childnode", "nodelabel")
            if not hasattr(t, name)
        ]
        if missing:
            raise ValueError(
                f"Unsupported RF tree struct at index {i}: missing fields {missing}."
            )

        node_cut_var = _coerce_1d("nodeCutVar", getattr(t, "nodeCutVar")).astype(int)
        node_cut_value = _coerce_1d("nodeCutValue", getattr(t, "nodeCutValue")).astype(
            np.float64
        )
        childnode = _coerce_1d("childnode", getattr(t, "childnode")).astype(int)
        nodelabel = _coerce_1d("nodelabel", getattr(t, "nodelabel")).astype(int)

        n_nodes = node_cut_var.size
        if not (node_cut_value.size == n_nodes == childnode.size == nodelabel.size):
            raise ValueError(
                f"Inconsistent tree field lengths at tree {i}: "
                f"var={node_cut_var.size}, cut={node_cut_value.size}, "
                f"child={childnode.size}, label={nodelabel.size}"
            )

        trees.append(
            MatlabRfTree(
                node_cut_var=node_cut_var,
                node_cut_value=node_cut_value,
                childnode=childnode,
                nodelabel=nodelabel,
                method=_coerce_method_str(getattr(t, "method", "")),
                oobe=float(getattr(t, "oobe", np.nan)),
            )
        )

    meta = {
        "model_file": str(p),
        "model_cell_index_1based": int(model_cell_index_1based),
        "n_trees": len(trees),
        "tree_method": trees[0].method if trees else "",
        "tree_oobe_mean": float(np.nanmean([tr.oobe for tr in trees])) if trees else np.nan,
    }
    return trees, meta


def _predict_tree_one(sample: np.ndarray, tree: MatlabRfTree) -> int:
    """
    Predict one sample for one tree.

    Tree traversal convention:
    - childnode(i) gives the left child (MATLAB 1-based node index)
    - right child is childnode(i) + 1
    - split uses feature nodeCutVar(i) and threshold nodeCutValue(i)
    - go left if x <= threshold, otherwise right
    """
    node = 1  # MATLAB root index
    n_nodes = tree.nodelabel.size

    for _ in range(n_nodes + 1):
        idx = node - 1
        if idx < 0 or idx >= n_nodes:
            raise ValueError(f"Tree traversal reached invalid node index: {node}")

        label = int(tree.nodelabel[idx])
        if label > 0:
            return label

        cut_var = int(tree.node_cut_var[idx])
        left = int(tree.childnode[idx])
        if cut_var <= 0 or left <= 0:
            raise ValueError(
                f"Internal node with invalid split/child at node {node}: "
                f"cut_var={cut_var}, left={left}"
            )

        feat_idx = cut_var - 1  # MATLAB 1-based -> Python 0-based
        if feat_idx < 0 or feat_idx >= sample.size:
            raise ValueError(
                f"Feature index out of range at node {node}: "
                f"cut_var={cut_var}, n_features={sample.size}"
            )

        x = float(sample[feat_idx])
        if np.isnan(x):
            raise ValueError(
                "NaN in feature vector is unsupported for faithful RF evaluation."
            )
        cut = float(tree.node_cut_value[idx])
        node = left if x <= cut else left + 1

    raise RuntimeError("Tree traversal exceeded node budget (possible malformed tree).")


def eval_matlab_rf(
    features: np.ndarray,
    trees: list[MatlabRfTree],
    class_labels: tuple[int, int] = (1, 2),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate MATLAB RF model on feature matrix.

    Parameters
    ----------
    features:
        Shape (n_samples, n_features) in exact MATLAB feature order.
    trees:
        Parsed forest trees.
    class_labels:
        Class labels for vote columns (default: pseudo=1, real=2).

    Returns
    -------
    pred:
        Shape (n_samples,), class label per event.
    votes:
        Shape (n_samples, 2), vote counts for class_labels order.
    vote_fraction:
        Shape (n_samples, 2), votes normalized by n_trees.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"features must be 2D (n_samples, n_features), got {x.shape}")
    if x.shape[0] == 0:
        return (
            np.zeros((0,), dtype=int),
            np.zeros((0, len(class_labels)), dtype=int),
            np.zeros((0, len(class_labels)), dtype=np.float64),
        )
    if not trees:
        raise ValueError("Empty forest.")

    label_to_col = {int(lbl): i for i, lbl in enumerate(class_labels)}
    votes = np.zeros((x.shape[0], len(class_labels)), dtype=int)

    for i in range(x.shape[0]):
        for tr in trees:
            lbl = _predict_tree_one(x[i, :], tr)
            if lbl not in label_to_col:
                raise ValueError(
                    f"Encountered unsupported class label {lbl}; "
                    f"expected one of {class_labels}."
                )
            votes[i, label_to_col[lbl]] += 1

    pred_cols = np.argmax(votes, axis=1)  # first max on ties, MATLAB-like
    pred = np.asarray([class_labels[j] for j in pred_cols], dtype=int)
    vote_fraction = votes.astype(np.float64) / float(len(trees))
    return pred, votes, vote_fraction

