import argparse
import json
import os
from itertools import product

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


DEFAULT_PARAM_NAMES = [
    "rad_plan",
    "log10_planet_metallicity",
    "tint",
    "semi_major",
    "ctoO",
    "log_Kzz",
]


def _decode_string_array(arr):
    if arr.dtype.kind in ("S", "O"):
        return np.vectorize(
            lambda x: x.decode("utf-8", "ignore") if isinstance(x, (bytes, np.bytes_)) else str(x)
        )(arr)
    return arr.astype(str)


def _strip_vector_tail(arr, ndims):
    """Strip vector tail (e.g., nlayers axis) by taking element 0 repeatedly."""
    out = arr
    while out.ndim > ndims:
        out = out[..., 0]
    return out


def _load_highlight_inputs(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        data = np.load(path)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as fp:
            data = np.array(json.load(fp), dtype=float)
    elif ext == ".csv":
        data = np.genfromtxt(path, delimiter=",", dtype=float)
    else:
        data = np.genfromtxt(path, delimiter=None, dtype=float)

    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        if data.size % 6 != 0:
            raise ValueError("Highlight input array must have 6 columns (or flat size divisible by 6).")
        data = data.reshape((-1, 6))

    if data.shape[1] < 6:
        raise ValueError("Highlight input array must have at least 6 columns.")
    if data.shape[1] > 6:
        data = data[:, :6]

    return data


def _nearest_index(values, x):
    values = np.asarray(values, dtype=float)
    return int(np.argmin(np.abs(values - float(x))))


def _make_tick_labels(values, max_labels=5):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n <= max_labels:
        idx = np.arange(n)
    else:
        idx = np.unique(np.linspace(0, n - 1, max_labels).astype(int))
    labels = [f"{values[i]:g}" for i in idx]
    return idx, labels


def load_failure_cube(h5_path):
    with h5py.File(h5_path, "r") as f:
        inputs = np.array(f["inputs"], dtype=float)

        # Build axis values directly from saved inputs.
        axis_values = []
        for col in range(inputs.shape[1]):
            col_vals = inputs[:, col]
            col_vals = col_vals[np.isfinite(col_vals)]
            axis_values.append(np.unique(col_vals))

        grid_shape = tuple(len(v) for v in axis_values)
        ndims = len(grid_shape)

        status = _decode_string_array(_strip_vector_tail(np.array(f["results"]["status"]), ndims))
        error = _decode_string_array(_strip_vector_tail(np.array(f["results"]["error"]), ndims))
        conv_tp = _strip_vector_tail(np.array(f["results"]["converged_TP"], dtype=np.uint8), ndims)
        conv_pc = _strip_vector_tail(np.array(f["results"]["converged_PC"], dtype=np.uint8), ndims)

        completed_flat = np.array(f["completed"], dtype=bool) if "completed" in f else None

    category = np.zeros(grid_shape, dtype=np.uint8)

    # 1: TP/PICASO failure.
    category[conv_tp == 0] = 1

    # 2: Photochem non-converged where TP converged.
    category[(conv_tp == 1) & (conv_pc == 0)] = 2

    # 3: explicit status/error issue (overrides 1/2).
    status_lower = np.char.lower(status.astype(str))
    error_lower = np.char.lower(error.astype(str))
    allowed_status = np.isin(status, ["Photochem-converged", "Photochem-not-converged"])
    hard_error = (
        (~allowed_status)
        | (np.char.find(status_lower, "error") >= 0)
        | (np.char.find(error_lower, "error") >= 0)
        | (np.char.str_len(np.char.strip(error.astype(str))) > 0)
    )
    category[hard_error] = 3

    # 4: job never completed.
    if completed_flat is not None and completed_flat.size == int(np.prod(grid_shape)):
        completed = completed_flat.reshape(grid_shape)
        category[~completed] = 4

    return {
        "category": category,
        "status": status,
        "error": error,
        "conv_tp": conv_tp,
        "conv_pc": conv_pc,
        "axis_values": axis_values,
        "grid_shape": grid_shape,
    }


def plot_full_pairwise_panel(h5_path, highlight_inputs_path, output_path, param_names=None):
    loaded = load_failure_cube(h5_path)
    category = loaded["category"]
    axis_values = loaded["axis_values"]
    ndims = len(axis_values)

    if param_names is None:
        param_names = DEFAULT_PARAM_NAMES[:ndims]
    if len(param_names) != ndims:
        raise ValueError(f"Expected {ndims} parameter names, got {len(param_names)}")

    highlight_inputs = _load_highlight_inputs(highlight_inputs_path)
    highlight_indices = np.zeros_like(highlight_inputs, dtype=int)
    for dim in range(ndims):
        vals = axis_values[dim]
        for k in range(highlight_inputs.shape[0]):
            highlight_indices[k, dim] = _nearest_index(vals, highlight_inputs[k, dim])

    cmap = ListedColormap(
        [
            "#2ca25f",  # 0 good
            "#de2d26",  # 1 TP/PICASO fail
            "#fdae6b",  # 2 Photochem non-converged
            "#6a51a3",  # 3 status/error issue
            "#969696",  # 4 incomplete
        ]
    )
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    fig, axs = plt.subplots(ndims, ndims, figsize=(3.4 * ndims, 3.1 * ndims))

    if ndims == 1:
        axs = np.array([[axs]])

    for i, j in product(range(ndims), repeat=2):
        ax = axs[i, j]

        if i > j:
            ax.axis("off")
            continue

        if i == j:
            vals = axis_values[i]
            ax.text(
                0.5,
                0.5,
                f"{param_names[i]}\nN={len(vals)}\n[{vals.min():g}, {vals.max():g}]",
                ha="center",
                va="center",
                fontsize=10,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        other_axes = tuple(k for k in range(ndims) if k not in (i, j))
        # Aggregate to a 2D map: worst-case failure category over all non-plotted axes.
        slab = np.max(category, axis=other_axes) if len(other_axes) > 0 else category

        im = ax.imshow(slab, origin="lower", aspect="auto", cmap=cmap, norm=norm)

        # Overlay user-specified points to highlight.
        ax.scatter(
            highlight_indices[:, j],
            highlight_indices[:, i],
            s=44,
            marker="o",
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
        )

        x_ticks, x_labels = _make_tick_labels(axis_values[j], max_labels=5)
        y_ticks, y_labels = _make_tick_labels(axis_values[i], max_labels=5)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=8)

        if i == ndims - 1:
            ax.set_xlabel(param_names[j])
        if j == 0:
            ax.set_ylabel(param_names[i])

    legend_items = [
        Patch(facecolor="#2ca25f", edgecolor="none", label="0 good"),
        Patch(facecolor="#de2d26", edgecolor="none", label="1 TP/PICASO fail"),
        Patch(facecolor="#fdae6b", edgecolor="none", label="2 Photochem not converged"),
        Patch(facecolor="#6a51a3", edgecolor="none", label="3 status/error issue"),
        Patch(facecolor="#969696", edgecolor="none", label="4 incomplete"),
        Patch(facecolor="white", edgecolor="black", label="highlighted inputs"),
    ]

    fig.legend(handles=legend_items, loc="lower center", ncol=3, frameon=False,bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Failure Panel (pairwise worst-case projections)", y=1.01)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _parse_param_names(arg, ndims):
    if arg is None:
        return DEFAULT_PARAM_NAMES[:ndims]
    names = [item.strip() for item in arg.split(",") if item.strip()]
    if len(names) != ndims:
        raise ValueError(f"--param-names must provide exactly {ndims} names, got {len(names)}")
    return names


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a full panel of pairwise failure maps from a Photochem/PICASO HDF5 grid, "
            "and overlay user-provided highlight input points."
        )
    )
    parser.add_argument("--h5", required=True, help="Path to HDF5 grid file (contains inputs/results/completed)")
    parser.add_argument("--highlights", required=True, help="Path to highlight inputs (.npy, .csv, .txt, or .json)")
    parser.add_argument("--out", default="failure_panel.png", help="Output image path")
    parser.add_argument(
        "--param-names",
        default=None,
        help="Comma-separated parameter names in grid order, e.g. rad,metal,tint,semi,ctoO,kzz",
    )

    args = parser.parse_args()

    loaded = load_failure_cube(args.h5)
    ndims = len(loaded["axis_values"])
    param_names = _parse_param_names(args.param_names, ndims)

    plot_full_pairwise_panel(
        h5_path=args.h5,
        highlight_inputs_path=args.highlights,
        output_path=args.out,
        param_names=param_names,
    )

    print(f"Saved failure panel to: {args.out}")


if __name__ == "__main__":
    main()
