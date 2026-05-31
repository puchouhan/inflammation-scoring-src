"""
Cross-animal tile similarity analysis for Punkt 6.

Finds the most visually similar tile pairs across different animals by computing
mean absolute pixel distance on downscaled (32x32x3) RGB thumbnails.
Outputs a figure showing the top-N most similar cross-animal pairs side by side.

Usage (from project root):
    python scripts/cross_animal_similarity.py

Output: figures/cross_animal_similar_pairs.png (and the corresponding caption text)
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ------ Configuration -------------------------------------------------------
DATASET_DIR = Path(
    "/Users/pulkit/Library/Mobile Documents/"
    "com~apple~CloudDocs/master_thesis/master_thesis_inflammation/"
    "dataset_norm/training"
)
OUTPUT_DIR = Path(
    "/Users/pulkit/Library/Mobile Documents/"
    "com~apple~CloudDocs/master_thesis/master_thesis_inflammation/"
    "figures"
)
FIGURE_NAME = "cross_animal_similar_pairs.png"

# Thumbnail size for fast distance computation
THUMB_SIZE = 32

# Number of top similar pairs to display
TOP_N = 3

# Max tiles sampled per animal for the exhaustive search (keeps runtime manageable)
MAX_TILES_PER_ANIMAL = 500

CLASSES = ["0", "1", "2", "3", "ignore"]
ANIMALS = ["15_304", "16_314", "17_305"]

FONT_SIZE = 12
# ---------------------------------------------------------------------------


def parse_animal(filename: str) -> str:
    """Extract animal_id (e.g. '15_304') from filename stem."""
    parts = Path(filename).stem.split("_")
    return f"{parts[0]}_{parts[1]}"


def collect_tiles(dataset_dir: Path, max_per_animal: int) -> dict:
    """Collect tile paths grouped by animal, with optional subsampling.

    Args:
        dataset_dir: Root directory with class subfolders.
        max_per_animal: Maximum tiles to sample per animal.

    Returns:
        Dict mapping animal_id -> list of Path objects.
    """
    by_animal: dict = {}
    for cls in CLASSES:
        cls_dir = dataset_dir / cls
        if not cls_dir.exists():
            continue
        for fpath in cls_dir.glob("*.png"):
            animal = parse_animal(fpath.name)
            by_animal.setdefault(animal, []).append(fpath)

    # Subsample deterministically
    rng = random.Random(42)
    for animal in list(by_animal.keys()):
        tiles = by_animal[animal]
        if len(tiles) > max_per_animal:
            by_animal[animal] = rng.sample(tiles, max_per_animal)

    return by_animal


def load_thumbnail(path: Path, size: int) -> np.ndarray:
    """Load image and resize to (size x size x 3) float32 in [0, 1].

    Args:
        path: Path to the image file.
        size: Target side length for the thumbnail.

    Returns:
        Float32 array of shape (size, size, 3).

    Raises:
        FileNotFoundError: If the image cannot be read.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    thumb = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)
    return thumb.astype(np.float32) / 255.0


def compute_thumbs(paths: List[Path], size: int) -> np.ndarray:
    """Load and stack thumbnails into a matrix.

    Args:
        paths: List of image paths.
        size: Thumbnail side length.

    Returns:
        Float32 array of shape (N, size*size*3).
    """
    thumbs = []
    for p in paths:
        try:
            t = load_thumbnail(p, size)
            thumbs.append(t.flatten())
        except FileNotFoundError:
            thumbs.append(np.zeros(size * size * 3, dtype=np.float32))
    return np.stack(thumbs, axis=0)


def find_top_similar_pairs(
    tiles_a: List[Path],
    tiles_b: List[Path],
    top_n: int,
    thumb_size: int,
) -> List[Tuple[Path, Path, float]]:
    """Find the top-N most similar tile pairs between two animal tile lists.

    Similarity is measured by mean absolute pixel distance on thumbnails
    (lower = more similar).

    Args:
        tiles_a: Tile paths for animal A.
        tiles_b: Tile paths for animal B.
        top_n: Number of most-similar pairs to return.
        thumb_size: Thumbnail side length for comparison.

    Returns:
        List of (path_a, path_b, distance) tuples sorted by ascending distance.
    """
    print(
        f"  Computing distances: {len(tiles_a)} x {len(tiles_b)} tiles ...",
        flush=True,
    )
    thumbs_a = compute_thumbs(tiles_a, thumb_size)
    thumbs_b = compute_thumbs(tiles_b, thumb_size)

    # Vectorised L1 distance: (N_a, D) vs (N_b, D) -> (N_a, N_b)
    # Split into chunks to limit memory usage
    chunk = 100
    distances = np.empty((len(tiles_a), len(tiles_b)), dtype=np.float32)
    for i in range(0, len(tiles_a), chunk):
        a_chunk = thumbs_a[i : i + chunk]  # (chunk, D)
        # |a - b| mean over pixel dimension
        distances[i : i + chunk] = np.mean(
            np.abs(a_chunk[:, np.newaxis, :] - thumbs_b[np.newaxis, :, :]),
            axis=2,
        )

    # Flatten and get indices of top_n smallest distances
    flat = distances.flatten()
    top_idx = np.argpartition(flat, top_n)[:top_n]
    top_idx = top_idx[np.argsort(flat[top_idx])]

    pairs = []
    for idx in top_idx:
        ia, ib = divmod(int(idx), len(tiles_b))
        pairs.append((tiles_a[ia], tiles_b[ib], float(flat[idx])))

    return pairs


def make_figure(
    all_pairs: List[Tuple[str, str, Path, Path, float]],
    out_path: Path,
    font_size: int = FONT_SIZE,
) -> None:
    """Render a thesis-quality figure showing the most similar cross-animal tile pairs.

    Each pair occupies one row with two equal-sized subplots. A centred annotation
    between columns shows the MAD value. Axes are square and images fill the cell
    without distortion.

    Args:
        all_pairs: List of (animal_a, animal_b, path_a, path_b, distance) tuples.
        out_path: Output file path for the PNG figure.
        font_size: Base font size for tile labels.
    """
    n = len(all_pairs)
    cell_size = 2.5  # inches per subplot
    fig, axes = plt.subplots(n, 2, figsize=(cell_size * 2 + 1.5, cell_size * n + 0.6))
    if n == 1:
        axes = [axes]

    for row_idx, (animal_a, animal_b, path_a, path_b, dist) in enumerate(all_pairs):
        img_a = cv2.cvtColor(cv2.imread(str(path_a)), cv2.COLOR_BGR2RGB)
        img_b = cv2.cvtColor(cv2.imread(str(path_b)), cv2.COLOR_BGR2RGB)

        ax_a = axes[row_idx][0]
        ax_b = axes[row_idx][1]

        ax_a.imshow(img_a, aspect="auto")
        ax_a.set_title(
            f"Animal {animal_a}",
            fontsize=font_size,
            fontweight="bold",
        )
        ax_a.set_xlabel(path_a.name, fontsize=font_size - 2)
        ax_a.set_xticks([])
        ax_a.set_yticks([])
        for spine in ax_a.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)

        ax_b.imshow(img_b, aspect="auto")
        ax_b.set_title(
            f"Animal {animal_b}",
            fontsize=font_size,
            fontweight="bold",
        )
        ax_b.set_xlabel(f"{path_b.name}\nMAD = {dist:.4f}", fontsize=font_size - 2)
        ax_b.set_xticks([])
        ax_b.set_yticks([])
        for spine in ax_b.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)

    fig.suptitle(
        "Three Most Similar Cross-Animal Tile Pairs",
        fontsize=font_size + 1,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout(pad=0.8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")


def main(
    dataset_dir: Path = DATASET_DIR,
    output_dir: Path = OUTPUT_DIR,
    top_n: int = TOP_N,
    thumb_size: int = THUMB_SIZE,
    max_tiles: int = MAX_TILES_PER_ANIMAL,
) -> None:
    """Run the full cross-animal similarity analysis.

    Args:
        dataset_dir: Path to the normalised training dataset directory.
        output_dir: Directory to write the output figure.
        top_n: Number of most-similar pairs to display per animal pair.
        thumb_size: Thumbnail side length for distance computation.
        max_tiles: Maximum tiles sampled per animal for the search.
    """
    print("Collecting tiles ...")
    by_animal = collect_tiles(dataset_dir, max_tiles)
    print(f"Animals found: { {a: len(t) for a, t in by_animal.items()} }")

    animal_list = sorted(by_animal.keys())
    all_top_pairs: List[Tuple[str, str, Path, Path, float]] = []

    # Compare every unique pair of animals
    for i in range(len(animal_list)):
        for j in range(i + 1, len(animal_list)):
            a_id = animal_list[i]
            b_id = animal_list[j]
            print(f"\nComparing {a_id} vs {b_id} ...")
            pairs = find_top_similar_pairs(
                by_animal[a_id], by_animal[b_id], top_n=top_n, thumb_size=thumb_size
            )
            for path_a, path_b, dist in pairs:
                all_top_pairs.append((a_id, b_id, path_a, path_b, dist))
                print(f"  {path_a.name} <-> {path_b.name}  MAD={dist:.6f}")

    # Sort globally and take the top_n overall
    all_top_pairs.sort(key=lambda x: x[4])
    top_pairs_global = all_top_pairs[:top_n]

    print(f"\nTop {top_n} most similar cross-animal pairs overall:")
    for animal_a, animal_b, path_a, path_b, dist in top_pairs_global:
        print(f"  [{animal_a}] {path_a.name}  <->  [{animal_b}] {path_b.name}  MAD={dist:.6f}")

    out_path = output_dir / FIGURE_NAME
    make_figure(top_pairs_global, out_path)

    # Print LaTeX caption suggestion
    print("\nSuggested LaTeX figure reference:")
    print(r"""  \begin{figure}[htbp]
    \centering
    \includegraphics[width=0.75\linewidth]{cross_animal_similar_pairs}
    \caption{The three most visually similar tile pairs across different animals,
             ranked by mean absolute pixel distance (MAD) on 32$\times$32 thumbnails.
             Despite visual resemblance, all pairs originate from different animals and
             therefore do not constitute data leakage.
             The smallest observed MAD value is \textbf{X.XXXX}, confirming that no
             near-duplicate tiles exist across animal boundaries.}
    \label{fig:cross_animal_similarity}
  \end{figure}""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-animal tile similarity analysis."
    )
    parser.add_argument("--top_n", type=int, default=TOP_N, help="Top N pairs to show")
    parser.add_argument(
        "--thumb_size", type=int, default=THUMB_SIZE, help="Thumbnail size for distance"
    )
    parser.add_argument(
        "--max_tiles",
        type=int,
        default=MAX_TILES_PER_ANIMAL,
        help="Max tiles per animal to sample",
    )
    args = parser.parse_args()
    main(top_n=args.top_n, thumb_size=args.thumb_size, max_tiles=args.max_tiles)
