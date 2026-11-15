"""Load the campus PNG with Assignment 2's Grid2D and check it.

Usage:
    python test_campus_map.py [maps/cmu_campus.png]
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'assignment2-handout' / 'mapper_py'))

from data_structures.grid import Cell, Grid2D  # noqa: E402
from utils import png_to_grid2d  # noqa: E402


def main():
    png = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'maps' / 'cmu_campus.png'
    meta = json.loads(png.with_suffix('.json').read_text())
    res, W, H = meta['resolution_m'], meta['width_cells'], meta['height_cells']

    grid = Grid2D(res, W, H, 0.001, 0.999)
    grid = png_to_grid2d(grid, str(png))

    # Grid row 0 is the bottom (south) row; the PNG's row 0 is the top (north)
    img = np.array(Image.open(png))[::-1, :]
    # Same tests as freeQ/occupiedQ, done for every cell at once
    logodds = np.asarray(grid.data).reshape(H, W)
    occ = logodds > grid.occ_thres
    free = logodds < grid.free_thres
    for r, c in [(0, 0), (H // 2, W // 2), (H - 1, W - 1)]:
        assert occ[r, c] == grid.occupiedQ(Cell(r, c)) and free[r, c] == grid.freeQ(Cell(r, c))

    assert occ.shape == (H, W)
    assert np.array_equal(occ, img < 50), 'occupied cells do not match the PNG'
    assert np.array_equal(free, img > 250), 'free cells do not match the PNG'
    assert not np.any(occ & free)
    assert 0.05 < free.mean() < 0.6, f'unexpected free fraction {free.mean():.2f}'

    # Point-to-cell lookups go through the course's own conversion
    r, c = np.argwhere(free)[len(np.argwhere(free)) // 2]
    p = grid.cell_to_point_row_col(r, c)
    assert grid.freeQ(grid.point_to_cell(p + 0.5 * res))

    print(f'Grid {W} x {H} at {res} m loaded: {free.mean():.1%} free, '
          f'{occ.mean():.1%} occupied. Free cell ({r}, {c}) is at '
          f'({p.x:.1f} m, {p.y:.1f} m).')

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(~free, cmap='gray_r', origin='lower',
              extent=[0, W * res, 0, H * res])
    ax.set_xlabel('x (m, east)')
    ax.set_ylabel('y (m, north)')
    ax.set_title('CMU campus as a Grid2D (white = walkable). '
                 'Map data (c) OpenStreetMap contributors')
    out = png.with_name(png.stem + '_grid2d.png')
    fig.savefig(out, dpi=100, bbox_inches='tight')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
