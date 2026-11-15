"""Checks that campus_grid.py builds the same grid as Assignment 2's png_to_grid2d.

Usage:
    python test_campus_grid.py
"""

import time

import cv2
import numpy as np

import campus_grid
from data_structures.grid import Point
from utils import png_to_grid2d


def main():
    t = time.perf_counter()
    grid, meta = campus_grid.load_campus_grid()
    t_fast = time.perf_counter() - t

    t = time.perf_counter()
    ref = campus_grid.empty_like(grid)
    ref = png_to_grid2d(ref, str(campus_grid.DEFAULT_MAP))
    t_ref = time.perf_counter() - t

    assert np.allclose(grid.data, ref.data), 'grid differs from png_to_grid2d'
    print(f'Same grid as png_to_grid2d ({grid.width} x {grid.height}); '
          f'load {t_fast:.2f} s vs {t_ref:.2f} s')

    free = campus_grid.free_mask(grid)
    r, c = np.argwhere(free)[0]
    x, y = (c + 0.5) * grid.resolution, (r + 0.5) * grid.resolution
    assert campus_grid.is_free(grid, x, y)
    assert not campus_grid.is_free(grid, -1.0, -1.0), 'outside the map must not be free'

    # A ray from a sidewalk point must stop at a building or at max range
    mapper, robot_grid = campus_grid.make_mapper(grid, max_range=8.0, num_rays=36)
    ends = mapper.add_obs(Point(x, y))
    assert len(ends) > 0
    assert all(abs(e - Point(x, y)) <= 8.0 + 1e-6 for e in ends)
    assert any(abs(v) > 0 for v in robot_grid.data), 'mapper did not update the grid'
    print(f'Ray casting from ({x:.1f} m, {y:.1f} m): {len(ends)} rays, robot map updated')

    # No sidewalk (free cell) lies inside a building footprint
    buildings = campus_grid.load_buildings()
    if buildings is not None:
        inside = int((free & buildings).sum())
        assert inside == 0, f'{inside} free cells lie inside buildings'
        print('No free cells inside buildings')

    # The robot cannot use stairs: no free cell lies on a staircase
    stairs = campus_grid.load_layer(campus_grid.DEFAULT_MAP, 'stairs')
    if stairs is not None:
        on_stairs = int((free & stairs).sum())
        assert on_stairs == 0, f'{on_stairs} free cells lie on stairs'
        print(f'No free cells on stairs ({int(stairs.sum()) * grid.resolution ** 2:.0f} m2 blocked)')

    # Every delivery point is on a free cell of the main sidewalk network
    entrances = campus_grid.load_entrances()
    assert entrances, 'no entrances saved with the map; rerun make_campus_map.py'
    n, labels = cv2.connectedComponents(free.astype(np.uint8), connectivity=8)
    main = np.argmax(np.bincount(labels[free]))
    for e in entrances:
        ex, ey = e['delivery_xy']
        assert campus_grid.is_free(grid, ex, ey), f"{e['building']} drop-off is not free"
        r, c = int(ey / grid.resolution), int(ex / grid.resolution)
        assert labels[r, c] == main, f"{e['building']} drop-off is cut off"

    # Every entrance knows the floor it opens onto
    for e in entrances:
        assert isinstance(e.get('floor'), str) and e['floor'], e
        assert e.get('floor_source') in ('user', 'osm', 'plan', 'terrain'), e
        assert isinstance(e.get('floor_index'), int), e
        if e.get('building_floors'):  # named floors must be real floors of the building
            assert e['floor'] in e['building_floors'], e
            assert e['building_floors'].index(e['floor']) == e['floor_index'], e
    n = {s: sum(e['floor_source'] == s for e in entrances) for s in ('osm', 'plan', 'terrain')}
    print(f"Floors known for all {len(entrances)} entrances ({n['osm']} OSM tags, "
          f"{n['plan']} CMU floor-plan names, {n['terrain']} terrain numbers)")

    # Elevation covers the map and looks like Pittsburgh ground level
    elev, elev_res = campus_grid.load_elevation()
    assert elev is not None, 'no elevation saved with the map; rerun make_campus_map.py'
    assert abs(elev.shape[1] * elev_res - grid.width * grid.resolution) < 1.0
    assert abs(elev.shape[0] * elev_res - grid.height * grid.resolution) < 1.0
    assert 200 < elev.min() and elev.max() < 400, 'elevation out of the expected range'

    # A terrain-aware delivery between the two entrances farthest apart ends at
    # the drop-off, stays on free cells, and is never steeper than the limit
    pts = np.array([e['delivery_xy'] for e in entrances])
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    route = campus_grid.plan_route(grid, pts[i], pts[j], elevation=elev, elev_res=elev_res)
    miss = np.linalg.norm(route[-1] - pts[j])
    assert miss < 1.0, f'route ends {miss:.2f} m from the drop-off'
    assert all(campus_grid.is_free(grid, px, py) for px, py in route)
    if stairs is not None:
        rows = (route[:, 1] / grid.resolution).astype(int)
        cols = (route[:, 0] / grid.resolution).astype(int)
        assert not stairs[rows, cols].any(), 'the route goes over stairs'
    _, _, stats = campus_grid.route_profile(route, elev, elev_res)
    assert stats['max_grade'] <= campus_grid.MAX_GRADE + 0.01, stats
    n_curb = sum(e['curbside'] for e in entrances)
    print(f'{len(entrances)} entrances usable ({n_curb} curbside); delivery '
          f'{entrances[i]["building"]} -> {entrances[j]["building"]}: '
          f"{stats['length_m']:.0f} m, climb {stats['climb_m']:.1f} m, "
          f"steepest {stats['max_grade'] * 100:.1f}%, ends {miss:.2f} m from the drop-off")

    # The heuristic never overestimates: with elevation, the route cost is at
    # least the straight-line distance plus the climb weight times the net gain
    z0, z1 = campus_grid.elevation_at(
        campus_grid.smooth_elevation(elev, elev_res), elev_res, route[[0, -1]])
    lower_bound = np.linalg.norm(route[-1] - route[0]) + \
        campus_grid.CLIMB_WEIGHT * max(0.0, z1 - z0)
    cost = stats['length_m'] + campus_grid.CLIMB_WEIGHT * stats['climb_m']
    assert cost >= lower_bound - 1e-6, (cost, lower_bound)
    print('All checks passed')


if __name__ == '__main__':
    main()
