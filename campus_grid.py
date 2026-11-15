"""CMU campus map as an Assignment 2 occupancy grid.

Ties the OpenStreetMap campus PNG (from make_campus_map.py) to the
Assignment 2 mapping code:

- load_campus_grid() turns the PNG into a Grid2D (the "ground truth" world).
- make_mapper() builds the Assignment 2 Sensor, Observer and Mapper on top of
  it, so a robot can ray-cast against campus and build its own map.
- plan_route() finds a sidewalk route between two points (A*).
- load_entrances() / find_entrance() give each building's delivery points
  (just outside its doors), so a delivery is plan_route() between two of them.

Coordinates are meters from the map's south-west corner: x points east, y
points north. That is the same convention Grid2D uses (cell (0, 0) at the
origin), so points from the simulator can be passed to Grid2D directly.

Run this file for a demo delivery: the robot drives from one building's
entrance to another's, maps the way with the Assignment 2 Mapper, and the
result is saved to docs/campus_grid_demo.png.

    python campus_grid.py                      (asks for buildings and floors)
    python campus_grid.py --list
    python campus_grid.py --from "Cohon University Center" --to "Baker Hall"
    python campus_grid.py --from "Hunt Library" --to "Wean Hall" --floor 5
    python campus_grid.py --from "Cohon University Center" --to "Doherty Hall" --floor A
"""

import argparse
import heapq
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
A2_MAPPER = HERE.parent / 'assignment2-handout' / 'mapper_py'
sys.path.insert(0, str(A2_MAPPER))

from data_structures.grid import Grid2D, Point  # noqa: E402
from data_structures.observer import Observer  # noqa: E402
from data_structures.sensor import Sensor  # noqa: E402
from mapper import Mapper  # noqa: E402

DEFAULT_MAP = HERE / 'maps' / 'cmu_campus.png'

# Same probabilities as the Assignment 2 tests
MIN_CLAMP = 0.001
MAX_CLAMP = 0.999


def load_png(png_path=DEFAULT_MAP):
    """Return (gray, meta): the map as a grayscale array (row 0 = south) and its settings."""
    png_path = Path(png_path)
    meta = json.loads(png_path.with_suffix('.json').read_text())
    gray = np.array(Image.open(png_path).convert('L'))[::-1, :]
    return gray, meta


def load_layer(png_path, name):
    """A boolean layer saved next to the map, e.g. 'buildings' or 'stairs'
    (row 0 = south), or None for maps made without it."""
    path = Path(png_path).with_name(Path(png_path).stem + f'_{name}.png')
    if not path.exists():
        return None
    return np.array(Image.open(path).convert('L'))[::-1, :] > 127


def load_buildings(png_path=DEFAULT_MAP):
    """Boolean building footprints (row 0 = south), or None for maps made without them."""
    return load_layer(png_path, 'buildings')


def load_entrances(png_path=DEFAULT_MAP, reachable_only=True):
    """Delivery entrances saved by make_campus_map.py (empty list if none).

    Each entrance is a dict with 'building', 'entrance' (OSM type, e.g. main),
    'source' ('osm' or 'estimated'), 'door_xy' and 'delivery_xy' (meters, map
    coordinates). The robot should stop at delivery_xy, just outside the door.
    By default, entrances the robot cannot reach on this map are left out,
    except that a door reachable only by stairs (or otherwise cut off) is kept
    with 'curbside': True and its delivery_xy moved to the closest point the
    robot can reach ('curbside_m' away from the door).
    """
    path = Path(png_path).with_name(Path(png_path).stem + '_entrances.json')
    entrances = json.loads(path.read_text()) if path.exists() else []
    if not reachable_only:
        return entrances
    usable = []
    for e in entrances:
        if e.get('reachable', True):
            usable.append(dict(e, curbside=False))
        elif 'curbside_xy' in e:
            usable.append(dict(e, curbside=True, delivery_xy=e['curbside_xy']))
    return usable


def building_names(entrances):
    """Sorted names of the buildings that have at least one delivery entrance."""
    return sorted({e['building'] for e in entrances})


def find_entrance(entrances, building, near_xy=None, floor=None, hard=None):
    """The entrance to deliver to for a building name (case-insensitive, partial ok).

    Prefers an exact name match, then a door the robot can reach (over a
    curbside drop-off, or a door for which hard(entrance) is True, e.g. one on
    ground too steep to drive to), then (when floor is given) the door whose
    floor is closest to it, then a 'main' entrance, then the entrance closest
    to near_xy (for example, where the robot starts).
    """
    key = building.lower()
    matches = [e for e in entrances if key in e['building'].lower()]
    if not matches:
        raise ValueError(f'No entrance found for {building!r}. Buildings with entrances: '
                         + ', '.join(building_names(entrances)))
    matches = [e for e in matches if e['building'].lower() == key] or matches

    target = floor_index(matches[0], floor) if floor is not None else None

    def rank(e):
        floor_gap = (abs(e['floor_index'] - target)
                     if target is not None and e.get('floor_index') is not None else 0)
        dist = (np.hypot(e['delivery_xy'][0] - near_xy[0], e['delivery_xy'][1] - near_xy[1])
                if near_xy is not None else 0.0)
        difficult = bool(e.get('curbside')) or (hard is not None and hard(e))
        return (difficult, floor_gap, e['entrance'] != 'main', dist)

    return min(matches, key=rank)


def drivable_gap(grid, xy, elevation=None, elev_res=1.0, step_m=0.5):
    """Meters from point xy to the nearest place the robot can drive to (the
    area plan_route uses, which respects stairs and the grade limit)."""
    prep = _prepare(grid, step_m, elevation, elev_res, SMOOTH_M, MAX_GRADE)
    rr, cc = np.nonzero(prep['main'])
    rc = prep['res_c']
    return float(np.min(np.hypot(cc * rc + rc / 2 - xy[0], rr * rc + rc / 2 - xy[1])))


def floor_index(e, floor):
    """Position of a requested floor name ('A', '3', 'LL') in an entrance's
    building, comparable with e['floor_index']. Raises ValueError for a floor
    the building does not have."""
    name = str(floor).strip().upper()
    floors = e.get('building_floors')
    if floors:
        if name not in floors:
            raise ValueError(f"{e['building']} has no floor {name!r}. Its floors "
                             f"(bottom to top): {', '.join(floors)}")
        return floors.index(name)
    if name.lstrip('-').isdigit():
        return int(name) - 1
    raise ValueError(f"{e['building']} has no floor plan; use a floor number")


def floor_text(e):
    """How the entrance's floor is known, e.g. 'floor A' or 'floor ~2 (...)'."""
    if e.get('floor') is None:
        return ''
    if e.get('floor_source') == 'osm':
        return f"floor {e['floor']}"
    if e.get('floor_source') == 'user':
        return f"floor {e['floor']} (set by hand)"
    if e.get('floor_source') == 'plan':
        return f"floor ~{e['floor']} (estimated from terrain, CMU floor-plan names)"
    return f"floor ~{e['floor']} (estimated from terrain)"


def floor_label(e):
    """Short floor label for figures: 'A' when known, '~A' when estimated."""
    if e.get('floor') is None:
        return ''
    return ('' if e.get('floor_source') in ('osm', 'user') else '~') + str(e['floor'])


def load_campus_grid(png_path=DEFAULT_MAP):
    """Load the campus PNG into an Assignment 2 Grid2D.

    Gives the same result as png_to_grid2d() but fills the grid with numpy
    instead of a Python loop over every cell. The cells are stored as a numpy
    array rather than a list (Grid2D indexes and assigns them the same way),
    which keeps a 10-million-cell map at 80 MB instead of hundreds.

    Returns:
        grid: (Grid2D) occupied cells at max_clamp, free at min_clamp, unknown at 0
        meta: (dict) resolution, size and location of the map
    """
    gray, meta = load_png(png_path)
    H, W = gray.shape
    if (W, H) != (meta['width_cells'], meta['height_cells']):
        raise ValueError(f'{png_path} is {W} x {H} but its .json says '
                         f"{meta['width_cells']} x {meta['height_cells']}")

    grid = empty_like_size(meta['resolution_m'], W, H)
    # Same thresholds as Assignment 2's png_to_grid2d: < 50 occupied, > 250 free
    logodds = np.zeros((H, W))
    logodds[gray < 50] = grid.max_clamp
    logodds[gray > 250] = grid.min_clamp
    # Grid2D stores cells row by row starting at row 0 (south), like to_index()
    grid.data = logodds.ravel()
    return grid, meta


def empty_like_size(res, W, H):
    """An all-unknown Grid2D whose cells are a numpy array."""
    grid = Grid2D(res, W, H, MIN_CLAMP, MAX_CLAMP)
    grid.data = np.zeros(grid.N)
    return grid


def empty_like(grid):
    """A new, all-unknown Grid2D with the same size and resolution."""
    return empty_like_size(grid.resolution, grid.width, grid.height)


def make_mapper(gt_grid, max_range=8.0, num_rays=180):
    """Assignment 2 mapping stack on the campus.

    The Observer ray-casts against the ground-truth campus grid (the "real
    world"); the Mapper updates a separate, initially unknown grid (what the
    robot has seen so far).

    Returns:
        mapper: (Mapper) call mapper.add_obs(Point(x, y)) at each robot position
        robot_grid: (Grid2D) the map the robot builds
    """
    robot_grid = empty_like(gt_grid)
    sensor = Sensor(max_range=max_range, num_rays=num_rays)
    mapper = Mapper(robot_grid, sensor, Observer(gt_grid))
    return mapper, robot_grid


def is_free(grid, x, y):
    """True if the point (x, y) in meters is inside the map and on a free cell."""
    cell = grid.point_to_cell(Point(x, y))
    return grid.inQ(cell) and grid.freeQ(cell)


def free_mask(grid):
    """Boolean (height, width) array of free cells, row 0 = south."""
    return np.asarray(grid.data).reshape(grid.height, grid.width) < grid.free_thres


def load_elevation(png_path=DEFAULT_MAP):
    """Ground elevation saved by make_campus_map.py, or (None, None) if there is none.

    Returns:
        elev: (H, W) float32 meters above sea level, row 0 = south (like Grid2D)
        res: meters per elevation sample
    """
    png_path = Path(png_path)
    path = png_path.with_name(png_path.stem + '_elevation.npy')
    if not path.exists():
        return None, None
    meta = json.loads(png_path.with_suffix('.json').read_text())
    return np.load(path)[::-1].astype(np.float32), meta.get('elevation_res_m', 1.0)


def elevation_at(elev, elev_res, xy):
    """Bilinear elevation (m) at points xy, an (N, 2) array in map meters."""
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    c = np.clip(xy[:, 0] / elev_res - 0.5, 0, elev.shape[1] - 1.001)
    r = np.clip(xy[:, 1] / elev_res - 0.5, 0, elev.shape[0] - 1.001)
    c0, r0 = c.astype(int), r.astype(int)
    fc, fr = c - c0, r - r0
    return ((1 - fr) * ((1 - fc) * elev[r0, c0] + fc * elev[r0, c0 + 1])
            + fr * ((1 - fc) * elev[r0 + 1, c0] + fc * elev[r0 + 1, c0 + 1]))


# Route cost settings (see plan_route)
CLIMB_WEIGHT = 10.0     # each meter climbed costs like 10 m of flat driving
DESCENT_WEIGHT = 2.0    # going down costs a little too (braking, stability)
RAMP_GRADE = 0.083      # 1:12, the steepest ramp allowed for wheelchairs (ADA)
MAX_GRADE = 0.20        # steeper than this is a wall, a drop or stairs: forbidden
STEEP_WEIGHT = 20.0     # extra cost per unit of grade above RAMP_GRADE
CENTER_WEIGHT = 0.5     # extra cost for driving within 1 m of the path edge


SMOOTH_M = 2.0          # terrain smoothing before measuring slope (removes curbs)

_PREPARED = {}


def _prepare(grid, step_m, elevation, elev_res, smooth_m, max_grade):
    """Coarse planning grid, cached per map and settings.

    Returns a dict with the drivable cells ('main'), their smoothed elevation
    ('z'), the edge penalty ('edge') and the coarse cell size ('res_c').
    """
    key = (id(grid), step_m, id(elevation), elev_res, smooth_m, max_grade)
    if key in _PREPARED:
        return _PREPARED[key]

    free = free_mask(grid).astype(np.uint8)
    # Distance to the nearest non-free cell, in meters (used to stay centered)
    clearance = cv2.distanceTransform(free, cv2.DIST_L2, 5) * grid.resolution
    k = max(1, int(round(step_m / grid.resolution)))
    coarse = clearance[k // 2::k, k // 2::k]
    ok = coarse > 0.3                       # at least 30 cm from the path edge
    Hc, Wc = ok.shape
    res_c = k * grid.resolution

    if elevation is not None:
        smooth = smooth_elevation(elevation, elev_res, smooth_m)
        cx, cy = np.meshgrid(np.arange(Wc) * res_c + res_c / 2, np.arange(Hc) * res_c + res_c / 2)
        z = elevation_at(smooth, elev_res, np.c_[cx.ravel(), cy.ravel()]).reshape(Hc, Wc)
    else:
        z = np.zeros((Hc, Wc))

    # Largest piece of sidewalk the robot can drive around in: 8-connected
    # through steps no steeper than max_grade (breadth-first search)
    label = np.zeros((Hc, Wc), dtype=np.int32)
    sizes = [0]
    sqrt2 = np.sqrt(2.0)
    for start in zip(*np.nonzero(ok)):
        if label[start]:
            continue
        lab = len(sizes)
        label[start] = lab
        stack, count = [start], 0
        while stack:
            r, c = stack.pop()
            count += 1
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if ((dr or dc) and 0 <= nr < Hc and 0 <= nc < Wc and ok[nr, nc]
                            and not label[nr, nc]
                            and abs(z[nr, nc] - z[r, c]) <= max_grade * res_c
                            * (sqrt2 if dr and dc else 1.0)):
                        label[nr, nc] = lab
                        stack.append((nr, nc))
        sizes.append(count)
    main = label == int(np.argmax(sizes))

    prep = {'main': main, 'z': z, 'res_c': res_c,
            'edge': CENTER_WEIGHT * np.clip(1.0 - coarse, 0.0, 1.0)}
    _PREPARED[key] = prep
    return prep


def smooth_elevation(elevation, elev_res, smooth_m=SMOOTH_M):
    """Gaussian-smoothed terrain, so a curb or a 1 m data bump does not read as
    a steep slope while real hills stay."""
    if elevation is None or smooth_m <= 0:
        return elevation
    return cv2.GaussianBlur(elevation, (0, 0), smooth_m / elev_res)


def plan_route(grid, start_xy, goal_xy, step_m=0.5, elevation=None, elev_res=1.0,
               climb_weight=CLIMB_WEIGHT, descent_weight=DESCENT_WEIGHT,
               max_grade=MAX_GRADE, smooth_m=SMOOTH_M):
    """A* along the sidewalks from start to goal.

    Plans on a coarser copy of the grid (step_m per cell) to stay fast, on the
    largest connected piece of sidewalk, from the free cells nearest to the
    requested start and goal.

    Cost of one step of horizontal length L (meters) that changes height by dz:

        L * (1 + CENTER_WEIGHT * edge closeness)        distance, keep off the edge
        + L * STEEP_WEIGHT * (grade - RAMP_GRADE)       only above ramp steepness
        + climb_weight * max(dz, 0)                     energy to climb
        + descent_weight * max(-dz, 0)                  care going down

    and steps steeper than max_grade are not allowed. Slopes are measured on
    terrain smoothed over smooth_m meters (see smooth_elevation). Without an
    elevation grid only the first line applies.

    Heuristic (admissible, so A* still finds the cheapest route):

        h = octile distance to the goal + climb_weight * max(0, z_goal - z)

    Every step costs at least its length, the octile distance is the shortest
    possible 8-connected path, and any route must climb at least the net height
    gain, so h never overestimates the remaining cost.

    Start and goal snap to the nearest cell the robot can drive to: the largest
    connected piece of sidewalk, joined only by steps within the grade limit.
    route[0] and route[-1] show where that is; compare them with start_xy and
    goal_xy to see how far from a door the robot actually gets.

    Returns:
        route: (N, 2) array of (x, y) points in meters, step_m apart
    """
    prep = _prepare(grid, step_m, elevation, elev_res, smooth_m, max_grade)
    main, z, edge, res_c = prep['main'], prep['z'], prep['edge'], prep['res_c']
    Hc, Wc = main.shape
    if elevation is None:
        climb_weight = descent_weight = 0.0
    rr, cc = np.nonzero(main)

    def nearest(xy):
        d = (cc * res_c + res_c / 2 - xy[0]) ** 2 + (rr * res_c + res_c / 2 - xy[1]) ** 2
        i = np.argmin(d)
        return int(rr[i]), int(cc[i])

    s, g = nearest(start_xy), nearest(goal_xy)
    zg = z[g]
    sqrt2 = np.sqrt(2.0)

    def heuristic(cell):
        dr, dc = abs(g[0] - cell[0]), abs(g[1] - cell[1])
        octile = (max(dr, dc) + (sqrt2 - 1.0) * min(dr, dc)) * res_c
        return octile + climb_weight * max(0.0, zg - z[cell])

    cost = {s: 0.0}
    came = {s: None}
    queue = [(heuristic(s), s)]
    closed = set()
    while queue:
        _, cur = heapq.heappop(queue)
        if cur == g:
            break
        if cur in closed:
            continue
        closed.add(cur)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nb = (cur[0] + dr, cur[1] + dc)
                if not ((dr or dc) and 0 <= nb[0] < Hc and 0 <= nb[1] < Wc and main[nb]):
                    continue
                L = (sqrt2 if dr and dc else 1.0) * res_c
                dz = z[nb] - z[cur]
                grade = abs(dz) / L
                if grade > max_grade:
                    continue
                step = (L * (1.0 + edge[nb])
                        + L * STEEP_WEIGHT * max(0.0, grade - RAMP_GRADE) * (elevation is not None)
                        + climb_weight * max(dz, 0.0) + descent_weight * max(-dz, 0.0))
                c = cost[cur] + step
                if c < cost.get(nb, np.inf):
                    cost[nb] = c
                    came[nb] = cur
                    heapq.heappush(queue, (c + heuristic(nb), nb))
    if g not in came:  # cannot happen: s and g are in the same connected piece
        raise RuntimeError('No sidewalk route between start and goal')

    cells = [g]
    while came[cells[-1]] is not None:
        cells.append(came[cells[-1]])
    cells = np.array(cells[::-1], dtype=float)
    return cells[:, ::-1] * res_c + res_c / 2


def route_profile(route, elevation, elev_res, smooth_m=SMOOTH_M):
    """Distance along the route, elevation, and summary numbers for printing.

    Uses the same smoothed terrain as plan_route, so the steepest grade here is
    the one the planner limited."""
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
    z = elevation_at(smooth_elevation(elevation, elev_res, smooth_m), elev_res, route)
    dz = np.diff(z)
    # Grade over ~2 m windows, so single-sample noise does not dominate
    w = 4
    grade = np.abs(z[w:] - z[:-w]) / np.maximum(d[w:] - d[:-w], 1e-6) if len(z) > w else [0.0]
    stats = {'length_m': float(d[-1]), 'climb_m': float(dz[dz > 0].sum()),
             'descent_m': float(-dz[dz < 0].sum()), 'max_grade': float(np.max(grade))}
    return d, z, stats


def demo(png_path=DEFAULT_MAP, pickup='Cohon University Center', dropoff='Baker Hall',
         out=HERE / 'docs' / 'campus_grid_demo.png', floor=None, from_floor=None):
    """Deliver from one building's entrance to another's, mapping on the way."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    t0 = time.perf_counter()
    gt, meta = load_campus_grid(png_path)
    res = gt.resolution
    W_m, H_m = gt.width * res, gt.height * res
    free = free_mask(gt)
    print(f'Loaded {gt.width} x {gt.height} grid at {res} m '
          f'({W_m:.0f} x {H_m:.0f} m, {free.mean():.1%} free) '
          f'in {time.perf_counter() - t0:.1f} s')

    entrances = load_entrances(png_path)
    if not entrances:
        raise SystemExit('This map has no entrances saved; rerun make_campus_map.py')
    # Leave by the pickup door nearest the destination, arrive by the nearest door
    elev, elev_res = load_elevation(png_path)

    def hard(e):  # a door on ground too steep to drive up to
        return drivable_gap(gt, e['delivery_xy'], elev, elev_res) > 2.0

    dest = find_entrance(entrances, dropoff, floor=floor, hard=hard)['delivery_xy']
    start = find_entrance(entrances, pickup, near_xy=dest, floor=from_floor, hard=hard)
    goal = find_entrance(entrances, dropoff, near_xy=start['delivery_xy'], floor=floor,
                         hard=hard)
    print(f"Delivery: {start['building']} -> {goal['building']} "
          f"({goal['entrance']} entrance, {goal['source']})")
    print(f"  Picks up at {start['building']}, {floor_text(start) or 'floor unknown'}")
    print(f"  Delivers to {goal['building']}, {floor_text(goal) or 'floor unknown'}")
    if from_floor is not None and start.get('floor_index') is not None:
        gap = abs(start['floor_index'] - floor_index(start, from_floor))
        if gap:
            print(f"  No pickup door on floor {str(from_floor).upper()} the robot can use; "
                  f"picking up on floor {start['floor']}, {gap} floor(s) away by elevator")
    floor_gap = None
    if floor is not None and goal.get('floor_index') is not None:
        floor = str(floor).strip().upper()
        floor_gap = abs(goal['floor_index'] - floor_index(goal, floor))
    if floor_gap:
        mine = [e for e in entrances
                if e['building'] == goal['building'] and e.get('floor') is not None]

        def names(es):
            return [e['floor'] for e in sorted(es, key=lambda e: e['floor_index'])]
        usable = list(dict.fromkeys(names([e for e in mine
                                           if not e.get('curbside') and not hard(e)])))
        unusable = [f for f in dict.fromkeys(names(mine)) if f not in usable]
        print(f"  No entrance on floor {floor} the robot can use. Usable doors are on floor"
              f"{'s' if len(usable) > 1 else ''} {', '.join(usable)}"
              + (f" (doors on floor {', '.join(unusable)} need stairs or are on "
                 f"ground too steep to drive)" if unusable else '')
              + f"; delivering on floor {goal['floor']}, so the last "
              f"{floor_gap} floor(s) need an elevator")
    for e, role in ((start, 'Pickup'), (goal, 'Drop-off')):
        if e.get('curbside'):
            print(f"  {role} door at {e['building']} is reachable only by stairs (or is cut "
                  f"off), so the robot stops curbside, {e['curbside_m']:.0f} m from it")

    t0 = time.perf_counter()
    route = plan_route(gt, start['delivery_xy'], goal['delivery_xy'],
                       elevation=elev, elev_res=elev_res)
    t_plan = time.perf_counter() - t0
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
    miss = np.hypot(*(route[-1] - np.array(goal['delivery_xy'])))
    print(f'Planned with A* in {t_plan:.1f} s; the route ends {miss:.2f} m from the drop-off point')
    for e, end, role in ((start, route[0], 'pickup'), (goal, route[-1], 'drop-off')):
        gap = np.hypot(*(end - np.array(e['delivery_xy'])))
        if gap > 2.0:
            print(f"  Note: the {role} point at {e['building']} is on ground steeper than "
                  f'{MAX_GRADE:.0%}; the closest place the robot can drive to is {gap:.0f} m away')
    flat = None
    if elev is not None:
        # Same planner ignoring elevation, to show what the terrain changes
        flat = plan_route(gt, start['delivery_xy'], goal['delivery_xy'])
        for name, r in (('terrain-aware', route), ('shortest (ignores slope)', flat)):
            _, _, s = route_profile(r, elev, elev_res)
            print(f"  {name:25s} {s['length_m']:5.0f} m, climb {s['climb_m']:4.1f} m, "
                  f"descent {s['descent_m']:4.1f} m, steepest {s['max_grade'] * 100:4.1f}%")
    else:
        print(f'  {seg[-1]:.0f} m (no elevation data: rerun make_campus_map.py)')

    # Drive it, sensing every 1 m with the Assignment 2 Sensor/Observer/Mapper
    stops = np.interp(np.arange(0, seg[-1], 1.0), seg, np.arange(len(route)))
    poses = [route[int(round(i))] for i in stops]
    mapper, robot_grid = make_mapper(gt, max_range=8.0, num_rays=180)
    t0 = time.perf_counter()
    for p in poses:
        mapper.add_obs(Point(float(p[0]), float(p[1])))
    print(f'Mapped from {len(poses)} positions in {time.perf_counter() - t0:.1f} s')

    # --- Figure -------------------------------------------------------------
    buildings = load_buildings(png_path)
    rgb = np.empty(free.shape + (3,), dtype=np.uint8)
    rgb[:] = (38, 52, 44)                          # grass, roads, everything else
    if buildings is not None:
        rgb[buildings] = (150, 158, 172)           # buildings
    rgb[free] = (244, 241, 234)                    # sidewalks

    # Square zoom window around the route, kept inside the map
    mid = (route.min(axis=0) + route.max(axis=0)) / 2
    side = min((route.max(axis=0) - route.min(axis=0)).max() + 24, W_m, H_m)
    lo = np.clip(mid - side / 2, 0, [W_m - side, H_m - side])
    hi = lo + side
    r0, r1 = int(lo[1] / res), int(np.ceil(hi[1] / res))
    c0, c1 = int(lo[0] / res), int(np.ceil(hi[0] / res))
    lodds = np.asarray(robot_grid.data).reshape(gt.height, gt.width)[r0:r1, c0:c1]
    prob = 1.0 / (1.0 + np.exp(-lodds))            # occupancy probability
    known = lodds != 0
    zoom = np.empty(lodds.shape + (3,), dtype=np.uint8)
    zoom[:] = (88, 98, 112)                        # never observed
    shade = (244 * (1 - prob[known]) + 20 * prob[known]).astype(np.uint8)
    zoom[known] = np.stack([shade] * 3, axis=1)
    zext = [c0 * res, c1 * res, r0 * res, r1 * res]

    stairs = load_layer(png_path, 'stairs')
    if stairs is not None:
        rgb[stairs & ~free] = (214, 69, 80)        # stairs (the robot cannot use them)

    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(2, 2, width_ratios=[W_m / H_m, 1.0], height_ratios=[2.2, 1])
    axs = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1])]
    ax = axs[0]
    ax.imshow(rgb, origin='lower', extent=[0, W_m, 0, H_m], interpolation='antialiased')
    if elev is not None:
        levels = np.arange(np.floor(elev.min() / 2) * 2, elev.max() + 2, 2.0)
        cs = ax.contour(elev, levels=levels, origin='lower', colors='#C9D3E3',
                        linewidths=0.4, alpha=0.6,
                        extent=[0, elev.shape[1] * elev_res, 0, elev.shape[0] * elev_res])
        ax.clabel(cs, levels[::5], fontsize=7, fmt='%d m')
    if flat is not None:
        ax.plot(*flat.T, color='#9AA5B1', lw=2, ls='--', label='shortest (ignores slope)')
    ax.plot(*route.T, color='#E8833A', lw=2.5, label='terrain-aware route')
    ax.plot([], [], 's', color='#D64550', label='stairs (not usable)')
    ent = np.array([e['delivery_xy'] for e in entrances])
    ax.scatter(*ent.T, s=16, c='#4CC9F0', edgecolors='#14213D', linewidths=0.5,
               zorder=3, label='building entrances (number = floor, ~ = estimated)')
    for e in entrances:  # floor number beside every entrance
        if e.get('floor') is not None:
            ax.annotate(floor_label(e),
                        e['delivery_xy'], xytext=(3, 3), textcoords='offset points',
                        fontsize=7, color='#DDEBF7', zorder=4)
    ax.plot(*route[0], 'o', color='#2BB673', ms=11, mec='white', mew=1.5,
            label=f"pickup: {start['building']}")
    callout = dict(boxstyle='round,pad=0.35', fc='#14213D', ec='white', lw=1, alpha=0.9)
    arrow = dict(arrowstyle='-', color='white', lw=1)
    for e, end, color, role in ((start, route[0], '#2BB673', 'Pickup'),
                                (goal, np.array(goal['delivery_xy']), '#E63946', 'Drop-off')):
        # Put the label on the side of the map with more room
        dx = 40 if end[0] < W_m / 2 else -40
        dy = 30 if end[1] < H_m / 2 else -30
        ax.annotate(f"{role}: {e['building']}\n{floor_text(e) or 'floor unknown'}",
                    end, xytext=(dx, dy), textcoords='offset points',
                    ha='left' if dx > 0 else 'right', fontsize=10, color='white',
                    bbox=dict(callout, ec=color, lw=1.5), arrowprops=arrow, zorder=6)
    ax.plot(*goal['delivery_xy'], '*', color='#E63946', ms=18, mec='white', mew=1,
            label=f"drop-off: {goal['building']}"
                  + (f" (floor {goal['floor']})" if goal.get('floor') is not None else ''))
    ax.add_patch(Rectangle(lo, *(hi - lo), fill=False, ec='#E8833A', lw=1.5, ls='--'))
    ax.set_title(f'Delivery on CMU campus ({gt.width} x {gt.height} Grid2D, {res} m cells)',
                 fontsize=14)
    # Legend under the map, so it never covers the pickup or drop-off
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=10,
              frameon=False)

    ax = axs[1]
    ax.imshow(zoom, origin='lower', extent=zext, interpolation='nearest')
    ax.plot(*route.T, color='#E8833A', lw=2, alpha=0.9, label='route driven')
    ax.add_patch(Circle(poses[-1], 8.0, fill=False, ec='#E8833A', lw=1.5, ls='--',
                        label='8 m sensor range'))
    ax.plot(*poses[-1], 's', color='#E8833A', ms=10, mec='white', mew=1.5)
    ax.plot(*goal['delivery_xy'], '*', color='#E63946', ms=18, mec='white', mew=1,
            label='drop-off at the door'
                  + (f" (floor {goal['floor']})" if goal.get('floor') is not None else ''))
    if floor_gap:
        ax.text(0.02, 0.98, f"Requested floor {floor}: no door the robot can use there.\n"
                            f"Delivered on floor {goal['floor']}; "
                            f"{floor_gap} floor(s) by elevator.",
                transform=ax.transAxes, va='top', fontsize=11, color='white',
                bbox=dict(boxstyle='round,pad=0.4', fc='#A04D10', ec='none', alpha=0.9))
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_title('Map built by the Assignment 2 Mapper while driving\n'
                 '(white = free, black = occupied, slate = unseen)', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    for ax in axs:
        ax.set_xlabel('x (m, east)')
        ax.set_aspect('equal')
    axs[0].set_ylabel('y (m, north)')

    ax = fig.add_subplot(gs[1, 1])
    if elev is not None:
        for r, color, ls, name in ((route, '#E8833A', '-', 'terrain-aware'),
                                   (flat, '#9AA5B1', '--', 'shortest (ignores slope)')):
            d, zz, s = route_profile(r, elev, elev_res)
            ax.plot(d, zz, color=color, ls=ls, lw=2,
                    label=f"{name}: {s['length_m']:.0f} m, +{s['climb_m']:.1f} m climb")
        ax.set_ylabel('elevation (m)')
        ax.legend(loc='best', fontsize=10)
    ax.set_xlabel('distance along route (m)')
    ax.set_title('Elevation profile (USGS 3DEP)', fontsize=13)
    ax.grid(alpha=0.3)
    fig.text(0.99, 0.01, 'Map data © OpenStreetMap contributors', ha='right', fontsize=9,
             color='#5B6576')
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches='tight')  # keeps the legend under the map
    print(f'Saved {out}')


def describe_building(name, entrances):
    """One line about a building's doors and floors, e.g. for --list."""
    mine = sorted((e for e in entrances if e['building'] == name and e.get('floor')),
                  key=lambda e: e['floor_index'])
    doors = list(dict.fromkeys(floor_label(e) for e in mine))
    plan = next((e['building_floors'] for e in mine if e.get('building_floors')), None)
    return (f'{name}  ({len(mine)} entrance{"s" if len(mine) != 1 else ""}'
            + (f"; doors on floor{'s' if len(doors) > 1 else ''} {', '.join(doors)}"
               if doors else '')
            + (f"; all floors {', '.join(plan)}" if plan else '') + ')')


def ask_building(prompt, names):
    """Ask for a building by its number in the list or (part of) its name."""
    while True:
        answer = input(prompt).strip()
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            return names[int(answer) - 1]
        matches = [n for n in names if answer.lower() in n.lower()] if answer else []
        exact = [n for n in matches if n.lower() == answer.lower()]
        if exact or len(matches) == 1:
            return (exact or matches)[0]
        if matches:
            print('  More than one building matches: ' + '; '.join(matches)
                  + '. Type more of the name or its number.')
        else:
            print(f'  No building matches {answer!r}. Type a number from the list '
                  f'or part of a name.')


def ask_floor(prompt, building, entrances):
    """Ask for a floor of a building; Enter means any floor (returns None)."""
    sample = next(e for e in entrances if e['building'] == building)
    while True:
        answer = input(prompt).strip()
        if not answer:
            return None
        try:
            floor_index(sample, answer)
            return answer.upper()
        except ValueError as err:
            print(f'  {err}')


def interactive(png_path=DEFAULT_MAP):
    """Ask in the terminal for pickup and drop-off buildings and floors, then deliver."""
    entrances = load_entrances(png_path)
    if not entrances:
        raise SystemExit('This map has no entrances saved; rerun make_campus_map.py')
    names = building_names(entrances)
    print('Buildings the robot can deliver between (~ = floor estimated):')
    for i, name in enumerate(names, 1):
        print(f'  {i:2d}. {describe_building(name, entrances)}')
    print('Type a number or part of a name. Floors are optional: press Enter to skip.\n')
    while True:
        pickup = ask_building('Pickup building: ', names)
        from_floor = ask_floor(f'Pickup floor at {pickup} (optional, Enter to skip): ',
                               pickup, entrances)
        dropoff = ask_building('Drop-off building: ', names)
        floor = ask_floor(f'Drop-off floor at {dropoff} (optional, Enter to skip): ',
                          dropoff, entrances)
        print()
        demo(png_path, pickup, dropoff, floor=floor, from_floor=from_floor)
        if input('\nAnother delivery? [y/N] ').strip().lower() not in ('y', 'yes'):
            return
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo delivery between two CMU buildings. Run with no options to be '
                    'asked for the buildings and floors.')
    parser.add_argument('--from', dest='pickup', default=None,
                        help='pickup building (partial names work)')
    parser.add_argument('--to', dest='dropoff', default=None,
                        help='drop-off building (partial names work)')
    parser.add_argument('--floor', default=None,
                        help="floor to deliver to, as CMU names it (e.g. 5, A, LL): picks the "
                             "drop-off door on that floor, or the closest floor the robot "
                             "can reach")
    parser.add_argument('--from-floor', default=None,
                        help='floor to pick up from (e.g. 1, A, LL), chosen the same way')
    parser.add_argument('--map', type=Path, default=DEFAULT_MAP)
    parser.add_argument('--list', action='store_true',
                        help='list the buildings you can deliver to and exit')
    args = parser.parse_args()
    if args.list:
        ents = load_entrances(args.map)
        for name in building_names(ents):
            print(describe_building(name, ents))
    elif args.pickup is None and args.dropoff is None:
        try:
            interactive(args.map)
        except (KeyboardInterrupt, EOFError):
            print('\nBye.')
    else:
        if args.pickup is None or args.dropoff is None:
            parser.error('give both --from and --to, or neither to be asked')
        try:
            demo(args.map, args.pickup, args.dropoff, floor=args.floor,
                 from_floor=args.from_floor)
        except ValueError as err:  # e.g. a floor the building does not have
            raise SystemExit(str(err))
