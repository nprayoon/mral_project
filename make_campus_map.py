"""Build a black-and-white occupancy map PNG of CMU's campus from OpenStreetMap.

White pixels are walkable (footways, paths, sidewalks); black pixels are
everything else (buildings, roads, grass). The PNG is in the format that
Assignment 2's png_to_grid2d() expects: < 50 means occupied, > 250 means free.

Building entrances (delivery destinations) are saved next to the map in
<name>_entrances.json. Entrances come from OpenStreetMap where they are mapped;
a building with none gets one estimated at the wall closest to a sidewalk.
Each door is joined to the nearest sidewalk by a short carved doorstep path,
and its delivery point sits just outside the door.

Map data (c) OpenStreetMap contributors, ODbL.

Usage:
    python make_campus_map.py                      # default CMU crop
    python make_campus_map.py --width-m 300 --res 0.2
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import osmnx as ox
from PIL import Image, ImageDraw
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon, box
from shapely.ops import nearest_points, unary_union
from shapely.prepared import prep

# Center of the default crop: CMU Pittsburgh campus, sized to take in Hunt Library
# (south), Tepper (north-west), Scaife, Roberts and Hamerschlag Halls (west)
# and Margaret Morrison (east).
CMU_LAT = 40.44323
CMU_LON = -79.94445

HERE = Path(__file__).resolve().parent

# Keep the tags that say whether a path under a building is an outdoor covered
# walkway (usable) or an indoor corridor (not). Parsing only: cached downloads
# are reused.
for _tag_name in ('covered', 'indoor', 'layer', 'level', 'tunnel', 'ramp', 'ramp:wheelchair'):
    if _tag_name not in ox.settings.useful_tags_way:
        ox.settings.useful_tags_way = list(ox.settings.useful_tags_way) + [_tag_name]

# Every crop that fits inside this radius reuses one cached download, so trying
# different crop sizes does not hit the Overpass server again. (It equals the
# radius the default 605 x 570 m crop asks for.)
MIN_RADIUS_M = 0.5 * (605.0 ** 2 + 570.0 ** 2) ** 0.5 + 50.0

# Public Overpass servers, tried in order. The main one is often overloaded.
OVERPASS_URLS = [
    'https://overpass-api.de/api',
    'https://overpass.kumi.systems/api',
    'https://overpass.private.coffee/api',
]


def with_retries(query, attempts=4, first_wait_s=5.0):
    """Run an OSMnx query, cycling through Overpass servers with exponential backoff."""
    wait = first_wait_s
    for attempt in range(attempts):
        for url in OVERPASS_URLS:
            ox.settings.overpass_url = url
            # Mirrors do not all serve the /status endpoint used for rate limiting
            ox.settings.overpass_rate_limit = url == OVERPASS_URLS[0]
            try:
                return query()
            except Exception as e:  # server busy, timeout, bad gateway, ...
                print(f'  {url} failed ({type(e).__name__}): {str(e).splitlines()[0][:100]}')
        if attempt < attempts - 1:
            print(f'  all servers busy, retrying in {wait:.0f} s')
            time.sleep(wait)
            wait *= 2
    raise RuntimeError('Every Overpass server failed. Try again in a few minutes.')


def download(lat, lon, radius_m):
    """Download the walking network and building footprints around a point."""
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(HERE / 'cache')
    radius_m = max(radius_m, MIN_RADIUS_M)

    # simplify=False keeps every OSM way separate, so a flight of stairs is not
    # merged into the footpath it joins (same download either way)
    graph = with_retries(lambda: ox.graph_from_point(
        (lat, lon), dist=radius_m, network_type='walk', simplify=False))
    paths = ox.graph_to_gdfs(graph, nodes=False)
    buildings = with_retries(lambda: ox.features_from_point(
        (lat, lon), tags={'building': True}, dist=radius_m))
    # Keep only polygon footprints (some building tags are on points)
    buildings = buildings[buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    entrances = with_retries(lambda: ox.features_from_point(
        (lat, lon), tags={'entrance': True, 'door': True}, dist=radius_m))
    entrances = entrances[entrances.geometry.type == 'Point']
    return paths, buildings, entrances


ELEVATION_URL = ('https://elevation.nationalmap.gov/arcgis/rest/services/'
                 '3DEPElevation/ImageServer/exportImage')


def download_elevation(origin, width_m, height_m, crs, res_m=1.0, attempts=4):
    """Bare-earth ground elevation (meters) from the USGS 3DEP service.

    Returns a float32 array covering the map at res_m per pixel, row 0 = north
    (the same orientation as the map PNG). 3DEP has 1 m lidar data for
    Pittsburgh; buildings are removed, so this is the ground the paths are on.
    """
    import io
    import requests

    x0, y0 = origin
    epsg = crs.to_epsg() if hasattr(crs, 'to_epsg') else int(str(crs).split(':')[-1])
    params = {
        'bbox': f'{x0},{y0},{x0 + width_m},{y0 + height_m}',
        'bboxSR': epsg, 'imageSR': epsg,
        'size': f'{int(round(width_m / res_m))},{int(round(height_m / res_m))}',
        'format': 'tiff', 'pixelType': 'F32',
        'interpolation': 'RSP_BilinearInterpolation', 'f': 'image',
    }
    wait = 5.0
    for attempt in range(attempts):
        try:
            r = requests.get(ELEVATION_URL, params=params, timeout=180)
            if r.ok and r.headers.get('content-type', '').startswith('image/tiff'):
                return np.array(Image.open(io.BytesIO(r.content)), dtype=np.float32)
            print(f'  elevation service answered {r.status_code}')
        except requests.RequestException as e:
            print(f'  elevation service failed ({type(e).__name__})')
        if attempt < attempts - 1:
            time.sleep(wait)
            wait *= 2
    raise RuntimeError('Could not download elevation from USGS 3DEP. Try again later.')


def to_metric(paths, buildings, entrances, lat, lon):
    """Project everything to UTM (meters) and return the projected center."""
    paths = ox.projection.project_gdf(paths)
    buildings = buildings.to_crs(paths.crs)
    entrances = entrances.to_crs(paths.crs)
    center = gpd.GeoSeries([Point(lon, lat)], crs='EPSG:4326').to_crs(paths.crs)[0]
    return paths, buildings, entrances, center


# Entrances a delivery robot should not use
SKIP_ENTRANCE = {'no', 'emergency', 'exit', 'service', 'staircase', 'garage'}
SKIP_ACCESS = {'no', 'private'}


def _tag(row, key):
    value = row.get(key)
    return value if isinstance(value, str) else ''


def _walk_points(walk, center, radius, spacing=0.5):
    """Points every `spacing` m along the walking network within `radius` of center,
    nearest first."""
    near = walk.intersection(center.buffer(radius))
    parts = getattr(near, 'geoms', [near])
    pts = []
    for line in parts:
        if isinstance(line, LineString) and line.length > 0:
            n = max(2, int(line.length / spacing) + 1)
            pts += [line.interpolate(t) for t in np.linspace(0.0, line.length, n)]
    # The exact nearest point first, so a door right on a path joins it directly
    pts.append(nearest_points(center, walk)[1])
    return sorted(pts, key=center.distance)


FLOOR_HEIGHT_M = 4.0    # typical floor-to-floor height of a campus building


def _ground_z(elev, dem_res, height_m, x, y):
    """Ground elevation at map point(s) (x, y), nearest elevation sample."""
    col = np.clip((np.asarray(x) / dem_res).astype(int), 0, elev.shape[1] - 1)
    row = np.clip(((height_m - np.asarray(y)) / dem_res).astype(int), 0, elev.shape[0] - 1)
    return elev[row, col]


def _building_base(poly, elev, dem_res, origin, height_m):
    """Lowest ground around a building (5th percentile of a ring 1-3 m outside
    its walls): the ground its first floor is on."""
    import shapely
    ring = poly.buffer(3.0).difference(poly.buffer(1.0))
    x0, y0, x1, y1 = ring.bounds
    xs, ys = np.meshgrid(np.arange(x0, x1, dem_res), np.arange(y0, y1, dem_res))
    keep = shapely.contains_xy(ring, xs.ravel(), ys.ravel())
    if not keep.any():
        return None
    z = _ground_z(elev, dem_res, height_m, xs.ravel()[keep] - origin[0],
                  ys.ravel()[keep] - origin[1])
    return float(np.percentile(z, 5))


def _floor_tag(value):
    """Floor count from a numeric OSM tag such as building:levels, if numeric."""
    try:
        return max(1, int(round(float(str(value).split(';')[0]))))
    except ValueError:
        return None


def _floor_name(value):
    """Floor name from an OSM door tag ('A', 'LL', '5'), or None if missing."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).split(';')[0].strip()
    if not s or s.lower() == 'nan':
        return None
    try:
        return str(int(round(float(s))))   # '1.0' -> '1'
    except ValueError:
        return s.upper()


def load_floor_plans(path=HERE / 'maps' / 'cmu_floor_plans.json'):
    """Official floor lists, keyed by the building name OSM uses.

    Returns {osm name: (plan name, main floors bottom to top)}; empty if the
    file is missing.
    """
    path = Path(path)
    if not path.exists():
        return {}
    plans = json.loads(path.read_text())['buildings']
    return {osm: (name, p['main_floors']) for name, p in plans.items() for osm in p['osm_names']}


def apply_floor_overrides(records, origin, path=HERE / 'maps' / 'floor_overrides.json'):
    """Set door floors by hand from floor_overrides.json (see its _about).

    A matched door gets 'floor_tag' = the given floor and 'floor_tag_source' =
    'user'. Returns the overrides that matched no door, so they can be reported.
    """
    path = Path(path)
    if not path.exists():
        return []
    unmatched = []
    for o in json.loads(path.read_text()).get('doors', []):
        group = [r for r in records if r['building'] == o['building']]
        hit = None
        if o.get('entrance_osm_id'):
            hit = next((r for r in group if r.get('entrance_osm_id') == o['entrance_osm_id']), None)
        elif o.get('utm_xy'):
            x, y = o['utm_xy'][0] - origin[0], o['utm_xy'][1] - origin[1]
            near = sorted(group, key=lambda r: np.hypot(r['door_xy'][0] - x, r['door_xy'][1] - y))
            if near and np.hypot(near[0]['door_xy'][0] - x, near[0]['door_xy'][1] - y) <= 10.0:
                hit = near[0]
        if hit is None:
            unmatched.append(o)
            continue
        hit['floor_tag'] = str(o['floor']).strip().upper()
        hit['floor_tag_source'] = 'user'
        if o.get('note'):
            hit['floor_note'] = o['note']
    return unmatched


def name_floors(records, plans):
    """Give every entrance the name of the floor it opens onto.

    - A door whose floor is set by hand (apply_floor_overrides) or tagged in
      OSM (level:ref like 'A', or level like '5') uses that name:
      floor_source 'user' or 'osm'.
    - Otherwise, for a building with an official floor list, the door's
      terrain height (record['terrain_floor'], 1 = lowest ground around the
      building) is counted along that list from the known door of the same
      building whose ground height is closest, or, with no known door, from
      the bottom of the list: floor_source 'plan'.
    - Otherwise the terrain count is used as a number: floor_source 'terrain'.

    Adds 'floor' (name), 'floor_index' (position in the building's floor list,
    for comparing floors), 'building_floors' and 'floor_plan'. Returns
    {building: number of known doors used to calibrate it}.
    """
    calibration = {}
    for building in {r['building'] for r in records}:
        group = [r for r in records if r['building'] == building]
        plan_name, floors = plans.get(building, (None, None))
        for r in group:
            tag = r.get('floor_tag')
            if tag is not None and floors and tag not in floors:
                # A tag naming a floor the building does not have (e.g. OSM
                # 'level 5' on Doherty Hall, whose top floor is 4): trust the plan
                r['floor_note'] = f"tag {tag!r} is not a floor of this building; ignored"
                r['floor_tag'] = None
        # Doors whose floor is known (OSM tag or set by hand) anchor the others:
        # an unknown door counts floors up or down from the known door whose
        # ground height is closest to its own
        anchors = [(r['terrain_floor'], floors.index(r['floor_tag'])) for r in group
                   if floors and r.get('floor_tag') in floors and r.get('terrain_floor') is not None]
        if anchors:
            calibration[building] = len(anchors)
        for r in group:
            tag, k = r.get('floor_tag'), r.get('terrain_floor')
            if tag is not None:
                name, src = tag, r.get('floor_tag_source', 'osm')
                idx = floors.index(tag) if floors and tag in floors else (
                    int(tag) - 1 if tag.lstrip('-').isdigit() else None)
            elif floors and k is not None:
                if anchors:
                    ka, ia = min(anchors, key=lambda a: (abs(a[0] - k), a[1]))
                    idx = ia + (k - ka)
                else:  # no known door: lowest ground = bottom floor of the list
                    idx = k - 1
                idx = min(max(idx, 0), len(floors) - 1)
                name, src = floors[idx], 'plan'
            elif k is not None:
                name, src, idx = str(k), 'terrain', k - 1
            else:
                name, src, idx = None, None, None
            r.update(floor=name, floor_index=idx, floor_source=src,
                     building_floors=floors, floor_plan=plan_name)
    return calibration


def find_entrances(buildings, entrances, paths, origin, width_m, height_m,
                   door_tol_m=3.0, max_doorstep_m=25.0, standoff_m=0.75,
                   elev=None, dem_res=1.0):
    """Pick delivery entrances for every building on the map.

    An OSM entrance belongs to a building when it lies within door_tol_m of
    the building's outline. A building with no usable OSM entrance gets an
    estimated one: in the middle of a walkway that passes through it (a
    building passage, like Scaife Hall's), otherwise at the point of its
    outline closest to the walking network.

    Returns a list of dicts in map coordinates (meters from the map's
    south-west corner): the door, the point on the sidewalk the doorstep path
    joins (doorstep_xy), and the delivery point standoff_m outside the door.

    Each entrance also gets 'floor_tag', the door's OSM floor tag (level:ref,
    level or addr:floor) if it has one, and, with an elevation grid,
    'terrain_floor': how many floors its ground is above the lowest ground
    around the building (1 = the lowest), at FLOOR_HEIGHT_M per floor.
    name_floors() turns these into floor names.
    """
    x0, y0 = origin
    walk = unary_union(list(paths.geometry))
    passage_rows = paths[paths.apply(is_passage, axis=1)]
    passages = unary_union(list(passage_rows.geometry)) if len(passage_rows) else None
    area = box(x0, y0, x0 + width_m, y0 + height_m)
    # Building interiors, shrunk slightly so a line that starts at a door on the
    # wall does not count as entering the building
    near = area.buffer(max_doorstep_m)
    blocked = prep(unary_union([g.buffer(-0.3) for g in buildings.geometry
                                if g.intersects(near)]))

    doors = []
    for idx, row in entrances.iterrows():
        kind = _tag(row, 'entrance') or 'door'
        if kind in SKIP_ENTRANCE or _tag(row, 'access') in SKIP_ACCESS:
            continue
        # level:ref carries letter floors like 'A'; level / addr:floor numbers
        floor = (_floor_name(row.get('level:ref')) or _floor_name(row.get('level'))
                 or _floor_name(row.get('addr:floor')))
        doors.append((row.geometry, kind, f'{idx[0]}/{idx[1]}', floor))

    def local(p):
        return [round(p.x - x0, 2), round(p.y - y0, 2)]

    records = []
    unnamed = 0
    for idx, row in buildings.iterrows():
        poly = row.geometry
        if not poly.intersects(area):
            continue
        name = _tag(row, 'name')
        if not name:
            unnamed += 1
            name = f'Unnamed building {unnamed}'

        found = [(p, kind, oid, 'osm') for p, kind, oid, _ in doors
                 if poly.boundary.distance(p) <= door_tol_m]
        tagged_floor = {oid: f for _, _, oid, f in doors if f is not None}
        base = (_building_base(poly, elev, dem_res, origin, height_m)
                if elev is not None else None)
        def floor_of(door_local, oid):
            """(OSM floor tag or None, floors above the lowest ground: 1, 2, ...)"""
            k = None
            if base is not None:
                z = float(_ground_z(elev, dem_res, height_m, *door_local))
                k = 1 + max(0, int(round((z - base) / FLOOR_HEIGHT_M)))
            return tagged_floor.get(oid), k
        if not found and passages is not None:
            # A walkway through the building (a building passage): the entrance
            # opens onto it, so deliver to the middle of the passage
            inside = passages.intersection(poly)
            pieces = [g for g in getattr(inside, 'geoms', [inside])
                      if isinstance(g, LineString) and g.length >= 2.0]
            if pieces:
                mid = max(pieces, key=lambda g: g.length).interpolate(0.5, normalized=True)
                if area.contains(mid):
                    tag, k = floor_of(local(mid), None)
                    records.append({
                        'building': name, 'building_osm_id': f'{idx[0]}/{idx[1]}',
                        'entrance': 'passage', 'source': 'estimated', 'entrance_osm_id': None,
                        'door_xy': local(mid), 'doorstep_xy': local(mid), 'doorstep_m': 0.0,
                        'delivery_xy': local(mid), 'floor_tag': tag, 'terrain_floor': k,
                    })
                    continue
        if not found:
            door = nearest_points(poly.boundary, walk)[0]
            found = [(door, 'estimated', None, 'estimated')]

        seen = set()
        for door, kind, oid, source in found:
            if (round(door.x, 1), round(door.y, 1)) in seen:
                continue  # the same door tagged twice
            seen.add((round(door.x, 1), round(door.y, 1)))
            # The doorstep joins the nearest sidewalk point that a straight line
            # from the door reaches without passing through any building
            clear = (p for p in _walk_points(walk, door, max_doorstep_m)
                     if door.distance(p) < 1e-6
                     or not blocked.intersects(LineString([door, p])))
            join = next(clear, None)
            if join is None:
                continue  # no sidewalk in sight of this door
            # Head away from the door toward a clear sidewalk point at least 1 m off,
            # so a door that sits on a path still gets a drop-off outside the wall
            aim = join if door.distance(join) >= 1.0 else next(
                (p for p in clear if door.distance(p) >= 1.0), None)
            if aim is None:
                continue
            dist = door.distance(join)
            n = door.distance(aim)
            ux, uy = (aim.x - door.x) / n, (aim.y - door.y) / n
            drop = Point(door.x + standoff_m * ux, door.y + standoff_m * uy)
            if not area.contains(drop):
                continue
            tag, k = floor_of(local(door), oid)
            records.append({
                'building': name,
                'building_osm_id': f'{idx[0]}/{idx[1]}',
                'entrance': kind,
                'source': source,
                'entrance_osm_id': oid,
                'door_xy': local(door),
                'doorstep_xy': local(join),
                'doorstep_m': round(dist, 2),
                'delivery_xy': local(drop),
                'floor_tag': tag,
                'terrain_floor': k,
            })
    return records


def _crosses(mask, r0, c0, r1, c1):
    """True if the straight pixel line from (r0, c0) to (r1, c1) touches the mask."""
    n = int(max(abs(r1 - r0), abs(c1 - c0))) * 2 + 2
    rr = np.linspace(r0, r1, n).round().astype(int)
    cc = np.linspace(c0, c1, n).round().astype(int)
    return bool(mask[rr, cc].any())


def _too_steep(elev, dem_res, res, r0, c0, r1, c1, max_grade):
    """True if a straight connector between two map pixels is steeper than
    max_grade anywhere (checked every meter on the elevation grid)."""
    if elev is None:
        return False
    length = np.hypot(r1 - r0, c1 - c0) * res
    n = max(2, int(length) + 1)
    s = res / dem_res
    rr = np.clip(np.linspace(r0, r1, n) * s, 0, elev.shape[0] - 1).astype(int)
    cc = np.clip(np.linspace(c0, c1, n) * s, 0, elev.shape[1] - 1).astype(int)
    z = elev[rr, cc]
    step = length / (n - 1)
    return bool(np.max(np.abs(np.diff(z))) / step > max_grade)


def bridge_gaps(img, bimg, res, max_len_m=60.0, width_m=2.0, min_area_m2=4.0,
                elev=None, dem_res=1.0, max_grade=0.12, simg=None):
    """Join cut-off pieces of sidewalk to the main network.

    OSM footpaths that pass through buildings are cut when the buildings are
    drawn over them, which leaves islands of sidewalk the robot cannot reach.
    Each island gets the shortest straight connector (at most max_len_m) to
    the largest piece that does not pass through a building and, when an
    elevation grid is given, is no steeper than max_grade (so a connector never
    stands in for the stairs that were left out).

    Returns the image with connectors drawn, and their lengths in meters.
    """
    walk = np.array(img)
    width_px = max(1, int(round(width_m / res)))
    # Grow the buildings by half the connector width, so checking a connector's
    # center line keeps its whole width out of every building
    grow = width_px // 2 + 1
    blocked = np.array(bimg) > 127
    if simg is not None:  # connectors must not cross stairs either
        blocked |= np.array(simg) > 127
    build = cv2.dilate(blocked.astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)) > 0
    lengths = []
    while True:
        n, lab = cv2.connectedComponents((walk > 250).astype(np.uint8), connectivity=8)
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        main = int(np.argmax(sizes))
        # Distance from every pixel to the main network, and which main pixel is nearest
        src = np.where(lab == main, 0, 255).astype(np.uint8)
        dist, nearest = cv2.distanceTransformWithLabels(
            src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        main_px = np.argwhere(src == 0)          # row-major, same order as the labels
        added = 0
        for k in range(1, n):
            if k == main or sizes[k] * res * res < min_area_m2:
                continue
            rr, cc = np.nonzero(lab == k)
            order = np.argsort(dist[rr, cc])
            for i in order[:4000:20]:            # closest island pixels first
                r0, c0 = rr[i], cc[i]
                if dist[r0, c0] * res > max_len_m:
                    break
                r1, c1 = main_px[nearest[r0, c0] - 1]
                if not _crosses(build, r0, c0, r1, c1) and not _too_steep(
                        elev, dem_res, res, r0, c0, r1, c1, max_grade):
                    cv2.line(walk, (int(c0), int(r0)), (int(c1), int(r1)), 255, width_px)
                    lengths.append(float(dist[r0, c0] * res))
                    added += 1
                    break
        if added == 0:
            break
    return Image.fromarray(walk), lengths


def clear_buildings(img, bimg, simg=None):
    """Mark every building (and stair) pixel occupied again, after doorsteps
    and connectors."""
    walk = np.array(img)
    walk[np.array(bimg) > 127] = 0
    if simg is not None:
        walk[np.array(simg) > 127] = 0
    return Image.fromarray(walk)


def mark_reachable(img, records, res, height_m, max_curbside_m=40.0):
    """Set record['reachable']: is the delivery point on the main sidewalk network?

    Entrances the robot cannot reach (only by stairs, through buildings or
    outside the map crop) are kept but marked unreachable. Those within
    max_curbside_m of the network also get a 'curbside_xy' drop-off point: the
    closest point the robot can reach, and its distance 'curbside_m'.
    """
    walk = (np.array(img) > 250).astype(np.uint8)
    n, lab = cv2.connectedComponents(walk, connectivity=8)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    main = int(np.argmax(sizes))
    H, W = walk.shape
    # Nearest point on the main network to every pixel, for curbside drop-offs
    src = np.where(lab == main, 0, 255).astype(np.uint8)
    dist, nearest = cv2.distanceTransformWithLabels(
        src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    main_px = np.argwhere(src == 0)
    for r in records:
        x, y = r['delivery_xy']
        row = min(H - 1, max(0, int((height_m - y) / res)))
        col = min(W - 1, max(0, int(x / res)))
        r['reachable'] = bool(lab[row, col] == main)
        if not r['reachable'] and dist[row, col] * res <= max_curbside_m:
            # The robot cannot get to this door (stairs only, or cut off): it can
            # still deliver to the closest point it can reach, e.g. the foot of the stairs
            cr, cc = main_px[nearest[row, col] - 1]
            r['curbside_xy'] = [round((cc + 0.5) * res, 2), round(height_m - (cr + 0.5) * res, 2)]
            r['curbside_m'] = round(float(dist[row, col] * res), 1)


def carve_doorsteps(img, records, res, height_m, width_m=2.0, gap_m=0.3):
    """Draw a walkable doorstep path from each door to the sidewalk it joins.

    The path stops gap_m short of the door so it does not cut into the building.
    """
    draw = ImageDraw.Draw(img)
    line_px = max(1, int(round(width_m / res)))
    for r in records:
        (dx, dy), (jx, jy) = r['door_xy'], r['doorstep_xy']
        d = ((jx - dx) ** 2 + (jy - dy) ** 2) ** 0.5
        if d <= gap_m:
            continue
        ex, ey = dx + gap_m * (jx - dx) / d, dy + gap_m * (jy - dy) / d
        pts = [(jx / res, (height_m - jy) / res), (ex / res, (height_m - ey) / res)]
        draw.line(pts, fill=255, width=line_px)


def _has(value, wanted):
    """True if an OSM tag value (a string, or a list after OSMnx merges edges) is in wanted."""
    values = value if isinstance(value, list) else [value]
    return any(isinstance(v, str) and v in wanted for v in values)


def is_steps(row):
    """Stairs: fine for people, impossible for a wheeled delivery robot.

    Steps with a ramp built in (ramp=yes or ramp:wheelchair=yes) are drivable,
    so they do not count. (A stroller ramp is only a narrow channel, so it does
    not make the steps drivable.)
    """
    return (_has(row.get('highway'), {'steps'})
            and not _has(row.get('ramp'), {'yes'})
            and not _has(row.get('ramp:wheelchair'), {'yes'}))


def beside_stairs(paths, steps, min_offset=0.8, max_offset=4.0, min_cos=0.85):
    """Paths that run alongside a staircase, like a ramp next to the steps.

    A path counts when it is roughly parallel to its nearest staircase and its
    middle lies between min_offset and max_offset meters to the side of the
    stairs' line. A path that continues a staircase end-on has (almost) no
    sideways offset, so it does not count.

    Returns a list of (path geometry, sideways offset in meters).
    """
    if len(steps) == 0:
        return []
    near = paths[paths.geometry.distance(unary_union(list(steps.geometry))) < max_offset]
    out = []
    for geom in near.geometry:
        if not isinstance(geom, LineString) or geom.length < 0.5:
            continue
        mid = geom.interpolate(0.5, normalized=True)
        stair = steps.geometry.iloc[int(steps.sindex.nearest(mid)[1][0])]
        (ax, ay), (bx, by) = stair.coords[0], stair.coords[-1]
        (px, py), (qx, qy) = geom.coords[0], geom.coords[-1]
        u = np.array([bx - ax, by - ay])
        v = np.array([qx - px, qy - py])
        if np.linalg.norm(u) < 1e-6 or np.linalg.norm(v) < 1e-6:
            continue
        u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
        if abs(u @ v) < min_cos:
            continue
        offset = abs(u[0] * (mid.y - ay) - u[1] * (mid.x - ax))  # sideways distance
        if min_offset <= offset <= max_offset:
            out.append((geom, offset))
    return out


def is_passage(row):
    """Outdoor walkways that OSM draws under a building outline: colonnades,
    arcades and other covered walkways, and building passages."""
    return (_has(row.get('covered'), {'colonnade', 'arcade', 'yes'})
            or _has(row.get('tunnel'), {'building_passage'}))


def rasterize(paths, buildings, center, width_m, height_m, res, sidewalk_m, steps):
    """Draw walkable paths in white, then buildings in black on top, then the
    covered walkways and building passages again on top of the buildings.

    Other paths under buildings (usually indoor corridors) stay blocked.

    `paths` should already exclude stairs; `steps` (the stairs) are only drawn
    into their own layer.

    Returns the occupancy image, an image with only the building footprints in
    white (passages cut out), and an image of the stairs, used for figures and
    checks.
    """
    W = int(round(width_m / res))
    H = int(round(height_m / res))
    x0 = center.x - width_m / 2.0   # left edge (meters)
    y1 = center.y + height_m / 2.0  # top edge (meters); image row 0 is north

    def to_px(coords):
        return [((x - x0) / res, (y1 - y) / res) for x, y in coords]

    img = Image.new('L', (W, H), 0)
    draw = ImageDraw.Draw(img)

    line_px = max(1, int(round(sidewalk_m / res)))

    def draw_paths(geoms, canvases):
        for geom in geoms:
            lines = geom.geoms if isinstance(geom, MultiLineString) else [geom]
            for line in lines:
                if isinstance(line, LineString):
                    pts = to_px(line.coords)
                    r = line_px / 2.0
                    for canvas, fill in canvases:
                        canvas.line(pts, fill=fill, width=line_px, joint='curve')
                        # Round the line ends so path junctions connect cleanly
                        for px, py in (pts[0], pts[-1]):
                            canvas.ellipse([px - r, py - r, px + r, py + r], fill=fill)

    draw_paths(paths.geometry, [(draw, 255)])

    bimg = Image.new('L', (W, H), 0)
    bdraw = ImageDraw.Draw(bimg)
    for geom in buildings.geometry:
        polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
        for poly in polys:
            if isinstance(poly, Polygon):
                draw.polygon(to_px(poly.exterior.coords), fill=0)
                bdraw.polygon(to_px(poly.exterior.coords), fill=255)

    # Covered walkways stay walkable, and are cut out of the building footprints
    passages = paths[paths.apply(is_passage, axis=1)]
    draw_paths(passages.geometry, [(draw, 255), (bdraw, 0)])

    # Stairs are never drivable. Short flights inside a footpath would otherwise
    # be bridged by the 3 m-wide path drawn on either side, so each flight is
    # blocked across a little more than the full path width (flat ends, so the
    # paths it joins stay open). This layer is applied again at the very end.
    simg = Image.new('L', (W, H), 0)
    sdraw = ImageDraw.Draw(simg)
    stair_px = line_px + max(2, int(round(0.4 / res)))
    for geom in steps.geometry:
        for line in (geom.geoms if isinstance(geom, MultiLineString) else [geom]):
            if isinstance(line, LineString):
                sdraw.line(to_px(line.coords), fill=255, width=stair_px)
    draw.bitmap((0, 0), simg, fill=0)

    # Ramps and paths running beside the stairs stay drivable: draw them again,
    # narrowed so they keep 0.5 m clear of the stairs' center line, and take
    # them out of the stairs layer
    for geom, offset in beside_stairs(paths, steps):
        w = max(1, int(round(min(sidewalk_m, 2.0 * (offset - 0.5)) / res)))
        pts = to_px(geom.coords)
        draw.line(pts, fill=255, width=w, joint='curve')
        sdraw.line(pts, fill=0, width=w, joint='curve')

    return img, bimg, simg


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--lat', type=float, default=CMU_LAT)
    parser.add_argument('--lon', type=float, default=CMU_LON)
    parser.add_argument('--width-m', type=float, default=605.0, help='east-west extent')
    parser.add_argument('--height-m', type=float, default=570.0, help='north-south extent')
    parser.add_argument('--res', type=float, default=0.1, help='meters per cell')
    parser.add_argument('--sidewalk-m', type=float, default=3.0, help='drawn path width')
    parser.add_argument('--elevation-res', type=float, default=1.0,
                        help='meters per elevation sample (3DEP has 1 m data here)')
    parser.add_argument('--out', type=Path, default=HERE / 'maps' / 'cmu_campus.png')
    args = parser.parse_args()

    radius = 0.5 * (args.width_m ** 2 + args.height_m ** 2) ** 0.5 + 50.0
    paths, buildings, entrances = download(args.lat, args.lon, radius)
    paths, buildings, entrances, center = to_metric(paths, buildings, entrances,
                                                    args.lat, args.lon)
    stairs = paths.apply(is_steps, axis=1)
    steps, paths = paths[stairs], paths[~stairs]
    img, bimg, simg = rasterize(paths, buildings, center, args.width_m, args.height_m,
                                args.res, args.sidewalk_m, steps)

    origin = (center.x - args.width_m / 2.0, center.y - args.height_m / 2.0)
    elev = download_elevation(origin, args.width_m, args.height_m, paths.crs,
                              res_m=args.elevation_res)
    print(f'Elevation: {elev.min():.1f} to {elev.max():.1f} m '
          f'({elev.max() - elev.min():.1f} m of relief) from USGS 3DEP')

    doors = find_entrances(buildings, entrances, paths, origin, args.width_m, args.height_m,
                           elev=elev, dem_res=args.elevation_res)
    plans = load_floor_plans()
    unmatched = apply_floor_overrides(doors, origin)
    calibration = name_floors(doors, plans)
    carve_doorsteps(img, doors, args.res, args.height_m)
    img, bridges = bridge_gaps(img, bimg, args.res, elev=elev, dem_res=args.elevation_res,
                               simg=simg)
    img = clear_buildings(img, bimg, simg)
    mark_reachable(img, doors, args.res, args.height_m)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    bimg.save(args.out.with_name(args.out.stem + '_buildings.png'))
    simg.save(args.out.with_name(args.out.stem + '_stairs.png'))
    np.save(args.out.with_name(args.out.stem + '_elevation.npy'), elev)
    args.out.with_name(args.out.stem + '_entrances.json').write_text(json.dumps(doors, indent=2))

    meta = {
        'resolution_m': args.res,
        'width_cells': img.width,
        'height_cells': img.height,
        'center_lat': args.lat,
        'center_lon': args.lon,
        'crs': str(paths.crs),
        'origin_xy_m': [center.x - args.width_m / 2.0, center.y - args.height_m / 2.0],
        'sidewalk_m': args.sidewalk_m,
        'elevation_res_m': args.elevation_res,
        'elevation_source': 'USGS 3DEP (bare earth), row 0 = north',
        'num_paths': int(len(paths)),
        'num_stairs': int(len(steps)),
        'num_buildings': int(len(buildings)),
        'attribution': 'Map data (c) OpenStreetMap contributors, ODbL',
    }
    args.out.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    print(f'Saved {args.out} ({img.width} x {img.height} cells at {args.res} m), '
          f'{len(paths)} path segments, {len(buildings)} buildings; left out '
          f'{len(steps)} stair segments ({steps.length.sum():.0f} m)')
    print(f'Joined {len(bridges)} cut-off sidewalk pieces to the network '
          f'({sum(bridges):.0f} m of connectors, longest {max(bridges, default=0):.1f} m)')
    ok = [d for d in doors if d['reachable']]
    n_osm = sum(d['source'] == 'osm' for d in ok)
    print(f'Saved {len(ok)} reachable delivery entrances for '
          f"{len({d['building'] for d in ok})} buildings "
          f'({n_osm} from OpenStreetMap, {len(ok) - n_osm} estimated)')
    curb = [d for d in doors if not d['reachable'] and 'curbside_xy' in d]
    print(f'{len(doors) - len(ok)} entrances the robot cannot reach (stairs only, through '
          f'buildings, or outside the map); {len(curb)} of them get a curbside drop-off')
    no_door = sorted({d['building'] for d in doors} - {d['building'] for d in ok})
    if no_door:
        print('Buildings with curbside delivery only: ' + ', '.join(no_door))
    from collections import Counter
    src = Counter(d['floor_source'] for d in doors)
    print(f"Floors: {src['user']} set by hand, {src['osm']} from OSM door tags, "
          f"{src['plan']} named from CMU floor plans, {src['terrain']} numbered from "
          f"terrain only (no floor plan for the building)")
    if calibration:
        print('Calibrated with known doors: ' + ', '.join(
            f'{b} ({n})' for b, n in sorted(calibration.items())))
    for o in unmatched:
        print(f"Floor override matched no door: {o['building']} "
              f"{o.get('entrance_osm_id') or o.get('utm_xy')}")


if __name__ == '__main__':
    main()
