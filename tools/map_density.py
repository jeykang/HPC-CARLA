#!/usr/bin/env python3
"""Dynamic urban-density metrics for ANY CARLA map — computed from the road
network (OpenDRIVE), never hardcoded per town.

Motivation: the map you drive on is a dominant difficulty factor (dense grids
like Town02 are brutal; open highway maps like Town05 are easy), but the route
XMLs store only 2-waypoint endpoints, so route geometry is degenerate and the old
difficulty score is blind to it. Every CARLA map — default OR user-added — ships
an OpenDRIVE description, so we can measure the road network itself.

Works two ways, same math:
  * OFFLINE:  point it at a .xodr file (carla_maps/OpenDrive/Town02.xodr).
  * IN-SIM :  density_from_opendrive(world.get_map().to_opendrive()) — for a
              custom map that's loaded but has no .xodr on disk.

Primary signal is JUNCTION DENSITY (intersections per km driven / per km^2):
high in dense urban grids, low on highways. Pure stdlib.
"""
import argparse, glob, math, os, sys
import xml.etree.ElementTree as ET


def _f(el, attr):
    try: return float(el.get(attr))
    except (TypeError, ValueError): return None


def density_from_root(root):
    """Compute urban-density metrics from a parsed OpenDRIVE <OpenDRIVE> root."""
    # strip any namespace so findall works regardless of schema decoration
    for e in root.iter():
        if '}' in e.tag: e.tag = e.tag.split('}', 1)[1]

    roads = root.findall('road')
    junctions = root.findall('junction')
    n_junc = len(junctions)
    n_roads = len(roads)
    total_len = sum(_f(r, 'length') or 0.0 for r in roads)
    # roads that are physical driving roads vs junction-internal connectors
    n_connectors = sum(1 for r in roads if (r.get('junction', '-1') or '-1') != '-1')

    # curvature + geometry bbox (fallback area)
    arc_len = geom_len = 0.0
    xs, ys = [], []
    for r in roads:
        pv = r.find('planView')
        if pv is None: continue
        for g in pv.findall('geometry'):
            L = _f(g, 'length') or 0.0; geom_len += L
            x, y = _f(g, 'x'), _f(g, 'y')
            if x is not None and y is not None: xs.append(x); ys.append(y)
            if g.find('arc') is not None: arc_len += L
            elif g.find('spiral') is not None: arc_len += 0.5 * L

    # area: prefer header bounds, else geometry bbox
    hdr = root.find('header')
    w = _f(hdr, 'west') if hdr is not None else None
    e = _f(hdr, 'east') if hdr is not None else None
    n = _f(hdr, 'north') if hdr is not None else None
    s = _f(hdr, 'south') if hdr is not None else None
    if None not in (w, e, n, s) and e > w and n > s:
        area_km2 = (e - w) * (n - s) / 1e6
    elif xs:
        area_km2 = max((max(xs) - min(xs)) * (max(ys) - min(ys)) / 1e6, 1e-6)
    else:
        area_km2 = None

    road_km = total_len / 1000.0
    return {
        'n_junctions': n_junc,
        'n_roads': n_roads,
        'road_km': road_km,
        'area_km2': area_km2,
        'junctions_per_road_km': (n_junc / road_km) if road_km else None,
        'junctions_per_km2': (n_junc / area_km2) if area_km2 else None,
        'road_km_per_km2': (road_km / area_km2) if area_km2 else None,
        'mean_road_len_m': (total_len / n_roads) if n_roads else None,
        'connector_frac': (n_connectors / n_roads) if n_roads else None,   # junction-internal share
        'frac_curved': (arc_len / geom_len) if geom_len else None,
    }


def density_from_opendrive(xodr_text):
    """From an OpenDRIVE string (e.g. carla.Map.to_opendrive())."""
    return density_from_root(ET.fromstring(xodr_text))


def density_from_file(path):
    return density_from_root(ET.parse(path).getroot())


def urban_density_score(m):
    """A single normalised-ish 'urban density' number for use as a difficulty axis.
    Junctions per km of road is the most drive-relevant (how often you hit an
    intersection); scaled to a convenient range."""
    v = m.get('junctions_per_road_km')
    return round(v, 4) if v is not None else None


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='*', default=None,
                    help='.xodr files (default: carla_maps/OpenDrive/*.xodr)')
    args = ap.parse_args(argv)
    paths = args.paths or sorted(glob.glob('carla_maps/OpenDrive/*.xodr'))
    if not paths:
        print('no .xodr files found', file=sys.stderr); return 2

    rows = []
    for p in paths:
        try:
            m = density_from_file(p); m['map'] = os.path.basename(p).replace('.xodr', '')
            rows.append(m)
        except Exception as ex:
            print(f'  {p}: parse error {ex}', file=sys.stderr)
    # rank by the drive-relevant density
    rows.sort(key=lambda r: -(r.get('junctions_per_road_km') or 0))
    hdr = ['map', 'n_junctions', 'road_km', 'area_km2', 'junctions_per_road_km',
           'junctions_per_km2', 'mean_road_len_m', 'frac_curved']
    print(f"{'map':10s} {'#junc':>6s} {'road_km':>8s} {'area_km2':>9s} "
          f"{'junc/road_km':>12s} {'junc/km2':>9s} {'mean_road_m':>11s} {'curved':>7s}")
    for r in rows:
        def g(k, f='{:.2f}'):
            v = r.get(k); return f.format(v) if isinstance(v, (int, float)) else '   -'
        print(f"{r['map']:10s} {r['n_junctions']:>6d} {g('road_km'):>8s} {g('area_km2'):>9s} "
              f"{g('junctions_per_road_km','{:.3f}'):>12s} {g('junctions_per_km2'):>9s} "
              f"{g('mean_road_len_m'):>11s} {g('frac_curved','{:.2f}'):>7s}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
