"""
experiments/comparison/run_comparison.py
========================================
Reads ``meta.json`` from this folder, generates a dataset with the specified
stage distribution, then runs three routing modes and produces an interactive
Folium comparison map with directional arrow-heads on every bus route.

Key design: the **bus** always drives on the full (unconstrained) road network.
Safety constraints only affect the **student walking BFS** — which stops the
student can reach on foot without crossing a dangerous road.

Modes
-----
  A  Constrained   – walking BFS avoids primary/trunk/secondary; same walk radius as B
  B  Unconstrained – walking BFS uses all edges; same walk radius as A
  C  Door-to-Door  – walk_radius=0 for all (bus visits every home)

Usage (from the repo root):
    python -m experiments.comparison.run_comparison          # uses meta.json defaults
    python -m experiments.comparison.run_comparison --iterations 50
"""

import os, sys, json, time, copy, math, argparse, datetime, statistics

# ── path fix: ensure repo root is on sys.path ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import folium
from folium import plugins, FeatureGroup

import detour_engine as _eng
import alns_engine   as _alns

from run_algorithm import (
    setup_graph, precompute_matrix, run_algorithm,
    DEFAULT_STAGE_WALK_LIMITS,
)
from data_loader   import load_mode1_input
from solution_state import ServiceSolution
from detour_engine  import (
    calculate_route_path_and_stats,
    calculate_route_time_from_matrix,
    walk_path_on_roads,
    walk_distance_on_roads,
    find_shortest_path_with_turns,
    compute_student_tmax,
    compute_direct_time,
    calculate_walk_penalty,
    _MATRIX_CACHE,
)

# Patch: fast snap for large graphs
_eng._FAST_SNAP_MODE = True

# ────────────────────────────────────────────────────────────────────
# Load input.json
# ────────────────────────────────────────────────────────────────────
_INPUT_PATH = os.path.join(_SCRIPT_DIR, "input.json")

def _load_meta(path=None):
    with open(path or _INPUT_PATH) as f:
        return json.load(f)


# ────────────────────────────────────────────────────────────────────
# Dataset generation (delegates to experiments.generate_dataset)
# ────────────────────────────────────────────────────────────────────
def _generate_dataset(meta):
    """Call the refactored generate_dataset() with meta.json values."""
    sys.path.insert(0, os.path.join(_ROOT, "experiments"))
    from generate_dataset import generate_dataset

    return generate_dataset(
        n_students=meta["n_students"],
        seed=meta["seed"],
        school=meta["school"],
        stage_dist=meta["stage_distribution"],
        annulus=meta.get("annulus"),
        buses_count=meta.get("buses", {}).get("count", 4),
        bus_capacity=meta.get("buses", {}).get("capacity", 60),
        constraints=meta.get("constraints"),
        iterations=meta.get("algorithm", {}).get("iterations", 30),
    )


# ────────────────────────────────────────────────────────────────────
# Cache helpers
# ────────────────────────────────────────────────────────────────────
def _reset_caches():
    _alns._student_candidate_cache.clear()
    _eng._MATRIX_CACHE.clear()
    _eng._MATRIX_CACHE_LENGTH.clear()
    _eng._path_cache.clear()
    _eng._WALK_DIST_CACHE.clear()
    _eng._STUDENT_NODE_CACHE.clear()
    _eng._WALK_GRAPH = None
    _eng._safe_nodes_cache.clear()


def _prebuild_ball_tree(G):
    print("  Pre-building BallTree …", end="", flush=True)
    t0 = time.time()
    _eng._get_or_build_ball_tree(G)
    print(f" {time.time()-t0:.1f}s")


# ────────────────────────────────────────────────────────────────────
# Input mutators  (same logic as visualize_comparison.py)
# ────────────────────────────────────────────────────────────────────
def _relax_ride_constraints(d):
    """Disable per-route ride-time caps so only the walking variable differs."""
    d["meta"].setdefault("constraints", {}).update(
        {"ride_time_multiplier": 999, "floor_minutes": 999, "ceiling_minutes": 999}
    )
    return d


def _make_constrained(data):
    """Mode A: safety constraints ON, ride-time constraints from meta.json."""
    return copy.deepcopy(data)


def _make_unconstrained(data):
    """Mode B: all-safe walking, ride-time constraints from meta.json."""
    d = copy.deepcopy(data)
    for s in d["data"]["students"]:
        s["walk_radius_override"] = 400
    return d


def _make_door_to_door(data):
    """Mode C: no walking (walk_radius=0), ride-time constraints from meta.json."""
    d = copy.deepcopy(data)
    for s in d["data"]["students"]:
        s["walk_radius_override"] = 0
    return d


# ────────────────────────────────────────────────────────────────────
# Crossing detection  — FIXED
# ────────────────────────────────────────────────────────────────────
#
# A "dangerous crossing" means the student's walking path passes
# through an edge whose highway type is a major road (primary, trunk,
# secondary, motorway).  Residential and tertiary roads are NOT
# dangerous — MID/HIGH students can cross them freely.
#
# The run-length heuristic:
#   • A consecutive run of dangerous edges < 150 m that is bounded by
#     non-dangerous edges on both sides = TRUE crossing (perpendicular).
#   • A longer run or one that starts/ends the path = walking ALONGSIDE
#     — not counted.
# ────────────────────────────────────────────────────────────────────

_CROSSING_RUN_MAX_M = 150.0

# Only these highway types count as genuinely dangerous to cross on foot.
_DANGEROUS_HW_TYPES = {'motorway', 'motorway_link', 'trunk', 'trunk_link',
                       'primary', 'primary_link', 'secondary', 'secondary_link'}


def _edge_is_dangerous(data_dict):
    """Return True if the edge represents a road dangerous to cross on foot."""
    hw = data_dict.get('highway', '')
    if isinstance(hw, list):
        hw = hw[0] if hw else ''
    return hw in _DANGEROUS_HW_TYPES


def _classify_walk_path(walk_path, G_con):
    """Detect true dangerous road crossings along a walk path.

    Uses the *constrained* graph as ground truth.  An edge is 'dangerous'
    only if its highway type is in _DANGEROUS_HW_TYPES (primary, trunk,
    secondary, motorway).  Tertiary and residential are safe to cross.
    """
    if len(walk_path) < 2:
        return []

    # Build per-edge metadata: (is_dangerous, length_m, u, v, highway)
    edges = []
    for i in range(len(walk_path) - 1):
        u, v = walk_path[i], walk_path[i + 1]
        ed = G_con.get_edge_data(u, v) or G_con.get_edge_data(v, u)
        if ed is None:
            edges.append((False, 0.0, u, v, ''))
            continue
        d = ed[0] if 0 in ed else list(ed.values())[0]
        dangerous = _edge_is_dangerous(d)
        hw = d.get('highway', '')
        if isinstance(hw, list):
            hw = hw[0] if hw else ''
        edges.append((dangerous, float(d.get('length', 0.0)), u, v, hw))

    crossings = []
    i = 0
    while i < len(edges):
        if not edges[i][0]:   # not dangerous — skip
            i += 1
            continue

        # Start of a dangerous run
        run_start   = i
        run_total_m = 0.0
        run_hws     = set()
        while i < len(edges) and edges[i][0]:
            run_total_m += edges[i][1]
            run_hws.add(edges[i][4])
            i += 1
        run_end = i  # exclusive

        # Find midpoint for marker
        accumulated = 0.0
        mid_u, mid_v = edges[run_start][2], edges[run_start][3]
        for j in range(run_start, run_end):
            accumulated += edges[j][1]
            if accumulated >= run_total_m / 2:
                mid_u, mid_v = edges[j][2], edges[j][3]
                break

        came_from_safe = (run_start == 0) or not edges[run_start - 1][0]
        goes_to_safe   = (run_end >= len(edges)) or not edges[run_end][0]

        if run_total_m < _CROSSING_RUN_MAX_M and came_from_safe and goes_to_safe:
            mid_lat = (G_con.nodes[mid_u]['y'] + G_con.nodes[mid_v]['y']) / 2
            mid_lon = (G_con.nodes[mid_u]['x'] + G_con.nodes[mid_v]['x']) / 2
            crossings.append({
                'u': mid_u, 'v': mid_v,
                'lat': mid_lat, 'lon': mid_lon,
                'run_length_m': round(run_total_m, 1),
                'road_types': ', '.join(sorted(run_hws)),
            })
    return crossings


def _count_unsafe_crossings(sol, G_con, G_solve):
    """Detect dangerous crossings for every served student.

    Walks on G_solve (the graph used for that mode's routing), but checks
    edge danger using G_con (the constrained ground-truth graph).
    """
    crossings = []
    for route in sol.routes:
        for stop in route.stops:
            if stop.stop_type == 'school':
                continue
            for student in stop.students:
                s_node = _eng.fast_nearest_node(G_solve, student.coords[1], student.coords[0])
                wp = walk_path_on_roads(G_solve, s_node, stop.node_id)
                for cx in _classify_walk_path(wp, G_con):
                    cx['student_id'] = student.id
                    crossings.append(cx)
    return crossings


# ────────────────────────────────────────────────────────────────────
# Dangerous / unclassified road helpers
# ────────────────────────────────────────────────────────────────────
def _extract_segments(G_con, center_lat, center_lon, kind="dangerous"):
    """Return coord lists for dangerous OR unclassified road segments within 5 km."""
    segments = []
    seen = set()
    for u, v, k, data in G_con.edges(keys=True, data=True):
        if kind == "dangerous":
            if data.get("is_safe_to_cross", True):
                continue
        elif kind == "unclassified":
            hw = data.get("highway", "")
            if isinstance(hw, list):
                hw = hw[0]
            if hw != "unclassified":
                continue
        if data.get("length", 0) < 20:
            continue
        ek = (min(u, v), max(u, v))
        if ek in seen:
            continue
        seen.add(ek)
        mid_lat = (G_con.nodes[u]["y"] + G_con.nodes[v]["y"]) / 2
        mid_lon = (G_con.nodes[u]["x"] + G_con.nodes[v]["x"]) / 2
        dlat = abs(mid_lat - center_lat) * 111.0
        dlon = abs(mid_lon - center_lon) * 111.0 * math.cos(math.radians(center_lat))
        if math.sqrt(dlat**2 + dlon**2) > 5.0:
            continue
        if "geometry" in data:
            coords = [(lat, lon) for lon, lat in data["geometry"].coords]
        else:
            coords = [(G_con.nodes[u]["y"], G_con.nodes[u]["x"]),
                       (G_con.nodes[v]["y"], G_con.nodes[v]["x"])]
        segments.append(coords)
    return segments


# ────────────────────────────────────────────────────────────────────
# MAP BUILDING  (with PolyLineTextPath arrows, like visualization.py)
# ────────────────────────────────────────────────────────────────────
_ROUTE_COLORS = {
    "A": ["#2196F3", "#1565C0", "#0D47A1", "#82B1FF"],
    "B": ["#4CAF50", "#2E7D32", "#1B5E20", "#A5D6A7"],
    "C": ["#FF9800", "#E65100", "#BF360C", "#FFCC80"],
}
# Matching folium-valid named colors for Icon markers (same order as _ROUTE_COLORS)
_ICON_COLORS = {
    "A": ["blue",   "darkblue",  "darkblue",  "lightblue"],
    "B": ["green",  "darkgreen", "darkgreen", "lightgreen"],
    "C": ["orange", "red",       "darkred",   "beige"],
}
_MODE_NAMES = {
    "A": "Constrained (Safe Walking)",
    "B": "Unconstrained (Any Walking)",
    "C": "Door-to-Door (No Walking)",
}

import networkx as nx
from detour_engine import (
    find_shortest_path_with_turns,
    get_bearing_of_path,
)

# Arrow text template — spaces pad between arrow glyphs
_ARROW_TEXT = "          \u27A4          "


def _compute_route_path(G, stops):
    """Compute the full node-level path between consecutive stops.

    Uses ``find_shortest_path_with_turns`` (bearing-aware A*) so that
    U-turns are penalised / banned — matching the solver's routing logic.

    The matrix cache only stores *times* (no paths), which makes the
    standard helper return ``(None, time)`` and breaks rendering.
    We work around this by chaining bearings across segments: when
    ``initial_bearing`` is not None the function skips the matrix
    shortcut and either hits the path cache or runs a full A*.
    """
    if len(stops) < 2:
        return []

    full_path = []
    last_bearing = None          # chain across segments

    for i in range(len(stops) - 1):
        u = stops[i].node_id
        v = stops[i + 1].node_id
        if u == v:
            if not full_path:
                full_path.append(u)
            continue

        # Use the turn-aware pathfinder.
        # Passing initial_bearing (even 0.0 on first call) bypasses
        # the matrix-only shortcut so we always get an actual path.
        bearing_arg = last_bearing if last_bearing is not None else 0.0
        seg, _ = find_shortest_path_with_turns(
            G, u, v, weight='travel_time', initial_bearing=bearing_arg,
        )

        if seg is None or len(seg) < 2:
            # Fallback: plain Dijkstra (at least draws *something*)
            try:
                seg = nx.shortest_path(G, u, v, weight='travel_time')
            except Exception:
                try:
                    seg = nx.shortest_path(G, u, v, weight='length')
                except Exception:
                    continue

        if not full_path:
            full_path.extend(seg)
        else:
            full_path.extend(seg[1:])

        last_bearing = get_bearing_of_path(G, seg)

    return full_path


def _build_path_coords(G, full_path, offset=0.0):
    """Convert a list of node IDs to (lat, lon) tuples following edge geometries."""
    coords = []
    for i in range(len(full_path) - 1):
        u, v = full_path[i], full_path[i + 1]
        ed = G.get_edge_data(u, v)
        if not ed:
            continue
        d = ed[0] if 0 in ed else list(ed.values())[0]
        if "geometry" in d:
            for lon, lat in d["geometry"].coords:
                coords.append((lat + offset, lon + offset))
        else:
            coords.append((G.nodes[u]["y"] + offset, G.nodes[u]["x"] + offset))
    last = full_path[-1]
    coords.append((G.nodes[last]["y"] + offset, G.nodes[last]["x"] + offset))
    return coords


def _build_walk_coords(G, wp):
    wcoords = []
    for wi in range(len(wp) - 1):
        u2, v2 = wp[wi], wp[wi + 1]
        ed2 = G.get_edge_data(u2, v2) or G.get_edge_data(v2, u2)
        if ed2:
            dd = ed2[0] if 0 in ed2 else list(ed2.values())[0]
            if "geometry" in dd:
                for lon, lat in dd["geometry"].coords:
                    wcoords.append((lat, lon))
            else:
                wcoords.append((G.nodes[u2]["y"], G.nodes[u2]["x"]))
        else:
            wcoords.append((G.nodes[u2]["y"], G.nodes[u2]["x"]))
    wcoords.append((G.nodes[wp[-1]]["y"], G.nodes[wp[-1]]["x"]))
    return wcoords


def _compute_pm_ride_time(route, stop, G):
    """Afternoon (school→home-stop) ride time using the reversed route sequence."""
    interior = [s for s in route.stops if s.stop_type != 'school']
    afternoon = [route.stops[0]] + interior[::-1] + [route.stops[-1]]
    target_idx = next((i for i, s in enumerate(afternoon) if s is stop), -1)
    if target_idx <= 0:
        return 0.0
    total = 0.0
    for i in range(target_idx):
        u = afternoon[i].node_id
        v = afternoon[i + 1].node_id
        t = _MATRIX_CACHE.get((u, v), None)
        if t is None:
            _, t = find_shortest_path_with_turns(G, u, v)
        if not math.isfinite(t):
            return float('inf')
        total += t
    return total


def _dir_cap_html(label, ride, direct, cap, k):
    """Compact per-direction ride-cap block with progress bar."""
    if direct is None or direct <= 0 or not math.isfinite(ride):
        ride_str = f"{ride:.1f}" if math.isfinite(ride) else "∞"
        return (
            f'<div style="margin:3px 0;font-size:11px;">'
            f'  <b>{label}:</b> ride&nbsp;<b>{ride_str}&nbsp;min</b> — direct N/A'
            f'</div>'
        )
    ratio = ride / direct
    ratio_color = 'green' if ratio <= 1.5 else ('darkorange' if ratio <= k else 'red')
    cap_safe = cap if math.isfinite(cap) else 999
    usage_pct = min(100, int(ride / cap_safe * 100)) if cap_safe > 0 else 0
    status = '✖ over cap' if ride > cap else '✔ ok'
    status_color = 'red' if ride > cap else 'green'
    return (
        f'<div style="margin:3px 0;font-size:11px;border-left:3px solid {ratio_color};padding-left:4px;">'
        f'  <b>{label}:</b> '
        f'  ride <b style="color:{ratio_color};">{ride:.1f}</b> / cap <b>{cap:.1f}</b> min'
        f'  &nbsp;<span style="color:{ratio_color};">({ratio:.2f}×)</span>'
        f'  <span style="color:{status_color};float:right;">{status}</span><br>'
        f'  direct {direct:.1f} min'
        f'  <div style="background:#eee;border-radius:3px;height:5px;margin-top:2px;">'
        f'    <div style="background:{ratio_color};width:{usage_pct}%;height:5px;border-radius:3px;"></div>'
        f'  </div>'
        f'</div>'
    )


def _add_route_layer(m, G, sol, mode_key, G_con, constraints=None):
    """Add route + walk FeatureGroups for one mode.  Returns (fg_routes, fg_walks, crossings, occupancies)."""
    con = constraints or {}
    ride_k       = float(con.get("ride_time_multiplier", 2.5))
    floor_min    = float(con.get("floor_minutes",        45))
    ceiling_min  = float(con.get("ceiling_minutes",      60))
    caps_enabled = bool(con.get("enabled",              True))

    show = mode_key in ("A", "B")   # show constrained + unconstrained by default
    fg_routes = FeatureGroup(name=f"{_MODE_NAMES[mode_key]} – Routes",        show=show)
    fg_walks  = FeatureGroup(name=f"{_MODE_NAMES[mode_key]} – Walking Paths", show=show)
    colors      = _ROUTE_COLORS[mode_key]
    icon_colors = _ICON_COLORS[mode_key]
    active  = [r for r in sol.routes if r.get_student_count() > 0]
    occupancies = []

    for ri, route in enumerate(active):
        c  = colors[ri % len(colors)]
        ic = icon_colors[ri % len(icon_colors)]

        # ── Bus route polyline with arrow-heads ──
        if len(route.stops) > 1:
            full_path = _compute_route_path(G, route.stops)
            if full_path and len(full_path) >= 2:
                offset = 0.00003 * (ri - 0.5)
                coords = _build_path_coords(G, full_path, offset)

                pl = folium.PolyLine(
                    coords, color=c, weight=5, opacity=0.8,
                    popup=f"Mode {mode_key} Route {route.route_id} "
                          f"({route.get_student_count()} students, "
                          f"{route.total_time:.0f} min, "
                          f"{route.total_distance:.1f} km)",
                )
                pl.add_to(fg_routes)

                # Directional arrows (same technique as visualization.py)
                plugins.PolyLineTextPath(
                    pl, _ARROW_TEXT, repeat=True, offset=6,
                    attributes={"fill": c, "font-weight": "bold", "font-size": "24"},
                ).add_to(fg_routes)

        # ── Stop markers ──
        student_count = 0
        for si, stop in enumerate(route.stops):
            if stop.stop_type == "school":
                continue
            n_stu = len(stop.students)
            student_count += n_stu
            folium.CircleMarker(
                location=stop.coords, radius=7,
                color=c, fill=True, fillColor=c, fillOpacity=0.85,
                popup=f"Mode {mode_key} {route.route_id} Stop-{si} ({n_stu} students)",
                tooltip=f"{mode_key}-{route.route_id} Stop {si} ({n_stu} students)",
            ).add_to(fg_routes)

            # ── Walk paths + student home markers ──
            for student in stop.students:
                s_node = _eng.fast_nearest_node(G, student.coords[1], student.coords[0])
                wp = walk_path_on_roads(G, s_node, stop.node_id)
                if len(wp) >= 2:
                    wcoords = _build_walk_coords(G, wp)
                    walk_dist = walk_distance_on_roads(G, s_node, stop.node_id)
                    folium.PolyLine(
                        wcoords, color=c, weight=2, opacity=0.6, dash_array="6,4",
                        tooltip=f"{student.id} walk: {walk_dist:.0f} m",
                    ).add_to(fg_walks)

                walk_m = walk_distance_on_roads(G, s_node, stop.node_id)

                # ── AM ride time: this stop → school (matrix-cache safe) ──
                stop_idx = next((i for i, s in enumerate(route.stops) if s is stop), -1)
                ride_time_am = 0.0
                if stop_idx != -1:
                    ride_time_am = calculate_route_time_from_matrix(route.stops[stop_idx:], G)
                    if ride_time_am >= 9999:
                        ride_time_am = float('inf')

                # ── PM ride time: school → this stop (reversed route) ──
                ride_time_pm = _compute_pm_ride_time(route, stop, G)

                # ── Direct times: AM = home→school, PM = school→home ──
                school_node = route.stops[-1].node_id
                direct_am = None
                direct_pm = None
                try:
                    _, direct_am = find_shortest_path_with_turns(G, s_node, school_node, weight='travel_time')
                    if not math.isfinite(direct_am):
                        direct_am = None
                except Exception:
                    pass
                try:
                    _, direct_pm = find_shortest_path_with_turns(G, school_node, s_node, weight='travel_time')
                    if not math.isfinite(direct_pm):
                        direct_pm = None
                except Exception:
                    pass

                # ── Per-direction caps ──
                k_eff = getattr(route, 'ride_time_multiplier', ride_k)
                fl    = getattr(route, 'floor_minutes',        floor_min)
                ce    = getattr(route, 'ceiling_minutes',      ceiling_min)

                def _cap(d):
                    if d is None or d <= 0:
                        return float('inf')
                    return max(fl, min(k_eff * d, d + ce))

                cap_html = (
                    _dir_cap_html('🟠 AM home→school', ride_time_am, direct_am, _cap(direct_am), k_eff) +
                    _dir_cap_html('🟦 PM school→home', ride_time_pm, direct_pm, _cap(direct_pm), k_eff)
                ) if caps_enabled else ''

                # ── Walk info ──
                stage_name = (
                    student.school_stage.name
                    if hasattr(student.school_stage, "name")
                    else str(student.school_stage)
                )
                walk_limit = student.walk_radius if hasattr(student, 'walk_radius') else 0
                walk_info  = (f"{walk_m:.0f}m / {walk_limit:.0f}m" if walk_limit > 0
                              else f"{walk_m:.0f}m (Door-to-Door)")

                popup_html = (
                    f'<div style="width:260px;font-size:12px;">'
                    f'<b>Student: {student.id}</b><br>'
                    f'Stage: {stage_name}<br>'
                    f'Home: {student.coords[0]:.5f}, {student.coords[1]:.5f}<br>'
                    f'Mode: {_MODE_NAMES[mode_key]}<br>'
                    f'<div style="margin-top:5px;border-top:1px solid #ccc;padding-top:5px;">'
                    f'<b>Route:</b> {route.route_id}<br>'
                    f'{cap_html}'
                    f'<div style="margin-top:3px;font-size:11px;">Walk to Stop: {walk_info}</div>'
                    f'</div></div>'
                )

                folium.Marker(
                    location=student.coords,
                    tooltip=f"{student.id} ({stage_name}) — AM {ride_time_am:.0f} min, walk {walk_m:.0f}m",
                    popup=folium.Popup(popup_html, max_width=300),
                    icon=folium.Icon(color=ic, icon='home', prefix='fa'),
                ).add_to(fg_walks)

        occupancies.append(student_count)

    crossings = _count_unsafe_crossings(sol, G_con, G)
    fg_routes.add_to(m)
    fg_walks.add_to(m)
    return fg_routes, fg_walks, crossings, occupancies


def _count_satisfied(sol, G, constraints):
    """Count served students whose ride time is within the cap.

    Respects ``bidirectional_check``:
      True  → satisfied when AM **or** PM ride ≤ cap (lenient: both must fail to fail)
      False → satisfied when AM ride ≤ cap (strict)

    Students without a finite direct time are counted as satisfied (no basis to reject).
    """
    con      = constraints or {}
    k        = float(con.get('ride_time_multiplier', 2.5))
    fl       = float(con.get('floor_minutes',        45))
    ce       = float(con.get('ceiling_minutes',      60))
    bidir    = bool(con.get('bidirectional_check',   True))
    caps_on  = bool(con.get('enabled',               True))

    if not caps_on:
        # caps disabled — every served student is trivially satisfied
        return sum(1 for s in sol.students if getattr(s, 'is_served', False))

    def _cap(d):
        if d is None or d <= 0 or not math.isfinite(d):
            return float('inf')
        return max(fl, min(k * d, d + ce))

    satisfied = 0
    for route in sol.routes:
        school_node = route.stops[-1].node_id
        for stop in route.stops:
            if stop.stop_type == 'school':
                continue
            stop_idx = next((i for i, s in enumerate(route.stops) if s is stop), -1)
            if stop_idx == -1:
                continue

            # AM ride: boarding stop → school
            ride_am = calculate_route_time_from_matrix(route.stops[stop_idx:], G)
            if ride_am >= 9999:
                ride_am = float('inf')

            for student in stop.students:
                s_node = _eng.fast_nearest_node(G, student.coords[1], student.coords[0])
                try:
                    _, direct_am = find_shortest_path_with_turns(
                        G, s_node, school_node, weight='travel_time')
                    if not math.isfinite(direct_am):
                        direct_am = None
                except Exception:
                    direct_am = None

                cap_am = _cap(direct_am)
                am_ok  = ride_am <= cap_am

                if not bidir:
                    if am_ok:
                        satisfied += 1
                else:
                    if am_ok:
                        satisfied += 1
                    else:
                        # Check PM direction
                        ride_pm = _compute_pm_ride_time(route, stop, G)
                        try:
                            _, direct_pm = find_shortest_path_with_turns(
                                G, school_node, s_node, weight='travel_time')
                            if not math.isfinite(direct_pm):
                                direct_pm = None
                        except Exception:
                            direct_pm = None
                        cap_pm = _cap(direct_pm)
                        if ride_pm <= cap_pm:
                            satisfied += 1
    return satisfied


def _add_unserved_layer(m, sol, mode_key):
    """Add a FeatureGroup with X-pin markers for every unserved student in *sol*."""
    show = mode_key in ("A", "B")
    fg = FeatureGroup(name=f"{_MODE_NAMES[mode_key]} – Unserved Students", show=show)
    for student in sol.students:
        if getattr(student, 'is_served', False):
            continue
        stage_name = (
            student.school_stage.name
            if hasattr(student.school_stage, 'name')
            else str(student.school_stage)
        )
        popup_html = (
            f'<div style="width:220px;font-size:12px;">'
            f'<b style="color:#c0392b;">&#x2716; Unserved</b><br>'
            f'<b>Student: {student.id}</b><br>'
            f'Stage: {stage_name}<br>'
            f'Home: {student.coords[0]:.5f}, {student.coords[1]:.5f}<br>'
            f'Mode: {_MODE_NAMES[mode_key]}'
            f'</div>'
        )
        folium.Marker(
            location=student.coords,
            tooltip=f"{student.id} ({stage_name}) — UNSERVED",
            popup=folium.Popup(popup_html, max_width=260),
            icon=folium.Icon(color='red', icon='times', prefix='fa'),
        ).add_to(fg)
    fg.add_to(m)
    return fg


def _add_crossing_markers(m, crossings_dict):
    """Collect all mode crossings into one FeatureGroup.  Returns fg or None."""
    all_cxs = [(mk, cx) for mk, cxs in crossings_dict.items() for cx in cxs]
    if not all_cxs:
        return None
    fg = FeatureGroup(name="Unsafe Crossings", show=True)
    seen = set()
    for mk, cx in all_cxs:
        lk = (round(cx["lat"], 6), round(cx["lon"], 6))
        if lk in seen:
            continue
        seen.add(lk)
        folium.CircleMarker(
            location=(cx["lat"], cx["lon"]), radius=6,
            color="red", fill=True, fillColor="yellow", fillOpacity=0.9, weight=2,
            tooltip=f"Unsafe crossing \u2013 {cx['student_id']}",
        ).add_to(fg)
    fg.add_to(m)
    return fg


def _build_custom_layer_control_js(
    map_var, fg_danger, fg_unclass,
    fgs_a, fgs_b, fgs_c,
    fg_crossings,
    fg_unserved_a=None, fg_unserved_b=None, fg_unserved_c=None,
):
    """Return JS that adds a titled, grouped layer-control widget to the map.

    Uses Folium's .get_name() to get each feature-group's actual JS variable
    name, so the control works on every freshly generated map.
    """
    va_r, va_w = fgs_a[0].get_name(), fgs_a[1].get_name()
    vb_r, vb_w = fgs_b[0].get_name(), fgs_b[1].get_name()
    vc_r, vc_w = fgs_c[0].get_name(), fgs_c[1].get_name()
    v_danger  = fg_danger.get_name()
    v_unclass = fg_unclass.get_name()

    crossings_row = ""
    if fg_crossings is not None:
        vc = fg_crossings.get_name()
        crossings_row = f"""
                row('Unsafe Crossings',
                    [{vc}],
                    map.hasLayer({vc}));"""

    unserved_rows = ""
    for fg_u, label in [
        (fg_unserved_a, 'Constrained – Unserved'),
        (fg_unserved_b, 'Unconstrained – Unserved'),
        (fg_unserved_c, 'Door-to-Door – Unserved'),
    ]:
        if fg_u is not None:
            vu = fg_u.get_name()
            unserved_rows += f"""
                row('{label}',
                    [{vu}],
                    map.hasLayer({vu}));"""

    return f"""
    window.addEventListener('load', function() {{
        var m = {map_var};
        var CustomCtrl = L.Control.extend({{
            options: {{ position: 'topright' }},
            onAdd: function(map) {{
                var c = L.DomUtil.create('div',
                    'leaflet-control-layers leaflet-control-layers-expanded');
                c.style.cssText =
                    'padding:10px 14px;min-width:210px;font-size:13px;' +
                    'font-family:Arial,sans-serif;line-height:1.4;';
                L.DomEvent.disableClickPropagation(c);
                L.DomEvent.disableScrollPropagation(c);
                // title
                var title = L.DomUtil.create('div', '', c);
                title.innerHTML = 'Route Views';
                title.style.cssText =
                    'font-weight:bold;font-size:14px;' +
                    'margin-bottom:8px;padding-bottom:6px;' +
                    'border-bottom:2px solid #bbb;';
                function sep() {{
                    var d = L.DomUtil.create('div', '', c);
                    d.style.cssText = 'border-top:1px solid #e0e0e0;margin:5px 0;';
                }}
                function row(label, layers, on) {{
                    var lbl = L.DomUtil.create('label', '', c);
                    lbl.style.cssText =
                        'display:flex;align-items:center;gap:7px;' +
                        'margin:5px 0;cursor:pointer;';
                    var cb = document.createElement('input');
                    cb.type = 'checkbox';
                    cb.checked = on;
                    cb.style.cssText =
                        'width:14px;height:14px;cursor:pointer;flex-shrink:0;';
                    cb.addEventListener('change', function() {{
                        layers.forEach(function(fg) {{
                            cb.checked ? map.addLayer(fg) : map.removeLayer(fg);
                        }});
                    }});
                    lbl.appendChild(cb);
                    var span = document.createElement('span');
                    span.textContent = label;
                    lbl.appendChild(span);
                }}
                row('Constrained (Safe Walking)',
                    [{va_r}, {va_w}], map.hasLayer({va_r}));
                row('Unconstrained (Any Walking)',
                    [{vb_r}, {vb_w}], map.hasLayer({vb_r}));
                row('Direct (No Walking)',
                    [{vc_r}, {vc_w}], map.hasLayer({vc_r}));
                sep();
                row('Dangerous Roads (unsafe to cross)',
                    [{v_danger}], map.hasLayer({v_danger}));
                row('Unclassified Roads (no student placement)',
                    [{v_unclass}], map.hasLayer({v_unclass}));{crossings_row}{unserved_rows}
                return c;
            }}
        }});
        new CustomCtrl().addTo(m);
    }});
    """


def _build_stats_html(all_stats, crossings_dict, occupancies_dict):
    blocks = ""
    for mk in ("A", "B", "C"):
        s = all_stats[mk]
        cx = len(crossings_dict.get(mk, []))
        occ = occupancies_dict.get(mk, [])
        avg_occ = (sum(occ) / len(occ)) if occ else 0
        cx_color = "#c0392b" if cx > 0 else "#27ae60"
        mc = _ROUTE_COLORS[mk][0]
        blocks += f"""
        <div style="margin-bottom:8px; padding-bottom:8px;
                    border-bottom:1px solid #e0e0e0;">
          <div style="font-weight:bold; color:{mc}; margin-bottom:3px;">
            {mk}: {_MODE_NAMES[mk]}
          </div>
          <table style="width:100%; border-collapse:collapse;
                        font-size:11px; text-align:center;">
            <tr style="color:#555;">
              <td style="text-align:left; padding:1px 4px;">Routes</td>
              <td style="text-align:left; padding:1px 4px;">Total Time</td>
              <td style="text-align:left; padding:1px 4px;">Distance</td>
              <td style="text-align:left; padding:1px 4px;">Avg Occ.</td>
              <td style="text-align:left; padding:1px 4px;">Served</td>
              <td style="text-align:left; padding:1px 4px;">Satisfied</td>
              <td style="text-align:left; padding:1px 4px;">Crossings</td>
            </tr>
            <tr style="font-weight:bold;">
              <td style="padding:1px 4px;">{s['routes']}</td>
              <td style="padding:1px 4px;">{s['total_time']:.0f} min</td>
              <td style="padding:1px 4px;">{s['total_dist']:.1f} km</td>
              <td style="padding:1px 4px;">{avg_occ:.1f}</td>
              <td style="padding:1px 4px;">{s['served']}/{s['total']}</td>
              <td style="padding:1px 4px;">{s.get('satisfied', '—')}/{s['served']}</td>
              <td style="padding:1px 4px; color:{cx_color};">{cx}</td>
            </tr>
          </table>
        </div>"""

    return f"""
    <div style="position:fixed; bottom:15px; right:15px; width:430px;
                max-height:calc(100vh - 80px); overflow-y:auto;
                background:white; border:2px solid #555; z-index:9999;
                padding:12px 14px; border-radius:6px; font-size:12px;
                font-family:Arial,sans-serif; box-shadow:2px 2px 8px rgba(0,0,0,.25);">
      <div style="font-weight:bold; font-size:13px; margin-bottom:10px;
                  padding-bottom:6px; border-bottom:2px solid #ccc;">
        Three-Mode Routing Comparison
      </div>
      {blocks}
      <div style="font-size:10px; color:#888; margin-top:4px;">
        Toggle layers via top-right control.<br>
        <span style="color:#e74c3c;">&#x2015;&#x2015;</span> Dangerous roads
        &nbsp;&nbsp;
        <span style="color:#7f8c8d;">&#x2508;&#x2508;</span> Unclassified roads
      </div>
    </div>
    """


# ────────────────────────────────────────────────────────────────────
# METRICS HELPERS
# ────────────────────────────────────────────────────────────────────
_WALK_SPEED_M_PER_MIN = 80.0  # comfortable pedestrian (≈ 4.8 km/h)


def _compute_walk_stats(sol, G, stage_walk):
    """Return walk-distance statistics for one solution.

    Uses ``walk_distance_on_roads`` for road-network accuracy, falling back
    to straight-line Haversine when the path isn't found.
    """
    dists = []
    utils = []
    for route in sol.routes:
        for stop in route.stops:
            if stop.stop_type == "school":
                continue
            for student in stop.students:
                s_node = _eng.fast_nearest_node(G, student.coords[1], student.coords[0])
                d = walk_distance_on_roads(G, s_node, stop.node_id)
                if d <= 0:          # fallback: straight-line
                    dlat = math.radians(stop.coords[0] - student.coords[0])
                    dlon = math.radians(stop.coords[1] - student.coords[1])
                    a = (math.sin(dlat / 2) ** 2
                         + math.cos(math.radians(student.coords[0]))
                         * math.cos(math.radians(stop.coords[0]))
                         * math.sin(dlon / 2) ** 2)
                    d = 6_371_000 * 2 * math.asin(math.sqrt(a))
                dists.append(d)
                # utilisation = fraction of walk budget actually used
                stage_name = (
                    student.school_stage.name
                    if hasattr(student.school_stage, "name")
                    else str(student.school_stage)
                )
                walk_max = stage_walk.get(stage_name, 0)
                if walk_max > 0:
                    utils.append(min(d / walk_max, 1.0))

    if not dists:
        return {"avg_walk_dist_m": 0, "median_walk_dist_m": 0,
                "max_walk_dist_m": 0, "min_walk_dist_m": 0,
                "avg_walk_time_min": 0, "avg_walk_utilisation_pct": None}
    return {
        "avg_walk_dist_m":        round(statistics.mean(dists),   1),
        "median_walk_dist_m":     round(statistics.median(dists), 1),
        "max_walk_dist_m":        round(max(dists),               1),
        "min_walk_dist_m":        round(min(dists),               1),
        "avg_walk_time_min":      round(statistics.mean(dists) / _WALK_SPEED_M_PER_MIN, 2),
        "avg_walk_utilisation_pct": round(statistics.mean(utils) * 100, 1) if utils else None,
    }


def _build_metrics(meta, stage_walk, all_stats, crossings_dict,
                   sol_a, sol_b, sol_c, G_unc, iters):
    """Assemble the full metrics dict that will be written to metrics.json."""
    mode_map = {
        "constrained":   ("A", sol_a),
        "unconstrained": ("B", sol_b),
        "door_to_door":  ("C", sol_c),
    }
    modes_out = {}
    for mode_key, (mk, sol) in mode_map.items():
        s   = all_stats[mk]
        cx  = len(crossings_dict.get(mk, []))
        n_routes = s["routes"]
        # walk stats: door-to-door has walk_radius=0, so no utilisation
        sw = stage_walk if mode_key != "door_to_door" else {k: 0 for k in stage_walk}
        walk = _compute_walk_stats(sol, G_unc, sw)
        
        # Build student list with ride time, direct potential, walk distance
        students_list = []
        for route in sol.routes:
            # Pre-compute per-stop ride-time using matrix-cache summation
            # (safe against cleared caches; falls back to lazy A* on miss)
            school_node = route.stops[-1].node_id
            for stop in route.stops:
                if stop.stop_type == "school":
                    continue

                stop_idx = next((i for i, s in enumerate(route.stops) if s is stop), -1)
                if stop_idx == -1:
                    continue

                # Ride time = sum of legs from this stop to school (matrix-safe)
                ride_time = calculate_route_time_from_matrix(route.stops[stop_idx:], G_unc)
                if ride_time >= 9999:
                    ride_time = None  # truly unreachable; treat as unknown

                for student in stop.students:
                    stage_name = (
                        student.school_stage.name
                        if hasattr(student.school_stage, "name")
                        else str(student.school_stage)
                    )

                    # Direct time home -> school (cached on student after first call)
                    direct_time = compute_direct_time(student, school_node, G_unc)
                    if not math.isfinite(direct_time):
                        direct_time = None

                    # Walk distance home -> assigned stop
                    s_node = _eng.fast_nearest_node(G_unc, student.coords[1], student.coords[0])
                    walk_dist = walk_distance_on_roads(G_unc, s_node, stop.node_id)
                    if walk_dist <= 0 or not math.isfinite(walk_dist):
                        walk_dist = 0.0

                    students_list.append({
                        "id": student.id,
                        "stage": stage_name,
                        "ride_time_min": round(ride_time, 2) if ride_time is not None else None,
                        "direct_potential_min": round(direct_time, 2) if direct_time is not None else None,
                        "walk_distance_m": round(walk_dist, 1),
                    })
        
        modes_out[mode_key] = {
            "routes_created":       n_routes,
            "students_served":      s["served"],
            "students_unserved":    s["total"] - s["served"],
            "total_route_time_min": round(s["total_time"], 2),
            "total_route_dist_km":  round(s["total_dist"],  2),
            "avg_route_time_min":   round(s["total_time"] / n_routes, 2) if n_routes else 0,
            "runtime_seconds":      round(s["runtime"],     2),
            "unsafe_crossings":     cx,
            "walk_stats":           walk,
            "students":             students_list,
        }

    # cross-mode comparisons
    t_con  = modes_out["constrained"]["total_route_time_min"]
    t_unc  = modes_out["unconstrained"]["total_route_time_min"]
    t_d2d  = modes_out["door_to_door"]["total_route_time_min"]
    cx_con = modes_out["constrained"]["unsafe_crossings"]
    cx_unc = modes_out["unconstrained"]["unsafe_crossings"]

    return {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": {
            "n_students":       meta.get("n_students"),
            "seed":             meta.get("seed"),
            "iterations":       iters,
            "buses_capacity":   meta.get("buses", {}).get("capacity"),
            "stage_walk_limits": stage_walk,
            "stage_distribution": {
                k: v for k, v in meta.get("stage_distribution", {}).items()
                if k != "_comment"
            },
        },
        "modes": modes_out,
        "comparison": {
            "efficiency_gain_vs_d2d_pct": (
                round((t_d2d - t_con) / t_d2d * 100, 1) if t_d2d else None
            ),
            "safety_cost_vs_unconstrained_pct": (
                round((t_con - t_unc) / t_unc * 100, 1) if t_unc else None
            ),
            "crossings_eliminated_vs_unconstrained": cx_unc - cx_con,
            "constrained_total_time_min":   t_con,
            "unconstrained_total_time_min": t_unc,
            "door_to_door_total_time_min":  t_d2d,
        },
    }


def _sanitise_floats(obj):
    """Recursively replace non-finite floats with None so json.dump stays valid."""
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitise_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise_floats(v) for v in obj]
    return obj


# ────────────────────────────────────────────────────────────────────
# PUBLIC API  (callable from thin launchers)
# ────────────────────────────────────────────────────────────────────
def run(input_path=None, output_path=None, iterations=None):
    """Run the three-mode comparison and save the map.

    Parameters
    ----------
    input_path  : str | None
        Path to an input.json file.  Defaults to the bundled input.json
        sitting next to this script.
    output_path : str | None
        Absolute path for the output HTML map.  Defaults to
        ``comparison_map.html`` next to the input file.
    iterations : int | None
        Override ALNS iteration count from input.json.
    """
    meta = _load_meta(input_path)
    iters  = iterations or meta.get("algorithm", {}).get("iterations", 30)

    # Resolve where to write the map
    if output_path:
        output = output_path
    else:
        rel = meta.get("output", "comparison_map.html")
        base = os.path.dirname(input_path) if input_path else _SCRIPT_DIR
        output = rel if os.path.isabs(rel) else os.path.join(base, rel)

    school_cfg = meta["school"]
    raw_walk = meta.get("stage_walk_limits", DEFAULT_STAGE_WALK_LIMITS)
    # Filter out _comment and other non-stage keys
    stage_walk = {k: v for k, v in raw_walk.items()
                  if k in ("KG", "ELEMENTARY", "MIDDLE", "HIGH")}

    print("=" * 60)
    print("  THREE-MODE ROUTING COMPARISON  (input.json)")
    print("=" * 60)
    print(f"  Students : {meta['n_students']}")
    print(f"  Seed     : {meta['seed']}")
    print(f"  Stages   : {meta['stage_distribution']}")
    print(f"  Walk lim : {stage_walk}")
    print(f"  Iters    : {iters}")
    print()

    # ── 1. Generate dataset ──
    print("[1/7] Generating dataset …")
    base_data = _generate_dataset(meta)
    base_data["meta"]["algorithm"]["iterations"] = iters

    # Print stage breakdown
    stage_counts = {}
    for s in base_data["data"]["students"]:
        stage_counts[s["school_stage"]] = stage_counts.get(s["school_stage"], 0) + 1
    print(f"  Stage breakdown: {stage_counts}")

    # ── 2. Constrained graph ──
    print("\n[2/7] Building CONSTRAINED graph …")
    G_con = setup_graph(base_data["meta"], unconstrained=False)
    _prebuild_ball_tree(G_con)

    import copy as _copy
    G_con_saved = _copy.deepcopy(G_con)

    # ── 3. Unconstrained graph ──
    print("[3/7] Building UNCONSTRAINED graph …")
    G_unc = setup_graph(base_data["meta"], unconstrained=True)
    G_con = G_con_saved
    _eng._BALL_TREE = None
    _eng._BALL_TREE_GRAPH_ID = None
    _eng._BALL_TREE_NODE_IDS = None

    # ── 4. Mode A: Constrained ──
    # Walking BFS uses G_con (safety-restricted edges).
    # Bus driving distances ALWAYS use G_unc (full road network).
    _ride_caps_on = meta.get("constraints", {}).get("enabled", True)
    print("\n" + "-" * 50)
    print("MODE A: Constrained (safety ON, stage walk radii)")
    print("-" * 50)
    data_a = _make_constrained(base_data)
    if not _ride_caps_on:
        _relax_ride_constraints(data_a)
    _reset_caches()
    _prebuild_ball_tree(G_con)
    sol_a, stats_a, school_a = run_algorithm(
        data_a, G_con, iterations=iters, stage_walk_limits=stage_walk,
        G_drive=G_unc)
    stats_a["label"] = "Mode-A"
    print(f"  [A] {stats_a['served']}/{stats_a['total']} served | "
          f"routes={stats_a['routes']} | time={stats_a['total_time']:.1f} min | "
          f"{stats_a['runtime']:.1f}s")

    # ── 5. Mode B: Unconstrained ──
    print("\n" + "-" * 50)
    print("MODE B: Unconstrained (all safe, same walk radius)")
    print("-" * 50)
    data_b = _make_unconstrained(base_data)
    if not _ride_caps_on:
        _relax_ride_constraints(data_b)
    _reset_caches()
    _prebuild_ball_tree(G_unc)
    sol_b, stats_b, school_b = run_algorithm(data_b, G_unc, iterations=iters,
                                              G_drive=G_unc)
    stats_b["label"] = "Mode-B"
    print(f"  [B] {stats_b['served']}/{stats_b['total']} served | "
          f"routes={stats_b['routes']} | time={stats_b['total_time']:.1f} min | "
          f"{stats_b['runtime']:.1f}s")

    # ── 6. Mode C: Door-to-Door ──
    print("\n" + "-" * 50)
    print("MODE C: Door-to-Door (walk=0, bus visits every home)")
    print("-" * 50)
    data_c = _make_door_to_door(base_data)
    if not _ride_caps_on:
        _relax_ride_constraints(data_c)
    _reset_caches()
    _prebuild_ball_tree(G_unc)
    sol_c, stats_c, school_c = run_algorithm(data_c, G_unc, iterations=iters,
                                              G_drive=G_unc)
    stats_c["label"] = "Mode-C"
    print(f"  [C] {stats_c['served']}/{stats_c['total']} served | "
          f"routes={stats_c['routes']} | time={stats_c['total_time']:.1f} min | "
          f"{stats_c['runtime']:.1f}s")

    # ── 7. Build map ──
    print("\n" + "-" * 50)
    print("BUILDING COMPARISON MAP")
    print("-" * 50)

    center = (school_cfg["latitude"], school_cfg["longitude"])
    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")

    # School marker
    folium.Marker(
        location=center, popup="<b>SCHOOL</b>", tooltip="School",
        icon=folium.Icon(color="darkgreen", icon="graduation-cap", prefix="fa"),
    ).add_to(m)

    # Dangerous roads layer
    fg_danger = FeatureGroup(name="\u26A0 Dangerous Roads (unsafe to cross)", show=True)
    danger_segs = _extract_segments(G_con, center[0], center[1], "dangerous")
    for seg in danger_segs:
        folium.PolyLine(seg, color="#e74c3c", weight=3, opacity=0.45,
                        dash_array="6,4").add_to(fg_danger)
    fg_danger.add_to(m)
    print(f"  Dangerous-road segments: {len(danger_segs)}")

    # Unclassified roads layer
    fg_unclass = FeatureGroup(name="Unclassified Roads (no student placement)", show=False)
    unclass_segs = _extract_segments(G_con, center[0], center[1], "unclassified")
    for seg in unclass_segs:
        folium.PolyLine(seg, color="#7f8c8d", weight=2, opacity=0.5,
                        dash_array="3,5", tooltip="Unclassified road").add_to(fg_unclass)
    fg_unclass.add_to(m)
    print(f"  Unclassified-road segments: {len(unclass_segs)}")

    crossings_dict, occupancies_dict, all_stats = {}, {}, {}

    # Clear path cache so rendering computes fresh turn-aware paths on G_unc
    _eng._path_cache.clear()
    _eng._MATRIX_CACHE.clear()
    _eng._MATRIX_CACHE_LENGTH.clear()

    # Bus routes are always rendered with G_unc (the bus drives on all roads)
    fgs = {}          # mk -> (fg_routes, fg_walks)
    fgs_unserved = {}  # mk -> fg_unserved
    for mk, sol, stats in [
        ("A", sol_a, stats_a),
        ("B", sol_b, stats_b),
        ("C", sol_c, stats_c),
    ]:
        print(f"  Drawing Mode {mk} …")
        fg_r, fg_w, cx, occ = _add_route_layer(m, G_unc, sol, mk, G_con,
                                                 constraints=meta.get("constraints"))
        fgs[mk] = (fg_r, fg_w)
        crossings_dict[mk] = cx
        occupancies_dict[mk] = occ
        all_stats[mk] = stats
        all_stats[mk]["satisfied"] = _count_satisfied(sol, G_unc, meta.get("constraints", {}))
        fgs_unserved[mk] = _add_unserved_layer(m, sol, mk)

    fg_crossings = _add_crossing_markers(m, crossings_dict)

    # Custom grouped layer control (title + 3 mode checkboxes, no radio buttons)
    map_var = f"map_{m._id}"
    ctrl_js = _build_custom_layer_control_js(
        map_var, fg_danger, fg_unclass,
        fgs["A"], fgs["B"], fgs["C"],
        fg_crossings,
        fg_unserved_a=fgs_unserved.get("A"),
        fg_unserved_b=fgs_unserved.get("B"),
        fg_unserved_c=fgs_unserved.get("C"),
    )
    m.get_root().script.add_child(folium.Element(ctrl_js))

    m.get_root().html.add_child(folium.Element(
        _build_stats_html(all_stats, crossings_dict, occupancies_dict)))

    m.save(output)
    fsize_kb = os.path.getsize(output) / 1024
    print(f"\n  Map saved: {output}  ({fsize_kb:.0f} KB)")

    # ── Metrics JSON ──
    metrics = _build_metrics(
        meta, stage_walk, all_stats, crossings_dict,
        sol_a, sol_b, sol_c, G_unc, iters,
    )
    metrics_path = os.path.join(os.path.dirname(output), "output.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(_sanitise_floats(metrics), f, indent=2, ensure_ascii=False)
    print(f"  Metrics  : {metrics_path}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    hdr = f"{'Mode':<30} {'Routes':>6} {'Time':>8} {'Dist':>8} {'Served':>8} {'Crossings':>10}"
    print(hdr)
    print("-" * len(hdr))
    for mk in ("A", "B", "C"):
        s = all_stats[mk]
        cx = len(crossings_dict.get(mk, []))
        print(f"{_MODE_NAMES[mk]:<30} {s['routes']:>6} {s['total_time']:>8.1f} "
              f"{s['total_dist']:>8.1f} {s['served']}/{s['total']:>5} {cx:>10}")
    print(f"\nStage distribution used: { {k:v for k,v in meta['stage_distribution'].items() if k != '_comment'} }")
    print(f"Walk limits used: {stage_walk}")
    print(f"\nOpen '{output}' in a browser to explore.")


# ────────────────────────────────────────────────────────────────────
# CLI entry-point
# ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Three-Mode Comparison runner (reads input.json)",
    )
    parser.add_argument(
        "--input", default=None,
        help="Path to input.json (default: input.json next to this script)",
    )
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="Override ALNS iterations from meta.json",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output HTML path (default: from meta.json)",
    )
    args = parser.parse_args()
    run(
        input_path=args.input,
        output_path=args.output,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
