"""
common.py – OSM Parsing, Graph, and Simulation Engine (Shared)
Defines constants, graph utilities, and helper functions used across all scripts.
"""

import xml.etree.ElementTree as ET
import math, random, heapq, time
from collections import defaultdict
import numpy as np
import pandas as pd

# ── File paths ─────────────────────────────────────────────────────────────────
OSM_FILE   = 'map.osm'                               # OpenStreetMap road network
EXCEL_FILE = 'Kochi_Shioe_evacuation_buildings.xlsx' # Shelter locations and capacities

# ── Simulation constants ───────────────────────────────────────────────────────
BASE_SPEED   = 1.2    # pedestrian free-flow speed (m/s)
ROAD_WIDTH   = 5.0    # effective road width (m); used to compute pedestrian density
REWARD_DEST  = 1.0    # terminal reward given to an agent upon reaching a shelter
GAMMA        = 0.99   # discount factor (higher = rewards propagate further back in time)
TOTAL_EPISODES  = 20000
MAX_STEPS_EP    = 600  # max steps per episode; slightly generous to allow rerouting
N_EXEC_AGENTS   = 3000
N_TRAIN_AGENTS  = N_EXEC_AGENTS
DT              = 5.0     # simulation timestep (seconds)
MAX_TIME        = 3600    # max simulation time (seconds = 1 hour)
ACTOR_PATH      = 'actor.npy'          # saved actor weights for execution
GRAPH_PATH      = 'graph_data.pkl'     # cached graph so OSM re-parsing can be skipped
SEED = 42


def haversine(c1, c2):
    """Great-circle distance (m) between two (lat, lon) points."""
    la1,lo1=c1; la2,lo2=c2; R=6371000; p=math.pi/180
    h=(math.sin((la2-la1)*p/2)**2
       +math.cos(la1*p)*math.cos(la2*p)*math.sin((lo2-lo1)*p/2)**2)
    return R*2*math.asin(math.sqrt(h))


def is_inside_polygon(lon, lat, polygon):
    """Ray-casting point-in-polygon test. Used to exclude a non-study region from the OSM parse."""
    n=len(polygon); inside=False; p1x,p1y=polygon[0]
    for i in range(n+1):
        p2x,p2y=polygon[i%n]
        if lat>min(p1y,p2y):
            if lat<=max(p1y,p2y):
                if lon<=max(p1x,p2x):
                    if p1y!=p2y:
                        xints=(lat-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x==p2x or lon<=xints:
                        inside=not inside
        p1x,p1y=p2x,p2y
    return inside


def find_nearest_node(target_lon, target_lat, node_coords, road_nodes):
    """Return the road node id closest to a given (lon, lat) point (Euclidean approx)."""
    best_node=None; min_dist=float('inf')
    for node_id in road_nodes:
        if node_id not in node_coords: continue
        lat,lon=node_coords[node_id]
        dist=(lon-target_lon)**2+(lat-target_lat)**2
        if dist<min_dist: min_dist=dist; best_node=node_id
    return best_node


def parse_osm(path=OSM_FILE):
    """
    Parse an OSM file into a walkable road graph.
    Steps:
      1. Load all node coordinates, excluding a designated non-study polygon.
      2. Build adjacency dict for walkable highway types (footway, residential, etc.).
      3. Keep only the largest connected component (removes isolated fragments).
    Returns: node_coords, adj (dict-of-dicts with haversine edge weights), road_nodes (set).
    """
    print("Parsing OSM...")
    t0=time.time(); tree=ET.parse(path); root=tree.getroot()
    EXCL=[(133.530,33.562),(133.565,33.562),(133.565,33.550),
          (133.545,33.557),(133.530,33.553)]
    node_coords={}
    for e in root:
        if e.tag!='node': continue
        lat=float(e.get('lat')); lon=float(e.get('lon'))
        if is_inside_polygon(lon,lat,EXCL): continue
        node_coords[e.get('id')]=(lat,lon)
    WALKABLE={'footway','unclassified','residential','tertiary','primary',
              'secondary','pedestrian','tertiary_link','path','service'}
    adj=defaultdict(dict)
    for w in root:
        if w.tag!='way': continue
        tags={t.get('k'):t.get('v') for t in w if t.tag=='tag'}
        if tags.get('highway','') not in WALKABLE: continue
        refs=[n.get('ref') for n in w if n.tag=='nd' and n.get('ref') in node_coords]
        for i in range(len(refs)-1):
            a,b=refs[i],refs[i+1]; d=haversine(node_coords[a],node_coords[b])
            adj[a][b]=min(adj[a].get(b,9e9),d); adj[b][a]=min(adj[b].get(a,9e9),d)
    # Keep only the largest connected component
    road_nodes=set(adj.keys()); visited,best=set(),set()
    for s in road_nodes:
        if s in visited: continue
        comp,q=set(),[s]
        while q:
            n=q.pop()
            if n in comp: continue
            comp.add(n); visited.add(n)
            q.extend(nb for nb in adj.get(n,{}) if nb not in comp)
        if len(comp)>len(best): best=comp
    adj={n:{nb:d for nb,d in nbdict.items() if nb in best}
         for n,nbdict in adj.items() if n in best}
    print(f"    Completed ({time.time()-t0:.1f}s)  Nodes: {len(best)}  Edges: {sum(len(v) for v in adj.values())//2}")
    return node_coords,adj,best


def load_evac_data(excel_path, node_coords, road_nodes):
    """
    Read shelter locations and capacities from Excel.
    Each row's (lon, lat) is snapped to the nearest road node.
    Returns: evac_nodes (set of node ids), evac_capacity (dict {node_id: capacity}).
    """
    print(f"Loading evacuation building data ({excel_path})...")
    df=pd.read_excel(excel_path); road_list=list(road_nodes)
    evac_nodes=set(); evac_capacity={}; success=0
    for idx,row in df.iterrows():
        if idx>86: break
        try:
            lon=float(row.iloc[3]); lat=float(row.iloc[4]); capacity=int(row.iloc[5])
            nn=find_nearest_node(lon,lat,node_coords,road_list)
            if nn:
                evac_nodes.add(nn); evac_capacity[nn]=evac_capacity.get(nn,0)+capacity; success+=1
        except Exception: pass
    print(f"  {success} facilities -> {len(evac_nodes)} road nodes set as evacuation destinations")
    return evac_nodes,evac_capacity


def compute_shelter_distances(evac_nodes, adj, road_nodes, return_source=False):
    """
    Multi-source Dijkstra from all shelters simultaneously.
    Returns dist[node] = shortest distance (m) to the nearest shelter.
    Used for potential-based reward shaping: phi(node) = -dist[node]/max_dist in [-1, 0].

    return_source: if True, also returns src[node] = which shelter node the
      shortest path came from. Needed by evac_env.py to look up the fullness
      of each agent's nearest shelter. Defaults to False for backward compatibility.
    """
    dist={}; src={}; heap=[]
    for shelter in evac_nodes:
        if shelter in road_nodes:
            dist[shelter]=0.0; src[shelter]=shelter; heapq.heappush(heap,(0.0,shelter))
    vis=set()
    while heap:
        d,u=heapq.heappop(heap)
        if u in vis: continue
        vis.add(u)
        for v,w in adj.get(u,{}).items():
            nd=d+w
            if nd<dist.get(v,float('inf')):
                # Propagate both distance and source shelter along shortest path
                dist[v]=nd; src[v]=src[u]; heapq.heappush(heap,(nd,v))
    max_d=max(dist.values()) if dist else 1.0
    for n in road_nodes:
        if n not in dist:
            dist[n]=max_d*2
            # Fallback for disconnected nodes: assign an arbitrary shelter to avoid KeyError
            src[n] = next(iter(evac_nodes)) if evac_nodes else n
    if return_source:
        return dist, max_d, src
    return dist, max_d


def compute_distances_from_node(source, adj, road_nodes):
    """
    Single-source Dijkstra from `source`.
    Returns (dist, hops):
      dist[node] = shortest-path distance (m) from source to node
      hops[node] = number of edges traversed along that shortest-distance path
    Used by evac_env.py to build per-shelter distance/hop tables (phi shaping,
    neighbor ordering, and hop-based progress) when each agent has its own
    randomly-assigned destination shelter.
    """
    dist = {source: 0.0}; hops = {source: 0}; heap = [(0.0, 0, source)]
    vis = set()
    while heap:
        d, h, u = heapq.heappop(heap)
        if u in vis: continue
        vis.add(u)
        for v, w in adj.get(u, {}).items():
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd; hops[v] = h + 1
                heapq.heappush(heap, (nd, h + 1, v))
    for n in road_nodes:
        if n not in dist:
            dist[n] = float('inf'); hops[n] = 10**6   # unreachable fallback
    return dist, hops


def dijkstra(start, targets, adj):
    """
    Single-source Dijkstra returning the shortest path (list of node ids)
    from start to the nearest node in targets.
    """
    dist={start:0.0}; prev={start:None}; heap=[(0.0,start)]; vis=set()
    while heap:
        d,u=heapq.heappop(heap)
        if u in vis: continue
        vis.add(u)
        if u in targets:
            path=[]; cur=u
            while cur is not None: path.append(cur); cur=prev[cur]
            return list(reversed(path))
        for v,w in adj.get(u,{}).items():
            nd=d+w
            if v not in dist or nd<dist[v]:
                dist[v]=nd; prev[v]=u; heapq.heappush(heap,(nd,v))
    return [start]


def build_graph_index(adj, road_nodes, shelter_dist=None):
    """
    Build integer-indexed graph structures for fast array-based lookups.

    Returns:
      node_list      : sorted list of node ids (index → node id)
      node_to_idx    : dict {node_id → index}
      neighbor_lists : list of lists; neighbor_lists[i] = [idx of neighbor, ...]
      max_degree     : maximum number of neighbors across all nodes (= action space size)

    shelter_dist: if provided, neighbors are sorted by ascending distance-to-shelter
      so that action index 0 always means "move toward the nearest shelter."
      Without this, action indices have no consistent spatial meaning across nodes,
      making it structurally hard for a shared-parameter network to generalize.
      If None, falls back to lexicographic sort (original behavior).
    """
    node_list=sorted(road_nodes)
    node_to_idx={n:i for i,n in enumerate(node_list)}
    n_nodes=len(node_list)
    neighbor_lists=[[] for _ in range(n_nodes)]
    for node_id in node_list:
        idx=node_to_idx[node_id]
        neighbor_ids = [nb for nb in adj.get(node_id,{}).keys() if nb in node_to_idx]
        if shelter_dist is not None:
            neighbor_ids.sort(key=lambda nb: shelter_dist.get(nb, float('inf')))
        else:
            neighbor_ids.sort()
        neighbor_lists[idx]=[node_to_idx[nb] for nb in neighbor_ids]
    max_degree=max((len(nb) for nb in neighbor_lists),default=1)
    return node_list,node_to_idx,neighbor_lists,max_degree


def build_grid_graph(rows=5, cols=5, cell_size_m=80.0, connect_diagonals=True):
    """
    Build a small synthetic rows x cols grid road network, as a drop-in
    replacement for parse_osm()'s output. Useful for quick, easy-to-reason-
    about experiments (e.g. a 5x5 grid) instead of the full Kochi map.

    Node ids are 'r{row}c{col}'. node_coords stores (row, col) scaled into
    degree-like units (cell_size_m / 111000) so it's still a (lat, lon)-style
    pair — this keeps existing plotting code (which expects (lat, lon)
    tuples) working unchanged. Edge weights in `adj` are the real distance
    in meters (cell_size_m for orthogonal neighbors, cell_size_m*sqrt(2) for
    diagonal neighbors), which is what all the distance/Dijkstra logic
    actually uses — node_coords is only ever used for plotting.

    Returns (node_coords, adj, road_nodes), matching parse_osm()'s signature.
    """
    node_coords = {}
    for r in range(rows):
        for c in range(cols):
            nid = f'r{r}c{c}'
            node_coords[nid] = (r * cell_size_m / 111000.0, c * cell_size_m / 111000.0)

    adj = defaultdict(dict)
    diag_dist = cell_size_m * math.sqrt(2)
    for r in range(rows):
        for c in range(cols):
            nid = f'r{r}c{c}'
            steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if connect_diagonals:
                steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dr, dc in steps:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    nb_id = f'r{nr}c{nc}'
                    dist = diag_dist if (dr != 0 and dc != 0) else cell_size_m
                    adj[nid][nb_id] = dist

    road_nodes = set(node_coords.keys())
    return node_coords, adj, road_nodes


def make_grid_evac_data(road_nodes, n_shelters=2, capacity_per_shelter=10**6, seed=None):
    """
    Randomly pick n_shelters nodes from road_nodes as evacuation shelters
    for a synthetic grid graph (see build_grid_graph). Capacity is set very
    high by default since evac_env.py's arrival logic does not enforce
    shelter capacity (destinations are assigned to agents without regard to
    capacity). Returns (evac_nodes, evac_capacity), matching
    load_evac_data()'s signature.
    """
    rng = random.Random(seed)
    nodes = sorted(road_nodes)
    chosen = rng.sample(nodes, min(n_shelters, len(nodes)))
    evac_nodes = set(chosen)
    evac_capacity = {n: capacity_per_shelter for n in chosen}
    return evac_nodes, evac_capacity


def get_all_edges(adj):
    """Return a deduplicated list of (a, b) edge tuples (a < b) from the adjacency dict."""
    edges=set()
    for a,nbdict in adj.items():
        for b in nbdict: edges.add((min(a,b),max(a,b)))
    return list(edges)