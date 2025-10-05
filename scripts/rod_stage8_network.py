import os, re, json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# -------------------- CONFIG --------------------
ISOLATED_DIR = r"isolated_symbols"
DECODED_CSV  = r"decoded_matches/decoded_symbols.csv"

OUT_DIR      = r"symbol_network_out"
os.makedirs(OUT_DIR, exist_ok=True)

# thresholds (tune as needed)
SCORE_THRESH       = 0.0   # keep only reasonably confident class matches
TEMPORAL_FRAME_GAP = 3      # connect across up to N previous frames
DIST_THRESH_PIX    = 40     # max distance (pixels) to treat as “same glyph across frames”
NEIGHBOR_RADIUS    = 55     # connect nearby symbols within same frame (sequence map)
MIN_SEQ_LEN        = 4      # only print/save sequences >= this length

# classes/colors for charts
CLASS_PALETTE = {
    "triangle": "#E4572E",
    "star6":    "#4E79A7",
    "spiral":   "#59A14F",
    "lines":    "#F28E2B",
    "circle":   "#B07AA1",
}

# ------------------------------------------------

print("Loading decoded symbols (Stage 6)…")
df = pd.read_csv(DECODED_CSV)
# df columns: file, class, score, tm_corr, hu_sim, width, height

# Parse frame,x,y from filenames like: symbol_0123_456_789.jpg
pattern = re.compile(r"symbol_(\d+)_(\d+)_(\d+)\.(?:jpg|png)$")
frame_idx, pos_x, pos_y = [], [], []

for f in df["file"]:
    m = pattern.search(f)
    if m:
        frame_idx.append(int(m.group(1)))
        pos_x.append(int(m.group(2)))
        pos_y.append(int(m.group(3)))
    else:
        frame_idx.append(np.nan); pos_x.append(np.nan); pos_y.append(np.nan)

df["frame"] = frame_idx
df["x"] = pos_x
df["y"] = pos_y
df = df.dropna(subset=["frame","x","y"]).copy()

# Filter by confidence + known classes (created in Stage 6)
known = set(CLASS_PALETTE.keys())
df = df[(df["class"].isin(known)) & (df["score"] >= SCORE_THRESH)].copy()
df["frame"] = df["frame"].astype(int)

print(f"Kept {len(df)} symbol instances after filters.")

# Build graph
G = nx.Graph()
# add nodes
for i, row in df.iterrows():
    nid = f"n{i}"
    G.add_node(
        nid,
        file=row["file"], cls=row["class"], score=float(row["score"]),
        frame=int(row["frame"]), x=int(row["x"]), y=int(row["y"])
    )

# index by frame for fast lookups
by_frame = defaultdict(list)
for n, d in G.nodes(data=True):
    by_frame[d["frame"]].append((n, d))

# temporal edges (persistence)
def dist(p, q):
    return np.hypot(p[0]-q[0], p[1]-q[1])

frames_sorted = sorted(by_frame.keys())
for f in frames_sorted:
    for dt in range(1, TEMPORAL_FRAME_GAP+1):
        prev = f - dt
        if prev not in by_frame: 
            continue
        curr_nodes = by_frame[f]
        prev_nodes = by_frame[prev]
        # connect “same class near same spot” across frames
        for n1, d1 in curr_nodes:
            for n2, d2 in prev_nodes:
                if d1["cls"] != d2["cls"]:
                    continue
                if dist((d1["x"], d1["y"]), (d2["x"], d2["y"])) <= DIST_THRESH_PIX:
                    G.add_edge(n1, n2, kind="temporal")

# spatial edges (within-frame adjacency)
for f in frames_sorted:
    items = by_frame[f]
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            n1, d1 = items[i]
            n2, d2 = items[j]
            if dist((d1["x"], d1["y"]), (d2["x"], d2["y"])) <= NEIGHBOR_RADIUS:
                G.add_edge(n1, n2, kind="spatial")

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Save graph for Gephi/Cytoscape
gexf_path = os.path.join(OUT_DIR, "symbol_network.gexf")
nx.write_gexf(G, gexf_path)
print(f"Saved Gephi network: {gexf_path}")

# Nodes/edges CSV
nodes_csv = os.path.join(OUT_DIR, "nodes.csv")
edges_csv = os.path.join(OUT_DIR, "edges.csv")
pd.DataFrame([
    {"id": n, **d} for n, d in G.nodes(data=True)
]).to_csv(nodes_csv, index=False)
pd.DataFrame([
    {"u": u, "v": v, **d} for u, v, d in G.edges(data=True)
]).to_csv(edges_csv, index=False)
print(f"Saved nodes and edges CSV -> {nodes_csv}, {edges_csv}")

# ---- TIMELINE: counts per class over frames ----
timeline = df.groupby(["frame","class"]).size().unstack(fill_value=0)
if df.empty:
    print("⚠️ No symbol data found after filters. Try lowering SCORE_THRESH or check decoded CSV.")
    exit()
timeline = timeline.reindex(range(int(df["frame"].min()), int(df["frame"].max())+1), fill_value=0)
plt.figure(figsize=(10,4))
for cls in known:
    if cls in timeline.columns:
        plt.plot(timeline.index, timeline[cls], label=cls, color=CLASS_PALETTE.get(cls, None))
plt.legend(); plt.title("Class counts per frame")
plt.xlabel("Frame"); plt.ylabel("Count")
tl_path = os.path.join(OUT_DIR, "class_timeline.png")
plt.tight_layout(); plt.savefig(tl_path, dpi=160); plt.close()
print(f"Saved timeline chart: {tl_path}")

# ---- COMPONENT SEQUENCES (ordered by time, then x) ----
def component_sequence(nodes):
    rows = []
    for n in nodes:
        d = G.nodes[n]
        rows.append((d["frame"], d["x"], d["cls"]))
    rows.sort(key=lambda t: (t[0], t[1]))
    return [r[2] for r in rows]

seq_counts = defaultdict(int)
seq_samples = {}
for comp in nx.connected_components(G):
    seq = component_sequence(comp)
    if len(seq) >= MIN_SEQ_LEN:
        key = " → ".join(seq[:12])  # truncate long sequences for readability
        seq_counts[key] += 1
        if key not in seq_samples:
            seq_samples[key] = list(comp)[:12]

seq_txt = os.path.join(OUT_DIR, "symbol_sequences.txt")
with open(seq_txt, "w", encoding="utf-8") as f:
    f.write("Top repeating sequences (by component count):\n\n")
    for s, c in sorted(seq_counts.items(), key=lambda x: -x[1])[:50]:
        f.write(f"×{c}  {s}\n")
print(f"Saved sequences: {seq_txt}")

# quick summary JSON
summary = {
    "nodes": G.number_of_nodes(),
    "edges": G.number_of_edges(),
    "frames": [int(df['frame'].min()), int(df['frame'].max())],
    "sequences_found": len(seq_counts),
    "top_sequence_example": next(iter(seq_counts)) if seq_counts else None
}
with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("Done.")
