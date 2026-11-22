"""Floor plans used in the paper (Sec. 5.1 and 5.4).

Nodes are free grid cells numbered 1..N in row-major order (row 0 first, increasing x within a
row); node coordinates are (col * spacing, row * spacing) in metres. Free cells are connected to
their 8 neighbours. Edge weights are in grid units: 1.0 orthogonal, sqrt(2) diagonal.
Adjacency uses a distance *tolerance* (the exact float comparison in the original prototype
dropped most edges; see Sec. 5.1).
"""
from dataclasses import dataclass, field
import math

SQRT2 = math.sqrt(2.0)


@dataclass
class GridMap:
    name: str
    cols: int
    rows: int
    spacing: float
    cells: list                     # list of (col, row), row-major
    pos: dict = field(default_factory=dict)   # node id -> (x, y) in metres
    adj: dict = field(default_factory=dict)   # node id -> list of (neighbour id, weight)
    cell_of: dict = field(default_factory=dict)

    @property
    def nodes(self):
        return list(self.pos.keys())

    @property
    def n_nodes(self):
        return len(self.pos)

    @property
    def n_directed_edges(self):
        return sum(len(v) for v in self.adj.values())

    @property
    def bbox(self):
        return (0.0, 0.0, (self.cols - 1) * self.spacing, (self.rows - 1) * self.spacing)

    def is_connected(self):
        start = next(iter(self.pos))
        seen, stack = {start}, [start]
        while stack:
            u = stack.pop()
            for v, _ in self.adj[u]:
                if v not in seen:
                    seen.add(v); stack.append(v)
        return len(seen) == self.n_nodes


def build_map(name, cols, rows, is_free, spacing=0.8, tol=1e-6):
    cells = [(c, r) for r in range(rows) for c in range(cols) if is_free(c, r)]
    m = GridMap(name=name, cols=cols, rows=rows, spacing=spacing, cells=cells)
    for i, (c, r) in enumerate(cells, start=1):
        m.pos[i] = (c * spacing, r * spacing)
        m.cell_of[i] = (c, r)
    ids = list(m.pos)
    for u in ids:
        xu, yu = m.pos[u]
        nb = []
        for v in ids:
            if v == u:
                continue
            xv, yv = m.pos[v]
            d = math.hypot(xu - xv, yu - yv)
            if abs(d - spacing) < tol:
                nb.append((v, 1.0))
            elif abs(d - spacing * SQRT2) < tol:
                nb.append((v, SQRT2))
        m.adj[u] = nb
    return m


def map1():
    """Primary map: 11x11 grid, corridors on rows/cols 0, 4, 5, 10; four room blocks are obstacles.
    72 nodes, 296 directed edges."""
    corr = (0, 4, 5, 10)
    return build_map("map1", 11, 11, lambda c, r: r in corr or c in corr)


MAP2_CORRIDOR_COLS = (0, 6, 12)
MAP2_CORRIDOR_ROWS = (0, 4, 8, 12, 16)


def map2_free_cells():
    cols, rows = 13, 17
    free = {(c, r) for r in range(rows) for c in range(cols)
            if r in MAP2_CORRIDOR_ROWS or c in MAP2_CORRIDOR_COLS}
    rooms = []
    for x0 in (1, 7):                    # blocks between corridor columns
        for y0 in (1, 5, 9, 13):         # blocks between corridor rows
            ry = y0 + 1                  # room occupies the middle row of the 3-row block
            room = [(x0 + 1 + i, ry) for i in range(3)]   # 3x1 room
            door = (x0, ry)              # single-cell doorway to the corridor on its left
            free.update(room); free.add(door)
            rooms.append({"cells": room, "door": door})
    return cols, rows, free, rooms


def map2():
    """Second map ("hospital wing"): 13x17 grid, corridor lattice (cols 0/6/12, rows 0/4/8/12/16)
    and eight 3x1 rooms each reachable only through a single-cell doorway.
    133 nodes, 376 directed edges."""
    cols, rows, free, _ = map2_free_cells()
    return build_map("map2", cols, rows, lambda c, r: (c, r) in free)


MAPS = {"map1": map1, "map2": map2}
