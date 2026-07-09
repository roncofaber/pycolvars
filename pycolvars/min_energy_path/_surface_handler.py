#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimum-energy path finder on an n-dimensional PMF energy grid.

Implements a drop-in replacement for the vendored MEPSAnd surface_handler.
The core algorithm is a bottleneck (minimax) path search via the minimum
spanning tree of the grid connectivity graph:

    edge weight = max(energy[i], energy[j])

The path between any two nodes in the MST is the minimax path — the one that
minimises the maximum energy encountered along the way.  For k-th alternative
paths, the bottleneck edge of the best path is removed from the graph and the
MST is rebuilt iteratively.
"""

import numpy as np
import networkx as nx


class _PathStore:
    def __init__(self):
        self.gnw_paths = []   # list of path entries; [n][7] is the trace array


class surface_handler:

    def __init__(self, raw_data):
        try:
            data = np.asarray(raw_data, dtype=float)
            self.coords = data[:, :-1]
            self.energy = data[:, -1]
            self._N = len(self.energy)
            self.propagation = _PathStore()
            self._origin = None
            self._target = None
            self._minima = None
            self._minima_set = set()
            self._G = None
            self._removed_edges = []
            self._build_neighbor_list()
            self.good = True
        except Exception:
            self.good = False

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    def _build_neighbor_list(self):
        N, d = self.coords.shape
        mins  = np.min(self.coords, axis=0)
        steps = np.zeros(d)
        for i in range(d):
            uniq = np.unique(self.coords[:, i])
            if len(uniq) > 1:
                steps[i] = np.min(np.diff(uniq))

        grid_indices = np.rint((self.coords - mins) / steps).astype(int)
        grid_dims    = tuple(np.max(grid_indices, axis=0) + 1)

        grid = np.full(grid_dims, -1, dtype=int)
        for i in range(N):
            grid[tuple(grid_indices[i])] = i

        neighbors = [[] for _ in range(N)]
        for i in range(N):
            idx = grid_indices[i]
            for dim in range(d):
                for delta in (-1, 1):
                    nidx = idx.copy()
                    nidx[dim] += delta
                    if 0 <= nidx[dim] < grid_dims[dim]:
                        j = int(grid[tuple(nidx)])
                        if j != -1:
                            neighbors[i].append(j)

        self._neighbors = neighbors

    def _build_graph(self):
        G = nx.Graph()
        for i in range(self._N):
            for j in self._neighbors[i]:
                if i < j:
                    w = float(max(self.energy[i], self.energy[j]))
                    G.add_edge(i, j, weight=w)
        self._G = G

    # ------------------------------------------------------------------
    # Local minimum detection
    # ------------------------------------------------------------------

    def _detect_minima(self):
        minima = [
            i for i in range(self._N)
            if all(self.energy[i] < self.energy[j] for j in self._neighbors[i])
        ]
        self._minima    = np.array(minima, dtype=int)
        self._minima_set = set(minima)

    # ------------------------------------------------------------------
    # Public API matching MEPSAnd surface_handler
    # ------------------------------------------------------------------

    def get_global_network(self):
        self._detect_minima()
        self._build_graph()

    def _select_minimum_in_range(self, coord_range):
        mask = np.ones(self._N, dtype=bool)
        for c in range(self.coords.shape[1]):
            mask &= self.coords[:, c] >= coord_range[c, 0]
            mask &= self.coords[:, c] <= coord_range[c, 1]
        candidates = np.where(mask)[0]
        in_range   = [i for i in candidates if i in self._minima_set]
        if not in_range:
            return -1
        return int(min(in_range, key=lambda i: self.energy[i]))

    def select_origin_by_range(self, coord_range, minimum=True):
        point = self._select_minimum_in_range(np.asarray(coord_range))
        if point >= 0:
            self._origin        = point
            self._removed_edges = []
            self.propagation.gnw_paths = []
        return point

    def select_target_by_range(self, coord_range, minimum=True):
        point = self._select_minimum_in_range(np.asarray(coord_range))
        if point >= 0:
            self._target        = point
            self._removed_edges = []
            self.propagation.gnw_paths = []
        return point

    # ------------------------------------------------------------------
    # Path finding
    # ------------------------------------------------------------------

    def _bottleneck_edge(self, path):
        """Return the edge on path with the highest max-endpoint energy."""
        best_w, best_e = -np.inf, None
        for u, v in zip(path[:-1], path[1:]):
            w = max(self.energy[u], self.energy[v])
            if w > best_w:
                best_w, best_e = w, (u, v)
        return best_e

    def _make_trace(self, path):
        """Build the trace array: trace[node] = step number (1-based), 0 if absent."""
        trace = np.zeros(self._N, dtype=float)
        for step, node in enumerate(path):
            trace[node] = step + 1
        return trace

    def get_npaths(self, n=0):
        assert self._origin is not None and self._target is not None, \
            "Set origin and target before calling get_npaths."

        while len(self.propagation.gnw_paths) <= n:
            k = len(self.propagation.gnw_paths)

            G_k = self._G.copy()
            for u, v in self._removed_edges[:k]:
                if G_k.has_edge(u, v):
                    G_k.remove_edge(u, v)

            mst = nx.minimum_spanning_tree(G_k)

            try:
                path = nx.shortest_path(mst, self._origin, self._target)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                break

            trace = self._make_trace(path)
            # index 7 carries the trace — pad with None to preserve MEPSAnd layout
            self.propagation.gnw_paths.append((None,) * 7 + (trace,))

            e = self._bottleneck_edge(path)
            if e:
                self._removed_edges.append(e)
