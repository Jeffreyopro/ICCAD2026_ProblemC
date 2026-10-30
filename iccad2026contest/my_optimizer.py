#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set

sys.path.insert(0, str(Path(__file__).parent))

# Lazy import: iccad2026_evaluate is loaded dynamically to prevent
# PyInstaller from following the import chain and bundling its
# non‑submission dependencies (liteloader, etc.).
import importlib as _importlib
import importlib.util  # noqa: F401  (needed for _importlib.util when imported directly)
_mod_name = "iccad2026_evaluate"
_contest_api_spec = _importlib.util.find_spec(_mod_name)
if _contest_api_spec is not None:
    contest_api = _importlib.util.module_from_spec(_contest_api_spec)
    _contest_api_spec.loader.exec_module(contest_api)
else:
    # Standalone (PyInstaller) fallback — minimal stub of the evaluator API.
    class _Fallback:
        class FloorplanOptimizer:
            def __init__(self, verbose: bool = False):
                self.verbose = verbose
            def solve(self, block_count, area_targets, b2b_connectivity,
                      p2b_connectivity, pins_pos, constraints,
                      target_positions=None):
                raise NotImplementedError

        @staticmethod
        def calculate_bbox_area(positions):
            if not positions:
                return 0.0
            x_min = min(p[0] for p in positions)
            y_min = min(p[1] for p in positions)
            x_max = max(p[0] + p[2] for p in positions)
            y_max = max(p[1] + p[3] for p in positions)
            return (x_max - x_min) * (y_max - y_min)

        @staticmethod
        def check_overlap(positions):
            violations = 0
            n = len(positions)
            for i in range(n):
                for j in range(i + 1, n):
                    x1, y1, w1, h1 = positions[i]
                    x2, y2, w2, h2 = positions[j]
                    overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                    overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                    if overlap_x > 1e-6 and overlap_y > 1e-6:
                        violations += 1
            return violations
    contest_api = _Fallback  # type: ignore

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None  # type: ignore
    _HAS_CUPY = False

IS_WINDOWS = sys.platform == 'win32'
if IS_WINDOWS:
    import os as _os_w
    _os_w.system('')  # enable VT100 emulation for colored terminal output
if not IS_WINDOWS:
    import concurrent.futures  # ProcessPoolExecutor — only safe on Linux/macOS


# ------------------------------------------------------------------
#  Experiment / tuning overrides (environment variables)
# ------------------------------------------------------------------
# The defaults below are the reduced Stage-III configuration used for
# the report.  The original (long) Stage-III budget can be restored with
#   FLOORSET_MAX_LEVELS=48 FLOORSET_STAGE3_SCALE=1.5
#   FLOORSET_PHASE3_FACTOR=0.25 FLOORSET_PHASE3_PATIENCE=3
# and the paper-style unified Fast-SA schedule with
#   FLOORSET_SCHEDULE=original
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default

# Pin BLAS threads to 1 for all paths (main process and workers)
for _env_key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_env_key, '1')




Rect = Tuple[float, float, float, float]

LEFT = 1
RIGHT = 2
TOP = 4
BOTTOM = 8

# Terminal colors
_C = "\033[0m"    # reset
_B = "\033[1m"    # bold
_GRY = "\033[90m" # gray
_R = "\033[91m"   # red
_G = "\033[92m"   # green
_Y = "\033[93m"   # yellow
_BLU = "\033[94m" # blue
_M = "\033[95m"   # magenta
_CYN = "\033[96m" # cyan


class BStarTree:
    """
    Serial B*-tree representation used by the CA-SA optimizer.

    Left child  -> placed to the right of parent
    Right child -> placed above parent
    """

    def __init__(self, areas: Sequence[float], widths: Sequence[float], heights: Sequence[float],
                 active: Optional[Sequence[bool]] = None):
        self.n = len(areas)
        self.areas = list(areas)
        self.widths = list(widths)
        self.heights = list(heights)
        self.parent = [-1] * self.n
        self.left = [-1] * self.n
        self.right = [-1] * self.n
        self.active = [True] * self.n if active is None else list(active)
        self.root = 0 if self.n else -1
        self._build_random_tree()

    def _build_random_tree(self) -> None:
        if self.n == 0:
            return
        self.parent = [-1] * self.n
        self.left = [-1] * self.n
        self.right = [-1] * self.n

        order = [i for i in range(self.n) if self.active[i]]
        if not order:
            self.root = -1
            return
        random.shuffle(order)
        order.sort(key=lambda block: self.areas[block], reverse=True)
        self.root = order[0]
        for idx, block in enumerate(order[1:], start=1):
            parent = order[(idx - 1) // 2]
            self.parent[block] = parent
            if idx % 2 == 1:
                self.left[parent] = block
            else:
                self.right[parent] = block

    def copy(self) -> "BStarTree":
        new = BStarTree.__new__(BStarTree)
        new.n = self.n
        new.areas = self.areas.copy()
        new.widths = self.widths.copy()
        new.heights = self.heights.copy()
        new.parent = self.parent.copy()
        new.left = self.left.copy()
        new.right = self.right.copy()
        new.root = self.root
        new.active = self.active.copy()
        return new

    def snapshot_state(self) -> Tuple[List[float], List[float], List[int], List[int], List[int], int]:
        # Lightweight reversible snapshot for in-place SA perturb / undo.
        return (
            self.widths.copy(),
            self.heights.copy(),
            self.parent.copy(),
            self.left.copy(),
            self.right.copy(),
            self.root,
        )

    def restore_state(
        self,
        snapshot: Tuple[List[float], List[float], List[int], List[int], List[int], int],
    ) -> None:
        # Restore a previous tree snapshot after a rejected in-place move.
        self.widths, self.heights, self.parent, self.left, self.right, self.root = (
            snapshot[0].copy(),
            snapshot[1].copy(),
            snapshot[2].copy(),
            snapshot[3].copy(),
            snapshot[4].copy(),
            snapshot[5],
        )

    def _insert_node(self, node: int, target: int, as_left: bool) -> None:
        if not (0 <= node < self.n and 0 <= target < self.n):
            return
        if not self.active[node] or not self.active[target]:
            return
        child_attr = self.left if as_left else self.right
        old_child = child_attr[target]
        child_attr[target] = node
        self.parent[node] = target
        self.left[node] = -1
        self.right[node] = -1
        if old_child != -1:
            self.left[node] = old_child
            self.parent[old_child] = node

    def _delete_node(self, node: int) -> None:
        if not (0 <= node < self.n):
            return
        if not self.active[node]:
            return
        parent = self.parent[node]
        left_child = self.left[node]
        right_child = self.right[node]

        if left_child == -1 and right_child == -1:
            replacement = -1
        elif left_child == -1:
            replacement = right_child
        elif right_child == -1:
            replacement = left_child
        else:
            replacement = left_child
            rightmost = left_child
            while self.right[rightmost] != -1:
                rightmost = self.right[rightmost]
            self.right[rightmost] = right_child
            self.parent[right_child] = rightmost

        if parent == -1:
            self.root = replacement
        elif self.left[parent] == node:
            self.left[parent] = replacement
        else:
            self.right[parent] = replacement

        if replacement != -1:
            self.parent[replacement] = parent

        self.parent[node] = -1
        self.left[node] = -1
        self.right[node] = -1

    def _is_ancestor(self, ancestor: int, node: int) -> bool:
        current = self.parent[node]
        while current != -1:
            if current == ancestor:
                return True
            current = self.parent[current]
        return False

    def _insert_leaf_node(self, node: int, target: int, as_left: bool) -> bool:
        if node == target or node < 0 or target < 0:
            return False
        if not (0 <= node < self.n and 0 <= target < self.n):
            return False
        if not self.active[node] or not self.active[target]:
            return False
        if self._is_ancestor(node, target):
            return False
        self._delete_node(node)
        self._insert_node(node, target, as_left)
        return True

    def _move_subtree_node(self, node: int, target: int, as_left: bool) -> bool:
        if node == target or node < 0 or target < 0:
            return False
        if not (0 <= node < self.n and 0 <= target < self.n):
            return False
        if not self.active[node] or not self.active[target]:
            return False
        if self._is_ancestor(node, target):
            return False
        parent = self.parent[node]
        if parent == -1:
            return False
        if self.left[parent] == node:
            self.left[parent] = -1
        elif self.right[parent] == node:
            self.right[parent] = -1
        else:
            return False
        self.parent[node] = -1

        child_attr = self.left if as_left else self.right
        old_child = child_attr[target]
        child_attr[target] = node
        self.parent[node] = target
        if old_child != -1:
            rightmost = node
            while self.right[rightmost] != -1:
                rightmost = self.right[rightmost]
            self.right[rightmost] = old_child
            self.parent[old_child] = rightmost
        return True

    def swap_nodes_single(self, block_a: int, block_b: int, locked: Iterable[int]) -> bool:
        # Main SA swap semantics: swap the two selected nodes and let each one
        # inherit the other node's children. This keeps the swap local to the
        # chosen nodes while not dragging each node's original children along.
        old_parent = self.parent.copy()
        old_left = self.left.copy()
        old_right = self.right.copy()
        old_root = self.root
        self.swap_nodes(block_a, block_b, locked)
        changed = (
            self.root != old_root
            or self.parent != old_parent
            or self.left != old_left
            or self.right != old_right
        )
        return changed

    def move_delete_insert(self, block: int, locked: Iterable[int]) -> None:
        # Legacy generic move helper. The optimizer currently uses _insert_boundary_aware instead.
        if self.n <= 1 or block in locked:
            return
        movable_targets = [i for i in range(self.n) if i != block]
        if not movable_targets:
            return
        self._delete_node(block)
        target = random.choice(movable_targets)
        self._insert_node(block, target, random.random() < 0.5)

    def swap_nodes(self, block_a: int, block_b: int, locked: Iterable[int]) -> None:
        locked_set = set(locked)
        if self.n <= 1 or block_a == block_b:
            return
        if block_a in locked_set or block_b in locked_set:
            return
        if not (0 <= block_a < self.n and 0 <= block_b < self.n):
            return
        if not self.active[block_a] or not self.active[block_b]:
            return

        old_parent = self.parent.copy()
        old_left = self.left.copy()
        old_right = self.right.copy()
        old_root = self.root

        def remap(node: int) -> int:
            if node == block_a:
                return block_b
            if node == block_b:
                return block_a
            return node

        for node in range(self.n):
            if node == block_a:
                source = block_b
            elif node == block_b:
                source = block_a
            else:
                source = node

            self.parent[node] = remap(old_parent[source])
            self.left[node] = remap(old_left[source])
            self.right[node] = remap(old_right[source])

        self.root = remap(old_root)

    def rotate_blocks(self, blocks: Iterable[int]) -> None:
        # Legacy width/height flip helper. Current move code mostly uses set_group_aspect instead.
        for block in blocks:
            self.widths[block], self.heights[block] = self.heights[block], self.widths[block]

    def set_group_aspect(self, blocks: Iterable[int], ratio: float) -> None:
        # Assign one shared aspect ratio to the whole block/group.
        # Each member keeps its own area, but the group cannot rotate
        # independently because every member uses the same ratio.
        ratio = max(0.2, min(5.0, ratio))
        block_list = list(blocks)
        if not block_list:
            return
        for block in block_list:
            area = self.areas[block]
            width = math.sqrt(area * ratio)
            height = area / max(width, 1e-9)
            self.widths[block] = width
            self.heights[block] = height

    def pack(
        self,
        anchors: Optional[Dict[int, Tuple[float, float]]] = None,
        boundary_codes: Optional[Dict[int, int]] = None,
    ) -> List[Rect]:
        # Convert the current B*-tree topology + aspect ratios into rectangles.
        # Left child  -> placed to the right of parent, then dropped to the contour.
        # Right child -> placed above parent, then legalized against the contour.
        anchors = anchors or {}
        positions: List[Rect] = [(0.0, 0.0, self.widths[i], self.heights[i]) for i in range(self.n)]
        if self.n == 0 or self.root == -1:
            return positions

        contour: List[Tuple[float, float]] = [(0.0, 0.0)]
        placed_rects: List[Rect] = []

        def get_contour_y(x_start: float, x_end: float) -> float:
            # Query the skyline height under interval [x_start, x_end).
            max_y = 0.0
            for idx, (cx_end, cy_top) in enumerate(contour):
                cx_start = contour[idx - 1][0] if idx > 0 else 0.0
                if x_start < cx_end and x_end > cx_start:
                    max_y = max(max_y, cy_top)
            return max_y

        def update_contour(x_start: float, x_end: float, y_top: float) -> None:
            # Replace the covered skyline segment with a new top edge at y_top.
            nonlocal contour
            new_contour: List[Tuple[float, float]] = []
            for idx, (cx_end, cy_top) in enumerate(contour):
                cx_start = contour[idx - 1][0] if idx > 0 else 0.0
                if cx_end <= x_start or cx_start >= x_end:
                    new_contour.append((cx_end, cy_top))
                    continue
                if cx_start < x_start:
                    new_contour.append((x_start, cy_top))
                if cx_end > x_end:
                    new_contour.append((cx_end, cy_top))

            insert_pos = 0
            for idx, (cx_end, _) in enumerate(new_contour):
                if cx_end <= x_start:
                    insert_pos = idx + 1
            new_contour.insert(insert_pos, (x_end, y_top))
            new_contour.sort(key=lambda item: item[0])

            merged: List[Tuple[float, float]] = []
            for x_val, y_val in new_contour:
                if merged and abs(merged[-1][1] - y_val) <= 1e-9:
                    merged[-1] = (x_val, y_val)
                else:
                    merged.append((x_val, y_val))
            contour = merged

        def legalize_y(x_pos: float, y_pos: float, width: float, height: float) -> float:
            # Drop the block to the current contour, then push it upward until no overlap remains.
            current_y = max(y_pos, get_contour_y(x_pos, x_pos + width))
            while True:
                overlap_tops: List[float] = []
                for placed_x, placed_y, placed_w, placed_h in placed_rects:
                    overlap_x = min(x_pos + width, placed_x + placed_w) - max(x_pos, placed_x)
                    overlap_y = min(current_y + height, placed_y + placed_h) - max(current_y, placed_y)
                    if overlap_x > 1e-9 and overlap_y > 1e-9:
                        overlap_tops.append(placed_y + placed_h)
                if not overlap_tops:
                    return current_y
                current_y = max(overlap_tops)

        def dfs(node: int) -> None:
            # DFS placement order matches the B*-tree semantics.
            if node == -1:
                return

            width = self.widths[node]
            height = self.heights[node]
            if node in anchors:
                x_pos, y_pos = anchors[node]
                positions[node] = (x_pos, y_pos, width, height)
                placed_rects.append(positions[node])
                update_contour(x_pos, x_pos + width, y_pos + height)
                dfs(self.left[node])
                dfs(self.right[node])
                return
            elif node == self.root:
                x_pos, y_pos = 0.0, 0.0
            else:
                parent = self.parent[node]
                px, py, pw, ph = positions[parent]
                if self.left[parent] == node:
                    x_pos = px + pw
                    y_pos = 0.0
                else:
                    x_pos = px
                    y_pos = py + ph

            y_pos = legalize_y(x_pos, y_pos, width, height)
            positions[node] = (x_pos, y_pos, width, height)
            placed_rects.append(positions[node])
            update_contour(x_pos, x_pos + width, y_pos + height)

            dfs(self.left[node])
            dfs(self.right[node])

        dfs(self.root)
        return positions


# ------------------------------------------------------------------
#  Cython acceleration (optional, falls back to pure Python)
# ------------------------------------------------------------------
try:
    if os.environ.get("FLOORSET_NO_CYTHON", "0") == "1":
        raise ImportError("disabled by FLOORSET_NO_CYTHON")
    import bstar_core
    _HAS_CYTHON = True
except ImportError:
    _HAS_CYTHON = False

if not _HAS_CYTHON:
    print(
        "[Cython] bstar_core NOT found - using the slow pure-Python fallback. "
        "Build it with: python setup.py build_ext --inplace "
        "(Linux) or .\\build_bstar_windows.ps1 (Windows + MSYS2 UCRT64 gcc).",
        file=sys.stderr,
    )

if _HAS_CYTHON:
    print("[Cython] bstar_core active — pack, copy, wirelength, contour accelerated", file=sys.stderr)
    # Patch BStarTree.pack to use Cython version
    def _cy_pack(self, anchors=None, boundary_codes=None):
        return bstar_core.cy_pack(
            self.n, self.root, self.parent, self.left, self.right,
            self.widths, self.heights, self.active, anchors or {},
        )
    BStarTree.pack = _cy_pack

    # Patch BStarTree.copy to use Cython version
    def _cy_copy(self):
        tup = bstar_core.cy_copy_tree(
            self.n, self.areas, self.widths, self.heights,
            self.parent, self.left, self.right, self.root, self.active,
        )
        new = BStarTree.__new__(BStarTree)
        new.n = self.n
        new.areas = tup[0]
        new.widths = tup[1]
        new.heights = tup[2]
        new.parent = tup[3]
        new.left = tup[4]
        new.right = tup[5]
        new.root = tup[6]
        new.active = tup[7]
        return new
    BStarTree.copy = _cy_copy


class MyOptimizer(contest_api.FloorplanOptimizer):
    """
    Constraint-aware simulated annealing without ML.

    Notes:
    - Uses a serial B*-tree representation.
    - Fixed-shape and preplaced geometry are enforced as hard constraints.
    - Preplaced blocks keep exact (x, y, w, h), but their surrounding tree
      topology may still change.
    - MIB blocks perturb together through group-wise shape moves.
    - Cluster and boundary constraints are pushed through extra cost terms.
    - Boundary violations occasionally trigger forced repair moves.
    """

    # ------------------------------------------------------------------
    #  Initialization & configuration
    # ------------------------------------------------------------------

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        # FAST-SA temperature schedule.
        self.initial_temp = 100.0
        self.final_temp = 1.5
        self.cooling_rate = 0.85
        self.fastsa_initial_accept_prob = 0.95
        self.fastsa_c = 100.0
        self.fastsa_k = _env_int("FLOORSET_FASTSA_K", 7)
        self.fastsa_probe_factor = 2
        self.fastsa_min_ratio = 0.02
        self.fastsa_max_ratio = 0.95
        self.fastsa_stage2_move_scale = _env_float("FLOORSET_STAGE2_SCALE", 1.0)  # 3
        # Reduced Stage-III budget (was 1.5 / 0.25 / 3 / 48).  Stage III is
        # polishing only: the solution is already constraint-feasible, so the
        # long tail of hill-climbing moves buys very little quality per second.
        self.fastsa_stage3_move_scale = _env_float("FLOORSET_STAGE3_SCALE", 1.0)
        self.fastsa_stage3_patience = _env_int("FLOORSET_STAGE3_PATIENCE", 2)
        self.fastsa_max_levels = _env_int("FLOORSET_MAX_LEVELS", 26)
        self.moves_per_temp_factor = 1.0
        self.phase2_part1_move_factor = _env_float("FLOORSET_PHASE2_PART1", 0.25)
        self.phase2_part2_move_factor = _env_float("FLOORSET_PHASE2_PART2", 1.0)
        self.phase2_part3_move_factor = _env_float("FLOORSET_PHASE2_PART3", 1.0)
        # Ablation control: 1 = collapse the three Stage-II sub-stages into one
        # uniform sub-stage with the same total move budget (isolates the effect
        # of the sub-stage decomposition from the budget itself).
        self.phase2_nosplit = _env_int("FLOORSET_PHASE2_NOSPLIT", 0) != 0
        self.phase3_move_factor = _env_float("FLOORSET_PHASE3_FACTOR", 0.15)
        self.phase3_patience = _env_int("FLOORSET_PHASE3_PATIENCE", 2)
        # "modified" = three-stage CA-SA (staged cost shaping + boundary relax);
        # "original" = paper-style unified Fast-SA (single full cost for all stages).
        self.fastsa_schedule_mode = os.environ.get("FLOORSET_SCHEDULE", "modified").strip().lower()
        self.experiment_seed = _env_int("FLOORSET_SEED", -1)
        self.stats_path = os.environ.get("FLOORSET_STATS", "").strip()
        import os as _os_rs
        _n_cores = max(1, _os_rs.cpu_count() or 1)
        _default_restarts = min(4, _n_cores) if IS_WINDOWS else _n_cores
        self.restarts = max(1, _env_int("FLOORSET_RESTARTS", _default_restarts))
        self.use_parallel_restarts = _env_int("FLOORSET_PARALLEL", 1) != 0
        self._restart_workers = self.restarts
        # Extra rescue tries after a move if the packed result still overlaps.
        self.boundary_repair_prob = 0.0005
        self.overlap_retry_count = 0
        # Cost-term weights.
        self.bbox_weight = 0.00001
        self.cluster_weight = 0.1 #0.1
        self.cluster_pre_sa_area_weight = 1.0
        self.cluster_pre_sa_hpwl_weight = 0.0
        self.cluster_pre_sa_outline_weight = 1.0
        self.boundary_weight = 0.1
        self.phase3_boundary_violation_weight = 0.25
        self.boundary_dimension_weight = 0.0
        self.outline_weight = 0.01 
        self.stage2_p2b_weight = 0.0
        self.area_wirelength_alpha = 0.2 #0.5
        self.preplaced_overlap_lambda = 1.0
        self.overlap_weight = 1_000_000.0
        self.base_time_limit = 50.0
        self.time_limit_per_block = 11
        self.max_time_limit = 1200.0
        # Optional acceleration toggle flags.
        self.use_gpu = False
        self.use_parallel_probe = True
        self.use_parallel_cluster = False  # clusters are small, sequential is faster
        self._cached_terminal_outline: Tuple[float, float] = (0.0, 0.0)
        self._cached_area_norm = 1.0
        self._cached_wirelength_norm = 1.0
        self._boundary_finalize_active = False
        self._phase3_preplaced_reinsert_active = False
        self._phase3_boundary_relax_active = False
        self._phase2_boundary_shape_only_active = False
        self._current_preplaced_overlap_lambda = 1.0
        self._phase2_boundary_ratio_mode_state: Optional[str] = None
        self._phase3_relaxed_geometry_boundary_blocks: set[int] = set()
        self._cached_geometry_locked_blocks: set[int] = set()
        self._cached_active_geometry_locked_blocks: set[int] = set()
        self._cached_geometry_locked_flags: List[bool] = []
        self._cached_movable_blocks: Tuple[int, ...] = ()
        self._cached_target_rects: List[Rect] = []
        self._cached_immovable_blocks: set[int] = set()
        self._cached_boundary_masks: List[int] = []
        self._cached_boundary_top_nodes: set[int] = set()
        self._cached_boundary_bottom_nodes: set[int] = set()
        self._cached_boundary_left_nodes: set[int] = set()
        self._cached_boundary_right_nodes: set[int] = set()
        self._cached_bottom_left_nodes: set[int] = set()
        self._cached_left_tail: Optional[int] = None
        self._cached_bottom_tail: Optional[int] = None
        self._cached_top_tail: Optional[int] = None
        self._cached_top_root: Optional[int] = None
        self._cached_right_root: Optional[int] = None
        self._cached_ordered_bottom: List[int] = []
        self._cached_ordered_left: List[int] = []
        self._cached_ordered_top: List[int] = []
        self._cached_ordered_right: List[int] = []
        self._cached_mib_forced_ratio_groups: List[Tuple[List[int], float, int]] = []
        self._cached_mib_blocks: set[int] = set()
        self._cached_mib_group_by_block: List[Optional[List[int]]] = []
        self._cached_cluster_blocks: set[int] = set()
        self._cached_macro_cluster_groups: List[List[int]] = []
        self._cached_macro_cluster_by_block: List[Optional[List[int]]] = []
        self._cached_macro_cluster_anchor_by_block: List[Optional[int]] = []
        self._cached_macro_cluster_non_anchors: set[int] = set()
        self._cached_macro_cluster_metadata: Dict[int, Dict[str, object]] = {}
        self._cached_mib_cluster_ratio_groups: List[Tuple[List[int], float]] = []
        self._cached_mib_hard_shape_blocks: set[int] = set()
        self._cached_cluster_satellites: set[int] = set()
        self._cached_cluster_representative_by_block: List[Optional[int]] = []
        self._stage2_hpwl_conn_weight: Optional[float] = None
        self._stage2_hpwl_outline_norm: Optional[float] = None

    def _build_restart_data(self, block_count, area_targets, b2b_connectivity,
                            p2b_connectivity, pins_pos, constraints, target_positions,
                            start_time, time_limit) -> dict:
        areas = [max(float(self._scalar(area_targets[i])), 1.0) for i in range(block_count)]
        constraints_rows = [self._constraint_row(constraints, i) for i in range(block_count)]
        target_rects = self._prepare_target_rects(target_positions, block_count)
        b2b_edges = self._prepare_edges(b2b_connectivity, 3)
        p2b_edges = self._prepare_edges(p2b_connectivity, 3)
        pin_positions = self._prepare_points(pins_pos)
        self._cached_terminal_outline = self._fixed_outline_dimensions(
            pin_positions, constraints_rows, target_rects,
        )
        self._boundary_finalize_active = False
        self._phase3_preplaced_reinsert_active = False
        self._phase3_boundary_relax_active = False
        self._phase2_boundary_shape_only_active = False
        self._current_preplaced_overlap_lambda = 0.0
        self._phase3_relaxed_geometry_boundary_blocks = set()
        self._cached_geometry_locked_blocks = set()
        self._cached_target_rects = []
        self._cached_immovable_blocks = set()
        self._cached_cluster_blocks = set()
        self._cached_macro_cluster_groups = []
        self._cached_macro_cluster_by_block = []
        self._cached_macro_cluster_anchor_by_block = []
        self._cached_macro_cluster_non_anchors = set()
        self._cached_cluster_satellites = set()
        self._cached_cluster_representative_by_block = []
        self._cached_macro_cluster_metadata = {}
        self._cached_mib_cluster_ratio_groups = []
        self._cached_group_clusters: Dict[int, List[int]] = {}
        fixed_blocks = {i for i, row in enumerate(constraints_rows) if row[0] != 0}
        preplaced_blocks = {i for i, row in enumerate(constraints_rows) if row[1] != 0}
        locked_blocks: set[int] = set()
        geometry_locked_blocks = {
            i for i in fixed_blocks | preplaced_blocks
            if target_rects[i][2] > 0.0 and target_rects[i][3] > 0.0
        }
        self._cached_geometry_locked_blocks = set(geometry_locked_blocks)
        self._cached_active_geometry_locked_blocks = set(geometry_locked_blocks)
        self._cached_geometry_locked_flags = [False] * block_count
        for block in geometry_locked_blocks:
            if 0 <= block < block_count:
                self._cached_geometry_locked_flags[block] = True
        mib_groups = self._group_indices(constraints_rows, 2)
        cluster_groups = self._group_indices(constraints_rows, 3)
        self._cached_group_clusters = cluster_groups
        boundary_codes = {i: int(row[4]) for i, row in enumerate(constraints_rows) if row[4] != 0}
        self._build_boundary_cache(boundary_codes, block_count, p2b_edges, pin_positions)
        self._phase3_relaxed_geometry_boundary_blocks = set()
        self._build_mib_geometry_cache(mib_groups, target_rects, geometry_locked_blocks)
        base_widths = []
        base_heights = []
        preplaced_anchors: Dict[int, Tuple[float, float]] = {}
        for block in range(block_count):
            tx, ty, tw, th = target_rects[block]
            if block in preplaced_blocks and tx >= 0.0 and ty >= 0.0:
                preplaced_anchors[block] = (tx, ty)
            if block in geometry_locked_blocks:
                base_widths.append(tw)
                base_heights.append(th)
            else:
                side = math.sqrt(areas[block])
                base_widths.append(side)
                base_heights.append(side)
        immovable_blocks = set(preplaced_anchors)
        self._cached_target_rects = list(target_rects)
        self._cached_immovable_blocks = set(immovable_blocks)
        self._build_cluster_cache(cluster_groups, boundary_codes, immovable_blocks, block_count)
        self._cached_movable_blocks = tuple(
            block for block in range(block_count)
            if block not in self._cached_macro_cluster_non_anchors
        )
        locked_blocks.update(self._cached_cluster_satellites)
        tree_active = [i not in self._cached_cluster_satellites for i in range(block_count)]
        return dict(
            areas=areas, locked_blocks=locked_blocks, mib_groups=mib_groups,
            cluster_groups=cluster_groups, boundary_codes=boundary_codes,
            geometry_locked_blocks=geometry_locked_blocks, immovable_blocks=immovable_blocks,
            target_rects=target_rects, b2b_edges=b2b_edges, p2b_edges=p2b_edges,
            pin_positions=pin_positions, preplaced_anchors=preplaced_anchors,
            tree_active=tree_active, block_count=block_count,
            base_widths=base_widths, base_heights=base_heights,
            start_time=start_time, time_limit=time_limit,
        )

    def _solve_one_restart(self, **sl) -> Dict[str, Any]:
        """Run one full SA restart. All solve-scope locals passed via **sl."""
        areas = sl["areas"]; locked_blocks = sl["locked_blocks"]; mib_groups = sl["mib_groups"]
        cluster_groups = sl["cluster_groups"]; boundary_codes = sl["boundary_codes"]
        geometry_locked_blocks = sl["geometry_locked_blocks"]; immovable_blocks = sl["immovable_blocks"]
        target_rects = sl["target_rects"]; b2b_edges = sl["b2b_edges"]; p2b_edges = sl["p2b_edges"]
        pin_positions = sl["pin_positions"]; preplaced_anchors = sl["preplaced_anchors"]
        tree_active = sl["tree_active"]; block_count = sl["block_count"]
        base_widths = sl["base_widths"]; base_heights = sl["base_heights"]
        start_time = sl["start_time"]; time_limit = sl["time_limit"]
        modified_schedule = self.fastsa_schedule_mode != "original"
        moves_total = 0
        stage_move_counts = {1: 0, 2: 0, 3: 0}
        # Effective block count: non-anchor local-cluster members are already
        # compacted during pre-SA, count each local cluster's anchor as 1.
        effective_n = max(1, block_count - len(self._cached_macro_cluster_non_anchors))
        best_positions: Optional[List[Rect]] = None; best_cost = float("inf")
        best_tree_global: Optional[BStarTree] = None
        best_anchors_global: Dict[int, Tuple[float, float]] = {}
        best_top_chain_locked_global: set[int] = set()
        best_finalized_positions: Optional[List[Rect]] = None; best_finalized_cost = float("inf")
        best_finalized_tree_global: Optional[BStarTree] = None
        best_finalized_anchors_global: Dict[int, Tuple[float, float]] = {}
        best_finalized_top_chain_locked_global: set[int] = set()
        best_phase3_finalized_positions: Optional[List[Rect]] = None; best_phase3_finalized_cost = float("inf")
        best_phase3_finalized_tree_global: Optional[BStarTree] = None
        best_phase3_finalized_anchors_global: Dict[int, Tuple[float, float]] = {}
        best_phase3_finalized_top_chain_locked_global: set[int] = set()
        tree = BStarTree(areas, base_widths, base_heights, active=tree_active)
        dynamic_anchors: Dict[int, Tuple[float, float]] = {}
        self._enforce_locked_geometry(tree, target_rects, geometry_locked_blocks)
        self._apply_hard_boundary_tree_rules(tree, boundary_codes)
        self._apply_cluster_hierarchy_rules(tree, cluster_groups, boundary_codes, immovable_blocks)
        self._build_cluster_local_layout_cache(tree, cluster_groups, mib_groups, b2b_edges)
        _combined_anchors = self._combine_anchors(preplaced_anchors, dynamic_anchors)
        current_positions = self._pack_compact_with_boundary_repair(
            tree, _combined_anchors,
            boundary_codes, immovable_blocks=immovable_blocks,
        )
        current_full_cost = self._cost(current_positions, b2b_edges, p2b_edges, pin_positions,
            cluster_groups, boundary_codes, areas, preplaced_anchors, dynamic_anchors, immovable_blocks, tree)
        current_accept_cost = current_full_cost; current_cost_mode = "full"; current_full_cost_valid = True
        current_phase3_boundary_violation = self._phase3_boundary_violation_cost(current_positions, boundary_codes)
        if current_full_cost < best_cost:
            best_cost = current_full_cost; best_positions = current_positions
            best_tree_global = tree.copy(); best_anchors_global = dict(dynamic_anchors)
        best_tree = tree.copy()
        best_restart_cost = current_full_cost; best_restart_positions = current_positions
        best_restart_anchors = dict(dynamic_anchors); best_restart_top_chain_locked: set[int] = set()
        top_chain_locked_blocks: set[int] = set()
        base_moves_per_level = max(20, int(self.moves_per_temp_factor * max(1, effective_n)))
        initial_temp, avg_cost_ratio = self._estimate_fastsa_parameters(
            tree, areas, locked_blocks, mib_groups, current_full_cost,
            preplaced_anchors, dynamic_anchors, geometry_locked_blocks, immovable_blocks,
            target_rects, b2b_edges, p2b_edges, pin_positions, cluster_groups, boundary_codes,
        )
        temp_level = 1; stage3_idle_levels = 0
        phase2_convergence_checked = False; phase2_post_convergence_full_cost = False
        max_temp_levels = max(
            self.fastsa_k + 8,
            min(self.fastsa_max_levels, self.fastsa_k + max(10, effective_n // 2)),
        )
        while temp_level <= max_temp_levels:
            if time.perf_counter() - start_time >= time_limit:
                break
            self._phase3_preplaced_reinsert_active = temp_level > 1
            self._phase3_boundary_relax_active = modified_schedule and temp_level > self.fastsa_k
            temp = self._fastsa_temperature(initial_temp, avg_cost_ratio, temp_level)
            moves_per_level = self._moves_for_fastsa_stage(base_moves_per_level, temp_level, effective_n)
            if modified_schedule and 1 < temp_level <= self.fastsa_k:
                n_phase2 = self.fastsa_k - 1
                per_third = n_phase2 // 3
                _split1 = 1 + per_third
                _split2 = 1 + 2 * per_third
                if self.phase2_nosplit:
                    # Ablation: one uniform Stage-II with the same total budget.
                    _pmf = (self.phase2_part1_move_factor
                            + self.phase2_part2_move_factor
                            + self.phase2_part3_move_factor) / 3.0
                    scheduled_cost_mode = "stage2_hpwl"
                elif temp_level <= _split1:
                    _pmf = self.phase2_part1_move_factor
                    scheduled_cost_mode = "stage2"
                elif temp_level <= _split2:
                    _pmf = self.phase2_part2_move_factor
                    scheduled_cost_mode = "stage2_boundary"
                else:
                    _pmf = self.phase2_part3_move_factor
                    scheduled_cost_mode = "stage2_hpwl"
                moves_per_level = max(24, int(moves_per_level * _pmf))
            else:
                # Paper-style Fast-SA: one unified (full) objective throughout.
                scheduled_cost_mode = "full"
            self._phase2_boundary_shape_only_active = scheduled_cost_mode in ("stage2", "stage2_hpwl", "stage2_boundary")
            cost_mode = scheduled_cost_mode
            if self._phase3_boundary_relax_active:
                self._current_preplaced_overlap_lambda = self.preplaced_overlap_lambda
            else:
                self._current_preplaced_overlap_lambda = 0.0
            freeze_boundary_topology = self._boundary_finalize_active
            if cost_mode != current_cost_mode:
                if (scheduled_cost_mode == "full" and current_cost_mode in ("stage2", "stage2_boundary", "stage2_hpwl")
                        and not phase2_convergence_checked):
                    current_positions = self._finalize_boundary_outline_fit(
                        tree, current_positions, boundary_codes, top_chain_locked_blocks,
                        preplaced_anchors, immovable_blocks, geometry_locked_blocks, target_rects)
                    #self._enforce_locked_geometry(tree, target_rects, geometry_locked_blocks)
                    #self._apply_hard_boundary_tree_rules(tree, boundary_codes)
                    current_positions = self._pack_compact_with_boundary_repair(
                        tree, self._combine_anchors(preplaced_anchors, dynamic_anchors),
                        boundary_codes, top_chain_locked_blocks, immovable_blocks)
                    self._boundary_finalize_active = True
                    self._current_preplaced_overlap_lambda = 0.0
                    phase2_convergence_checked = True; phase2_post_convergence_full_cost = True
                    current_full_cost = self._cost(current_positions, b2b_edges, p2b_edges, pin_positions,
                        cluster_groups, boundary_codes, areas, preplaced_anchors, dynamic_anchors, immovable_blocks, tree)
                    current_full_cost_valid = True; current_accept_cost = current_full_cost; current_cost_mode = "full"; cost_mode = "full"
                    current_phase3_boundary_violation = self._phase3_boundary_violation_cost(current_positions, boundary_codes)
                    best_tree = tree.copy(); best_restart_cost = current_full_cost; best_restart_positions = current_positions
                    best_restart_anchors = dict(dynamic_anchors); best_restart_top_chain_locked = set(top_chain_locked_blocks)
                    if current_full_cost < best_finalized_cost:
                        best_finalized_cost = current_full_cost; best_finalized_positions = current_positions
                        best_finalized_tree_global = tree.copy(); best_finalized_anchors_global = dict(dynamic_anchors)
                        best_finalized_top_chain_locked_global = set(top_chain_locked_blocks)
                    if self._phase3_boundary_relax_active and current_full_cost < best_phase3_finalized_cost:
                        best_phase3_finalized_cost = current_full_cost; best_phase3_finalized_positions = current_positions
                        best_phase3_finalized_tree_global = tree.copy(); best_phase3_finalized_anchors_global = dict(dynamic_anchors)
                        best_phase3_finalized_top_chain_locked_global = set(top_chain_locked_blocks)
                current_accept_cost = self._cost(current_positions, b2b_edges, p2b_edges, pin_positions,
                    cluster_groups, boundary_codes, areas, preplaced_anchors, dynamic_anchors, immovable_blocks, tree,
                    mode=cost_mode)
                if cost_mode == "full" and not current_full_cost_valid:
                    current_full_cost = self._cost(current_positions, b2b_edges, p2b_edges, pin_positions,
                        cluster_groups, boundary_codes, areas, preplaced_anchors, dynamic_anchors, immovable_blocks, tree)
                    current_full_cost_valid = True
                current_cost_mode = cost_mode
            level_ratios: List[float] = []; level_improved = False
            for _ in range(moves_per_level):
                moves_total += 1
                if temp_level == 1:
                    stage_move_counts[1] += 1
                elif temp_level <= self.fastsa_k:
                    stage_move_counts[2] += 1
                else:
                    stage_move_counts[3] += 1
                if time.perf_counter() - start_time >= time_limit:
                    break
                phase2_outline_converged = scheduled_cost_mode in ("stage2", "stage2_hpwl", "stage2_boundary") and self._outline_is_converged(current_positions)
                if scheduled_cost_mode in ("stage2", "stage2_hpwl", "stage2_boundary") and phase2_outline_converged and not phase2_convergence_checked:
                    current_positions = self._finalize_boundary_outline_fit(
                        tree, current_positions, boundary_codes, top_chain_locked_blocks,
                        preplaced_anchors, immovable_blocks, geometry_locked_blocks, target_rects)
                    #self._enforce_locked_geometry(tree, target_rects, geometry_locked_blocks)
                    #self._apply_hard_boundary_tree_rules(tree, boundary_codes)
                    current_positions = self._pack_compact_with_boundary_repair(
                        tree, self._combine_anchors(preplaced_anchors, dynamic_anchors),
                        boundary_codes, top_chain_locked_blocks, immovable_blocks)
                    phase2_post_convergence_full_cost = True; cost_mode = "full"
                    current_accept_cost = self._cost(current_positions, b2b_edges, p2b_edges, pin_positions,
                        cluster_groups, boundary_codes, areas, preplaced_anchors, dynamic_anchors, immovable_blocks, tree,
                        mode="full")
                    current_full_cost = current_accept_cost; current_full_cost_valid = True; current_cost_mode = "full"
                    current_phase3_boundary_violation = self._phase3_boundary_violation_cost(current_positions, boundary_codes)
                    self._boundary_finalize_active = True; self._current_preplaced_overlap_lambda = 0.0
                    phase2_convergence_checked = True; phase2_outline_converged = True
                    best_tree = tree.copy(); best_restart_cost = current_full_cost; best_restart_positions = current_positions
                    best_restart_anchors = dict(dynamic_anchors); best_restart_top_chain_locked = set(top_chain_locked_blocks)
                    if current_full_cost < best_finalized_cost:
                        best_finalized_cost = current_full_cost; best_finalized_positions = current_positions
                        best_finalized_tree_global = tree.copy(); best_finalized_anchors_global = dict(dynamic_anchors)
                        best_finalized_top_chain_locked_global = set(top_chain_locked_blocks)
                    if self._phase3_boundary_relax_active and current_full_cost < best_phase3_finalized_cost:
                        best_phase3_finalized_cost = current_full_cost; best_phase3_finalized_positions = current_positions
                        best_phase3_finalized_tree_global = tree.copy(); best_phase3_finalized_anchors_global = dict(dynamic_anchors)
                        best_phase3_finalized_top_chain_locked_global = set(top_chain_locked_blocks)
                trial_anchors = dict(dynamic_anchors)
                tree_snapshot = tree.snapshot_state(); force_accept = False; moved_blocks: List[int]
                # 1% stretch-touching-boundary move in Phase 2/3
                _stretch_block = None
                if temp_level > 1 and random.random() < 0.01:
                    _stretch_block, _stretch_x = self._find_touching_boundary_stretch(
                        current_positions, boundary_codes, locked_blocks)
                if _stretch_block is not None:
                    moved_blocks = self._apply_stretch_move(
                        tree, _stretch_block, _stretch_x, boundary_codes, mib_groups)
                elif ((not self._boundary_finalize_active) and not self._phase2_boundary_shape_only_active
                        and random.random() < self.boundary_repair_prob):
                    self._phase2_boundary_ratio_mode_state = (self._phase2_boundary_ratio_mode(current_positions)
                        if scheduled_cost_mode in ("stage2", "stage2_hpwl", "stage2_boundary") and not phase2_post_convergence_full_cost else None)
                    moved_blocks, force_accept = self._top_right_repair_move(tree, current_positions, boundary_codes, locked_blocks)
                    if not moved_blocks:
                        preferred_block = None
                        if temp_level > 1:
                            preferred_block = self._pick_exposed_outline_block(current_positions, boundary_codes, locked_blocks, scheduled_cost_mode)
                        _contour_filled = False
                        if preferred_block is not None and random.random() < 0.5:
                            _as_left_skip = set(locked_blocks)
                            _contour_filled = self._phase2_fill_contour_gap(tree, current_positions, boundary_codes, locked_blocks, preferred_block, as_left=random.random() < self._contour_fill_as_left_probability(current_positions, boundary_codes, _as_left_skip))
                        if _contour_filled:
                            moved_blocks = [preferred_block]; force_accept = True
                        else:
                            moved_blocks = self._apply_regular_move(tree, areas, locked_blocks, mib_groups, cluster_groups,
                                boundary_codes, geometry_locked_blocks, freeze_boundary_topology, top_chain_locked_blocks,
                                stage2_boundary_focus=phase2_outline_converged, stage2_mode=(scheduled_cost_mode == "stage2"),
                                phase3_free_boundary_blocks=(modified_schedule and temp_level > self.fastsa_k), stage2_hpwl_mode=(scheduled_cost_mode in ("stage2_hpwl", "stage2_boundary")),
                                preferred_block=preferred_block)
                            if preferred_block is not None and 0 <= preferred_block < len(self._cached_macro_cluster_anchor_by_block) and self._cached_macro_cluster_anchor_by_block[preferred_block] is not None:
                                force_accept = True
                else:
                    self._phase2_boundary_ratio_mode_state = (self._phase2_boundary_ratio_mode(current_positions)
                        if scheduled_cost_mode in ("stage2", "stage2_hpwl", "stage2_boundary") and not phase2_post_convergence_full_cost else None)
                    preferred_block = None
                    if temp_level > 1:
                        preferred_block = self._pick_exposed_outline_block(current_positions, boundary_codes, locked_blocks, scheduled_cost_mode)
                    _contour_filled = False
                    if preferred_block is not None and random.random() < 0.5:
                        _as_left_skip = set(locked_blocks)
                        _contour_filled = self._phase2_fill_contour_gap(tree, current_positions, boundary_codes, locked_blocks, preferred_block, as_left=random.random() < self._contour_fill_as_left_probability(current_positions, boundary_codes, _as_left_skip))
                    if _contour_filled:
                        moved_blocks = [preferred_block]; force_accept = True
                    else:
                        moved_blocks = self._apply_regular_move(tree, areas, locked_blocks, mib_groups, cluster_groups,
                            boundary_codes, geometry_locked_blocks, freeze_boundary_topology, top_chain_locked_blocks,
                            stage2_boundary_focus=phase2_outline_converged, stage2_mode=(scheduled_cost_mode == "stage2"),
                            phase3_free_boundary_blocks=(modified_schedule and temp_level > self.fastsa_k), stage2_hpwl_mode=(scheduled_cost_mode in ("stage2_hpwl", "stage2_boundary")),
                            preferred_block=preferred_block)
                        if preferred_block is not None and 0 <= preferred_block < len(self._cached_macro_cluster_anchor_by_block) and self._cached_macro_cluster_anchor_by_block[preferred_block] is not None:
                            force_accept = True
                #self._enforce_locked_geometry(tree, target_rects, geometry_locked_blocks)
                #if self._boundary_finalize_active and not self._phase3_boundary_relax_active:
                    #self._apply_hard_boundary_tree_rules(tree, boundary_codes)
                if not self._boundary_finalize_active:
                    self._enforce_top_chain_constraints(tree, boundary_codes)
                trial_positions = self._pack_compact_with_boundary_repair(
                    tree, self._combine_anchors(preplaced_anchors, trial_anchors),
                    boundary_codes, top_chain_locked_blocks, immovable_blocks)
                if trial_positions is None:
                    self._phase2_boundary_ratio_mode_state = None; tree.restore_state(tree_snapshot); continue
                self._phase2_boundary_ratio_mode_state = None
                if cost_mode == "full":
                    trial_accept_cost = self._cost(trial_positions, b2b_edges, p2b_edges, pin_positions,
                        cluster_groups, boundary_codes, areas, preplaced_anchors, trial_anchors, immovable_blocks, tree)
                    trial_full_cost = trial_accept_cost
                else:
                    trial_accept_cost = self._cost(trial_positions, b2b_edges, p2b_edges, pin_positions,
                        cluster_groups, boundary_codes, areas, preplaced_anchors, trial_anchors, immovable_blocks, tree,
                        mode=cost_mode)
                    trial_full_cost = None
                trial_phase3_boundary_violation = self._phase3_boundary_violation_cost(trial_positions, boundary_codes)
                delta = trial_accept_cost - current_accept_cost
                level_ratios.append(self._normalize_cost_ratio(delta, current_accept_cost))
                accept = force_accept
                if self._phase3_boundary_relax_active and trial_phase3_boundary_violation > current_phase3_boundary_violation + 1e-12:
                    accept = False
                if not accept:
                    accept = delta <= 0 or random.random() < math.exp(-delta / max(temp, 1e-9))
                if accept:
                    dynamic_anchors = trial_anchors; current_positions = trial_positions
                    current_accept_cost = trial_accept_cost; current_cost_mode = cost_mode
                    current_phase3_boundary_violation = trial_phase3_boundary_violation
                    if trial_full_cost is None:
                        current_full_cost_valid = False
                    else:
                        current_full_cost = trial_full_cost; current_full_cost_valid = True
                        if current_full_cost < best_restart_cost:
                            best_tree = tree.copy(); best_restart_cost = current_full_cost; best_restart_positions = current_positions
                            best_restart_anchors = dict(dynamic_anchors); best_restart_top_chain_locked = set(top_chain_locked_blocks)
                            level_improved = True
                        if current_full_cost < best_cost:
                            best_cost = current_full_cost; best_positions = current_positions
                            best_tree_global = tree.copy(); best_anchors_global = dict(dynamic_anchors)
                            best_top_chain_locked_global = set(top_chain_locked_blocks)
                        if self._boundary_finalize_active and current_full_cost < best_finalized_cost:
                            best_finalized_cost = current_full_cost; best_finalized_positions = current_positions
                            best_finalized_tree_global = tree.copy(); best_finalized_anchors_global = dict(dynamic_anchors)
                            best_finalized_top_chain_locked_global = set(top_chain_locked_blocks)
                        if self._phase3_boundary_relax_active and current_full_cost < best_phase3_finalized_cost:
                            best_phase3_finalized_cost = current_full_cost; best_phase3_finalized_positions = current_positions
                            best_phase3_finalized_tree_global = tree.copy(); best_phase3_finalized_anchors_global = dict(dynamic_anchors)
                            best_phase3_finalized_top_chain_locked_global = set(top_chain_locked_blocks)
                else:
                    tree.restore_state(tree_snapshot)
            if level_ratios:
                avg_cost_ratio = max(self.fastsa_min_ratio,
                    min(self.fastsa_max_ratio, sum(level_ratios) / len(level_ratios)))
            if temp_level > self.fastsa_k:
                if level_improved:
                    stage3_idle_levels = 0
                else:
                    stage3_idle_levels += 1
                if temp <= self.final_temp and stage3_idle_levels >= self.phase3_patience:
                    break
            n_phase2 = self.fastsa_k - 1; per_third = n_phase2 // 3; _split1 = 1 + per_third
            if scheduled_cost_mode == "stage2" and temp_level == _split1:
                s2 = self._cost_terms(current_positions, b2b_edges, p2b_edges, pin_positions,
                    cluster_groups, boundary_codes, immovable_blocks, tree, mode="stage2")[0]
                hpwl = self._cost_terms(current_positions, b2b_edges, p2b_edges, pin_positions,
                    cluster_groups, boundary_codes, immovable_blocks, tree, mode="stage2_hpwl")[0]
                conn_raw = hpwl["b2b"] + hpwl["p2b"]; outline_raw = hpwl["outline"]
                if conn_raw > 1e-12 and outline_raw > 1e-12:
                    self._stage2_hpwl_conn_weight = 0.25 * outline_raw / conn_raw; self._stage2_hpwl_outline_norm = outline_raw
                else:
                    self._stage2_hpwl_conn_weight = 1.0; self._stage2_hpwl_outline_norm = 1.0
            elif scheduled_cost_mode == "stage2_hpwl" and temp_level == self.fastsa_k:
                hpwl = self._cost_terms(current_positions, b2b_edges, p2b_edges, pin_positions,
                    cluster_groups, boundary_codes, immovable_blocks, tree, mode="stage2_hpwl")[0]
                connection_raw = hpwl["b2b"] + hpwl["p2b"]; hpwl_total = connection_raw + hpwl["outline"]
            temp_level += 1
        if time.perf_counter() - start_time >= time_limit:
            pass  # time ran out, return whatever we found
        if not phase2_convergence_checked:
            current_positions = self._finalize_boundary_outline_fit(
                tree, current_positions, boundary_codes, top_chain_locked_blocks,
                preplaced_anchors, immovable_blocks, geometry_locked_blocks, target_rects)
            #self._enforce_locked_geometry(tree, target_rects, geometry_locked_blocks)
            #self._apply_hard_boundary_tree_rules(tree, boundary_codes)
            current_positions = self._pack_compact_with_boundary_repair(
                tree, self._combine_anchors(preplaced_anchors, dynamic_anchors),
                boundary_codes, top_chain_locked_blocks, immovable_blocks)
            self._boundary_finalize_active = True; self._current_preplaced_overlap_lambda = 0.0; phase2_convergence_checked = True
            current_full_cost = self._cost(current_positions, b2b_edges, p2b_edges, pin_positions,
                cluster_groups, boundary_codes, areas, preplaced_anchors, dynamic_anchors, immovable_blocks, tree)
            current_accept_cost = current_full_cost; current_cost_mode = "full"; current_full_cost_valid = True
            current_phase3_boundary_violation = self._phase3_boundary_violation_cost(current_positions, boundary_codes)
            best_tree = tree.copy(); best_restart_cost = current_full_cost; best_restart_positions = current_positions
            best_restart_anchors = dict(dynamic_anchors); best_restart_top_chain_locked = set(top_chain_locked_blocks)
            if current_full_cost < best_finalized_cost:
                best_finalized_cost = current_full_cost; best_finalized_positions = current_positions
                best_finalized_tree_global = tree.copy(); best_finalized_anchors_global = dict(dynamic_anchors)
                best_finalized_top_chain_locked_global = set(top_chain_locked_blocks)
            if self._phase3_boundary_relax_active and current_full_cost < best_phase3_finalized_cost:
                best_phase3_finalized_cost = current_full_cost; best_phase3_finalized_positions = current_positions
                best_phase3_finalized_tree_global = tree.copy(); best_phase3_finalized_anchors_global = dict(dynamic_anchors)
                best_phase3_finalized_top_chain_locked_global = set(top_chain_locked_blocks)
        tree = best_tree; dynamic_anchors = best_restart_anchors; current_positions = best_restart_positions
        current_full_cost = best_restart_cost; top_chain_locked_blocks = set(best_restart_top_chain_locked)
        if current_full_cost < best_cost:
            best_cost = current_full_cost; best_positions = current_positions
            best_tree_global = tree.copy(); best_anchors_global = dict(dynamic_anchors)
            best_top_chain_locked_global = set(top_chain_locked_blocks)

        # Per-restart Phase 3 suppressed; final winner printed by solve()

        _runtime_elapsed = time.perf_counter() - start_time
        _stats = {
            "block_count": block_count,
            "effective_n": effective_n,
            "schedule": self.fastsa_schedule_mode,
            "max_levels": self.fastsa_max_levels,
            "moves_total": moves_total,
            "stage1_moves": stage_move_counts[1],
            "stage2_moves": stage_move_counts[2],
            "stage3_moves": stage_move_counts[3],
            "best_cost": best_cost,
            "best_phase3_finalized_cost": best_phase3_finalized_cost,
            "runtime_seconds": _runtime_elapsed,
        }
        if self.stats_path:
            try:
                import json as _json_stats
                with open(self.stats_path, "a", encoding="utf-8") as _sf:
                    _sf.write(_json_stats.dumps(_stats) + "\n")
            except Exception:
                pass
        return {
            "best_cost": best_cost, "best_positions": best_positions,
            "best_tree_global": best_tree_global, "best_anchors_global": best_anchors_global,
            "best_top_chain_locked_global": best_top_chain_locked_global,
            "best_finalized_cost": best_finalized_cost, "best_finalized_positions": best_finalized_positions,
            "best_finalized_tree_global": best_finalized_tree_global,
            "best_finalized_anchors_global": best_finalized_anchors_global,
            "best_finalized_top_chain_locked_global": best_finalized_top_chain_locked_global,
            "best_phase3_finalized_cost": best_phase3_finalized_cost,
            "best_phase3_finalized_positions": best_phase3_finalized_positions,
            "best_phase3_finalized_tree_global": best_phase3_finalized_tree_global,
            "best_phase3_finalized_anchors_global": best_phase3_finalized_anchors_global,
            "best_phase3_finalized_top_chain_locked_global": best_phase3_finalized_top_chain_locked_global,
            "runtime_seconds": _runtime_elapsed,
            "stats": _stats,
            "preplaced_anchors": preplaced_anchors,
            "boundary_codes": boundary_codes,
            "immovable_blocks": immovable_blocks,
            "target_rects": target_rects,
            "geometry_locked_blocks": geometry_locked_blocks,
        }

    def solve(
        self,
        block_count,
        area_targets,
        b2b_connectivity,
        p2b_connectivity,
        pins_pos,
        constraints,
        target_positions=None,
    ) -> List[Rect]:
        # Main contest entry point.
        # Pipeline:
        # 1. preprocess inputs
        # 2. build one balanced B*-tree start
        # 3. enforce boundary chains on that start
        # 4. run FAST-SA with copy/perturb/pack/cost/accept
        # 5. return the best packed rectangles seen
        start_time = time.perf_counter()
        time_limit = min(self.max_time_limit, self.base_time_limit + self.time_limit_per_block * block_count)
        if self.experiment_seed >= 0:
            # Reproducible experiments: same seed for every schedule variant.
            random.seed(self.experiment_seed + int(block_count))

        n_restarts = self.restarts
        par_mode = self.use_parallel_restarts and n_restarts > 1
        print(f"[solve] Running {n_restarts} restart{'s' if n_restarts != 1 else ''} "
              f"{'in parallel' if par_mode else 'sequentially'} "
              f"on {os.cpu_count() or 1} core{'s' if (os.cpu_count() or 1) != 1 else ''}")

        best_positions: Optional[List[Rect]] = None
        best_cost = float("inf")
        best_tree_global: Optional[BStarTree] = None
        best_anchors_global: Dict[int, Tuple[float, float]] = {}
        best_top_chain_locked_global: set[int] = set()
        best_finalized_positions: Optional[List[Rect]] = None
        best_finalized_cost = float("inf")
        best_finalized_tree_global: Optional[BStarTree] = None
        best_finalized_anchors_global: Dict[int, Tuple[float, float]] = {}
        best_finalized_top_chain_locked_global: set[int] = set()
        best_phase3_finalized_positions: Optional[List[Rect]] = None
        best_phase3_finalized_cost = float("inf")
        best_phase3_finalized_tree_global: Optional[BStarTree] = None
        best_phase3_finalized_anchors_global: Dict[int, Tuple[float, float]] = {}
        best_phase3_finalized_top_chain_locked_global: set[int] = set()
        best_official_cost = float("inf")
        best_official_tree_global: Optional[BStarTree] = None
        best_official_anchors_global: Dict[int, Tuple[float, float]] = {}
        best_official_top_chain_locked_global: set[int] = set()
        best_official_positions: Optional[List[Rect]] = None

        if par_mode:
            # Fork BEFORE any preprocessing — each worker builds its own data
            # from raw tensors, so zero COW page faults from shared Python objects.
            opt_state = {k: v for k, v in self.__dict__.items() if not k.startswith('_gpu_')}
            import concurrent.futures
            import multiprocessing
            n_workers = min(n_restarts, self._restart_workers)
            _all_results = [None] * n_restarts
            _done = 0
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=multiprocessing.get_context('spawn' if IS_WINDOWS else 'fork'),
            ) as executor:
                _futures = {
                    executor.submit(_fork_worker_from_raw, opt_state, i,
                        block_count, area_targets, b2b_connectivity,
                        p2b_connectivity, pins_pos, constraints, target_positions,
                        start_time, time_limit): i
                    for i in range(n_restarts)
                }
                for _fut in concurrent.futures.as_completed(_futures):
                    _ri = _futures[_fut]
                    _all_results[_ri] = _fut.result()
                    _done += 1
                    print(f"\r  {_GRY}Phase 3 [{_done}/{n_restarts}] restarts complete{_C}", end="", flush=True)
                print()
        else:
            _restart_data = self._build_restart_data(
                block_count, area_targets, b2b_connectivity, p2b_connectivity,
                pins_pos, constraints, target_positions, start_time, time_limit,
            )
            if self.use_gpu and _HAS_CUPY:
                self._init_gpu_data(
                    _restart_data['b2b_edges'], _restart_data['p2b_edges'],
                    _restart_data['pin_positions'],
                )
            _all_results = [self._solve_one_restart(**_restart_data) for _ in range(n_restarts)]

        if par_mode:
            # Workers each built their own preprocessing state in their own
            # process; the parent still needs the same caches for the final
            # repack / boundary-repair fallback below.
            self._build_restart_data(
                block_count, area_targets, b2b_connectivity, p2b_connectivity,
                pins_pos, constraints, target_positions, start_time, time_limit,
            )

        # Shared context for the final repack fallback (also works in
        # ProcessPool mode, where the locals of _build_restart_data do not exist).
        _ctx = _all_results[0] if _all_results else {}
        preplaced_anchors = _ctx.get("preplaced_anchors", {})
        boundary_codes = _ctx.get("boundary_codes", {})
        immovable_blocks = _ctx.get("immovable_blocks", set())
        target_rects = _ctx.get("target_rects", [])
        geometry_locked_blocks = _ctx.get("geometry_locked_blocks", set())

        def _finalize_safe(positions):
            """Never return overlapping rectangles: fall back to the best
            directly-packed state if a tree repack became inconsistent."""
            if positions is None:
                positions = best_positions if best_positions is not None else []
            if positions and best_positions is not None:
                try:
                    if contest_api.check_overlap(positions) > 0 and contest_api.check_overlap(best_positions) == 0:
                        positions = list(best_positions)
                except Exception:
                    pass
            if positions:
                positions = self._enforce_locked_rect_geometry(
                    positions, target_rects, geometry_locked_blocks, immovable_blocks,
                )
            return positions

        # First pass: collect HPWL and area from all restart results
        try:
            from iccad2026_evaluate import calculate_hpwl_b2b, calculate_hpwl_p2b, calculate_bbox_area
        except ImportError:
            _b2b_conn_ref = b2b_connectivity; _p2b_conn_ref = p2b_connectivity; _pins_ref = pins_pos
            def calculate_hpwl_b2b(positions, b2b_conn=_b2b_conn_ref):
                if b2b_conn is None or len(b2b_conn) == 0: return 0.0
                total = 0.0
                for edge in b2b_conn:
                    if edge[0] == -1: continue
                    i, j, weight = int(edge[0]), int(edge[1]), float(edge[2])
                    if i < len(positions) and j < len(positions):
                        x1 = positions[i][0] + positions[i][2] / 2
                        y1 = positions[i][1] + positions[i][3] / 2
                        x2 = positions[j][0] + positions[j][2] / 2
                        y2 = positions[j][1] + positions[j][3] / 2
                        total += weight * (abs(x2 - x1) + abs(y2 - y1))
                return total
            def calculate_hpwl_p2b(positions, p2b_conn=_p2b_conn_ref, pins_p=_pins_ref):
                if p2b_conn is None or len(p2b_conn) == 0: return 0.0
                total = 0.0
                for edge in p2b_conn:
                    if edge[0] == -1: continue
                    pin_idx, block_idx, weight = int(edge[0]), int(edge[1]), float(edge[2])
                    if block_idx < len(positions) and pin_idx < len(pins_p):
                        px, py = float(pins_p[pin_idx][0]), float(pins_p[pin_idx][1])
                        bx = positions[block_idx][0] + positions[block_idx][2] / 2
                        by = positions[block_idx][1] + positions[block_idx][3] / 2
                        total += weight * (abs(px - bx) + abs(py - by))
                return total
            def calculate_bbox_area(positions):
                if not positions: return 0.0
                x_min = min(p[0] for p in positions)
                y_min = min(p[1] for p in positions)
                x_max = max(p[0] + p[2] for p in positions)
                y_max = max(p[1] + p[3] for p in positions)
                return (x_max - x_min) * (y_max - y_min)

        _hpwls: List[float] = []
        _areas: List[float] = []
        for result in _all_results:
            rcost = result["best_cost"]
            if rcost < best_cost:
                best_cost = rcost; best_positions = result["best_positions"]
                best_tree_global = result["best_tree_global"]
                best_anchors_global = result["best_anchors_global"]
                best_top_chain_locked_global = result["best_top_chain_locked_global"]
            rcost = result["best_finalized_cost"]
            if rcost < best_finalized_cost:
                best_finalized_cost = rcost; best_finalized_positions = result["best_finalized_positions"]
                best_finalized_tree_global = result["best_finalized_tree_global"]
                best_finalized_anchors_global = result["best_finalized_anchors_global"]
                best_finalized_top_chain_locked_global = result["best_finalized_top_chain_locked_global"]
            rcost = result["best_phase3_finalized_cost"]
            if rcost < best_phase3_finalized_cost:
                best_phase3_finalized_cost = rcost; best_phase3_finalized_positions = result["best_phase3_finalized_positions"]
                best_phase3_finalized_tree_global = result["best_phase3_finalized_tree_global"]
                best_phase3_finalized_anchors_global = result["best_phase3_finalized_anchors_global"]
                best_phase3_finalized_top_chain_locked_global = result["best_phase3_finalized_top_chain_locked_global"]
            # Collect HPWL and area for baseline averaging (use original tensors)
            _pos3 = result.get("best_phase3_finalized_positions")
            if _pos3 is not None:
                _hpwls.append(
                    calculate_hpwl_b2b(_pos3, b2b_connectivity) + calculate_hpwl_p2b(_pos3, p2b_connectivity, pins_pos)
                )
                _areas.append(calculate_bbox_area(_pos3))
        # Second pass: average HPWL/area across restarts as shared baseline, pick official-cost winner
        try:
            from iccad2026_evaluate import evaluate_solution
        except ImportError:
            def evaluate_solution(solution, baseline, constraints_ref, b2b_ref, p2b_ref, pins_ref,
                                  areas_ref, target_pos_ref, median_runtime=1.0):
                class _Metrics:
                    def __init__(self):
                        self.cost = float("inf")
                        self.is_feasible = False
                        self.violations_relative = 1.0
                return _Metrics()
        if _hpwls:
            _avg_hpwl = sum(_hpwls) / len(_hpwls)
            _avg_area = sum(_areas) / len(_areas)
            _hpwl_bl = _avg_hpwl / 2
            _area_bl = _avg_area * 0.67
            _shared_baseline = {"hpwl_baseline": _hpwl_bl, "area_baseline": _area_bl}
            _rows = []; _winner_ri = None
            for _ri, result in enumerate(_all_results):
                _pos3 = result.get("best_phase3_finalized_positions")
                if _pos3 is not None:
                    _rt = result.get("runtime_seconds", 1.0)
                    _metrics = evaluate_solution(
                        {"positions": _pos3, "runtime": _rt},
                        _shared_baseline, constraints, b2b_connectivity, p2b_connectivity, pins_pos,
                        area_targets, target_positions,
                        median_runtime=_rt,
                    )
                    _oc = _metrics.cost
                    _h_loc = calculate_hpwl_b2b(_pos3, b2b_connectivity) + calculate_hpwl_p2b(_pos3, p2b_connectivity, pins_pos)
                    _a_loc = calculate_bbox_area(_pos3)
                    _hg = (_h_loc - _hpwl_bl) / _hpwl_bl * 100
                    _ag = (_a_loc - _area_bl) / _area_bl * 100
                    _rows.append((_ri, _h_loc, _a_loc, _hg, _ag, _oc, _metrics.is_feasible, _metrics.violations_relative))
                    if _oc < best_official_cost:
                        best_official_cost = _oc; _winner_ri = _ri
                        best_official_tree_global = result.get("best_phase3_finalized_tree_global") or result.get("best_finalized_tree_global") or result.get("best_tree_global")
                        best_official_anchors_global = result.get("best_phase3_finalized_anchors_global") or result.get("best_finalized_anchors_global") or result.get("best_anchors_global") or {}
                        best_official_top_chain_locked_global = result.get("best_phase3_finalized_top_chain_locked_global") or result.get("best_finalized_top_chain_locked_global") or result.get("best_top_chain_locked_global") or set()
                        best_official_positions = _pos3
            # Pretty table
            print(f"{_B}[merge]{_C}  {_CYN}HPWL baseline={_hpwl_bl:.2f}, Area baseline={_area_bl:.0f}{_C}")
            _sep = f"  {_GRY}{'─'*74}{_C}"
            print(_sep)
            _hdr = f"  {_B}{'R#':>3}{_C} │ {_B}{'HPWL':>8}{_C} │ {_B}{'Area':>8}{_C} │ {_B}{'HPWL Δ':>7}{_C} │ {_B}{'Area Δ':>7}{_C} │ {_B}{'Cost':>8}{_C} │ {_B}{'Feas':>4}{_C} │ {_B}{'Viol':>5}{_C}"
            print(_hdr)
            print(_sep)
            for _ri, _h_loc, _a_loc, _hg, _ag, _oc, _ifeas, _viol in _rows:
                _hcol = _G if _hg <= 0 else _R
                _acol = _G if _ag <= 0 else _R
                _chk = f"{_G}OK{_C}" if _ifeas else f"{_R}NO{_C}"
                _is_winner = _ri == _winner_ri
                _mrk = " \u25c0" if _is_winner else "  "
                _row_color = _B if _is_winner else ""
                print(f"  {_row_color}{_ri:>3}{_C} │ {_row_color}{_h_loc:>8.2f}{_C} │ {_row_color}{_a_loc:>8.0f}{_C} │ {_hcol}{_hg:>+6.1f}%{_C} │ {_acol}{_ag:>+6.1f}%{_C} │ {_row_color}{_Y if _is_winner else ''}{_oc:>8.4f}{_C} │ {_chk} │ {_row_color}{_viol:>5.2f}{_C} {_mrk}")
            print(_sep)
        if best_official_tree_global is not None:
            print(f"  {_B}{_Y}Winner{_C}: restart {_B}{_winner_ri}{_C} (cost={_G}{best_official_cost:.4f}{_C})")

            best_phase3_finalized_tree_global = best_official_tree_global
            best_phase3_finalized_anchors_global = best_official_anchors_global
            best_phase3_finalized_top_chain_locked_global = best_official_top_chain_locked_global
            if best_official_positions is not None:
                best_phase3_finalized_positions = best_official_positions
        if best_phase3_finalized_tree_global is not None:
            final_tree = best_phase3_finalized_tree_global.copy()
            # Use the positions from the restart directly (already fully processed)
            # to avoid repack inconsistencies. Fall back to tree repack if missing.
            if best_phase3_finalized_positions is not None:
                final_positions = list(best_phase3_finalized_positions)
            else:
                self._current_preplaced_overlap_lambda = self.preplaced_overlap_lambda
                self._phase3_preplaced_reinsert_active = True
                self._phase3_boundary_relax_active = True
                final_positions = self._pack_compact_with_boundary_repair(
                    final_tree,
                    self._combine_anchors(preplaced_anchors, best_phase3_finalized_anchors_global),
                    boundary_codes,
                    best_phase3_finalized_top_chain_locked_global,
                    immovable_blocks,
                )
                final_positions = self._enforce_locked_rect_geometry(
                    final_positions,
                    target_rects,
                    geometry_locked_blocks,
                    immovable_blocks,
                )
            return _finalize_safe(final_positions)

        if best_finalized_tree_global is not None:
            self._current_preplaced_overlap_lambda = self.preplaced_overlap_lambda
            self._phase3_preplaced_reinsert_active = True
            self._phase3_boundary_relax_active = True
            final_tree = best_finalized_tree_global.copy()
            final_positions = self._pack_compact_with_boundary_repair(
                    final_tree,
                    self._combine_anchors(preplaced_anchors, best_finalized_anchors_global),
                    boundary_codes,
                    best_finalized_top_chain_locked_global,
                    immovable_blocks,
                )
            final_positions = self._enforce_locked_rect_geometry(
                final_positions,
                target_rects,
                geometry_locked_blocks,
                immovable_blocks,
            )
            return _finalize_safe(final_positions)

        final_positions = best_positions if best_positions is not None else []
        return _finalize_safe(final_positions)

    # ------------------------------------------------------------------
    #  Boundary tree operations (chain topology enforcement)
    # ------------------------------------------------------------------
    def _apply_hard_boundary_tree_rules(
        self,
        tree: BStarTree,
        boundary_codes: Dict[int, int],
    ) -> None:
        # Enforce the structural boundary model:
        # - bottom-chain lives on repeated left children
        # - left-chain lives on repeated right children
        # - top-chain lives on repeated left children
        # - right-chain lives on repeated right children
        # - BL node becomes the root when it exists
        # - TL is ordered to the tail of the left-chain
        # - BR is ordered to the tail of the bottom-chain
        if tree.n == 0 or not boundary_codes:
            return

        ordered_bottom = self._ordered_bottom_chain_blocks(boundary_codes)
        ordered_left = self._ordered_left_chain_blocks(boundary_codes)
        ordered_top = self._ordered_top_chain_blocks(boundary_codes)
        ordered_right = self._ordered_right_chain_blocks(boundary_codes)

        if ordered_bottom:
            chain_root = ordered_bottom[0]
            # Bottom-left constrained block must stay as the B*-tree root.
            if tree.root != chain_root:
                tree.swap_nodes(tree.root, chain_root, ())

            parent = chain_root
            for block in ordered_bottom[1:]:
                self._force_child_relation(tree, parent, block, as_left=True, boundary_codes=boundary_codes)
                parent = block

        if ordered_left:
            left_root = ordered_left[0]
            if tree.root != left_root and not ordered_bottom:
                tree.swap_nodes(tree.root, left_root, ())

            parent = left_root
            for block in ordered_left[1:]:
                self._force_child_relation(tree, parent, block, as_left=False, boundary_codes=boundary_codes)
                parent = block

        if ordered_top:
            parent = ordered_top[0]
            for block in ordered_top[1:]:
                self._force_child_relation(tree, parent, block, as_left=True, boundary_codes=boundary_codes)
                parent = block

            for block, code in boundary_codes.items():
                if not (code & TOP):
                    continue
                self._clear_child_slot(tree, block, as_left=False, boundary_codes=boundary_codes)

        if ordered_right:
            anchor = self._right_root_block(boundary_codes)
            if anchor is None:
                anchor = ordered_right[0]
                parent = anchor
                for block in ordered_right[1:]:
                    self._force_child_relation(tree, parent, block, as_left=False, boundary_codes=boundary_codes)
                    parent = block
            else:
                first = ordered_right[0]
                self._force_child_relation(tree, anchor, first, as_left=False, boundary_codes=boundary_codes)
                parent = first
                for block in ordered_right[1:]:
                    self._force_child_relation(tree, parent, block, as_left=False, boundary_codes=boundary_codes)
                    parent = block

    def _force_child_relation(
        self,
        tree: BStarTree,
        parent: int,
        node: int,
        as_left: bool,
        boundary_codes: Dict[int, int],
    ) -> None:
        # Force one exact parent->child relation while relocating any old occupant away first.
        if parent == node or parent == -1 or node == -1:
            return

        current_child = tree.left[parent] if as_left else tree.right[parent]
        guard = 0
        max_relocations = max(1, tree.n * 2)
        while current_child != -1 and current_child != node and guard < max_relocations:
            relocated = self._relocate_node(tree, current_child, boundary_codes, forbidden_parent=parent)
            if not relocated:
                break
            current_child = tree.left[parent] if as_left else tree.right[parent]
            guard += 1

        if tree.parent[node] == parent and ((tree.left[parent] == node) if as_left else (tree.right[parent] == node)):
            return

        tree._delete_node(node)
        tree._insert_node(node, parent, as_left)

    def _clear_child_slot(
        self,
        tree: BStarTree,
        parent: int,
        as_left: bool,
        boundary_codes: Dict[int, int],
    ) -> None:
        # Clear one child slot by relocating the occupant elsewhere.
        current_child = tree.left[parent] if as_left else tree.right[parent]
        guard = 0
        max_relocations = max(1, tree.n * 2)
        while current_child != -1 and guard < max_relocations:
            relocated = self._relocate_node(tree, current_child, boundary_codes, forbidden_parent=parent)
            if not relocated:
                break
            current_child = tree.left[parent] if as_left else tree.right[parent]
            guard += 1

    def _enforce_top_chain_constraints(
        self,
        tree: BStarTree,
        boundary_codes: Dict[int, int],
    ) -> None:
        # Hard invariant: TL / TOP / TR nodes cannot keep right children.
        top_nodes = [block for block, code in boundary_codes.items() if code & TOP]
        for block in top_nodes:
            self._clear_child_slot(tree, block, as_left=False, boundary_codes=boundary_codes)

    def _relocate_node(
        self,
        tree: BStarTree,
        node: int,
        boundary_codes: Dict[int, int],
        forbidden_parent: Optional[int] = None,
    ) -> bool:
        candidates = [
            block for block in range(tree.n)
            if block != node and block != forbidden_parent and (0 <= block < tree.n) and tree.active[block]
        ]
        candidates.sort(key=lambda block: 1 if block in boundary_codes else 0)

        for target in candidates:
            target_mask = self._structural_boundary_mask(boundary_codes, target)
            if (target_mask & LEFT) and tree.left[target] == -1:
                tree._delete_node(node)
                tree._insert_node(node, target, True)
                return True
            if (target_mask & BOTTOM) and tree.right[target] == -1:
                tree._delete_node(node)
                tree._insert_node(node, target, False)
                return True
            if (target_mask & RIGHT) and tree.right[target] == -1:
                tree._delete_node(node)
                tree._insert_node(node, target, False)
                return True
            if tree.left[target] == -1:
                tree._delete_node(node)
                tree._insert_node(node, target, True)
                return True
            if tree.right[target] == -1 and not (target_mask & TOP):
                tree._delete_node(node)
                tree._insert_node(node, target, False)
                return True
        return False

    def _structural_boundary_mask(self, boundary_codes: Dict[int, int], block: int) -> int:
        # LEFT/BOTTOM/TOP/RIGHT all affect the active boundary-chain structure.
        if 0 <= block < len(self._cached_boundary_masks):
            return self._cached_boundary_masks[block]
        return boundary_codes.get(block, 0) & (LEFT | BOTTOM | TOP | RIGHT)

    def _is_boundary_constraint_block(self, boundary_codes: Dict[int, int], block: int) -> bool:
        return self._structural_boundary_mask(boundary_codes, block) != 0

    def _is_corner_constraint_block(self, boundary_codes: Dict[int, int], block: int) -> bool:
        return self._structural_boundary_mask(boundary_codes, block).bit_count() >= 2

    def _phase3_can_free_boundary(self, boundary_codes: Dict[int, int], block: int) -> bool:
        return self._phase3_boundary_relax_active and self._is_boundary_constraint_block(boundary_codes, block)

    def _phase3_can_free_boundary_geometry(self, block: int) -> bool:
        return self._phase3_boundary_relax_active and block in self._phase3_relaxed_geometry_boundary_blocks

    def _active_geometry_locked_blocks(self, geometry_locked_blocks: Iterable[int]) -> set[int]:
        if geometry_locked_blocks is self._cached_geometry_locked_blocks:
            locked = set(self._cached_active_geometry_locked_blocks)
        else:
            locked = set(geometry_locked_blocks)
        if self._phase3_boundary_relax_active:
            locked.difference_update(self._phase3_relaxed_geometry_boundary_blocks)
        return locked

    def _default_sampling_window(
        self,
        blocks: Iterable[int],
    ) -> Tuple[float, float]:
        return 1.0 / 3.0, 3.0

    def _commit_tree_state(self, tree: BStarTree, source: BStarTree) -> None:
        tree.parent = source.parent.copy()
        tree.left = source.left.copy()
        tree.right = source.right.copy()
        tree.widths = source.widths.copy()
        tree.heights = source.heights.copy()
        tree.root = source.root

    def _left_tail_block(self, boundary_codes: Dict[int, int]) -> Optional[int]:
        # Use TL if present; otherwise fall back to the last block on the left chain.
        if self._cached_left_tail is not None:
            return self._cached_left_tail
        if self._cached_ordered_left:
            return self._cached_ordered_left[-1]
        return None

    def _bottom_tail_block(self, boundary_codes: Dict[int, int]) -> Optional[int]:
        # Use BR if present; otherwise fall back to the last block on the bottom chain.
        if self._cached_bottom_tail is not None:
            return self._cached_bottom_tail
        if self._cached_ordered_bottom:
            return self._cached_ordered_bottom[-1]
        return None

    def _top_root_block(self, boundary_codes: Dict[int, int]) -> Optional[int]:
        # Unique TL block, if present. Otherwise use the end of the left chain as the anchor.
        return self._cached_top_root

    def _top_tail_block(self, boundary_codes: Dict[int, int]) -> Optional[int]:
        # Use TR if present; otherwise fall back to the last block on the top chain.
        if self._cached_top_tail is not None:
            return self._cached_top_tail
        if self._cached_ordered_top:
            return self._cached_ordered_top[-1]
        return None

    def _right_root_block(self, boundary_codes: Dict[int, int]) -> Optional[int]:
        # Right chain hangs from BR when present. Otherwise use the end of the bottom chain.
        return self._cached_right_root

    def _right_tail_block(self, boundary_codes: Dict[int, int]) -> Optional[int]:
        # Use BR if present; otherwise fall back to the last block on the right chain,
        # or to the right-chain anchor when the pure-right chain is empty.
        if self._cached_bottom_tail is not None:
            return self._cached_bottom_tail
        if self._cached_ordered_right:
            return self._cached_ordered_right[-1]
        if self._cached_right_root is not None:
            return self._cached_right_root
        return None

    def _ordered_bottom_chain_blocks(self, boundary_codes: Dict[int, int]) -> List[int]:
        # Bottom chain starts at BL when possible and ends at BR when possible.
        return list(self._cached_ordered_bottom)

    def _ordered_left_chain_blocks(self, boundary_codes: Dict[int, int]) -> List[int]:
        # Left chain starts at BL when possible and ends at TL when possible.
        return list(self._cached_ordered_left)

    def _ordered_top_chain_blocks(self, boundary_codes: Dict[int, int]) -> List[int]:
        # Top chain starts at TL (or the end of the left chain) and ends at TR.
        return list(self._cached_ordered_top)

    def _ordered_right_chain_blocks(self, boundary_codes: Dict[int, int]) -> List[int]:
        # Right chain contains only pure RIGHT blocks.
        # BR is the anchor for the chain, and BR stays only on the bottom chain.
        return list(self._cached_ordered_right)

    def _insert_boundary_aware(
        self,
        tree: BStarTree,
        block: int,
        boundary_codes: Dict[int, int],
        locked: Iterable[int],
        force_top_chain: bool = False,
        ignore_finalize: bool = False,
    ) -> None:
        block_mask = self._structural_boundary_mask(boundary_codes, block)
        if force_top_chain:
            block_mask = TOP
        if self._boundary_finalize_active and (block_mask != 0 or force_top_chain) and not ignore_finalize:
            return

        locked_set = set(locked)
        candidates = [i for i in range(tree.n) if i != block and i not in locked_set and tree.active[i]]
        random.shuffle(candidates)
        left_tail = self._left_tail_block(boundary_codes)
        bottom_tail = self._bottom_tail_block(boundary_codes)
        top_tail = self._top_tail_block(boundary_codes)
        right_anchor = self._right_root_block(boundary_codes)

        if block_mask == LEFT:
            if block == left_tail:
                candidates = [i for i in candidates if self._structural_boundary_mask(boundary_codes, i) & LEFT]
            else:
                candidates = [
                    i
                    for i in candidates
                    if (self._structural_boundary_mask(boundary_codes, i) & LEFT)
                    and i != left_tail
                ]
        elif block_mask == BOTTOM:
            if block == bottom_tail:
                candidates = [i for i in candidates if self._structural_boundary_mask(boundary_codes, i) & BOTTOM]
            else:
                candidates = [
                    i
                    for i in candidates
                    if (self._structural_boundary_mask(boundary_codes, i) & BOTTOM)
                    and i != bottom_tail
                ]
        elif block_mask == TOP:
            if block == top_tail:
                candidates = [i for i in candidates if (self._structural_boundary_mask(boundary_codes, i) & TOP) and i not in self._cached_immovable_blocks]
            else:
                candidates = [
                    i
                    for i in candidates
                    if (self._structural_boundary_mask(boundary_codes, i) & TOP)
                    and i != top_tail
                    and i not in self._cached_immovable_blocks
                ]
        elif block_mask == RIGHT:
            chain_members = [
                i
                for i in candidates
                if self._structural_boundary_mask(boundary_codes, i) == RIGHT
                and i not in self._cached_immovable_blocks
            ]
            candidates = chain_members
            if right_anchor is not None and right_anchor != block and right_anchor not in candidates and right_anchor not in self._cached_immovable_blocks:
                candidates = [right_anchor] + candidates

        if not candidates:
            return

        target = None
        as_left = True
        for candidate in candidates:
            target_mask = self._structural_boundary_mask(boundary_codes, candidate)
            if block_mask == LEFT:
                if tree.right[candidate] == -1:
                    target = candidate
                    as_left = False
                    break
                continue
            if block_mask == BOTTOM:
                if tree.left[candidate] == -1:
                    target = candidate
                    as_left = True
                    break
                continue
            if block_mask == TOP:
                if tree.left[candidate] == -1:
                    target = candidate
                    as_left = True
                    break
                continue
            if block_mask == RIGHT:
                if candidate == right_anchor and tree.right[candidate] == -1:
                    target = candidate
                    as_left = False
                    break
                if self._structural_boundary_mask(boundary_codes, candidate) == RIGHT and tree.right[candidate] == -1:
                    target = candidate
                    as_left = False
                    break
                continue
            if self._boundary_finalize_active and target_mask != 0:
                continue
            open_sides = self._allowed_insert_sides_for_target(target_mask, tree, candidate)
            if open_sides:
                target = candidate
                as_left = random.choice(open_sides)
                break

        if target is None:
            return

        tree._insert_leaf_node(block, target, as_left)

    def _allowed_insert_sides_for_target(
        self,
        target_mask: int,
        tree: BStarTree,
        candidate: int,
    ) -> List[bool]:
        # Child-slot rule for attaching a non-boundary block near boundary nodes:
        # - LEFT / BOTTOM / BL: both sides
        # - RIGHT / BR: no side
        # - TOP / TL: no side
        # - TR: no side
        open_sides: List[bool] = []
        # has_left = tree.left[candidate] == -1
        # has_right = tree.right[candidate] == -1
        if (target_mask & TOP) or (target_mask & RIGHT):
            return open_sides
        # if target_mask in (TOP, LEFT | TOP):
        #     if has_left:
        #         open_sides.append(True)
        #     return open_sides
        # if target_mask in (RIGHT, RIGHT | BOTTOM):
        #     if has_right:
        #         open_sides.append(False)
        if (target_mask & LEFT) and not (target_mask & BOTTOM):
            open_sides.append(True)
            return open_sides
        if (target_mask & BOTTOM) and not (target_mask & LEFT):
            open_sides.append(False)
            return open_sides
        open_sides.append(True)
        open_sides.append(False)
        return open_sides

    # ------------------------------------------------------------------
    #  FAST-SA parameter estimation & temperature schedule
    # ------------------------------------------------------------------
    def _estimate_fastsa_parameters(
        self,
        tree: BStarTree,
        areas: Sequence[float],
        locked_blocks: Iterable[int],
        mib_groups: Dict[int, List[int]],
        current_cost: float,
        preplaced_anchors: Dict[int, Tuple[float, float]],
        dynamic_anchors: Dict[int, Tuple[float, float]],
        geometry_locked_blocks: Iterable[int],
        immovable_blocks: Iterable[int],
        target_rects: Sequence[Rect],
        b2b_edges: List[Tuple[int, int, float]],
        p2b_edges: List[Tuple[int, int, float]],
        pin_positions: List[Tuple[float, float]],
        cluster_groups: Dict[int, List[int]],
        boundary_codes: Dict[int, int],
    ) -> Tuple[float, float]:
        # Probe a handful of random moves to estimate:
        # - a starting temperature that gives the target uphill acceptance rate
        # - a normalized average cost-change ratio used by later FAST-SA stages
        probe_count = max(12, min(60, self.fastsa_probe_factor * max(1, tree.n)))
        uphill_deltas: List[float] = []
        delta_ratios: List[float] = []
        probe_areas: List[float] = []
        probe_wirelengths: List[float] = []

        if self.use_parallel_probe and not IS_WINDOWS:
            _probe_parallel_ok = False
            try:
                import pickle
                pickle.dumps(_probe_worker)
                _probe_parallel_ok = True
            except Exception:
                pass
            if _probe_parallel_ok:
                try:
                    import concurrent.futures
                    import multiprocessing
                    n_workers = min(probe_count, multiprocessing.cpu_count())
                    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                        locked_set = set(locked_blocks) if not isinstance(locked_blocks, set) else locked_blocks
                        geo_set = set(geometry_locked_blocks) if not isinstance(geometry_locked_blocks, set) else geometry_locked_blocks
                        immovable_set = set(immovable_blocks) if not isinstance(immovable_blocks, set) else immovable_blocks
                        args_list = [
                            (self, tree.copy(), list(areas), set(locked_set), dict(mib_groups),
                             dict(cluster_groups), dict(boundary_codes), set(geo_set),
                             set(immovable_set), list(target_rects), list(b2b_edges), list(p2b_edges),
                             list(pin_positions), dict(preplaced_anchors), dict(dynamic_anchors),
                             current_cost)
                            for _ in range(probe_count)
                        ]
                        results = list(executor.map(_probe_worker, args_list))
                    for delta, probe_area, probe_wl in results:
                        if delta > 0:
                            uphill_deltas.append(delta)
                        delta_ratios.append(self._normalize_cost_ratio(delta, current_cost))
                        probe_areas.append(probe_area)
                        probe_wirelengths.append(probe_wl)
                except Exception:
                    pass
        if not self.use_parallel_probe or len(probe_areas) < probe_count:
            # Sequential fallback (also the normal path when parallel is disabled)
            for _ in range(probe_count - len(probe_areas)):
                trial_anchors = dict(dynamic_anchors)
                tree_snapshot = tree.snapshot_state()
                moved_blocks = self._apply_regular_move(
                    tree,
                    areas,
                    locked_blocks,
                    mib_groups,
                    cluster_groups,
                    boundary_codes,
                    geometry_locked_blocks,
                    stage2_mode=True,
                )
                self._enforce_top_chain_constraints(tree, boundary_codes)
                trial_positions = tree.pack(
                    self._combine_anchors(preplaced_anchors, trial_anchors),
                    boundary_codes,
                )
                trial_positions = self._pack_compact_with_boundary_repair(
                    tree,
                    self._combine_anchors(preplaced_anchors, trial_anchors),
                    boundary_codes,
                    immovable_blocks=immovable_blocks,
                )
                trial_cost = self._cost(
                    trial_positions,
                    b2b_edges,
                    p2b_edges,
                    pin_positions,
                    cluster_groups,
                    boundary_codes,
                    areas,
                    preplaced_anchors,
                    trial_anchors,
                    immovable_blocks,
                    tree,
                )
                probe_areas.append(contest_api.calculate_bbox_area(trial_positions))
                hpwl_b2b, hpwl_p2b = self._wirelength_cost(
                    trial_positions,
                    b2b_edges,
                    p2b_edges,
                    pin_positions,
                )
                probe_wirelengths.append(hpwl_b2b + hpwl_p2b)
                delta = trial_cost - current_cost
                if delta > 0:
                    uphill_deltas.append(delta)
                delta_ratios.append(self._normalize_cost_ratio(delta, current_cost))
                tree.restore_state(tree_snapshot)

        avg_uphill = sum(uphill_deltas) / len(uphill_deltas) if uphill_deltas else max(current_cost * 0.1, 1.0)
        self._cached_area_norm = max(sum(probe_areas) / len(probe_areas), 1.0) if probe_areas else 1.0
        self._cached_wirelength_norm = (
            max(sum(probe_wirelengths) / len(probe_wirelengths), 1.0) if probe_wirelengths else 1.0
        )
        initial_temp = max(
            self.initial_temp,
            -avg_uphill / math.log(max(min(self.fastsa_initial_accept_prob, 0.999), 1e-6)),
        )
        avg_cost_ratio = max(
            self.fastsa_min_ratio,
            min(
                self.fastsa_max_ratio,
                sum(delta_ratios) / len(delta_ratios) if delta_ratios else self.fastsa_min_ratio,
            ),
        )
        return initial_temp, avg_cost_ratio

    def _fastsa_temperature(self, initial_temp: float, avg_cost_ratio: float, temp_level: int) -> float:
        # 3-stage FAST-SA temperature rule.
        if temp_level <= 1:
            return max(initial_temp, self.final_temp)
        if temp_level <= self.fastsa_k:
            return max(
                initial_temp * avg_cost_ratio / max(temp_level * self.fastsa_c, 1e-9),
                1e-6,
            )
        return max(initial_temp * avg_cost_ratio / max(temp_level, 1e-9), 1e-6)

    def _moves_for_fastsa_stage(self, base_moves_per_level: int, temp_level: int, block_count: int) -> int:
        def map_range_log(x: float) -> float:
            """
            Map input x from [1, 100] to output y in [1.1, 2].
            Logarithmic scaling.
            """
            y_min, y_max = 0.1, 2.0
            x = max(1.0, float(x))
            return y_min + (y_max - y_min) * math.log(x) / math.log(100)
        # Spend fewer moves in Stage II and more in Stage III polishing.
        if temp_level == 1:
            return base_moves_per_level
        if temp_level <= self.fastsa_k:
            return max(24, int(base_moves_per_level * self.fastsa_stage2_move_scale * map_range_log(block_count - 20)))
        if self.fastsa_schedule_mode == "original":
            # Paper-style Fast-SA has no extra Stage-III decimation; the caller
            # sets FLOORSET_STAGE3_SCALE to match the modified budget.
            return max(24, int(base_moves_per_level * self.fastsa_stage3_move_scale))
        return max(24, int(base_moves_per_level * self.fastsa_stage3_move_scale * self.phase3_move_factor))

    def _normalize_cost_ratio(self, delta: float, reference_cost: float) -> float:
        # Keep the FAST-SA ratio numerically stable.
        scale = max(abs(reference_cost), 1.0)
        ratio = abs(delta) / scale
        return max(self.fastsa_min_ratio, min(self.fastsa_max_ratio, ratio))

    def _pre_stage2_top_chain_insertion(
        self,
        tree: BStarTree,
        areas: Sequence[float],
        boundary_codes: Dict[int, int],
        locked_blocks: Iterable[int],
    ) -> List[int]:
        # Before Stage II, grow the top chain until its estimated width
        # (sqrt of total top-chain area) is close to the bottom chain's
        # estimated width. Extra unconstrained blocks are inserted into the
        # top chain until sqrt(sum(bottom areas)) - sqrt(sum(top areas)) < 10.
        if tree.n == 0:
            return []
        top_right = self._top_tail_block(boundary_codes)
        if top_right is None:
            return []

        locked = set(locked_blocks)
        inserted_blocks: List[int] = []
        bottom_chain = self._ordered_bottom_chain_blocks(boundary_codes)
        top_chain = self._ordered_top_chain_blocks(boundary_codes)
        bottom_area = sum(areas[block] for block in bottom_chain if 0 <= block < len(areas))
        top_area = sum(areas[block] for block in top_chain if 0 <= block < len(areas))

        while True:
            if math.sqrt(max(bottom_area, 0.0)) - math.sqrt(max(top_area, 0.0)) < 10.0:
                break

            candidates = [
                block
                for block in range(tree.n)
                if block not in locked
                and block not in inserted_blocks
                and self._structural_boundary_mask(boundary_codes, block) == 0
            ]
            if not candidates:
                break

            top_chain = self._ordered_top_chain_blocks(boundary_codes)
            top_chain = [block for block in top_chain if 0 <= block < tree.n]
            if not top_chain:
                break

            # Insert only immediately before the TR tail so the extra blocks
            # stay on the top chain end and do not interfere with the left chain.
            splice_targets = [block for block in top_chain if block != top_right]
            if not splice_targets:
                break

            # Prefer larger blocks first so the estimated top-chain width closes
            # the bottom/top gap in fewer insertions.
            candidates.sort(key=lambda block: areas[block], reverse=True)
            block = candidates[0]
            target = splice_targets[-1]
            tree._delete_node(block)
            tree._insert_node(block, target, True)
            inserted_blocks.append(block)
            top_area += areas[block]

        return inserted_blocks

    # ------------------------------------------------------------------
    #  Packing, compaction & contour operations
    # ------------------------------------------------------------------
    def _snap_right_top_boundary_clusters(
        self,
        positions: List[Rect],
        boundary_codes: Dict[int, int],
    ) -> List[Rect]:
        """Snap same-cluster blocks on TOP/RIGHT chains toward the TR corner block."""
        current = list(positions)
        tol = 1e-6

        # Build block-to-group lookup from the constraint-based cluster groups
        block_to_group: Dict[int, int] = {}
        for gid, members in self._cached_group_clusters.items():
            for m in members:
                block_to_group[m] = gid

        # Find TR corner blocks (TOP & RIGHT)
        tr_blocks = [b for b, code in boundary_codes.items() if (code & TOP) and (code & RIGHT)]
        if not tr_blocks:
            return current

        for tr_block in tr_blocks:
            tr_group = block_to_group.get(tr_block)
            if tr_group is None:
                continue
            tx, ty, tw, th = current[tr_block]

            # --- TOP chain: snap blocks to the right (toward TR) ---
            if self._cached_ordered_top:
                try:
                    tr_idx = self._cached_ordered_top.index(tr_block)
                except ValueError:
                    tr_idx = -1
                if tr_idx >= 0:
                    for i in range(tr_idx - 1, -1, -1):
                        block = self._cached_ordered_top[i]
                        if block_to_group.get(block) != tr_group:
                            break
                        if block in self._cached_immovable_blocks:
                            break
                        ref = self._cached_ordered_top[i + 1]
                        rx, ry, rw, rh = current[ref]
                        bx, by, bw, bh = current[block]
                        new_bx = rx - bw
                        if abs(bx - new_bx) > tol:
                            current[block] = (new_bx, by, bw, bh)
                else:
                    break

            # --- RIGHT chain: snap blocks upward (toward TR) ---
            if self._cached_ordered_right:
                chain = self._cached_ordered_right
                snap_y = ty  # Primary reference: TR corner's top Y
                for i in range(len(chain) - 1, -1, -1):
                    block = chain[i]
                    if block_to_group.get(block) != tr_group:
                        break
                    if block in self._cached_immovable_blocks:
                        continue
                    bx, by, bw, bh = current[block]
                    new_by = snap_y - bh
                    if abs(by - new_by) <= tol:
                        snap_y = by
                        continue
                    # Adjust downward only if the snapped position actually
                    # overlaps a same-cluster TOP chain block.
                    for _tb in self._cached_ordered_top:
                        if block_to_group.get(_tb) == tr_group:
                            tbx, tby, tbw, tbh = current[_tb]
                            if (bx < tbx + tbw and bx + bw > tbx and
                                new_by < tby + tbh and new_by + bh > tby):
                                new_by = max(new_by, tby + tbh)
                    overlap = False
                    for other in range(len(current)):
                        if other == block:
                            continue
                        ox, oy, ow, oh = current[other]
                        if (bx < ox + ow - tol and bx + bw > ox + tol and
                            new_by < oy + oh - tol and new_by + bh > oy + tol):
                            overlap = True
                            break
                    if not overlap:
                        current[block] = (bx, new_by, bw, bh)
                        snap_y = new_by
                    else:
                        continue

        return current

    def _pack_compact_with_boundary_repair(
        self,
        tree: BStarTree,
        anchors: Dict[int, Tuple[float, float]],
        boundary_codes: Dict[int, int],
        extra_top_blocks: Optional[Iterable[int]] = None,
        immovable_blocks: Optional[Iterable[int]] = None,
    ) -> List[Rect]:
        # Pack, do one geometric bottom/left preprocessing pass against preplaced
        # blocks, then run contour-based compaction. Compaction itself must not
        # modify the B*-tree topology.
        self._enforce_cluster_mib_geometry(tree)
        self._enforce_locked_geometry(
            tree,
            self._cached_target_rects,
            self._cached_geometry_locked_blocks,
        )
        self._refresh_cluster_metadata_from_tree(tree)
        positions = tree.pack(anchors, boundary_codes)
        positions = self._expand_cluster_satellites(positions)
        positions = self._preprocess_left_bottom_boundary_shapes(
            tree,
            positions,
            boundary_codes,
            anchors,
            immovable_blocks,
        )
        if self._phase3_boundary_relax_active or self._phase2_boundary_shape_only_active:
            compacted = self._compact_positions(
                positions,
                boundary_codes,
                extra_top_blocks,
                immovable_blocks,
            )
            compacted = self._snap_right_top_boundary_clusters(compacted, boundary_codes)
            return self._enforce_locked_rect_geometry(
                compacted,
                self._cached_target_rects,
                self._cached_geometry_locked_blocks,
                self._cached_immovable_blocks or immovable_blocks,
            )
        positions = self._compact_positions(
            positions,
            boundary_codes,
            extra_top_blocks,
            immovable_blocks,
        )
        positions = self._snap_right_top_boundary_clusters(positions, boundary_codes)
        return self._enforce_locked_rect_geometry(
            positions,
            self._cached_target_rects,
            self._cached_geometry_locked_blocks,
            self._cached_immovable_blocks or immovable_blocks,
        )

    def _enforce_cluster_mib_geometry(self, tree: BStarTree) -> None:
        touched_metadata: Dict[int, Dict[str, object]] = {}
        for blocks, ratio in self._cached_mib_cluster_ratio_groups:
            valid_blocks = [block for block in blocks if 0 <= block < tree.n]
            if valid_blocks:
                tree.set_group_aspect(valid_blocks, ratio)
                for block in valid_blocks:
                    metadata = self._cached_macro_cluster_metadata.get(block)
                    if metadata is None:
                        continue
                    local_rects = metadata.get("local_rects")
                    if not isinstance(local_rects, dict) or block not in local_rects:
                        continue
                    x_pos, y_pos, _, _ = local_rects[block]
                    local_rects[block] = (x_pos, y_pos, tree.widths[block], tree.heights[block])
                    touched_metadata[id(metadata)] = metadata
        for metadata in touched_metadata.values():
            local_rects = metadata.get("local_rects")
            if isinstance(local_rects, dict):
                all_blocks = sorted(local_rects.keys(), key=lambda b: (local_rects[b][0], local_rects[b][1]))
                for i in range(1, len(all_blocks)):
                    bi = all_blocks[i]
                    xi, yi, wi, hi = local_rects[bi]
                    new_xi = 0.0
                    for j in range(i):
                        bj = all_blocks[j]
                        xj, yj, wj, hj = local_rects[bj]
                        if yi < yj + hj - 1e-9 and yi + hi > yj + 1e-9:
                            new_xi = max(new_xi, xj + wj)
                    xi = new_xi
                    local_rects[bi] = (xi, yi, wi, hi)
                all_blocks.sort(key=lambda b: (local_rects[b][1], local_rects[b][0]))
                for i in range(1, len(all_blocks)):
                    bi = all_blocks[i]
                    xi, yi, wi, hi = local_rects[bi]
                    new_yi = 0.0
                    for j in range(i):
                        bj = all_blocks[j]
                        xj, yj, wj, hj = local_rects[bj]
                        if xi < xj + wj - 1e-9 and xi + wi > xj + 1e-9:
                            new_yi = max(new_yi, yj + hj)
                    yi = new_yi
                    local_rects[bi] = (xi, yi, wi, hi)
            self._refresh_cluster_metadata_bbox(metadata)
        self._refresh_cluster_metadata_from_tree(tree)

    def _refresh_cluster_metadata_from_tree(self, tree: BStarTree) -> None:
        touched: Dict[int, Dict[str, object]] = {}
        for block, metadata in self._cached_macro_cluster_metadata.items():
            if not (0 <= block < tree.n):
                continue
            metadata_id = id(metadata)
            local_rects = metadata.get("local_rects")
            if not isinstance(local_rects, dict) or block not in local_rects:
                continue
            x_pos, y_pos, _, _ = local_rects[block]
            local_rects[block] = (x_pos, y_pos, tree.widths[block], tree.heights[block])
            touched[metadata_id] = metadata
        for metadata in touched.values():
            self._refresh_cluster_metadata_bbox(metadata)

    def _refresh_cluster_metadata_bbox(self, metadata: Dict[str, object]) -> None:
        local_rects = metadata.get("local_rects")
        if not isinstance(local_rects, dict):
            return
        bbox_w = max((rect[0] + rect[2] for rect in local_rects.values()), default=0.0)
        bbox_h = max((rect[1] + rect[3] for rect in local_rects.values()), default=0.0)
        metadata["bbox_w"] = bbox_w
        metadata["bbox_h"] = bbox_h
        rebuilt = self._cluster_metadata_from_local(
            list(metadata.get("group", [])),
            local_rects,  # type: ignore[arg-type]
        )
        metadata["x_contour"] = rebuilt["x_contour"]
        metadata["y_contour"] = rebuilt["y_contour"]
        metadata["left_contour"] = rebuilt["left_contour"]
        metadata["bottom_contour"] = rebuilt["bottom_contour"]

    def _expand_cluster_satellites(self, positions: List[Rect]) -> List[Rect]:
        if not self._cached_cluster_satellites or not positions:
            return positions
        current = list(positions)
        for group in self._cached_macro_cluster_groups:
            if len(group) <= 1:
                continue
            anchor = group[0]
            if not (0 <= anchor < len(current)):
                continue
            metadata = self._cached_macro_cluster_metadata.get(anchor)
            if metadata is None:
                continue
            local_rects = metadata.get("local_rects")
            if not isinstance(local_rects, dict):
                continue
            anchor_local = local_rects.get(anchor)
            if anchor_local is None:
                continue
            origin_x = current[anchor][0] - anchor_local[0]
            origin_y = current[anchor][1] - anchor_local[1]
            for member in group:
                member_local = local_rects.get(member)
                if member_local is None:
                    continue
                lx, ly, w, h = member_local
                current[member] = (origin_x + lx, origin_y + ly, w, h)
        return current

    def _preprocess_left_bottom_boundary_shapes(
        self,
        tree: BStarTree,
        positions: List[Rect],
        boundary_codes: Dict[int, int],
        anchors: Dict[int, Tuple[float, float]],
        immovable_blocks: Optional[Iterable[int]] = None,
    ) -> List[Rect]:
        # One-shot preprocessing before compaction:
        # allow only LEFT/BOTTOM family shape morphing against preplaced blocks.
        if not positions:
            return positions
        preplaced = set(immovable_blocks or [])
        if not preplaced:
            return positions

        current = list(positions)
        tol = 1e-9
        for block, code in boundary_codes.items():
            if not (0 <= block < len(current)):
                continue
            if not (code & (LEFT | BOTTOM)):
                continue
            if block in self._cached_geometry_locked_blocks:
                continue
            if block in self._cached_mib_hard_shape_blocks:
                continue

            for fixed in preplaced:
                if fixed == block or not (0 <= fixed < len(current)):
                    continue

                candidate = self._try_shape_fix_before_compaction(
                    tree,
                    current,
                    block,
                    fixed,
                    boundary_codes,
                    anchors,
                    preplaced,
                )
                if candidate is not None:
                    current = candidate
        return current

    def _try_shape_fix_before_compaction(
        self,
        tree: BStarTree,
        positions: List[Rect],
        block: int,
        fixed_block: int,
        boundary_codes: Dict[int, int],
        anchors: Dict[int, Tuple[float, float]],
        preplaced_blocks: Iterable[int],
    ) -> Optional[List[Rect]]:
        # Bottom/left preprocessing shape-fix before compaction. It edits only
        # the candidate block's shape at the original packed coordinate; if no
        # candidate clears every preplaced block, the original tree shape is
        # restored and normal compaction handles the placement.
        if not (0 <= block < len(positions) and 0 <= fixed_block < len(positions)):
            return None
        if block in self._cached_geometry_locked_blocks:
            return None

        block_group = (
            self._cached_mib_group_by_block[block]
            if 0 <= block < len(self._cached_mib_group_by_block)
            else None
        )
        if block in self._cached_mib_hard_shape_blocks:
            return None

        tol = 1e-9
        bx, by, bw, bh = positions[block]
        px, py, pw, ph = positions[fixed_block]
        area = max(tree.areas[block], tol)

        # Only attempt shape fix if the two blocks actually overlap
        overlap_x = min(bx + bw, px + pw) - max(bx, px)
        overlap_y = min(by + bh, py + ph) - max(by, py)
        if not (overlap_x > tol and overlap_y > tol):
            return None

        original_dims = (
            [(member, tree.widths[member], tree.heights[member]) for member in block_group]
            if block_group is not None
            else [(block, tree.widths[block], tree.heights[block])]
        )
        candidates: List[Tuple[float, float, float]] = []

        def add_candidate(new_w: float, new_h: float) -> None:
            if new_w <= tol or new_h <= tol:
                return
            ratio = new_w / max(new_h, 1e-9)
            check_blocks = block_group if block_group is not None else [block]
            if not self._group_ratio_within_bounds(
                tree,
                check_blocks,
                ratio,
                boundary_codes,
                1.0 / 5.0,
                5.0,
            ):
                return
            score = abs(new_w - bw) + abs(new_h - bh)
            candidates.append((score, new_w, new_h))

        if px > bx + tol and bx + bw > px + tol:
            new_w = max(px - bx, tol)
            add_candidate(new_w, area / new_w)
        if py > by + tol and by + bh > py + tol:
            new_h = max(py - by, tol)
            add_candidate(area / new_h, new_h)
        candidates.sort(key=lambda item: item[0])

        preplaced = [
            fixed
            for fixed in preplaced_blocks
            if fixed != block and 0 <= fixed < len(positions)
        ]

        def overlaps_any_preplaced_at_original_pack(rects: List[Rect], test_blocks: Iterable[int]) -> bool:
            for test_block in test_blocks:
                if not (0 <= test_block < len(rects)):
                    continue
                tx, ty, tw, th = rects[test_block]
                for fixed in preplaced:
                    fx, fy, fw, fh = rects[fixed]
                    overlap_x = min(tx + tw, fx + fw) - max(tx, fx)
                    overlap_y = min(ty + th, fy + fh) - max(ty, fy)
                    if overlap_x > tol and overlap_y > tol:
                        return True
            return False

        for _, new_w, new_h in candidates:
            candidate_positions = list(positions)
            if block_group is not None:
                ratio = new_w / max(new_h, tol)
                tree.set_group_aspect(block_group, ratio)
                for member in block_group:
                    if 0 <= member < len(candidate_positions):
                        mx, my, _, _ = candidate_positions[member]
                        candidate_positions[member] = (mx, my, tree.widths[member], tree.heights[member])
                test_blocks = block_group
            else:
                tree.widths[block], tree.heights[block] = new_w, new_h
                candidate_positions[block] = (bx, by, new_w, new_h)
                test_blocks = [block]
            if not overlaps_any_preplaced_at_original_pack(candidate_positions, test_blocks):
                return candidate_positions

        for member, width, height in original_dims:
            tree.widths[member] = width
            tree.heights[member] = height
        return None

    def _dynamic_top_boundary_order(
        self,
        positions: List[Rect],
        boundary_codes: Dict[int, int],
        extra_top_blocks: Optional[Iterable[int]] = None,
    ) -> List[int]:
        extra_top = set(extra_top_blocks or [])
        top_blocks = {
            block for block, code in boundary_codes.items()
            if (code & TOP) and 0 <= block < len(positions)
        }
        top_blocks.update(block for block in extra_top if 0 <= block < len(positions))
        if not self._phase3_boundary_relax_active:
            top_boundary_order = self._ordered_top_chain_blocks(boundary_codes)
            top_tail = self._top_tail_block(boundary_codes)
            top_order = [idx for idx in top_boundary_order if idx in top_blocks]
            inserted_top = sorted(idx for idx in extra_top if idx in top_blocks and idx not in top_order)
            if top_tail is not None and top_tail in top_order:
                tail_index = top_order.index(top_tail)
                top_order[tail_index:tail_index] = inserted_top
            else:
                top_order.extend(inserted_top)
            top_order.extend(idx for idx in sorted(top_blocks) if idx not in top_order)
            return top_order
        return sorted(top_blocks, key=lambda idx: (positions[idx][0], positions[idx][1], idx))

    def _dynamic_right_boundary_order(
        self,
        positions: List[Rect],
        boundary_codes: Dict[int, int],
    ) -> List[int]:
        right_blocks = {
            block
            for block, code in boundary_codes.items()
            if (code & RIGHT) and 0 <= block < len(positions)
        }
        if not self._phase3_boundary_relax_active:
            ordered = [idx for idx in self._ordered_right_chain_blocks(boundary_codes) if idx in right_blocks]
            ordered.extend(
                idx
                for idx in sorted(right_blocks, key=lambda idx: (positions[idx][1], positions[idx][0], idx))
                if idx not in ordered
            )
            return ordered
        return sorted(right_blocks, key=lambda idx: (positions[idx][1], positions[idx][0], idx))

    def _boundary_snap_violation_count(
        self,
        positions: List[Rect],
        boundary_codes: Dict[int, int],
        tol: float = 1e-6,
    ) -> int:
        if not positions or not boundary_codes:
            return 0
        bbox_w = max((x + w for x, _, w, _ in positions), default=0.0)
        bbox_h = max((y + h for _, y, _, h in positions), default=0.0)
        violations = 0
        for block, code in boundary_codes.items():
            if not (0 <= block < len(positions)):
                continue
            x_pos, y_pos, width, height = positions[block]
            # Count each required boundary side separately. Corner blocks are
            # therefore checked on both of their constrained sides.
            if (code & LEFT) and abs(x_pos - 0.0) > tol:
                violations += 1
            if (code & BOTTOM) and abs(y_pos - 0.0) > tol:
                violations += 1
            if (code & TOP) and abs((y_pos + height) - bbox_h) > tol:
                violations += 1
            if (code & RIGHT) and abs((x_pos + width) - bbox_w) > tol:
                violations += 1
        return violations

    def _boundary_snap_error(
        self,
        positions: List[Rect],
        boundary_codes: Dict[int, int],
    ) -> float:
        # Continuous boundary mismatch measure used to decide whether another
        # snap iteration actually improved the current bbox matching.
        if not positions or not boundary_codes:
            return 0.0
        bbox_w = max((x + w for x, _, w, _ in positions), default=0.0)
        bbox_h = max((y + h for _, y, _, h in positions), default=0.0)
        error = 0.0
        for block, code in boundary_codes.items():
            if not (0 <= block < len(positions)):
                continue
            x_pos, y_pos, width, height = positions[block]
            # if code & LEFT:
            #     error += abs(x_pos - 0.0)
            # if code & BOTTOM:
            #     error += abs(y_pos - 0.0)
            if code & TOP:
                error += abs((y_pos + height) - bbox_h)
            if code & RIGHT:
                error += abs((x_pos + width) - bbox_w)
        return error

    def _repair_left_bottom_chain_by_compaction(
        self,
        tree: BStarTree,
        positions: List[Rect],
        boundary_codes: Dict[int, int],
        tol: float = 1e-6,
    ) -> bool:
        # If compaction leaves a LEFT-chain block away from the bottom-most
        # vertical order or a BOTTOM-chain block away from the left-most
        # horizontal order, rebuild that chain order and repack.
        if self._phase3_boundary_relax_active:
            return False
        if not positions:
            return False

        changed = False

        def reorder_chain(
            chain: List[int],
            key_fn,
            first_mask: int,
            last_mask: int,
        ) -> List[int]:
            if not chain:
                return []
            first = next(
                (block for block in chain if (boundary_codes.get(block, 0) & first_mask) == first_mask),
                None,
            )
            last = next(
                (block for block in chain if (boundary_codes.get(block, 0) & last_mask) == last_mask),
                None,
            )
            middle = [block for block in chain if block != first and block != last]
            middle.sort(key=key_fn)
            ordered: List[int] = []
            if first is not None:
                ordered.append(first)
            ordered.extend(middle if first is not None else sorted([b for b in chain if b != last], key=key_fn))
            if last is not None and last not in ordered:
                ordered.append(last)
            return ordered

        left_chain = [block for block in self._cached_ordered_left if 0 <= block < len(positions)]
        left_changed = False
        # Reorder the left chain if compaction pulled it away from x = 0.
        if left_chain and any(positions[block][0] > tol for block in left_chain):
            desired_left = reorder_chain(
                left_chain,
                key_fn=lambda block: (positions[block][1], positions[block][0], block),
                first_mask=LEFT | BOTTOM,
                last_mask=LEFT | TOP,
            )
            if desired_left and desired_left != left_chain:
                self._cached_ordered_left = desired_left
                changed = True
                left_changed = True

        bottom_chain = [block for block in self._cached_ordered_bottom if 0 <= block < len(positions)]
        bottom_changed = False
        # Reorder the bottom chain if compaction lifted it away from y = 0.
        if bottom_chain and any(positions[block][1] > tol for block in bottom_chain):
            desired_bottom = reorder_chain(
                bottom_chain,
                key_fn=lambda block: (positions[block][0], positions[block][1], block),
                first_mask=LEFT | BOTTOM,
                last_mask=RIGHT | BOTTOM,
            )
            if desired_bottom and desired_bottom != bottom_chain:
                self._cached_ordered_bottom = desired_bottom
                changed = True
                bottom_changed = True

        if left_changed and self._cached_left_tail is None:
            self._cached_top_root = self._cached_ordered_left[-1] if self._cached_ordered_left else None
            if self._cached_ordered_top:
                top_members = [block for block in self._cached_ordered_top if block != self._cached_top_root]
                if self._cached_top_root is not None:
                    self._cached_ordered_top = [self._cached_top_root] + top_members
        if bottom_changed and self._cached_bottom_tail is None:
            self._cached_right_root = self._cached_ordered_bottom[-1] if self._cached_ordered_bottom else None

        if changed: 
            self._apply_hard_boundary_tree_rules(tree, boundary_codes)
        return changed

    def _compact_positions(
        self,
        positions: List[Rect],
        boundary_codes: Dict[int, int],
        extra_top_blocks: Optional[Iterable[int]] = None,
        immovable_blocks: Optional[Iterable[int]] = None,
    ) -> List[Rect]:
        # Contour-based geometric compaction:
        # 1. base left/down push uses X-contour and Y-contour only
        # 2. preplaced blocks are not inserted into the contours; if a move
        #    overlaps a preplaced block, shift right/up around that obstacle
        # 3. top/right refinement then iterates against bbox without changing topology
        if not positions:
            return positions
        tol = 1e-9
        current = list(positions)
        extra_top = set(extra_top_blocks or [])
        immovable = set(immovable_blocks or [])
        top_blocks = {block for block, code in boundary_codes.items() if code & TOP}
        top_blocks.update(extra_top)
        right_blocks = {
            block
            for block, code in boundary_codes.items()
            if (code & RIGHT)
        }
        delayed_top = set(top_blocks)
        delayed_right = set(right_blocks)
        macro_groups = [
            [block for block in group if 0 <= block < len(current)]
            for group in self._cached_macro_cluster_groups
            if group
        ]
        macro_anchors = {group[0] for group in macro_groups if group}
        base_indices = [
            idx
            for idx in range(len(current))
            if (
                idx not in delayed_top
                and idx not in delayed_right
                and idx not in immovable
                and (idx not in self._cached_macro_cluster_non_anchors)
            )
        ]
        cluster_group_by_block: Dict[int, List[int]] = {}
        for group in macro_groups:
            for block in group:
                cluster_group_by_block[block] = group
        immovable_support: List[int] = sorted(immovable)

        if _HAS_CYTHON:
            def vertical_overlap(rect_a: Rect, rect_b: Rect) -> bool:
                return bstar_core.cy_vertical_overlap(rect_a, rect_b, tol)
        else:
            def vertical_overlap(rect_a: Rect, rect_b: Rect) -> bool:
                _, ay, _, ah = rect_a
                _, by, _, bh = rect_b
                return min(ay + ah, by + bh) - max(ay, by) > tol

        if _HAS_CYTHON:
            def horizontal_overlap(rect_a: Rect, rect_b: Rect) -> bool:
                return bstar_core.cy_horizontal_overlap(rect_a, rect_b, tol)
        else:
            def horizontal_overlap(rect_a: Rect, rect_b: Rect) -> bool:
                ax, _, aw, _ = rect_a
                bx, _, bw, _ = rect_b
                return min(ax + aw, bx + bw) - max(ax, bx) > tol

        def contour_overlap_preplaced_map(
            rects: List[Rect],
            indices: Iterable[int],
            support: Iterable[int],
        ) -> Dict[int, int]:
            mapping: Dict[int, int] = {}
            support_list = [idx for idx in support if 0 <= idx < len(rects)]
            for idx in indices:
                if idx in immovable or not self._is_boundary_constraint_block(boundary_codes, idx):
                    continue
                rect = rects[idx]
                for fixed in support_list:
                    if vertical_overlap(rect, rects[fixed]) and horizontal_overlap(rect, rects[fixed]):
                        mapping[idx] = fixed
                        break
            return mapping

        if _HAS_CYTHON:
            def query_x_contour(contour: List[Tuple[float, float]], y_start: float, y_end: float) -> float:
                return bstar_core.cy_query_x_contour(contour, y_start, y_end, tol)
        else:
            def query_x_contour(contour: List[Tuple[float, float]], y_start: float, y_end: float) -> float:
                max_x = 0.0
                for idx, (cy_end, cx_right) in enumerate(contour):
                    cy_start = contour[idx - 1][0] if idx > 0 else 0.0
                    if y_start < cy_end - tol and y_end > cy_start + tol:
                        max_x = max(max_x, cx_right)
                return max_x

        if _HAS_CYTHON:
            def update_x_contour(contour: List[Tuple[float, float]], y_start: float, y_end: float, x_right: float) -> List[Tuple[float, float]]:
                return bstar_core.cy_update_x_contour(contour, y_start, y_end, x_right, tol)
        else:
            def update_x_contour(contour: List[Tuple[float, float]], y_start: float, y_end: float, x_right: float) -> List[Tuple[float, float]]:
                new_contour: List[Tuple[float, float]] = []
                replaced_max = x_right
                for idx, (cy_end, cx_right) in enumerate(contour):
                    cy_start = contour[idx - 1][0] if idx > 0 else 0.0
                    if cy_end <= y_start + tol or cy_start >= y_end - tol:
                        new_contour.append((cy_end, cx_right))
                    else:
                        if cx_right > replaced_max:
                            replaced_max = cx_right
                        if cy_start < y_start - tol:
                            new_contour.append((y_start, cx_right))
                        if cy_end > y_end + tol:
                            new_contour.append((cy_end, cx_right))
                new_contour.append((y_end, replaced_max))
                new_contour.sort(key=lambda item: item[0])
                merged: List[Tuple[float, float]] = []
                for y_val, x_val in new_contour:
                    if merged and abs(merged[-1][1] - x_val) <= tol:
                        merged[-1] = (y_val, x_val)
                    else:
                        merged.append((y_val, x_val))
                return merged

        if _HAS_CYTHON:
            def query_y_contour(contour: List[Tuple[float, float]], x_start: float, x_end: float) -> float:
                return bstar_core.cy_query_y_contour(contour, x_start, x_end, tol)
        else:
            def query_y_contour(contour: List[Tuple[float, float]], x_start: float, x_end: float) -> float:
                max_y = 0.0
                for idx, (cx_end, cy_top) in enumerate(contour):
                    cx_start = contour[idx - 1][0] if idx > 0 else 0.0
                    if x_start < cx_end - tol and x_end > cx_start + tol:
                        max_y = max(max_y, cy_top)
                return max_y

        if _HAS_CYTHON:
            def update_y_contour(contour: List[Tuple[float, float]], x_start: float, x_end: float, y_top: float) -> List[Tuple[float, float]]:
                return bstar_core.cy_update_y_contour(contour, x_start, x_end, y_top, tol)
        else:
            def update_y_contour(contour: List[Tuple[float, float]], x_start: float, x_end: float, y_top: float) -> List[Tuple[float, float]]:
                new_contour: List[Tuple[float, float]] = []
                replaced_max = y_top
                for idx, (cx_end, cy_top) in enumerate(contour):
                    cx_start = contour[idx - 1][0] if idx > 0 else 0.0
                    if cx_end <= x_start + tol or cx_start >= x_end - tol:
                        new_contour.append((cx_end, cy_top))
                    else:
                        if cy_top > replaced_max:
                            replaced_max = cy_top
                        if cx_start < x_start - tol:
                            new_contour.append((x_start, cy_top))
                        if cx_end > x_end + tol:
                            new_contour.append((cx_end, cy_top))
                new_contour.append((x_end, replaced_max))
                new_contour.sort(key=lambda item: item[0])
                merged: List[Tuple[float, float]] = []
                for x_val, y_val in new_contour:
                    if merged and abs(merged[-1][1] - y_val) <= tol:
                        merged[-1] = (x_val, y_val)
                    else:
                        merged.append((x_val, y_val))
                return merged

        def compact_local_cluster(group: List[int], rects: List[Rect]) -> Dict[int, Rect]:
            # Dictionary to store compacted positions for each block
            local: Dict[int, Rect] = {}
            if not group:
                return local

            # Normalize coordinates so the cluster starts at (0,0)
            min_x = min(rects[block][0] for block in group)
            min_y = min(rects[block][1] for block in group)
            local_rects = {
                block: (
                    rects[block][0] - min_x,  # normalized x
                    rects[block][1] - min_y,  # normalized y
                    rects[block][2],          # width
                    rects[block][3]           # height
                )
                for block in group
            }

            max_local_rounds = max(3, min(len(group) * 3, 24))

            for _ in range(max_local_rounds):
                moved = False

                # ---- X-contour update ----
                # Sort primarily by x, then y, then index (tie-breaker)
                order_x = sorted(group, key=lambda idx: (local_rects[idx][0], local_rects[idx][1], idx))
                x_contour_local: List[Tuple[float, float]] = [(float('inf'), 0.0)]
                for block in order_x:
                    x_pos, y_pos, width, height = local_rects[block]
                    new_x = query_x_contour(x_contour_local, y_pos, y_pos + height)
                    # Check if block moved beyond tolerance
                    if abs(new_x - x_pos) > tol:
                        moved = True
                    # Update block position
                    local_rects[block] = (new_x, y_pos, width, height)
                    # Update contour with new block boundary
                    x_contour_local = update_x_contour(x_contour_local, y_pos, y_pos + height, new_x + width)

                # ---- Y-contour update ----
                # Sort primarily by y, then x, then index (tie-breaker)
                order_y = sorted(group, key=lambda idx: (local_rects[idx][1], local_rects[idx][0], idx))
                y_contour_local: List[Tuple[float, float]] = [(float('inf'), 0.0)]
                for block in order_y:
                    x_pos, y_pos, width, height = local_rects[block]
                    new_y = query_y_contour(y_contour_local, x_pos, x_pos + width)
                    # Check if block moved beyond tolerance
                    if abs(new_y - y_pos) > tol:
                        moved = True
                    # Update block position
                    local_rects[block] = (x_pos, new_y, width, height)
                    # Update contour with new block boundary
                    y_contour_local = update_y_contour(y_contour_local, x_pos, x_pos + width, new_y + height)

                # Stop if no block moved in this round
                if not moved:
                    break

            return local_rects


        def build_cluster_metadata(group: List[int], rects: List[Rect]) -> Dict[str, object]:
            anchor = group[0]
            cached = self._cached_macro_cluster_metadata.get(anchor)
            if cached is None:
                local_rects = compact_local_cluster(group, rects)
                cached = self._cluster_metadata_from_local(group, local_rects)
            local_rects = cached["local_rects"]  # type: ignore[assignment]
            anchor_local = local_rects[anchor]  # type: ignore[index]
            origin_x = rects[anchor][0] - anchor_local[0]
            origin_y = rects[anchor][1] - anchor_local[1]
            metadata = dict(cached)
            metadata["origin"] = (origin_x, origin_y)
            return metadata

        cluster_metadata_by_block: Dict[int, Dict[str, object]] = {}
        for group in macro_groups:
            metadata = build_cluster_metadata(group, current)
            local_rects = metadata["local_rects"]  # type: ignore[assignment]
            anchor = group[0]
            anchor_local = local_rects[anchor]  # type: ignore[index]
            origin_x = current[anchor][0] - anchor_local[0]
            origin_y = current[anchor][1] - anchor_local[1]
            metadata["origin"] = (origin_x, origin_y)
            for block in group:
                lx, ly, width, height = local_rects[block]  # type: ignore[index]
                current[block] = (origin_x + lx, origin_y + ly, width, height)
                cluster_metadata_by_block[block] = metadata

        def shift_cluster_right_of_preplaced(
            origin_x: float,
            origin_y: float,
            local_rects: Dict[int, Rect],
            rects: List[Rect],
            support: List[int],
        ) -> float:
            current_x = origin_x
            while True:
                shifted_x = current_x
                for fixed in support:
                    if not (0 <= fixed < len(rects)):
                        continue
                    fx, fy, fw, fh = rects[fixed]
                    for _, (lx, ly, width, height) in local_rects.items():
                        gx = current_x + lx
                        gy = origin_y + ly
                        overlap_x = min(gx + width, fx + fw) - max(gx, fx)
                        overlap_y = min(gy + height, fy + fh) - max(gy, fy)
                        if overlap_x > tol and overlap_y > tol:
                            shifted_x = max(shifted_x, fx + fw - lx)
                if shifted_x <= current_x + tol:
                    return current_x
                current_x = shifted_x

        def shift_cluster_above_preplaced(
            origin_x: float,
            origin_y: float,
            local_rects: Dict[int, Rect],
            rects: List[Rect],
            support: List[int],
        ) -> float:
            current_y = origin_y
            while True:
                shifted_y = current_y
                for fixed in support:
                    if not (0 <= fixed < len(rects)):
                        continue
                    fx, fy, fw, fh = rects[fixed]
                    for _, (lx, ly, width, height) in local_rects.items():
                        gx = origin_x + lx
                        gy = current_y + ly
                        overlap_x = min(gx + width, fx + fw) - max(gx, fx)
                        overlap_y = min(gy + height, fy + fh) - max(gy, fy)
                        if overlap_x > tol and overlap_y > tol:
                            shifted_y = max(shifted_y, fy + fh - ly)
                if shifted_y <= current_y + tol:
                    return current_y
                current_y = shifted_y

        if _HAS_CYTHON:
            def shift_right_of_preplaced(
                x_pos: float, y_pos: float, width: float, height: float,
                rects: List[Rect], support: List[int],
            ) -> float:
                return bstar_core.cy_shift_right_of_preplaced(x_pos, y_pos, width, height, rects, support, tol)
        else:
            def shift_right_of_preplaced(
                x_pos: float,
                y_pos: float,
                width: float,
                height: float,
                rects: List[Rect],
                support: List[int],
            ) -> float:
                current_x = x_pos
                while True:
                    shifted_x = current_x
                    for other in support:
                        if not (0 <= other < len(rects)):
                            continue
                        other_rect = rects[other]
                        if not vertical_overlap((current_x, y_pos, width, height), other_rect):
                            continue
                        overlap_x = min(current_x + width, other_rect[0] + other_rect[2]) - max(current_x, other_rect[0])
                        if overlap_x > tol:
                            shifted_x = max(shifted_x, other_rect[0] + other_rect[2])
                    if shifted_x <= current_x + tol:
                        return current_x
                    current_x = shifted_x

        if _HAS_CYTHON:
            def shift_above_preplaced(
                x_pos: float, y_pos: float, width: float, height: float,
                rects: List[Rect], support: List[int],
            ) -> float:
                return bstar_core.cy_shift_above_preplaced(x_pos, y_pos, width, height, rects, support, tol)
        else:
            def shift_above_preplaced(
                x_pos: float,
                y_pos: float,
                width: float,
                height: float,
                rects: List[Rect],
                support: List[int],
            ) -> float:
                current_y = y_pos
                while True:
                    shifted_y = current_y
                    for other in support:
                        if not (0 <= other < len(rects)):
                            continue
                        other_rect = rects[other]
                        if not horizontal_overlap((x_pos, current_y, width, height), other_rect):
                            continue
                        overlap_y = min(current_y + height, other_rect[1] + other_rect[3]) - max(current_y, other_rect[1])
                        if overlap_y > tol:
                            shifted_y = max(shifted_y, other_rect[1] + other_rect[3])
                    if shifted_y <= current_y + tol:
                        return current_y
                    current_y = shifted_y

        def legalize_x_against_immovable(
            rects: List[Rect],
            x_pos: float,
            y_pos: float,
            width: float,
            height: float,
        ) -> float:
            return shift_right_of_preplaced(x_pos, y_pos, width, height, rects, immovable_support)

        def legalize_y_against_immovable(
            rects: List[Rect],
            x_pos: float,
            y_pos: float,
            width: float,
            height: float,
        ) -> float:
            return shift_above_preplaced(x_pos, y_pos, width, height, rects, immovable_support)

        def contour_left_push(
            rects: List[Rect],
            indices: List[int],
            support: Optional[List[int]] = None,
            include_support_in_contour: bool = False,
            contour_support: Optional[List[int]] = None,
            skip_left_boundary_obstacle_shift: bool = False,
            skip_preplaced_shift: bool = False,
        ) -> List[Rect]:
            new_rects = list(rects)
            x_contour: List[Tuple[float, float]] = [(0.0, 0.0)]
            active_support = support or []
            contour_blocks = contour_support if contour_support is not None else active_support
            overlap_fixed_map = contour_overlap_preplaced_map(rects, indices, contour_blocks)

            def boundary_priority(idx: int) -> int:
                return 0 if self._is_boundary_constraint_block(boundary_codes, idx) else 1

            def x_order_key(idx: int) -> Tuple[float, float, int, int, float, float, int]:
                if idx in immovable:
                    x_val, y_val, _, _ = rects[idx]
                    return (x_val, y_val, 0, 0, x_val, y_val, idx)
                fixed = overlap_fixed_map.get(idx)
                if fixed is not None:
                    fx, fy, _, _ = rects[fixed]
                    x_val, y_val, _, _ = rects[idx]
                    return (fx, fy, 1, boundary_priority(idx), x_val, y_val, idx)
                x_val, y_val, _, _ = rects[idx]
                return (x_val, boundary_priority(idx), y_val, 2, x_val, y_val, idx)

            order = sorted(indices, key=x_order_key)
            if include_support_in_contour:
                all_order = sorted(
                    list(indices) + [idx for idx in contour_blocks if idx not in indices],
                    key=x_order_key,
                )
            else:
                all_order = order
            contour_support_set = set(contour_blocks) if include_support_in_contour else set()
            processed_cluster_members: set[int] = set()
            for idx in all_order:
                if idx in processed_cluster_members:
                    continue
                _, y_pos, width, height = rects[idx]
                if idx in contour_support_set:
                    new_x = rects[idx][0]
                    x_contour = update_x_contour(x_contour, y_pos, y_pos + height, new_x + width)
                    continue

                group = cluster_group_by_block.get(idx)
                if group is None:
                    bc = boundary_codes.get(idx, 0)
                    if bc & LEFT:
                        new_x = rects[idx][0]  # preserve x for left boundary
                    else:
                        new_x = query_x_contour(x_contour, y_pos, y_pos + height)
                    if skip_preplaced_shift:
                        new_rects[idx] = (new_x, y_pos, width, height)
                        x_contour = update_x_contour(x_contour, y_pos, y_pos + height, new_x + width)
                    else:
                        block_support = [] if (skip_left_boundary_obstacle_shift and (bc & LEFT)) else active_support
                        if bc & BOTTOM and not (bc & LEFT):
                            # Bottom boundary: shift RIGHT
                            new_x = shift_right_of_preplaced(new_x, y_pos, width, height, new_rects, block_support)
                            new_rects[idx] = (new_x, y_pos, width, height)
                            x_contour = update_x_contour(x_contour, y_pos, y_pos + height, new_x + width)
                        else:
                            # Left boundary / other: shift ABOVE
                            new_y = shift_above_preplaced(new_x, y_pos, width, height, new_rects, block_support)
                            new_rects[idx] = (new_x, new_y, width, height)
                            x_contour = update_x_contour(x_contour, new_y, new_y + height, new_x + width)
                    continue

                metadata = cluster_metadata_by_block[idx]
                local_rects = metadata["local_rects"]  # type: ignore[assignment]
                bbox_w = float(metadata["bbox_w"])  # type: ignore[arg-type]
                bbox_h = float(metadata["bbox_h"])  # type: ignore[arg-type]
                anchor_local = local_rects[group[0]]
                origin_y = new_rects[group[0]][1] - anchor_local[1]
                left_contour = metadata["left_contour"]
                need_x = 0.0
                gi = 0
                for si in range(1, len(left_contour)):
                    seg_y_start = origin_y + left_contour[si-1][0]
                    seg_y_end = origin_y + left_contour[si][0]
                    seg_min_x = left_contour[si][1]
                    if seg_min_x >= float('inf'):
                        continue
                    while gi < len(x_contour) and x_contour[gi][0] <= seg_y_start + tol:
                        gi += 1
                    seg_max_x = 0.0
                    j = gi - 1 if gi > 0 else 0
                    while j < len(x_contour):
                        cy_end = x_contour[j][0]
                        cy_start = x_contour[j-1][0] if j > 0 else 0.0
                        if seg_y_end <= cy_start + tol:
                            break
                        if seg_y_start < cy_end - tol and seg_y_end > cy_start + tol:
                            seg_max_x = max(seg_max_x, x_contour[j][1])
                        j += 1
                    need = seg_max_x - seg_min_x
                    if need > need_x:
                        need_x = need
                origin_x = need_x
                if any(boundary_codes.get(block, 0) & LEFT for block in group if 0 <= block < len(boundary_codes)):
                    if origin_x > tol:
                        origin_x = rects[group[0]][0] - anchor_local[0]
                if not skip_preplaced_shift:
                    origin_x = shift_cluster_right_of_preplaced(origin_x, origin_y, local_rects, new_rects, active_support)

                for block in sorted(group, key=lambda b: local_rects[b][0]):
                    lx, ly, w_local, h_local = local_rects[block]
                    gx = origin_x + lx
                    gy = origin_y + ly
                    new_rects[block] = (gx, gy, w_local, h_local)
                    x_contour = update_x_contour(x_contour, gy, gy + h_local, gx + w_local)
                    processed_cluster_members.add(block)

            return new_rects

        def contour_down_push(
            rects: List[Rect],
            indices: List[int],
            support: Optional[List[int]] = None,
            include_support_in_contour: bool = False,
            contour_support: Optional[List[int]] = None,
            skip_preplaced_shift: bool = False,
        ) -> List[Rect]:
            new_rects = list(rects)
            y_contour: List[Tuple[float, float]] = [(0.0, 0.0)]
            active_support = support or []
            contour_blocks = contour_support if contour_support is not None else active_support
            overlap_fixed_map = contour_overlap_preplaced_map(rects, indices, contour_blocks)

            def boundary_priority(idx: int) -> int:
                return 0 if self._is_boundary_constraint_block(boundary_codes, idx) else 1

            def y_order_key(idx: int) -> Tuple[float, float, int, int, float, float, int]:
                if idx in immovable:
                    x_val, y_val, _, _ = rects[idx]
                    return (y_val, x_val, 0, 0, y_val, x_val, idx)
                fixed = overlap_fixed_map.get(idx)
                if fixed is not None:
                    fx, fy, _, _ = rects[fixed]
                    x_val, y_val, _, _ = rects[idx]
                    return (fy, fx, 1, boundary_priority(idx), y_val, x_val, idx)
                x_val, y_val, _, _ = rects[idx]
                return (y_val, boundary_priority(idx), x_val, 2, y_val, x_val, idx)

            order = sorted(indices, key=y_order_key)
            if include_support_in_contour:
                all_order = sorted(
                    list(indices) + [idx for idx in contour_blocks if idx not in indices],
                    key=y_order_key,
                )
            else:
                all_order = order
            contour_support_set = set(contour_blocks) if include_support_in_contour else set()
            processed_cluster_members: set[int] = set()
            for idx in all_order:
                if idx in processed_cluster_members:
                    continue
                x_pos, _, width, height = rects[idx]
                if idx in contour_support_set:
                    new_y = rects[idx][1]
                    y_contour = update_y_contour(y_contour, x_pos, x_pos + width, new_y + height)
                    continue

                group = cluster_group_by_block.get(idx)
                if group is None:
                    bc = boundary_codes.get(idx, 0)
                    if bc & BOTTOM:
                        new_y = rects[idx][1]  # preserve y for bottom boundary
                    else:
                        new_y = query_y_contour(y_contour, x_pos, x_pos + width)
                    if skip_preplaced_shift:
                        new_rects[idx] = (x_pos, new_y, width, height)
                        y_contour = update_y_contour(y_contour, x_pos, x_pos + width, new_y + height)
                    else:
                        if bc & BOTTOM and not (bc & LEFT):
                            # Bottom boundary: shift RIGHT
                            new_x = shift_right_of_preplaced(x_pos, new_y, width, height, new_rects, active_support)
                            new_rects[idx] = (new_x, new_y, width, height)
                            y_contour = update_y_contour(y_contour, new_x, new_x + width, new_y + height)
                        else:
                            # Left boundary / other: shift ABOVE
                            new_y = shift_above_preplaced(x_pos, new_y, width, height, new_rects, active_support)
                            new_rects[idx] = (x_pos, new_y, width, height)
                            y_contour = update_y_contour(y_contour, x_pos, x_pos + width, new_y + height)
                    continue

                metadata = cluster_metadata_by_block[idx]
                local_rects = metadata["local_rects"]  # type: ignore[assignment]
                bbox_w = float(metadata["bbox_w"])  # type: ignore[arg-type]
                bbox_h = float(metadata["bbox_h"])  # type: ignore[arg-type]
                anchor_local = local_rects[group[0]]
                origin_x = new_rects[group[0]][0] - anchor_local[0]
                bottom_contour = metadata["bottom_contour"]
                need_y = 0.0
                gi = 0
                for si in range(1, len(bottom_contour)):
                    seg_x_start = origin_x + bottom_contour[si-1][0]
                    seg_x_end = origin_x + bottom_contour[si][0]
                    seg_min_y = bottom_contour[si][1]
                    if seg_min_y >= float('inf'):
                        continue
                    while gi < len(y_contour) and y_contour[gi][0] <= seg_x_start + tol:
                        gi += 1
                    seg_max_y = 0.0
                    j = gi - 1 if gi > 0 else 0
                    while j < len(y_contour):
                        cx_end = y_contour[j][0]
                        cx_start = y_contour[j-1][0] if j > 0 else 0.0
                        if seg_x_end <= cx_start + tol:
                            break
                        if seg_x_start < cx_end - tol and seg_x_end > cx_start + tol:
                            seg_max_y = max(seg_max_y, y_contour[j][1])
                        j += 1
                    need = seg_max_y - seg_min_y
                    if need > need_y:
                        need_y = need
                origin_y = need_y
                if not skip_preplaced_shift:
                    origin_y = shift_cluster_above_preplaced(origin_x, origin_y, local_rects, new_rects, active_support)

                for block in sorted(group, key=lambda b: local_rects[b][1]):
                    lx, ly, w_local, h_local = local_rects[block]
                    gx = origin_x + lx
                    gy = origin_y + ly
                    new_rects[block] = (gx, gy, w_local, h_local)
                    y_contour = update_y_contour(y_contour, gx, gx + w_local, gy + h_local)
                    processed_cluster_members.add(block)

            return new_rects

        if base_indices:
            max_base_rounds = max(1, min(max(1, len(base_indices)), 8))

            # Phase 1: no preplaced overlap check, compact until convergence
            current = contour_left_push(
                current, base_indices, support=immovable_support,
                skip_left_boundary_obstacle_shift=True, skip_preplaced_shift=True,
            )
            current = contour_down_push(
                current, base_indices, support=immovable_support, skip_preplaced_shift=True,
            )
            for _ in range(max_base_rounds):
                candidate = contour_left_push(
                    current, base_indices, support=immovable_support, skip_preplaced_shift=True,
                )
                candidate = contour_down_push(
                    candidate, base_indices, support=immovable_support, skip_preplaced_shift=True,
                )
                moved = False
                for idx, (cx, cy, _, _) in enumerate(current):
                    nx, ny, _, _ = candidate[idx]
                    if abs(nx - cx) > tol or abs(ny - cy) > tol:
                        moved = True
                        break
                current = candidate
                if not moved:
                    break

            # Phase 2: with preplaced overlap check, compact until convergence
            current = contour_left_push(
                current, base_indices, support=immovable_support,
                skip_left_boundary_obstacle_shift=True, skip_preplaced_shift=False,
            )
            current = contour_down_push(
                current, base_indices, support=immovable_support, skip_preplaced_shift=False,
            )
            for _ in range(max_base_rounds):
                candidate = contour_left_push(
                    current, base_indices, support=immovable_support, skip_preplaced_shift=False,
                )
                candidate = contour_down_push(
                    candidate, base_indices, support=immovable_support, skip_preplaced_shift=False,
                )
                moved = False
                for idx, (cx, cy, _, _) in enumerate(current):
                    nx, ny, _, _ = candidate[idx]
                    if abs(nx - cx) > tol or abs(ny - cy) > tol:
                        moved = True
                        break
                current = candidate
                if not moved:
                    break

        if _HAS_CYTHON:
            def bbox_dims(rects: List[Rect]) -> Tuple[float, float, float]:
                return bstar_core.cy_bbox_dims(rects)
        else:
            def bbox_dims(rects: List[Rect]) -> Tuple[float, float, float]:
                if not rects:
                    return 0.0, 0.0, 0.0
                bbox_w = max((x + w for x, _, w, _ in rects), default=0.0)
                bbox_h = max((y + h for _, y, _, h in rects), default=0.0)
                return bbox_w * bbox_h, bbox_w, bbox_h

        if _HAS_CYTHON:
            def overlaps_any(rects: List[Rect], block: int) -> bool:
                return bstar_core.cy_overlaps_any(rects, block, tol)
        else:
            def overlaps_any(rects: List[Rect], block: int) -> bool:
                x1, y1, w1, h1 = rects[block]
                for other, (x2, y2, w2, h2) in enumerate(rects):
                    if other == block:
                        continue
                    overlap_x = min(x1 + w1, x2 + w2) - max(x1, x2)
                    overlap_y = min(y1 + h1, y2 + h2) - max(y1, y2)
                    if overlap_x > tol and overlap_y > tol:
                        return True
                return False

        def push_top_down(rects: List[Rect]) -> List[Rect]:
            new_rects = list(rects)
            placed = {idx for idx in range(len(new_rects)) if idx not in delayed_top}
            top_order = self._dynamic_top_boundary_order(new_rects, boundary_codes, extra_top)
            for idx in top_order:
                if idx in immovable:
                    continue
                x_pos, _, width, height = new_rects[idx]
                new_y = 0.0
                for other in placed:
                    other_rect = new_rects[other]
                    if horizontal_overlap((x_pos, 0.0, width, height), other_rect):
                        new_y = max(new_y, other_rect[1] + other_rect[3])
                new_y = legalize_y_against_immovable(new_rects, x_pos, new_y, width, height)
                new_rects[idx] = (x_pos, new_y, width, height)
                placed.add(idx)
            return new_rects

        def align_top_to_bbox(rects: List[Rect]) -> List[Rect]:
            new_rects = list(rects)
            _, _, bbox_h = bbox_dims(new_rects)
            for idx in self._dynamic_top_boundary_order(new_rects, boundary_codes, extra_top):
                if idx in immovable:
                    continue
                x_pos, current_y, width, height = new_rects[idx]
                target_y = bbox_h - height
                if abs((current_y + height) - bbox_h) <= tol:
                    continue
                candidate_y = target_y
                guard = 0
                while guard < len(new_rects) + 2:
                    guard += 1
                    candidate = list(new_rects)
                    candidate[idx] = (x_pos, candidate_y, width, height)
                    if not overlaps_any(candidate, idx):
                        new_rects = candidate
                        break
                    shifted_y = candidate_y
                    for other, other_rect in enumerate(new_rects):
                        if other == idx:
                            continue
                        if not horizontal_overlap((x_pos, candidate_y, width, height), other_rect):
                            continue
                        overlap_y = min(candidate_y + height, other_rect[1] + other_rect[3]) - max(candidate_y, other_rect[1])
                        if overlap_y > tol:
                            shifted_y = min(shifted_y, other_rect[1] - height)
                    if shifted_y >= candidate_y - tol:
                        break
                    candidate_y = shifted_y
            return new_rects

        def push_right_left(rects: List[Rect]) -> List[Rect]:
            new_rects = list(rects)
            if not delayed_right:
                return new_rects
            right_order = [idx for idx in self._dynamic_right_boundary_order(new_rects, boundary_codes) if idx in delayed_right]
            fixed_support = [idx for idx in range(len(new_rects)) if idx not in delayed_right and idx not in immovable]
            for idx in right_order:
                if idx in immovable:
                    continue
                _, y_pos, width, height = new_rects[idx]
                new_x = 0.0
                probe = new_rects[idx]
                for other in fixed_support:
                    other_rect = new_rects[other]
                    if vertical_overlap(probe, other_rect):
                        new_x = max(new_x, other_rect[0] + other_rect[2])
                new_x = legalize_x_against_immovable(new_rects, new_x, y_pos, width, height)
                new_rects[idx] = (new_x, y_pos, width, height)
                fixed_support.append(idx)
            return new_rects

        def align_right_to_bbox(rects: List[Rect]) -> List[Rect]:
            new_rects = list(rects)
            _, bbox_w, _ = bbox_dims(new_rects)
            for idx in self._dynamic_right_boundary_order(new_rects, boundary_codes):
                if idx not in delayed_right or idx in immovable:
                    continue
                current_x, y_pos, width, height = new_rects[idx]
                target_x = bbox_w - width
                if abs((current_x + width) - bbox_w) <= tol:
                    continue
                candidate_x = target_x
                guard = 0
                while guard < len(new_rects) + 2:
                    guard += 1
                    candidate = list(new_rects)
                    candidate[idx] = (candidate_x, y_pos, width, height)
                    if not overlaps_any(candidate, idx):
                        new_rects = candidate
                        break
                    shifted_x = candidate_x
                    for other, other_rect in enumerate(new_rects):
                        if other == idx:
                            continue
                        if not vertical_overlap((candidate_x, y_pos, width, height), other_rect):
                            continue
                        overlap_x = min(candidate_x + width, other_rect[0] + other_rect[2]) - max(candidate_x, other_rect[0])
                        if overlap_x > tol:
                            shifted_x = min(shifted_x, other_rect[0] - width)
                    if shifted_x >= candidate_x - tol:
                        break
                    candidate_x = shifted_x
            return new_rects

        current_area, current_w, current_h = bbox_dims(current)
        max_refine_rounds = max(2, min(len(current), 8))
        for i in range(max_refine_rounds):
            candidate = push_top_down(current)
            candidate = align_top_to_bbox(candidate)
            candidate = push_right_left(candidate)
            candidate = align_right_to_bbox(candidate)
            candidate_area, candidate_w, candidate_h = bbox_dims(candidate)
            if candidate_area + tol < current_area:
                improved = True
            elif abs(candidate_area - current_area) <= tol:
                improved = candidate_w + tol < current_w or candidate_h + tol < current_h
            else:
                improved = False
            if not improved:
                if i != 0:
                    break
            else:
                current = candidate
                current_area, current_w, current_h = candidate_area, candidate_w, candidate_h

        # Snap LEFT-boundary blocks to x=0.
        left_blocks = [
            idx for idx, code in boundary_codes.items()
            if (code & LEFT) and idx < len(current) and idx not in immovable
        ]
        for idx in left_blocks:
            x, y, w, h = current[idx]
            if x <= tol:
                continue
            clear = True
            for other in range(len(current)):
                if other == idx:
                    continue
                ox, oy, ow, oh = current[other]
                if (0.0 < ox + ow and w > ox and
                        y < oy + oh and y + h > oy):
                    clear = False
                    break
            if clear:
                current[idx] = (0.0, y, w, h)

        rightmost_block: Optional[int] = None
        topmost_block: Optional[int] = None
        rightmost_edge = float("-inf")
        topmost_edge = float("-inf")
        for block, _ in enumerate(current):
            if self._is_boundary_constraint_block(boundary_codes, block):
                continue
            candidate = self._cluster_bbox_edges_for_outline_pick(block, current)
            if candidate is None:
                continue
            block_right, block_top, candidate_block = candidate
            if block_right > rightmost_edge:
                rightmost_edge = block_right
                rightmost_block = candidate_block
            if block_top > topmost_edge:
                topmost_edge = block_top
                topmost_block = candidate_block
        self._cached_compaction_rightmost_block = rightmost_block
        self._cached_compaction_topmost_block = topmost_block
        self._cached_compaction_rightmost_edge = 0.0 if rightmost_block is None else rightmost_edge
        self._cached_compaction_topmost_edge = 0.0 if topmost_block is None else topmost_edge
        return current

    def _resolve_preplaced_conflicts(
        self,
        tree: BStarTree,
        positions: List[Rect],
        boundary_codes: Dict[int, int],
        preplaced_anchors: Dict[int, Tuple[float, float]],
        immovable_blocks: Iterable[int],
        geometry_locked_blocks: Set[int],
        top_chain_locked_blocks: Optional[Iterable[int]] = None,
        discard_on_non_boundary_failure: bool = False,
    ) -> Optional[List[Rect]]:
        """
        Resolve conflicts where packed positions overlap with preplaced obstacles.
        Two strategies in order of preference:
        1) shape-fix to push the block off the obstacle.
        2) re-insert the block as a child of the preplaced block it overlaps.
        Returns the updated positions list, or None on failure.
        """
        return positions

    def _preferred_preplaced_child_side(self, code: int) -> Optional[bool]:
        if (code & BOTTOM) and not (code & TOP):
            return True
        if (code & LEFT) and not (code & RIGHT):
            return False
        return None

    def _random_reinsert_block(self, tree: BStarTree, block: int) -> bool:
        targets = [candidate for candidate in range(tree.n) if candidate != block]
        if not targets:
            return False
        random.shuffle(targets)
        for target in targets:
            sides = [True, False]
            random.shuffle(sides)
            for as_left in sides:
                if tree._insert_leaf_node(block, target, as_left):
                    return True
        return False

    # ------------------------------------------------------------------
    #  Ratio bounds & sampling helpers
    # ------------------------------------------------------------------
    def _ratio_bounds_for_code(self, code: int) -> Tuple[float, float]:
        # Boundary-constrained blocks keep their own aspect-ratio windows.
        if (code & LEFT) and (code & BOTTOM):
            return 1.0, 1.0
        if (code & LEFT) and not (code & TOP) and self._phase2_boundary_ratio_mode_state == "left":
            return 1.0, 2.0
        if (code & BOTTOM) and not (code & RIGHT) and self._phase2_boundary_ratio_mode_state == "bottom":
            return 1.0 / 2.0, 1.0
        if code & LEFT:
            return 1.0 / 2, 1.0 / 1.5
        if code & BOTTOM:
            return 1.5, 2
        if code & TOP:
            return 1.5, 2
        return 1.0 / 3.0, 3.0

    def _ratio_bounds_for_block(
        self,
        block: int,
        boundary_codes: Dict[int, int],
    ) -> Tuple[float, float]:
        if self._phase3_can_free_boundary(boundary_codes, block):
            return 1.0 / 3.0, 3.0
        return self._ratio_bounds_for_code(boundary_codes.get(block, 0))

    def _effective_ratio_bounds(
        self,
        blocks: Iterable[int],
        boundary_codes: Dict[int, int],
        fallback_low: float,
        fallback_high: float,
    ) -> Tuple[float, float]:
        # Boundary windows override the generic unconstrained window when they conflict.
        low = fallback_low
        high = fallback_high
        has_boundary = False
        for block in blocks:
            code = boundary_codes.get(block, 0)
            if code == 0:
                continue
            has_boundary = True
            block_low, block_high = self._ratio_bounds_for_block(block, boundary_codes)
            low = max(low, block_low)
            high = min(high, block_high)
        if has_boundary:
            if low > high:
                return 1.0, 1.0
            return low, high
        return fallback_low, fallback_high

    def _dims_within_ratio_bounds(
        self,
        width: float,
        height: float,
        block: int,
        boundary_codes: Dict[int, int],
        fallback_low: float = 1.0 / 3.0,
        fallback_high: float = 3.0,
        use_fallback_only: bool = False,
    ) -> bool:
        if width <= 0.0 or height <= 0.0:
            return False
        if use_fallback_only:
            low, high = fallback_low, fallback_high
        else:
            low, high = self._effective_ratio_bounds([block], boundary_codes, fallback_low, fallback_high)
        ratio = width / max(height, 1e-9)
        return low - 1e-9 <= ratio <= high + 1e-9

    def _group_ratio_within_bounds(
        self,
        tree: BStarTree,
        blocks: Iterable[int],
        ratio: float,
        boundary_codes: Dict[int, int],
        fallback_low: float,
        fallback_high: float,
    ) -> bool:
        if ratio <= 0.0:
            return False
        for block in blocks:
            if block in self._cached_geometry_locked_blocks or block in self._cached_mib_hard_shape_blocks:
                return False
            if not (0 <= block < tree.n):
                return False
            area = max(tree.areas[block], 1e-9)
            width = math.sqrt(area * ratio)
            height = math.sqrt(area / ratio)
            if not self._dims_within_ratio_bounds(
                width,
                height,
                block,
                boundary_codes,
                fallback_low,
                fallback_high,
                use_fallback_only=True,
            ):
                return False
        return True

    def _sample_group_ratio(
        self,
        blocks: Iterable[int],
        boundary_codes: Dict[int, int],
        low: float,
        high: float,
    ) -> float:
        # Sample one aspect ratio in log-space so thin/wide values are explored more evenly.
        low, high = self._effective_ratio_bounds(blocks, boundary_codes, low, high)
        if abs(low - high) <= 1e-9:
            return low
        return math.exp(random.uniform(math.log(low), math.log(high)))

    def _find_group_for_block(self, block: int, mib_groups: Dict[int, List[int]]) -> Optional[List[int]]:
        # Return the MIB group that contains this block, if any.
        for group in mib_groups.values():
            if block in group:
                return group
        return None

    def _find_cluster_for_block(self, block: int, cluster_groups: Dict[int, List[int]]) -> Optional[List[int]]:
        # Return the cluster that contains this block, if any.
        for group in cluster_groups.values():
            if block in group:
                return group
        return None

    # ------------------------------------------------------------------
    #  Cluster constraint handling (caching, hierarchy, local SA)
    # ------------------------------------------------------------------
    def _build_cluster_cache(
        self,
        cluster_groups: Dict[int, List[int]],
        boundary_codes: Dict[int, int],
        preplaced_blocks: Iterable[int],
        block_count: int,
    ) -> None:
        self._cached_cluster_blocks = set()
        self._cached_macro_cluster_groups = []
        self._cached_macro_cluster_by_block = [None] * block_count
        self._cached_macro_cluster_anchor_by_block = [None] * block_count
        self._cached_macro_cluster_non_anchors = set()
        self._cached_cluster_satellites = set()
        self._cached_cluster_representative_by_block = [None] * block_count

        preplaced = set(preplaced_blocks)

        for group in cluster_groups.values():
            members = [block for block in group if 0 <= block < block_count]
            if not members:
                continue
            self._cached_cluster_blocks.update(members)
            filtered = [
                block for block in members
                if block not in preplaced and boundary_codes.get(block, 0) == 0 
            ]
            if filtered:
                anchor = random.choice(filtered)
                cached_group = [anchor] + [block for block in filtered if block != anchor]
                self._cached_macro_cluster_groups.append(cached_group)
                self._cached_macro_cluster_non_anchors.update(cached_group[1:])
                # Satellites: all cluster members except the anchor/representative
                for block in cached_group[1:]:
                    self._cached_cluster_satellites.add(block)
                for block in cached_group:
                    self._cached_macro_cluster_by_block[block] = cached_group
                    self._cached_macro_cluster_anchor_by_block[block] = anchor
                    if block == anchor:
                        self._cached_cluster_representative_by_block[block] = block
                    else:
                        self._cached_cluster_representative_by_block[block] = anchor

    def _apply_cluster_hierarchy_rules(
        self,
        tree: BStarTree,
        cluster_groups: Dict[int, List[int]],
        boundary_codes: Dict[int, int],
        preplaced_blocks: Iterable[int],
    ) -> None:
        for group in self._cached_macro_cluster_groups:
            members = [block for block in group if 0 <= block < tree.n and tree.active[block]]
            if len(members) <= 1:
                continue
            anchor = members[0]
            attached = [anchor]
            remaining = members[1:]
            for block in remaining:
                target = random.choice(attached)
                sides = [True, False]
                random.shuffle(sides)
                moved = False
                for as_left in sides:
                    if tree._insert_leaf_node(block, target, as_left):
                        moved = True
                        break
                if moved:
                    attached.append(block)

    def _build_cluster_local_layout_cache(
        self,
        tree: BStarTree,
        cluster_groups: Dict[int, List[int]],
        mib_groups: Dict[int, List[int]],
        b2b_edges: List[Tuple[int, int, float]],
    ) -> None:
        # Small pre-SA for eligible clusters. It optimizes local area plus
        # intra-cluster HPWL, then caches local contours so global SA can treat
        # the cluster as a rigid macro-block during compaction.
        self._cached_macro_cluster_metadata = {}
        self._cached_mib_cluster_ratio_groups = []
        mib_by_block: Dict[int, List[int]] = {}
        mib_ratio_by_key: Dict[Tuple[int, ...], float] = {}
        for blocks in mib_groups.values():
            group_list = list(blocks)
            for block in group_list:
                mib_by_block[block] = group_list
            hard_anchor = next((block for block in group_list if block in self._cached_mib_hard_shape_blocks), None)
            if hard_anchor is not None and 0 <= hard_anchor < tree.n:
                mib_ratio_by_key[tuple(sorted(group_list))] = tree.widths[hard_anchor] / max(tree.heights[hard_anchor], 1e-9)

        # Phase 1: Apply all MIB dimension updates to the tree (sequential).
        for group in self._cached_macro_cluster_groups:
            for mib_group in mib_groups.values():
                cluster_mib_members = [block for block in mib_group if block in group]
                if not cluster_mib_members:
                    continue
                key = tuple(sorted(mib_group))
                ratio = mib_ratio_by_key.get(key)
                if ratio is None:
                    continue
                for member in cluster_mib_members:
                    area = max(tree.areas[member], 1e-9)
                    tree.widths[member] = math.sqrt(area * ratio)
                    tree.heights[member] = math.sqrt(area / ratio)

        # Phase 2: Collect per-group data for local SA.
        groups_data: List[Tuple[List[int], List[Tuple[int, int, float]]]] = []
        for group in self._cached_macro_cluster_groups:
            local_edges = self._cluster_internal_b2b_edges(group, b2b_edges)
            groups_data.append((group, local_edges))

        # Phase 3: Run local SA for each cluster (parallel or sequential).
        local_rects_list = []
        _cluster_parallel_ok = False
        if self.use_parallel_cluster and not IS_WINDOWS:
            try:
                import pickle
                pickle.dumps(_cluster_local_sa_worker)
                _cluster_parallel_ok = True
            except Exception:
                pass
            if _cluster_parallel_ok:
                try:
                    import concurrent.futures
                    import multiprocessing as _mp
                    n_workers = min(len(groups_data), _mp.cpu_count())
                    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                        args_list = [
                            (self, tree, group, local_edges)
                            for group, local_edges in groups_data
                        ]
                        local_rects_list = list(
                            executor.map(_cluster_local_sa_worker, args_list)
                        )
                except Exception:
                    local_rects_list = []
        if not local_rects_list:
            local_rects_list = [
                self._local_cluster_sa(tree, group, local_edges)
                for group, local_edges in groups_data
            ]

        # Phase 4: Sequential post-processing (MIB, re-compact, metadata).
        for (group, _), local_rects in zip(groups_data, local_rects_list):
            had_mib_dim_change = False
            for mib_group in mib_groups.values():
                cluster_mib_members = [block for block in mib_group if block in local_rects]
                if not cluster_mib_members:
                    continue
                key = tuple(sorted(mib_group))
                ratio = mib_ratio_by_key.get(key)
                if ratio is None:
                    anchor = cluster_mib_members[0]
                    _, _, width, height = local_rects[anchor]
                    if height <= 1e-9:
                        continue
                    ratio = width / height
                    mib_ratio_by_key[key] = ratio
                for member in cluster_mib_members:
                    area = max(tree.areas[member], 1e-9)
                    x_pos, y_pos, _, _ = local_rects[member]
                    local_rects[member] = (
                        x_pos,
                        y_pos,
                        math.sqrt(area * ratio),
                        math.sqrt(area / ratio),
                    )
                    had_mib_dim_change = True
            if had_mib_dim_change:
                all_blocks = sorted(local_rects.keys(), key=lambda b: (local_rects[b][0], local_rects[b][1]))
                for i in range(1, len(all_blocks)):
                    bi = all_blocks[i]
                    xi, yi, wi, hi = local_rects[bi]
                    new_xi = 0.0
                    for j in range(i):
                        bj = all_blocks[j]
                        xj, yj, wj, hj = local_rects[bj]
                        if yi < yj + hj - 1e-9 and yi + hi > yj + 1e-9:
                            new_xi = max(new_xi, xj + wj)
                    xi = new_xi
                    local_rects[bi] = (xi, yi, wi, hi)
                all_blocks.sort(key=lambda b: (local_rects[b][1], local_rects[b][0]))
                for i in range(1, len(all_blocks)):
                    bi = all_blocks[i]
                    xi, yi, wi, hi = local_rects[bi]
                    new_yi = 0.0
                    for j in range(i):
                        bj = all_blocks[j]
                        xj, yj, wj, hj = local_rects[bj]
                        if xi < xj + wj - 1e-9 and xi + wi > xj + 1e-9:
                            new_yi = max(new_yi, yj + hj)
                    yi = new_yi
                    local_rects[bi] = (xi, yi, wi, hi)
            metadata = self._cluster_metadata_from_local(group, local_rects)
            anchor = group[0]
            self._cached_macro_cluster_metadata[anchor] = metadata
            for block in group:
                self._cached_macro_cluster_metadata[block] = metadata
                lx, ly, width, height = local_rects[block]
                tree.widths[block] = width
                tree.heights[block] = height
                mib_group = mib_by_block.get(block)
                if mib_group and height > 1e-9:
                    key = tuple(sorted(mib_group))
                    ratio = mib_ratio_by_key.get(key, width / height)
                    if (
                        not any(set(existing) == set(mib_group) for existing, _ in self._cached_mib_cluster_ratio_groups)
                    ):
                        self._cached_mib_cluster_ratio_groups.append((mib_group, ratio))
                    self._cached_mib_hard_shape_blocks.update(mib_group)
        self._enforce_cluster_mib_geometry(tree)

    def _cluster_internal_b2b_edges(
        self,
        group: List[int],
        b2b_edges: List[Tuple[int, int, float]],
    ) -> List[Tuple[int, int, float]]:
        members = set(group)
        return [
            (block_i, block_j, weight)
            for block_i, block_j, weight in b2b_edges
            if block_i in members and block_j in members
        ]

    def _local_cluster_sa(
        self,
        tree: BStarTree,
        group: List[int],
        local_edges: List[Tuple[int, int, float]],
    ) -> Dict[int, Rect]:
        def compact_local(rects: Dict[int, Rect]) -> Dict[int, Rect]:
            local = dict(rects)
            tol = 1e-9
            for _ in range(4):
                moved = False
                placed: List[int] = []
                for block in sorted(group, key=lambda idx: (local[idx][0], local[idx][1], idx)):
                    _, y_pos, width, height = local[block]
                    new_x = 0.0
                    for other in placed:
                        ox, oy, ow, oh = local[other]
                        if min(y_pos + height, oy + oh) - max(y_pos, oy) > tol:
                            new_x = max(new_x, ox + ow)
                    if abs(new_x - local[block][0]) > tol:
                        moved = True
                    local[block] = (new_x, y_pos, width, height)
                    placed.append(block)
                placed = []
                for block in sorted(group, key=lambda idx: (local[idx][1], local[idx][0], idx)):
                    x_pos, _, width, height = local[block]
                    new_y = 0.0
                    for other in placed:
                        ox, oy, ow, oh = local[other]
                        if min(x_pos + width, ox + ow) - max(x_pos, ox) > tol:
                            new_y = max(new_y, oy + oh)
                    if abs(new_y - local[block][1]) > tol:
                        moved = True
                    local[block] = (x_pos, new_y, width, height)
                    placed.append(block)
                if not moved:
                    break
            return local

        def local_area(rects: Dict[int, Rect]) -> float:
            max_x = max((x + w for x, _, w, _ in rects.values()), default=0.0)
            max_y = max((y + h for _, y, _, h in rects.values()), default=0.0)
            return max_x * max_y

        def local_hpwl(rects: Dict[int, Rect]) -> float:
            hpwl = 0.0
            for block_i, block_j, weight in local_edges:
                if block_i not in rects or block_j not in rects:
                    continue
                x1, y1, w1, h1 = rects[block_i]
                x2, y2, w2, h2 = rects[block_j]
                cx1 = x1 + w1 / 2.0
                cy1 = y1 + h1 / 2.0
                cx2 = x2 + w2 / 2.0
                cy2 = y2 + h2 / 2.0
                hpwl += weight * (abs(cx2 - cx1) + abs(cy2 - cy1))
            return hpwl

        total_area = sum(tree.areas[block] for block in group)
        cluster_scale = math.sqrt(max(total_area, 1.0))

        def local_outline_penalty(rects: Dict[int, Rect]) -> float:
            bbox_w = max((x + w for x, _, w, _ in rects.values()), default=0.0)
            bbox_h = max((y + h for _, y, _, h in rects.values()), default=0.0)
            if bbox_w <= 0.0 or bbox_h <= 0.0:
                return 0.0
            o_w, o_h = self._cached_terminal_outline
            aspect = o_w / max(o_h, 1e-9)
            target_w = math.sqrt(max(total_area, 1e-9) * aspect)
            target_h = max(total_area, 1e-9) / target_w
            return abs(bbox_w - target_w) + abs(bbox_h - target_h)

        def local_cost(rects: Dict[int, Rect]) -> float:
            area_cost = self.cluster_pre_sa_area_weight * local_area(rects)
            hpwl_cost = self.cluster_pre_sa_hpwl_weight * cluster_scale * local_hpwl(rects)
            outline_cost = self.cluster_pre_sa_outline_weight * cluster_scale * local_outline_penalty(rects)
            return area_cost + hpwl_cost + outline_cost

        def shape_trial(rects: Dict[int, Rect], block: int) -> Dict[int, Rect]:
            width = rects[block][2]
            height = rects[block][3]
            area = max(tree.areas[block], 1e-9)
            ratio = width / max(height, 1e-9)
            ratio *= math.exp(random.uniform(-0.35, 0.35))
            ratio = min(3.0, max(1.0 / 3.0, ratio))
            trial = dict(rects)
            trial[block] = (rects[block][0], rects[block][1], math.sqrt(area * ratio), math.sqrt(area / ratio))
            return trial

        def swap_trial(rects: Dict[int, Rect], block_a: int, block_b: int) -> Dict[int, Rect]:
            trial = dict(rects)
            ax, ay, aw, ah = trial[block_a]
            bx, by, bw, bh = trial[block_b]
            trial[block_a] = (bx, by, aw, ah)
            trial[block_b] = (ax, ay, bw, bh)
            return trial

        def insert_trial(rects: Dict[int, Rect], block: int, target: int) -> Dict[int, Rect]:
            trial = dict(rects)
            tx, ty, tw, th = trial[target]
            _, _, width, height = trial[block]
            if random.random() < 0.5:
                trial[block] = (tx + tw, ty, width, height)
            else:
                trial[block] = (tx, ty + th, width, height)
            return trial

        def pick_square_outline_candidate(rects: Dict[int, Rect], candidates: List[int]) -> Optional[int]:
            if not candidates:
                return None
            if random.random() >= 0.5:
                return random.choice(candidates)
            candidate_set = set(candidates)
            target_side = math.sqrt(max(total_area, 1e-9))
            rightmost = max(
                candidate_set,
                key=lambda block: (rects[block][0] + rects[block][2], rects[block][1], block),
            )
            topmost = max(
                candidate_set,
                key=lambda block: (rects[block][1] + rects[block][3], rects[block][0], block),
            )
            bbox_w = max((x + w for x, _, w, _ in rects.values()), default=0.0)
            bbox_h = max((y + h for _, y, _, h in rects.values()), default=0.0)
            right_excess = max(0.0, bbox_w - target_side)
            top_excess = max(0.0, bbox_h - target_side)
            denom = right_excess + top_excess
            outline_sum = target_side * 2.0
            if denom <= 1e-12 or outline_sum <= 1e-12:
                return rightmost if random.random() < 0.5 else topmost
            choose_right_prob = (target_side / outline_sum) * (right_excess / denom)
            return rightmost if random.random() < choose_right_prob else topmost

        local_rects = {
            block: (0.0, 0.0, tree.widths[block], tree.heights[block])
            for block in group
        }
        best = compact_local(local_rects)
        best_area = local_area(best)
        best_cost = local_cost(best)
        optimize_outline = self.cluster_pre_sa_outline_weight > 0.0
        if total_area <= 0.0 or (best_area <= total_area * 1.05 and not local_edges and not optimize_outline):
            return best

        current = dict(best)
        current_area = best_area
        current_cost = best_cost

        # Phase 1: multi-start high-temp exploration for outline matching
        high_temp_runs = 4
        best_high_temp = dict(best)
        best_high_temp_cost = best_cost
        base_temp = max(1.0, current_area * 0.08)
        ht_shape_movable = [
            block for block in group
            if block not in self._cached_mib_hard_shape_blocks and block not in self._cached_geometry_locked_blocks
        ]
        ht_topology_movable = list(group)
        ht_can_shape = bool(ht_shape_movable)
        ht_can_topology = len(ht_topology_movable) >= 2
        if ht_can_shape or ht_can_topology:
            for run in range(high_temp_runs):
                cur = dict(best)
                cur_cost = best_cost
                temp = base_temp
                no_imp = 0
                max_ht_steps = max(16, len(group) * 6)
                ht_patience = max(6, len(group) * 4)
                for step in range(max_ht_steps):
                    if no_imp >= ht_patience:
                        break
                    if ht_can_shape and ht_can_topology:
                        roll = random.random()
                        if roll < 0.4:
                            move = "shape"
                        elif roll < 0.7:
                            move = "insert"
                        else:
                            move = "swap"
                    elif ht_can_shape:
                        move = "shape"
                    elif ht_can_topology:
                        move = "insert" if random.random() < 0.5 else "swap"
                    else:
                        break
                    if move == "shape":
                        block = pick_square_outline_candidate(cur, ht_shape_movable)
                        if block is None:
                            continue
                        trial = shape_trial(cur, block)
                    elif move == "swap":
                        block_a = pick_square_outline_candidate(cur, ht_topology_movable)
                        if block_a is None:
                            continue
                        others = [b for b in ht_topology_movable if b != block_a]
                        if not others:
                            continue
                        trial = swap_trial(cur, block_a, random.choice(others))
                    else:
                        block = pick_square_outline_candidate(cur, ht_topology_movable)
                        if block is None:
                            continue
                        targets = [t for t in ht_topology_movable if t != block]
                        if not targets:
                            continue
                        trial = insert_trial(cur, block, random.choice(targets))
                    trial = compact_local(trial)
                    trial_cost = local_cost(trial)
                    delta = trial_cost - cur_cost
                    if delta <= 0.0 or random.random() < math.exp(-delta / max(temp, 1e-9)):
                        cur = trial
                        cur_cost = trial_cost
                        if trial_cost < cur_cost:
                            no_imp = 0
                        else:
                            no_imp += 1
                        if trial_cost < best_high_temp_cost:
                            best_high_temp = trial
                            best_high_temp_cost = trial_cost
                            no_imp = 0
                    else:
                        no_imp += 1
                    temp *= 0.92
                if best_high_temp_cost < best_cost:
                    best = best_high_temp
                    best_cost = best_high_temp_cost

        # Phase 2: low temperature refinement on best result
        current = dict(best)
        current_cost = best_cost
        temp = max(0.5, current_area * 0.008)
        no_improve_steps = 0
        shape_movable = [
            block for block in group
            if block not in self._cached_mib_hard_shape_blocks and block not in self._cached_geometry_locked_blocks
        ]
        topology_movable = list(group)
        can_shape = bool(shape_movable)
        can_topology = len(topology_movable) >= 2
        if not can_shape and not can_topology:
            return best
        max_steps = max(32, len(group) * 18)
        patience_limit = max(12, len(group) * 8)
        for step in range(max_steps):
            if no_improve_steps >= patience_limit:
                break
            if can_shape and can_topology:
                move_roll = random.random()
                if move_roll < 0.4:
                    move = "shape"
                elif move_roll < 0.7:
                    move = "insert"
                else:
                    move = "swap"
            elif can_shape:
                move = "shape"
            elif can_topology:
                move = "insert" if random.random() < 0.5 else "swap"
            else:
                break
            if move == "shape":
                block = pick_square_outline_candidate(current, shape_movable)
                if block is None:
                    continue
                trial = shape_trial(current, block)
            elif move == "swap":
                block_a = pick_square_outline_candidate(current, topology_movable)
                if block_a is None:
                    continue
                other_candidates = [block for block in topology_movable if block != block_a]
                if not other_candidates:
                    continue
                block_b = random.choice(other_candidates)
                trial = swap_trial(current, block_a, block_b)
            else:
                block = pick_square_outline_candidate(current, topology_movable)
                if block is None:
                    continue
                targets = [target for target in topology_movable if target != block]
                if not targets:
                    continue
                trial = insert_trial(current, block, random.choice(targets))
            trial = compact_local(trial)
            trial_area = local_area(trial)
            trial_cost = local_cost(trial)
            delta = trial_cost - current_cost
            if delta <= 0.0 or random.random() < math.exp(-delta / max(temp, 1e-9)):
                current = trial
                current_area = trial_area
                current_cost = trial_cost
                if trial_cost < best_cost:
                    best = trial
                    best_area = trial_area
                    best_cost = trial_cost
                    no_improve_steps = 0
                    if best_area <= total_area * 1.03 and not local_edges and not optimize_outline:
                        break
                else:
                    no_improve_steps += 1
            else:
                no_improve_steps += 1
            temp *= 0.97
        return best

    def _cluster_metadata_from_local(self, group: List[int], local_rects: Dict[int, Rect]) -> Dict[str, object]:
        # Dictionary to store compacted positions for each block
        local: Dict[int, Rect] = {}
        if not group:
            return local
        y_edges = sorted({0.0} | {y for _, y, _, _ in local_rects.values()} |
                        {y + h for _, y, _, h in local_rects.values()})
        x_edges = sorted({0.0} | {x for x, _, _, _ in local_rects.values()} |
                        {x + w for x, _, w, _ in local_rects.values()})

        bbox_w = max((x + w for x, _, w, _ in local_rects.values()), default=0.0)
        bbox_h = max((y + h for _, y, _, h in local_rects.values()), default=0.0)

        # Initialize with sentinel
        x_contour: List[Tuple[float, float]] = [(0.0, 0.0)]
        left_contour: List[Tuple[float, float]] = [(0.0, 0.0)]
        prev_y = 0.0
        for y_end in y_edges[1:]:
            max_x = 0.0
            min_x = float('inf')
            for x_pos, y_pos, width, height in local_rects.values():
                if prev_y < y_pos + height and y_end > y_pos:
                    max_x = max(max_x, x_pos + width)
                    min_x = min(min_x, x_pos)
            x_contour.append((y_end, max_x))
            left_contour.append((y_end, min_x))
            prev_y = y_end
        # Add final cap at bbox_h.
        x_contour.append((bbox_h, bbox_w))
        left_contour.append((bbox_h, 0.0))

        y_contour: List[Tuple[float, float]] = [(0.0, 0.0)]
        bottom_contour: List[Tuple[float, float]] = [(0.0, 0.0)]
        prev_x = 0.0
        for x_end in x_edges[1:]:
            max_y = 0.0
            min_y = float('inf')
            for x_pos, y_pos, width, height in local_rects.values():
                if prev_x < x_pos + width and x_end > x_pos:
                    max_y = max(max_y, y_pos + height)
                    min_y = min(min_y, y_pos)
            y_contour.append((x_end, max_y))
            bottom_contour.append((x_end, min_y))
            prev_x = x_end
        # Add final cap at bbox_w.
        y_contour.append((bbox_w, bbox_h))
        bottom_contour.append((bbox_w, 0.0))


        return {
            "group": list(group),
            "local_rects": dict(local_rects),
            "x_contour": x_contour,
            "y_contour": y_contour,
            "left_contour": left_contour,
            "bottom_contour": bottom_contour,
            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
            "origin": (0.0, 0.0),
        }



    def _cluster_adjacent_target_ok(
        self,
        tree: BStarTree,
        target: int,
        same_cluster_blocks: Iterable[int],
    ) -> bool:
        """
        A cluster-constrained move target is valid only when:
        - The target node's parent or either child already touches the same cluster
        - AND none of those neighbors are boundary-constrained or preplaced blocks
        """
        same_cluster = set(same_cluster_blocks)
        if not same_cluster:
            return True

        # Collect parent and children
        neighbors = [tree.parent[target], tree.left[target], tree.right[target]]

        for neighbor in neighbors:
            if neighbor == -1:
                continue
            # Must be in the same cluster
            if neighbor in same_cluster:
                # Reject if neighbor is boundary-constrained or preplaced
                if self._cached_boundary_masks[neighbor] != 0 or (neighbor in self._cached_immovable_blocks):
                    return False
                return True

        return False

    def _touches_macro_cluster(self, blocks: Iterable[int]) -> bool:
        return any(
            0 <= block < len(self._cached_macro_cluster_anchor_by_block)
            and self._cached_macro_cluster_anchor_by_block[block] is not None
            for block in blocks
        )

    def _cluster_bbox_edges_for_outline_pick(
        self,
        block: int,
        positions: List[Rect],
    ) -> Optional[Tuple[float, float, int]]:
        if not (0 <= block < len(positions)):
            return None
        anchor = (
            self._cached_macro_cluster_anchor_by_block[block]
            if 0 <= block < len(self._cached_macro_cluster_anchor_by_block)
            else None
        )
        if anchor is not None:
            group = (
                self._cached_macro_cluster_by_block[anchor]
                if 0 <= anchor < len(self._cached_macro_cluster_by_block)
                else None
            )
            members = [member for member in (group or [block]) if 0 <= member < len(positions)]
            if not members:
                return None
            right_edge = max(positions[member][0] + positions[member][2] for member in members)
            top_edge = max(positions[member][1] + positions[member][3] for member in members)
            return right_edge, top_edge, anchor
        x_pos, y_pos, width, height = positions[block]
        return x_pos + width, y_pos + height, block


    def _group_has_hard_mib_shape(self, blocks: Iterable[int]) -> bool:
        # If a MIB group contains a fixed-shape/preplaced member, no member of
        # that group is allowed to rotate or perturb aspect ratio independently.
        return any(block in self._cached_mib_hard_shape_blocks for block in blocks)

    # ------------------------------------------------------------------
    #  SA move generator (aspect / rotate / swap / insert)
    # ------------------------------------------------------------------
    def _apply_regular_move(
        self,
        tree: BStarTree,
        areas: Sequence[float],
        locked_blocks: Iterable[int],
        mib_groups: Dict[int, List[int]],
        cluster_groups: Dict[int, List[int]],
        boundary_codes: Dict[int, int],
        geometry_locked_blocks: Optional[Iterable[int]] = None,
        freeze_boundary_topology: bool = False,
        top_chain_locked_blocks: Optional[Iterable[int]] = None,
        stage2_boundary_focus: bool = False,
        stage2_mode: bool = False,
        phase3_free_boundary_blocks: bool = False,
        stage2_hpwl_mode: bool = False,
        preferred_block: Optional[int] = None,
    ) -> List[int]:
        # SA move generator.
        # Returns the blocks touched by the move so overlap-retry can focus on them.
        locked = locked_blocks if isinstance(locked_blocks, set) else set(locked_blocks)
        top_chain_locked = (
            top_chain_locked_blocks
            if isinstance(top_chain_locked_blocks, set)
            else set(top_chain_locked_blocks or [])
        )
        boundary_masks = self._cached_boundary_masks
        geometry_locked_flags = self._cached_geometry_locked_flags
        movable_blocks = list(self._cached_movable_blocks)
        if not movable_blocks:
            return []

        topology_ban_boundary_involvement = stage2_mode or stage2_hpwl_mode or phase3_free_boundary_blocks
        enforce_cluster_target_rule = False

        def insert_unconstrained(node: int) -> None:
            targets: List[int] = []
            cluster = self._find_cluster_for_block(node, cluster_groups) if enforce_cluster_target_rule else None
            same_cluster: List[int] = []
            if cluster is not None:
                for member in cluster:
                    if member != node:
                        same_cluster.append(member)
            for candidate in self._cached_movable_blocks:
                if candidate == node or candidate in locked:
                    continue
                if topology_ban_boundary_involvement and boundary_masks[candidate] != 0:
                    continue
                if boundary_masks[candidate] & TOP or boundary_masks[candidate] & RIGHT:
                    continue
                if same_cluster and not self._cluster_adjacent_target_ok(tree, candidate, same_cluster):
                    continue
                targets.append(candidate)
            if topology_ban_boundary_involvement:
                pass
            if not targets:
                return
            random.shuffle(targets)
            for target in targets:
                sides = [True, False]
                random.shuffle(sides)
                for as_left in sides:
                    if tree._insert_leaf_node(node, target, as_left):
                        return

        if stage2_boundary_focus:
            focus_blocks: List[int] = []
            for candidate in movable_blocks:
                if boundary_masks[candidate] != 0 or candidate in top_chain_locked:
                    focus_blocks.append(candidate)
            if focus_blocks:
                movable_blocks = focus_blocks

        # Separate boundary and free blocks; 80% chance pick free, 20% boundary.
        boundary_movable = [b for b in movable_blocks if boundary_masks[b] != 0]
        free_movable = [b for b in movable_blocks if boundary_masks[b] == 0]

        move_type = random.random()
        if preferred_block is not None and preferred_block in movable_blocks:
            block = preferred_block
        elif free_movable and (not boundary_movable or random.random() < 0.80):
            block = random.choice(free_movable)
        elif boundary_movable:
            block = random.choice(boundary_movable)
        else:
            block = random.choice(movable_blocks)
        free_blocks: List[int] = []
        for candidate in movable_blocks:
            if boundary_masks[candidate] == 0:
                free_blocks.append(candidate)
        structural_mask = boundary_masks[block]
        left_tail = self._left_tail_block(boundary_codes)
        bottom_tail = self._bottom_tail_block(boundary_codes)
        top_tail = self._top_tail_block(boundary_codes)

        block_group = self._find_group_for_block(block, mib_groups)
        cluster_members = self._find_cluster_for_block(block, cluster_groups)
        same_cluster_neighbors: List[int] = []
        if cluster_members is not None:
            for member in cluster_members:
                if member != block:
                    same_cluster_neighbors.append(member)
        group_masks: List[int] = []
        if block_group is not None:
            for member in block_group:
                group_masks.append(boundary_masks[member])
        else:
            group_masks.append(structural_mask)
        touches_boundary = any(mask != 0 for mask in group_masks)
        relaxed_boundary_blocks: List[int] = []
        for member in (block_group if block_group is not None else [block]):
            if phase3_free_boundary_blocks and self._phase3_can_free_boundary(boundary_codes, member):
                relaxed_boundary_blocks.append(member)
        has_relaxed_boundary = bool(relaxed_boundary_blocks)

        if stage2_mode:
            if move_type < 0.05:
                move_type = 0.0
            elif move_type < 0.10:
                move_type = 0.24
            elif move_type < 0.20:
                move_type = 0.34
            else:
                move_type = 0.80
        elif stage2_hpwl_mode:
            # 1/3 shape (aspect+rotate), 1/3 swap, 1/3 insertion
            if move_type < 1/3:
                move_type = random.random() * 0.33
            elif move_type < 2/3:
                move_type = 0.34 + random.random() * 0.32
            else:
                move_type = 0.66 + random.random() * 0.34
        elif phase3_free_boundary_blocks:
            # 1/3 shape (aspect+rotate), 1/3 swap, 1/3 insertion
            if move_type < 1/3:
                move_type = random.random() * 0.33
            elif move_type < 2/3:
                move_type = 0.34 + random.random() * 0.32
            else:
                move_type = 0.66 + random.random() * 0.34

        # Boundary blocks: only shape/rotate moves before boundary finalize.
        if touches_boundary and not has_relaxed_boundary and not self._boundary_finalize_active:
            move_type = min(move_type, 0.32)

        if move_type < 0.23:
            # Aspect-ratio perturbation.
            blocks = block_group if block_group is not None else [block]
            if self._touches_macro_cluster(blocks):
                return list(blocks)
            for member in blocks:
                if 0 <= member < len(geometry_locked_flags) and geometry_locked_flags[member]:
                    return list(blocks)
            if self._group_has_hard_mib_shape(blocks):
                return list(blocks)
            low, high = self._default_sampling_window(blocks)
            ratio = self._sample_group_ratio(blocks, boundary_codes, low, high)
            tree.set_group_aspect(blocks, ratio)
            return list(blocks)
        elif move_type < 0.33:
            # "Rotate" branch currently implemented as another ratio resample.
            blocks = block_group if block_group is not None else [block]
            if self._touches_macro_cluster(blocks):
                return list(blocks)
            for member in blocks:
                if 0 <= member < len(geometry_locked_flags) and geometry_locked_flags[member]:
                    return list(blocks)
            if self._group_has_hard_mib_shape(blocks):
                return list(blocks)
            low, high = self._default_sampling_window(blocks)
            ratio = self._sample_group_ratio(blocks, boundary_codes, low, high)
            tree.set_group_aspect(blocks, ratio)
            return list(blocks)
        elif move_type < 0.66:
            # Topology swap.
            if topology_ban_boundary_involvement and touches_boundary:
                return list(block_group if block_group is not None else [block])
            if stage2_mode and touches_boundary and not has_relaxed_boundary:
                return list(block_group if block_group is not None else [block])
            if freeze_boundary_topology and block_group is not None and any(
                self._structural_boundary_mask(boundary_codes, member) != 0
                and not (phase3_free_boundary_blocks and self._phase3_can_free_boundary(boundary_codes, member))
                for member in block_group
            ):
                return list(block_group)
            if freeze_boundary_topology and structural_mask != 0 and not has_relaxed_boundary:
                return [block]
            if block in top_chain_locked and not has_relaxed_boundary and not topology_ban_boundary_involvement:
                swap_candidates = []
                for candidate in movable_blocks:
                    if candidate == block:
                        continue
                    if stage2_mode:
                        if boundary_masks[candidate] == 0:
                            swap_candidates.append(candidate)
                    else:
                        if boundary_masks[candidate] == 0 or (boundary_masks[candidate] & TOP):
                            swap_candidates.append(candidate)
            elif structural_mask == 0:
                swap_candidates = []
                for candidate in free_blocks:
                    if candidate != block:
                        swap_candidates.append(candidate)
            else:
                if has_relaxed_boundary:
                    swap_candidates = []
                    for candidate in movable_blocks:
                        if candidate != block:
                            swap_candidates.append(candidate)
                else:
                    swap_candidates = []
                    for candidate in movable_blocks:
                        if candidate != block and boundary_masks[candidate] == structural_mask:
                            swap_candidates.append(candidate)
                    if structural_mask == LEFT:
                        filtered_candidates = []
                        for candidate in swap_candidates:
                            if (candidate == left_tail) == (block == left_tail):
                                filtered_candidates.append(candidate)
                        swap_candidates = filtered_candidates
                    elif structural_mask == BOTTOM:
                        filtered_candidates = []
                        for candidate in swap_candidates:
                            if (candidate == bottom_tail) == (block == bottom_tail):
                                filtered_candidates.append(candidate)
                        swap_candidates = filtered_candidates
                    elif structural_mask == TOP:
                        filtered_candidates = []
                        for candidate in swap_candidates:
                            if (candidate == top_tail) == (block == top_tail):
                                filtered_candidates.append(candidate)
                        swap_candidates = filtered_candidates
            if topology_ban_boundary_involvement:
                filtered_candidates: List[int] = []
                for candidate in swap_candidates:
                    if boundary_masks[candidate] == 0:
                        filtered_candidates.append(candidate)
                swap_candidates = filtered_candidates
            if enforce_cluster_target_rule and same_cluster_neighbors:
                swap_candidates = [
                    candidate
                    for candidate in swap_candidates
                    if self._cluster_adjacent_target_ok(tree, candidate, same_cluster_neighbors)
                ]
            # Don't swap TOP/RIGHT boundary blocks with preplaced TOP/RIGHT blocks.
            if structural_mask in (TOP, RIGHT) and self._cached_immovable_blocks:
                swap_candidates = [c for c in swap_candidates if c not in self._cached_immovable_blocks]
            if swap_candidates:
                other = random.choice(swap_candidates)
                if tree.swap_nodes_single(block, other, locked):
                    return [block, other]
            return [block]
        else:
            # Topology insert/delete with boundary-aware placement.
            if topology_ban_boundary_involvement and touches_boundary:
                return list(block_group if block_group is not None else [block])
            if stage2_mode and touches_boundary and not has_relaxed_boundary:
                return list(block_group if block_group is not None else [block])
            if freeze_boundary_topology and block_group is not None and any(
                self._structural_boundary_mask(boundary_codes, member) != 0
                and not (phase3_free_boundary_blocks and self._phase3_can_free_boundary(boundary_codes, member))
                for member in block_group
            ):
                return list(block_group)
            if freeze_boundary_topology and structural_mask != 0 and not has_relaxed_boundary:
                return [block]
            if block_group is not None:
                for member in block_group:
                    if topology_ban_boundary_involvement:
                        insert_unconstrained(member)
                    elif phase3_free_boundary_blocks and self._phase3_can_free_boundary(boundary_codes, member):
                        insert_unconstrained(member)
                    else:
                        self._insert_boundary_aware(
                            tree,
                            member,
                            boundary_codes,
                            locked,
                            force_top_chain=(member in top_chain_locked),
                        )
                return list(block_group)
            if has_relaxed_boundary:
                insert_unconstrained(block)
            else:
                if topology_ban_boundary_involvement:
                    insert_unconstrained(block)
                else:
                    self._insert_boundary_aware(
                        tree,
                        block,
                        boundary_codes,
                        locked,
                        force_top_chain=(block in top_chain_locked),
                    )
            return [block]

    # ------------------------------------------------------------------
    #  GPU / CPU cost helpers
    # ------------------------------------------------------------------
    def _init_gpu_data(
        self,
        b2b_edges: List[Tuple[int, int, float]],
        p2b_edges: List[Tuple[int, int, float]],
        pin_positions: List[Tuple[float, float]],
    ) -> None:
        if not (self.use_gpu and _HAS_CUPY):
            return
        if b2b_edges:
            self._gpu_b2b_src = cp.array([e[0] for e in b2b_edges], dtype=cp.int32)
            self._gpu_b2b_dst = cp.array([e[1] for e in b2b_edges], dtype=cp.int32)
            self._gpu_b2b_w = cp.array([e[2] for e in b2b_edges], dtype=cp.float64)
        if p2b_edges:
            self._gpu_p2b_pin = cp.array([e[0] for e in p2b_edges], dtype=cp.int32)
            self._gpu_p2b_blk = cp.array([e[1] for e in p2b_edges], dtype=cp.int32)
            self._gpu_p2b_w = cp.array([e[2] for e in p2b_edges], dtype=cp.float64)
        if pin_positions:
            self._gpu_pins = cp.array(pin_positions, dtype=cp.float64)

    def _gpu_wirelength_cost(
        self,
        positions: List[Rect],
    ) -> Tuple[float, float]:
        if not _HAS_CUPY:
            return 0.0, 0.0
        n = len(positions)
        if n == 0:
            return 0.0, 0.0
        pos = cp.array(positions, dtype=cp.float64)
        cx = pos[:, 0] + pos[:, 2] / 2.0
        cy = pos[:, 1] + pos[:, 3] / 2.0

        hpwl_b2b = 0.0
        if hasattr(self, '_gpu_b2b_src') and len(self._gpu_b2b_src) > 0:
            hpwl_b2b = float(cp.sum(
                self._gpu_b2b_w * (
                    cp.abs(cx[self._gpu_b2b_src] - cx[self._gpu_b2b_dst])
                    + cp.abs(cy[self._gpu_b2b_src] - cy[self._gpu_b2b_dst])
                )
            ))

        hpwl_p2b = 0.0
        if hasattr(self, '_gpu_p2b_pin') and len(self._gpu_p2b_pin) > 0:
            px = self._gpu_pins[self._gpu_p2b_pin, 0]
            py = self._gpu_pins[self._gpu_p2b_pin, 1]
            bx = cx[self._gpu_p2b_blk]
            by = cy[self._gpu_p2b_blk]
            hpwl_p2b = float(cp.sum(
                self._gpu_p2b_w * (cp.abs(px - bx) + cp.abs(py - by))
            ))

        return hpwl_b2b, hpwl_p2b

    def _gpu_bbox_area(self, positions: List[Rect]) -> float:
        if not _HAS_CUPY:
            return contest_api.calculate_bbox_area(positions)
        if not positions:
            return 0.0
        pos = cp.array(positions, dtype=cp.float64)
        x_min = float(cp.min(pos[:, 0]))
        y_min = float(cp.min(pos[:, 1]))
        x_max = float(cp.max(pos[:, 0] + pos[:, 2]))
        y_max = float(cp.max(pos[:, 1] + pos[:, 3]))
        return (x_max - x_min) * (y_max - y_min)

    def _gpu_outline_penalty(self, positions: List[Rect]) -> float:
        if not positions:
            return 0.0
        outline_width, outline_height = self._cached_terminal_outline
        if outline_width <= 0.0 or outline_height <= 0.0:
            return 0.0
        pos = cp.array(positions, dtype=cp.float64)
        x_min = float(cp.min(pos[:, 0]))
        y_min = float(cp.min(pos[:, 1]))
        x_max = float(cp.max(pos[:, 0] + pos[:, 2]))
        y_max = float(cp.max(pos[:, 1] + pos[:, 3]))
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        return max(0.0, bbox_w - outline_width) + max(0.0, bbox_h - outline_height)

    # ------------------------------------------------------------------
    #  Cost functions (scalar objective, term decomposition, printing)
    # ------------------------------------------------------------------
    def _cost(
        self,
        positions: List[Rect],
        b2b_edges: List[Tuple[int, int, float]],
        p2b_edges: List[Tuple[int, int, float]],
        pin_positions: List[Tuple[float, float]],
        cluster_groups: Dict[int, List[int]],
        boundary_codes: Dict[int, int],
        areas: Sequence[float],
        preplaced_anchors: Dict[int, Tuple[float, float]],
        dynamic_anchors: Dict[int, Tuple[float, float]],
        immovable_blocks: Iterable[int],
        tree: Optional[BStarTree] = None,
        mode: str = "full",
    ) -> float:
        # Full scalar objective used by SA.
        terms, total_cost = self._cost_terms(
            positions,
            b2b_edges,
            p2b_edges,
            pin_positions,
            cluster_groups,
            boundary_codes,
            immovable_blocks,
            tree,
            mode,
        )
        return total_cost

    def _phase3_boundary_violation_cost(
        self,
        positions: List[Rect],
        boundary_codes: Dict[int, int],
    ) -> float:
        return (
            float(self._boundary_snap_violation_count(positions, boundary_codes))
            * self.phase3_boundary_violation_weight
        )

    def _cost_terms(
        self,
        positions: List[Rect],
        b2b_edges: List[Tuple[int, int, float]],
        p2b_edges: List[Tuple[int, int, float]],
        pin_positions: List[Tuple[float, float]],
        cluster_groups: Dict[int, List[int]],
        boundary_codes: Dict[int, int],
        immovable_blocks: Iterable[int],
        tree: Optional[BStarTree] = None,
        mode: str = "full",
    ) -> Tuple[Dict[str, float], float]:
        use_gpu = self.use_gpu and _HAS_CUPY
        outline_width, outline_height = self._cached_terminal_outline
        if outline_width <= 0.0 or outline_height <= 0.0:
            outline_area = 0.0
        else:
            outline_area = outline_width * outline_height
        preplaced_overlap_penalty = 0
        if mode in ("stage2", "stage2_hpwl", "stage2_boundary"):
            if mode == "stage2":
                bbox_area = self._bbox_area_for_positions(positions)
                outline_penalty = self._outline_penalty_for_positions(positions) * self.outline_weight
                hpwl_p2b = 0
                terms = {
                    "p2b": hpwl_p2b * self.stage2_p2b_weight,
                    "bbox": bbox_area * self.bbox_weight,
                    "outline": outline_penalty,
                    "preplaced_overlap": preplaced_overlap_penalty,
                }
                return terms, sum(terms.values())
            outline_penalty = self._outline_penalty_for_positions(positions) * self.outline_weight
            if mode == "stage2_boundary":
                filtered_b2b = [(i, j, w) for (i, j, w) in b2b_edges
                               if i in boundary_codes or j in boundary_codes] if boundary_codes else []
                hpwl_b2b, hpwl_p2b = self._wirelength_cost(positions, filtered_b2b, p2b_edges, pin_positions)
            elif use_gpu:
                hpwl_b2b, hpwl_p2b = self._gpu_wirelength_cost(positions)
            else:
                hpwl_b2b, hpwl_p2b = self._wirelength_cost(positions, b2b_edges, p2b_edges, pin_positions)
            conn_weight = self._stage2_hpwl_conn_weight if self._stage2_hpwl_conn_weight is not None else 1.0
            p2b_term = conn_weight * hpwl_p2b / max(self._cached_wirelength_norm, 1.0)
            b2b_term = conn_weight * hpwl_b2b / max(self._cached_wirelength_norm, 1.0)
            cluster_penalty = (self._cluster_penalty(positions, cluster_groups) * self.cluster_weight
                               if mode == "stage2_hpwl" and cluster_groups else 0.0)
            terms = {
                "b2b": b2b_term,
                "p2b": p2b_term,
                "outline": outline_penalty,
                "cluster": cluster_penalty,
                "preplaced_overlap": preplaced_overlap_penalty,
            }
            return terms, sum(terms.values())
        # full mode: compute bbox once, reuse for area + outline + boundary
        if use_gpu:
            bbox_area = self._gpu_bbox_area(positions)
            outline_penalty = self._gpu_outline_penalty(positions) * self.outline_weight
            hpwl_b2b, hpwl_p2b = self._gpu_wirelength_cost(positions)
        else:
            x_min, y_min, x_max, y_max = float('inf'), float('inf'), float('-inf'), float('-inf')
            for x, y, w, h in positions:
                if x < x_min: x_min = x
                if y < y_min: y_min = y
                if x + w > x_max: x_max = x + w
                if y + h > y_max: y_max = y + h
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min
            bbox_area = max(0.0, bbox_w * bbox_h)
            outline_penalty = self.outline_weight * (
                max(0.0, bbox_w - outline_width) + max(0.0, bbox_h - outline_height)
            )
            hpwl_b2b, hpwl_p2b = self._wirelength_cost(positions, b2b_edges, p2b_edges, pin_positions)
        total_wirelength = hpwl_b2b + hpwl_p2b
        area_term = self.area_wirelength_alpha * (bbox_area / max(self._cached_area_norm, 1.0))
        wirelength_term = (1.0 - self.area_wirelength_alpha) * (
            total_wirelength / max(self._cached_wirelength_norm, 1.0)
        )
        overlap_penalty = 0
        if cluster_groups:
            cluster_penalty = self._cluster_penalty(positions, cluster_groups) * self.cluster_weight
        else:
            cluster_penalty = 0.0
        boundary_penalty = 0
        if boundary_codes:
            phase3_boundary_violation_penalty = (
                float(self._boundary_snap_violation_count(positions, boundary_codes))
                * self.phase3_boundary_violation_weight
            )
        else:
            phase3_boundary_violation_penalty = 0.0

        terms = {
            "area": area_term,
            "wirelength": wirelength_term,
            "overlap": overlap_penalty,
            "cluster": cluster_penalty,
            "boundary": boundary_penalty,
            "phase3_boundary_violation": phase3_boundary_violation_penalty,
            "outline": outline_penalty,
            "preplaced_overlap": preplaced_overlap_penalty,
        }
        return terms, sum(terms.values())

    def _print_final_cost_breakdown(
        self,
        positions: List[Rect],
        b2b_edges: List[Tuple[int, int, float]],
        p2b_edges: List[Tuple[int, int, float]],
        pin_positions: List[Tuple[float, float]],
        cluster_groups: Dict[int, List[int]],
        boundary_codes: Dict[int, int],
        immovable_blocks: Iterable[int],
        tree: Optional[BStarTree] = None,
    ) -> None:
        terms, total = self._cost_terms(
            positions,
            b2b_edges,
            p2b_edges,
            pin_positions,
            cluster_groups,
            boundary_codes,
            immovable_blocks,
            tree,
            mode="full",
        )
        print(f"  {_B}{_CYN}{'─'*14} Phase 3 {'─'*18}{_C}")
        term_colors = {
            "area": _BLU, "bbox": _BLU, "outline": _Y,
            "wirelength": _CYN, "hpwl": _CYN,
            "cluster": _Y, "boundary": _M,
            "b2b_boundary": _M, "p2b": _G,
        }
        displayed = {"phase3_boundary_violation": "boundary"}
        lines = []
        for key, value in sorted(terms.items()):
            if key in {"boundary", "preplaced_overlap"} or (abs(value) < 1e-12):
                continue
            display_key = displayed.get(key, key)
            color = term_colors.get(display_key, "")
            pct = 100.0 * value / total if abs(total) > 1e-12 else 0.0
            lines.append(f"  {color}{display_key}{_C}: {value:.6f} ({pct:.2f}%)")
        for line in lines:
            print(line)
        print(f"  {_B}{_G}total{_C}: {total:.6f} (100.00%)")
        print(f"  {'─'*40}")
        import os as _os; n_cores = _os.cpu_count() or '?'
        par_status = "disabled on Windows" if IS_WINDOWS else f"gpu={self.use_gpu}, probe={self.use_parallel_probe}, cluster={self.use_parallel_cluster}"
        print(f"  {_GRY}parallel{_C}: {par_status}")
        overlap_count = contest_api.check_overlap(positions)
        flag = "FEASIBLE" if overlap_count == 0 else "INFEASIBLE"
        color_status = _G if overlap_count == 0 else _R
        print(f"  {color_status}status{_C}: {color_status}{flag}{_C} (overlaps={overlap_count})")

    def _wirelength_cost(
        self,
        positions: List[Rect],
        b2b_edges: List[Tuple[int, int, float]],
        p2b_edges: List[Tuple[int, int, float]],
        pin_positions: List[Tuple[float, float]],
    ) -> Tuple[float, float]:
        centers = [(x + w / 2.0, y + h / 2.0) for x, y, w, h in positions]

        hpwl_b2b = 0.0
        for block_i, block_j, weight in b2b_edges:
            x1, y1 = centers[block_i]
            x2, y2 = centers[block_j]
            hpwl_b2b += weight * (abs(x2 - x1) + abs(y2 - y1))

        hpwl_p2b = 0.0
        n_pins = len(pin_positions)
        n_centers = len(centers)
        for pin_idx, block_idx, weight in p2b_edges:
            if pin_idx >= n_pins or block_idx >= n_centers:
                continue
            px, py = pin_positions[pin_idx]
            bx, by = centers[block_idx]
            hpwl_p2b += weight * (abs(px - bx) + abs(py - by))

        return hpwl_b2b, hpwl_p2b

    def _cluster_penalty(self, positions: List[Rect], cluster_groups: Dict[int, List[int]]) -> float:
        # Cluster score is based on how many touching connected-components are formed.
        penalty = 0.0
        for blocks in cluster_groups.values():
            if len(blocks) <= 1:
                continue
            components = self._count_touching_components(positions, blocks)
            penalty += max(0.0, float(components - 1) / float(len(blocks)))
        return penalty

    def _rectangles_touch(self, rect_a: Rect, rect_b: Rect, tol: float = 1e-6) -> bool:
        # Touching counts as overlap or edge/corner contact within tolerance.
        ax, ay, aw, ah = rect_a
        bx, by, bw, bh = rect_b
        overlap_x = min(ax + aw, bx + bw) - max(ax, bx)
        overlap_y = min(ay + ah, by + bh) - max(ay, by)
        return overlap_x >= -tol and overlap_y >= -tol

    def _count_touching_components(self, positions: List[Rect], blocks: List[int]) -> int:
        # Count connected components in the touching graph of one cluster.
        remaining = set(blocks)
        components = 0
        while remaining:
            components += 1
            start = remaining.pop()
            stack = [start]
            while stack:
                current = stack.pop()
                for other in list(remaining):
                    if self._rectangles_touch(positions[current], positions[other]):
                        remaining.remove(other)
                        stack.append(other)
        return components

    def _top_right_repair_move(
        self,
        tree: BStarTree,
        positions: List[Rect],
        boundary_codes: Dict[int, int],
        locked_blocks: Iterable[int],
    ) -> Tuple[List[int], bool]:
        # Repair move for TOP constraints only.
        # Method 1: if an unconstrained block is already on the top contour, swap with it.
        # Method 2: otherwise insert the constrained block toward that contour and force accept.
        if not positions or not boundary_codes:
            return [], False

        locked = set(locked_blocks)
        y_contour = self._y_contour_blocks(positions)

        violations: List[Tuple[int, str]] = []
        for block, code in boundary_codes.items():
            if block in locked:
                continue
            if self._structural_boundary_mask(boundary_codes, block) != 0:
                continue
            if (code & TOP) and block not in y_contour:
                violations.append((block, "top"))

        if not violations:
            return [], False

        block, mode = random.choice(violations)
        if mode != "top":
            return [], False

        boundary_candidates = [
            idx
            for idx in y_contour
            if idx != block and boundary_codes.get(idx, 0) == 0 and idx not in locked
        ]
        if boundary_candidates:
            other = random.choice(boundary_candidates)
            tree.swap_nodes(block, other, locked)
            return [block, other], True

        insert_targets = [idx for idx in y_contour if idx != block and idx not in locked and tree.right[idx] == -1]
        if insert_targets:
            target = random.choice(insert_targets)
            tree._delete_node(block)
            tree._insert_node(block, target, False)
            return [block], True
        return [], False

    def _interval_fully_exposed(
        self,
        start: float,
        end: float,
        covered: List[Tuple[float, float]],
        tol: float = 1e-6,
    ) -> bool:
        # Return True only when the entire interval is exposed.
        if end - start <= tol:
            return False
        if not covered:
            return True

        merged: List[Tuple[float, float]] = []
        for left, right in sorted(covered):
            left = max(left, start)
            right = min(right, end)
            if right - left <= tol:
                continue
            if not merged or left > merged[-1][1] + tol:
                merged.append((left, right))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], right))

        covered_length = sum(right - left for left, right in merged)
        return covered_length <= tol

    def _y_contour_blocks(self, positions: List[Rect], tol: float = 1e-6) -> set[int]:
        # Top contour: blocks whose top edge has some exposed x-interval.
        contour_blocks: set[int] = set()
        for idx, (x_pos, y_pos, width, height) in enumerate(positions):
            top = y_pos + height
            covered: List[Tuple[float, float]] = []
            for other_idx, (ox, oy, ow, oh) in enumerate(positions):
                if other_idx == idx:
                    continue
                other_bottom = oy
                other_top = oy + oh
                if other_bottom > top + tol:
                    continue
                if other_top <= top + tol:
                    continue
                overlap_start = max(x_pos, ox)
                overlap_end = min(x_pos + width, ox + ow)
                if overlap_end - overlap_start > tol:
                    covered.append((overlap_start, overlap_end))
            if self._interval_fully_exposed(x_pos, x_pos + width, covered, tol):
                contour_blocks.add(idx)
        return contour_blocks

    def _x_contour_blocks(self, positions: List[Rect], tol: float = 1e-6) -> set[int]:
        """Right contour: blocks whose right edge has some exposed y-interval."""
        contour_blocks: set[int] = set()
        for idx, (x_pos, y_pos, width, height) in enumerate(positions):
            right = x_pos + width
            covered: List[Tuple[float, float]] = []
            for oidx, (ox, oy, ow, oh) in enumerate(positions):
                if oidx == idx:
                    continue
                other_right = ox + ow
                if other_right <= right + tol:
                    continue
                if ox >= right - tol:
                    continue
                overlap_start = max(y_pos, oy)
                overlap_end = min(y_pos + height, oy + oh)
                if overlap_end - overlap_start > tol:
                    covered.append((overlap_start, overlap_end))
            if self._interval_fully_exposed(y_pos, y_pos + height, covered, tol):
                contour_blocks.add(idx)
        return contour_blocks

    def _contour_lowest_block(self, positions: List[Rect], boundary_codes: Dict[int, int],
                              skip_blocks: set[int]) -> Optional[int]:
        """Find the non-TOP-boundary contour block with the lowest top (deepest dip inside outline)."""
        contour_blocks = self._y_contour_blocks(positions)
        _, outline_h = self._cached_terminal_outline
        dip_block = None
        dip_y = float('inf')
        for block in contour_blocks:
            if block in skip_blocks or (boundary_codes.get(block, 0) & TOP):
                continue
            top = positions[block][1] + positions[block][3]
            if top >= outline_h:
                continue
            if top < dip_y:
                dip_y = top
                dip_block = block
        return dip_block

    def _contour_innermost_block(self, positions: List[Rect], boundary_codes: Dict[int, int],
                                  skip_blocks: set[int]) -> Optional[int]:
        """Find the non-RIGHT-boundary contour block with the innermost right edge (deepest gap from right inside outline)."""
        contour_blocks = self._x_contour_blocks(positions)
        outline_w, _ = self._cached_terminal_outline
        dip_block = None
        dip_right = float('inf')
        for block in contour_blocks:
            if block in skip_blocks or (boundary_codes.get(block, 0) & RIGHT):
                continue
            right_edge = positions[block][0] + positions[block][2]
            if right_edge >= outline_w:
                continue
            if right_edge < dip_right:
                dip_right = right_edge
                dip_block = block
        return dip_block

    def _phase2_fill_contour_gap(self, tree: BStarTree, positions: List[Rect],
                                  boundary_codes: Dict[int, int],
                                  locked_blocks: Iterable[int],
                                  filler_block: int,
                                  as_left: bool = False) -> bool:
        """Insert filler_block as RIGHT (as_left=False) or LEFT (as_left=True) child of the
        lowest (Y) or innermost (X) contour block to fill outline gaps."""
        locked = set(locked_blocks)
        if as_left:
            dip_block = self._contour_innermost_block(positions, boundary_codes, locked)
        else:
            dip_block = self._contour_lowest_block(positions, boundary_codes, locked)
        # Redirect non-anchor cluster satellite to its active anchor.
        if dip_block is not None and dip_block in self._cached_macro_cluster_non_anchors:
            _anchor = self._cached_macro_cluster_anchor_by_block.get(dip_block)
            if _anchor is not None and tree.active[_anchor]:
                dip_block = _anchor
        if dip_block is None or filler_block == dip_block or not tree.active[filler_block]:
            return False
        if self._is_boundary_constraint_block(boundary_codes, filler_block):
            return False
        tree._delete_node(filler_block)
        tree._insert_node(filler_block, dip_block, as_left)
        return True

    def _contour_fill_as_left_probability(self, positions: List[Rect],
                                           boundary_codes: Dict[int, int],
                                           skip_blocks: set[int]) -> float:
        """Probability of using X-contour fill (LEFT child) vs Y-contour fill, proportional to gap sizes."""
        outline_w, outline_h = self._cached_terminal_outline
        dip_y = self._contour_lowest_block(positions, boundary_codes, skip_blocks)
        dip_x = self._contour_innermost_block(positions, boundary_codes, skip_blocks)
        y_gap = outline_h - (positions[dip_y][1] + positions[dip_y][3]) if dip_y is not None else 0.0
        x_gap = outline_w - (positions[dip_x][0] + positions[dip_x][2]) if dip_x is not None else 0.0
        total = y_gap + x_gap
        if total <= 1e-9:
            return 0.5
        return x_gap / total

    def _contour_constraint_penalty(self, positions: List[Rect], boundary_codes: Dict[int, int]) -> float:
        # Count how many TOP constrained blocks fail to lie on the top contour.
        if not positions or not boundary_codes:
            return 0.0

        need_top = any(code & TOP for code in boundary_codes.values())
        y_contour = self._y_contour_blocks(positions) if need_top else set()

        penalty = 0.0
        for block, code in boundary_codes.items():
            if (code & TOP) and block not in y_contour:
                penalty += 1.0
        return penalty


    def _outline_penalty(
        self,
        positions: List[Rect],
    ) -> float:
        # Penalize only the amount by which the bbox exceeds the fixed outline
        # on width or height.
        if not positions:
            return 0.0
        outline_width, outline_height = self._cached_terminal_outline
        if outline_width <= 0.0 or outline_height <= 0.0:
            return 0.0

        x_min = min(p[0] for p in positions)
        y_min = min(p[1] for p in positions)
        x_max = max(p[0] + p[2] for p in positions)
        y_max = max(p[1] + p[3] for p in positions)
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        return max(0.0, bbox_width - outline_width) + max(0.0, bbox_height - outline_height)

    def _bbox_area_for_positions(
        self,
        positions: List[Rect],
    ) -> float:
        if not positions:
            return 0.0
        x_min = min(p[0] for p in positions)
        y_min = min(p[1] for p in positions)
        x_max = max(p[0] + p[2] for p in positions)
        y_max = max(p[1] + p[3] for p in positions)
        return max(0.0, (x_max - x_min) * (y_max - y_min))

    def _outline_penalty_for_positions(self, positions: List[Rect]) -> float:
        if not positions:
            return 0.0

        w_out, h_out = self._cached_terminal_outline
        if w_out <= 0.0 or h_out <= 0.0:
            return 0.0

        # 1. iterate one time to find BBox, reducing the cost of scanning positions repeatly
        x1, y1, x2, y2 = float('inf'), float('inf'), float('-inf'), float('-inf')
        for x, y, w, h in positions:
            if x < x1: x1 = x
            if y < y1: y1 = y
            if x + w > x2: x2 = x + w
            if y + h > y2: y2 = y + h

        bbox_w = x2 - x1
        bbox_h = y2 - y1

        # 2. compute the outside value (Violation)
        dw = max(0.0, bbox_w - w_out)
        dh = max(0.0, bbox_h - h_out)

        # 3. weight computation
        inv_total = 1.0 / (w_out + h_out)
        return (h_out * dw + w_out * dh) * inv_total

    def _outline_is_converged(
        self,
        positions: List[Rect],
    ) -> bool:
        # Stage-II boundary-focused search starts once the current bbox area
        # is reasonably close to the terminal-derived fixed outline area.
        if not positions:
            return False
        outline_width, outline_height = self._cached_terminal_outline
        outline_area = outline_width * outline_height
        if outline_area <= 0.0:
            return False

        x_min = min(p[0] for p in positions)
        y_min = min(p[1] for p in positions)
        x_max = max(p[0] + p[2] for p in positions)
        y_max = max(p[1] + p[3] for p in positions)
        bbox_area = max(0.0, (x_max - x_min) * (y_max - y_min))
        return abs(bbox_area - outline_area) <= max(100.0, 0.05 * outline_area)

    def _pick_exposed_outline_block(
        self,
        positions: List[Rect],
        boundary_codes: Dict[int, int],
        locked_blocks: Iterable[int],
        scheduled_cost_mode,
        probability: float = 0.02,
    ) -> Optional[int]:
        # Adjust probability for stage2
        if scheduled_cost_mode == "stage2":
            probability = 0.75
        if not positions or random.random() >= probability:
            return None

        locked = set(locked_blocks)
        outline_width, outline_height = self._cached_terminal_outline

        candidates: List[Tuple[float, int]] = []
        seen_anchors: set[int] = set()
        for block, _ in enumerate(positions):
            if block in locked:
                continue
            if self._is_boundary_constraint_block(boundary_codes, block):
                continue
            if block in self._cached_macro_cluster_non_anchors:
                continue
            candidate = self._cluster_bbox_edges_for_outline_pick(block, positions)
            if candidate is None:
                continue
            right_edge, top_edge, candidate_block = candidate
            if candidate_block in seen_anchors:
                continue
            seen_anchors.add(candidate_block)
            right_excess = max(0.0, right_edge - outline_width)
            top_excess = max(0.0, top_edge - outline_height)
            weight = right_excess + top_excess
            if weight > 1e-9:
                candidates.append((weight, candidate_block))

        if not candidates:
            return None

        total_weight = sum(w for w, _ in candidates)
        r = random.random() * total_weight
        cumul = 0.0
        for weight, block in candidates:
            cumul += weight
            if r <= cumul:
                return block
        return candidates[-1][1]

    # ------------------------------------------------------------------
    #  Phase II / III transition helpers
    # ------------------------------------------------------------------
    def _finalize_boundary_outline_fit(
        self,
        tree: BStarTree,
        positions: List[Rect],
        boundary_codes: Dict[int, int],
        top_chain_locked_blocks: Iterable[int],
        preplaced_anchors: Dict[int, Tuple[float, float]],
        immovable_blocks: Iterable[int],
        geometry_locked_blocks: Iterable[int],
        target_rects: Sequence[Rect],
    ) -> List[Rect]:
        # Phase-II convergence now only freezes boundary order/topology.
        # It no longer deterministically solves TL/TR/BR by rewriting
        # boundary block aspect ratios, because that was stamping the same
        # top/bottom/left chain shapes across runs and starving Phase III of
        # useful boundary-shape exploration.
        return positions

    def _phase2_boundary_ratio_mode(self, positions: List[Rect]) -> Optional[str]:
        # When Phase II outline has not converged, relax only one family of
        # boundary aspect ratios depending on whether height or width mismatch
        # is currently larger.
        if not positions or self._outline_is_converged(positions):
            return None
        outline_width, outline_height = self._cached_terminal_outline
        if outline_width <= 0.0 or outline_height <= 0.0:
            return None

        x_min = min(p[0] for p in positions)
        y_min = min(p[1] for p in positions)
        x_max = max(p[0] + p[2] for p in positions)
        y_max = max(p[1] + p[3] for p in positions)
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        width_gap = bbox_width - outline_width
        height_gap = bbox_height - outline_height

        if height_gap > width_gap:
            return "left"
        if width_gap > height_gap:
            return "bottom"
        return None

    def _last_adjustable_chain_block(
        self,
        ordered_chain: List[int],
        tail_block: Optional[int],
        allowed_blocks: Iterable[int],
        geometry_locked_blocks: Iterable[int],
    ) -> Optional[int]:
        allowed = set(allowed_blocks)
        locked = set(geometry_locked_blocks)
        if not ordered_chain:
            return None
        end_idx = ordered_chain.index(tail_block) if tail_block in ordered_chain else len(ordered_chain) - 1
        for block in reversed(ordered_chain[: end_idx + 1]):
            if block in allowed and block not in locked:
                return block
        return None

    # ------------------------------------------------------------------
    #  Outline dimension & terminal helpers
    # ------------------------------------------------------------------
    def _fixed_outline_dimensions(
        self,
        pin_positions: List[Tuple[float, float]],
        constraints_rows: List[List[float]],
        target_rects: Sequence[Rect],
    ) -> Tuple[float, float]:
        # Base outline comes from terminals, but hard preplaced blocks on the
        # outer boundary override the corresponding outline edge. Corner
        # preplaced blocks therefore drive both dimensions when applicable:
        # - TL contributes to outline height
        # - BR contributes to outline width
        # - TR contributes to both width and height
        outline_width, outline_height = self._terminal_outline_dimensions(pin_positions)

        top_preplaced = []
        right_preplaced = []
        has_top_override = False
        has_right_override = False
        for block, row in enumerate(constraints_rows):
            if len(row) < 5 or row[1] == 0:
                continue
            tx, ty, tw, th = target_rects[block]
            if tx < 0.0 or ty < 0.0 or tw <= 0.0 or th <= 0.0:
                continue
            code = int(row[4])
            top_edge = ty + th
            right_edge = tx + tw
            if code & TOP:
                top_preplaced.append(top_edge)
                has_top_override = True
            if code & RIGHT:
                right_preplaced.append(right_edge)
                has_right_override = True

        # Hard preplaced boundary blocks override the terminal-derived outline
        # rather than competing with it via max(...). Corner blocks therefore
        # fully define the corresponding dimension(s).
        if has_top_override and top_preplaced:
            outline_height = max(top_preplaced)
        if has_right_override and right_preplaced:
            outline_width = max(right_preplaced)
        return outline_width, outline_height

    def _terminal_outline_dimensions(
        self,
        pin_positions: List[Tuple[float, float]],
    ) -> Tuple[float, float]:
        # height = (highest y on left/right boundary terminals + terminal height) / 2
        # width  = (rightmost x on bottom/top boundary terminals + terminal width) / 2
        # Terminal size is not exposed in the public input, so width/height are 0.
        valid_pins = [
            (float(x_pos), float(y_pos))
            for x_pos, y_pos in pin_positions
            if x_pos >= 0.0 and y_pos >= 0.0
        ]
        if not valid_pins:
            return 0.0, 0.0

        tol = 1e-6
        min_x = min(x_pos for x_pos, _ in valid_pins)
        max_x = max(x_pos for x_pos, _ in valid_pins)
        min_y = min(y_pos for _, y_pos in valid_pins)
        max_y = max(y_pos for _, y_pos in valid_pins)

        left_right_terminals = [
            (x_pos, y_pos)
            for x_pos, y_pos in valid_pins
            if abs(x_pos - min_x) <= tol or abs(x_pos - max_x) <= tol
        ]
        top_bottom_terminals = [
            (x_pos, y_pos)
            for x_pos, y_pos in valid_pins
            if abs(y_pos - min_y) <= tol or abs(y_pos - max_y) <= tol
        ]

        global_max_x = max_x
        global_max_y = max_y
        outline_height = (max(y_pos for _, y_pos in left_right_terminals) + global_max_y) / 2.0
        outline_width = (max(x_pos for x_pos, _ in top_bottom_terminals) + global_max_x) / 2.0
        return outline_width, outline_height

    def _pins_on_side(
        self,
        pin_positions: List[Tuple[float, float]],
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        if not pin_positions:
            return [], [], [], []
        tol = 1e-6
        min_x = min(x_pos for x_pos, _ in pin_positions)
        max_x = max(x_pos for x_pos, _ in pin_positions)
        min_y = min(y_pos for _, y_pos in pin_positions)
        max_y = max(y_pos for _, y_pos in pin_positions)

        bottom = sorted(
            [idx for idx, (x_pos, y_pos) in enumerate(pin_positions) if abs(y_pos - min_y) <= tol],
            key=lambda idx: (pin_positions[idx][0], pin_positions[idx][1], idx),
        )
        top = sorted(
            [idx for idx, (x_pos, y_pos) in enumerate(pin_positions) if abs(y_pos - max_y) <= tol],
            key=lambda idx: (pin_positions[idx][0], pin_positions[idx][1], idx),
        )
        left = sorted(
            [idx for idx, (x_pos, y_pos) in enumerate(pin_positions) if abs(x_pos - min_x) <= tol],
            key=lambda idx: (pin_positions[idx][1], pin_positions[idx][0], idx),
        )
        right = sorted(
            [idx for idx, (x_pos, y_pos) in enumerate(pin_positions) if abs(x_pos - max_x) <= tol],
            key=lambda idx: (pin_positions[idx][1], pin_positions[idx][0], idx),
        )
        return bottom, left, top, right

    def _ordered_boundary_blocks_from_terminals(
        self,
        candidate_blocks: Iterable[int],
        pin_indices: List[int],
        p2b_edges: List[Tuple[int, int, float]],
        fixed_first: Optional[int] = None,
        fixed_last: Optional[int] = None,
    ) -> List[int]:
        candidates = set(candidate_blocks)
        if not candidates:
            return []

        pin_to_blocks: Dict[int, List[int]] = {}
        for pin_idx, block_idx, _ in p2b_edges:
            if block_idx in candidates:
                pin_to_blocks.setdefault(pin_idx, []).append(block_idx)

        ordered: List[int] = []
        seen: set[int] = set()
        if fixed_first is not None and fixed_first in candidates:
            ordered.append(fixed_first)
            seen.add(fixed_first)

        for pin_idx in pin_indices:
            for block in pin_to_blocks.get(pin_idx, []):
                if block in seen:
                    continue
                if fixed_last is not None and block == fixed_last:
                    continue
                ordered.append(block)
                seen.add(block)

        for block in sorted(candidates):
            if block in seen:
                continue
            if fixed_last is not None and block == fixed_last:
                continue
            ordered.append(block)
            seen.add(block)

        if fixed_last is not None and fixed_last in candidates and fixed_last not in seen:
            ordered.append(fixed_last)
        return ordered

    def _build_boundary_cache(
        self,
        boundary_codes: Dict[int, int],
        block_count: int,
        p2b_edges: List[Tuple[int, int, float]],
        pin_positions: List[Tuple[float, float]],
    ) -> None:
        # Cache all boundary-structure metadata that is static for one testcase.
        masks = [0] * block_count
        for block, code in boundary_codes.items():
            if 0 <= block < block_count:
                masks[block] = code & (LEFT | BOTTOM | TOP | RIGHT)
        self._cached_boundary_masks = masks

        self._cached_boundary_bottom_nodes = {block for block, code in boundary_codes.items() if code & BOTTOM}
        self._cached_boundary_left_nodes = {block for block, code in boundary_codes.items() if code & LEFT}
        self._cached_boundary_top_nodes = {block for block, code in boundary_codes.items() if code & TOP}
        self._cached_boundary_right_nodes = {
            block
            for block, code in boundary_codes.items()
            if (code & RIGHT) and not (code & TOP) and not (code & BOTTOM)
        }
        self._cached_bottom_left_nodes = {
            block for block, code in boundary_codes.items() if (code & BOTTOM) and (code & LEFT)
        }

        self._cached_left_tail = next(
            (block for block, code in boundary_codes.items() if (code & LEFT) and (code & TOP)),
            None,
        )
        self._cached_bottom_tail = next(
            (block for block, code in boundary_codes.items() if (code & BOTTOM) and (code & RIGHT)),
            None,
        )
        self._cached_top_tail = next(
            (block for block, code in boundary_codes.items() if (code & TOP) and (code & RIGHT)),
            None,
        )

        bottom_pins, left_pins, top_pins, right_pins = self._pins_on_side(pin_positions)

        bottom_nodes = list(self._cached_boundary_bottom_nodes)
        if bottom_nodes:
            chain_root = next((block for block in bottom_nodes if boundary_codes[block] & LEFT), bottom_nodes[0])
            self._cached_ordered_bottom = self._ordered_boundary_blocks_from_terminals(
                bottom_nodes,
                bottom_pins,
                p2b_edges,
                fixed_first=chain_root,
                fixed_last=self._cached_bottom_tail if self._cached_bottom_tail != chain_root else None,
            )
        else:
            self._cached_ordered_bottom = []

        left_nodes = list(self._cached_boundary_left_nodes)
        if left_nodes:
            chain_root = next((block for block in left_nodes if boundary_codes[block] & BOTTOM), left_nodes[0])
            self._cached_ordered_left = self._ordered_boundary_blocks_from_terminals(
                left_nodes,
                left_pins,
                p2b_edges,
                fixed_first=chain_root,
                fixed_last=self._cached_left_tail if self._cached_left_tail != chain_root else None,
            )
        else:
            self._cached_ordered_left = []

        if self._cached_boundary_top_nodes:
            top_nodes = list(self._cached_boundary_top_nodes)
            if self._cached_left_tail is not None:
                self._cached_top_root = self._cached_left_tail
            elif self._cached_ordered_left:
                self._cached_top_root = self._cached_ordered_left[-1]
            else:
                self._cached_top_root = top_nodes[0]
            ordered_top_members = self._ordered_boundary_blocks_from_terminals(
                top_nodes,
                top_pins,
                p2b_edges,
                fixed_first=self._cached_top_root if self._cached_top_root in top_nodes else None,
                fixed_last=self._cached_top_tail if self._cached_top_tail != self._cached_top_root else None,
            )
            if self._cached_top_root is not None and self._cached_top_root not in ordered_top_members:
                self._cached_ordered_top = [self._cached_top_root] + ordered_top_members
            else:
                self._cached_ordered_top = ordered_top_members
        else:
            self._cached_top_root = None
            self._cached_ordered_top = []

        self._cached_ordered_right = self._ordered_boundary_blocks_from_terminals(
            self._cached_boundary_right_nodes,
            right_pins,
            p2b_edges,
        )
        if self._cached_bottom_tail is not None:
            self._cached_right_root = self._cached_bottom_tail
        elif self._cached_ordered_bottom:
            self._cached_right_root = self._cached_ordered_bottom[-1]
        else:
            self._cached_right_root = None

        self._refine_boundary_order_by_clusters(boundary_codes)

    def _refine_boundary_order_by_clusters(
        self,
        boundary_codes: Dict[int, int],
    ) -> None:
        """Refine boundary chain order so same-constraint-cluster blocks are
        contiguous, respecting terminal order and excluding clusters with
        interleaved preplaced blocks."""
        if not self._cached_group_clusters:
            return

        def _find_corner(chain: List[int], mask: int) -> Optional[int]:
            for b in chain:
                if boundary_codes.get(b, 0) & mask == mask:
                    return b
            return None

        # Build block -> cluster map
        _b2c: Dict[int, int] = {}
        for _gid, _members in self._cached_group_clusters.items():
            for _m in _members:
                _b2c[_m] = _gid

        def _eligible(gid: int, chain: List[int]) -> bool:
            members = [b for b in chain if _b2c.get(b) == gid]
            if not members:
                return False
            # Cluster member itself is preplaced → ineligible
            for m in members:
                if m in self._cached_immovable_blocks:
                    return False
            if len(members) <= 1:
                return True
            idxs = [i for i, b in enumerate(chain) if b in members]
            lo, hi = min(idxs), max(idxs)
            # Preplaced block between cluster members → ineligible
            for i in range(lo + 1, hi):
                if chain[i] not in members and chain[i] in self._cached_immovable_blocks:
                    return False
            return True

        def _clusters_on_chain(chain: List[int]):
            return {_b2c[b] for b in chain if _b2c.get(b) is not None}

        def _refine_one(chain: List[int], cf: Optional[int], cl: Optional[int]):
            if len(chain) <= 2:
                return chain
            clusters = _clusters_on_chain(chain)
            eligible = {g for g in clusters if _eligible(g, chain)}
            if not eligible:
                return chain

            # Check whether corner blocks are actually in the chain
            chain_set = set(chain)
            cf_in_chain = cf is not None and cf in chain_set
            cl_in_chain = cl is not None and cl in chain_set

            cf_g = _b2c.get(cf) if cf is not None else None
            cl_g = _b2c.get(cl) if cl is not None else None

            # Build per-cluster member sets, preserving original relative order
            cluster_members: Dict[int, List[int]] = {}
            for b in chain:
                g = _b2c.get(b)
                if g is not None and g in eligible:
                    cluster_members.setdefault(g, []).append(b)
            singleton_eligible = {g for g, ms in cluster_members.items() if len(ms) <= 1}
            # Preserve corner clusters even with ≤1 member on this chain:
            # they must be placed at the chain ends for snap logic.
            for g in list(singleton_eligible):
                if g is not None and (g == cf_g or g == cl_g):
                    singleton_eligible.discard(g)
            eligible -= singleton_eligible
            for g in singleton_eligible:
                del cluster_members[g]
            if not eligible:
                return chain

            cf_set = set(cluster_members.get(cf_g, [])) if cf_g is not None else set()
            cl_set = set(cluster_members.get(cl_g, [])) if cl_g is not None else set()

            new_chain: List[int] = []
            seen: set[int] = set()

            # Corner-first block + its cluster
            if cf is not None and cf_in_chain:
                new_chain.append(cf)
                seen.add(cf)
                for b in chain:
                    if b not in seen and b in cf_set:
                        new_chain.append(b)
                        seen.add(b)
            elif cf is not None and not cf_in_chain:
                # Virtual corner: flush all cluster members together at front
                for m in cluster_members.get(cf_g, []):
                    if m not in seen:
                        new_chain.append(m)
                        seen.add(m)

            # Middle blocks
            for b in chain:
                if b in seen or (cl_in_chain and b == cl):
                    continue
                g = _b2c.get(b)
                if g is not None and g in eligible and b in cluster_members.get(g, []):
                    if g == cf_g or g == cl_g:
                        continue  # already handled by corner sections
                    # Flush this cluster's members together
                    for m in chain:
                        if m not in seen and _b2c.get(m) == g:
                            new_chain.append(m)
                            seen.add(m)
                else:
                    new_chain.append(b)
                    seen.add(b)

            # Corner-last block + its cluster
            if cl is not None and cl_in_chain:
                for b in chain:
                    if b not in seen and b in cl_set:
                        new_chain.append(b)
                        seen.add(b)
                if cl not in seen:
                    new_chain.append(cl)
                    seen.add(cl)
            elif cl is not None and not cl_in_chain:
                # Virtual corner: flush all cluster members together at end
                for m in cluster_members.get(cl_g, []):
                    if m not in seen:
                        new_chain.append(m)
                        seen.add(m)

            # Safety: append anything missed
            for b in chain:
                if b not in seen:
                    new_chain.append(b)
            return new_chain

        # Refine each boundary chain
        cf_left = _find_corner(self._cached_ordered_left, LEFT | BOTTOM)
        cl_left = _find_corner(self._cached_ordered_left, LEFT | TOP)
        self._cached_ordered_left = _refine_one(self._cached_ordered_left, cf_left, cl_left)

        cf_bottom = _find_corner(self._cached_ordered_bottom, LEFT | BOTTOM)
        cl_bottom = _find_corner(self._cached_ordered_bottom, RIGHT | BOTTOM)
        self._cached_ordered_bottom = _refine_one(self._cached_ordered_bottom, cf_bottom, cl_bottom)

        cf_top = _find_corner(self._cached_ordered_top, LEFT | TOP)
        cl_top = _find_corner(self._cached_ordered_top, RIGHT | TOP)
        self._cached_ordered_top = _refine_one(self._cached_ordered_top, cf_top, cl_top)

        # RIGHT chain: BR/TR not in the chain, pass as virtual corners
        self._cached_ordered_right = _refine_one(
            self._cached_ordered_right,
            self._cached_bottom_tail,  # BR — virtual corner-first
            self._cached_top_tail,     # TR — virtual corner-last
        )

    def _debug_print_boundary_orders(self, boundary_codes: Dict[int, int]) -> None:
        if not self.verbose or not boundary_codes:
            return

        def fmt(blocks: List[int]) -> str:
            return "[" + ", ".join(str(block) for block in blocks) + "]"

        print("Boundary chain order:")
        print(f"  bottom: {fmt(self._ordered_bottom_chain_blocks(boundary_codes))}")
        print(f"  left:   {fmt(self._ordered_left_chain_blocks(boundary_codes))}")
        print(f"  top:    {fmt(self._ordered_top_chain_blocks(boundary_codes))}")
        print(f"  right:  {fmt(self._ordered_right_chain_blocks(boundary_codes))}")

    def _boundary_chain_violation(
        self,
        tree: BStarTree,
        boundary_codes: Dict[int, int],
    ) -> float:
        # Structural checker for the active LEFT/BOTTOM chain model.
        if tree.n == 0 or tree.root == -1 or not boundary_codes:
            return 0.0

        bottom_nodes = {block for block, code in boundary_codes.items() if code & BOTTOM}
        left_nodes = {block for block, code in boundary_codes.items() if code & LEFT}
        top_nodes = {block for block, code in boundary_codes.items() if code & TOP}
        right_nodes = {
            block
            for block, code in boundary_codes.items()
            if (code & RIGHT) and not (code & TOP) and not (code & BOTTOM)
        }
        bottom_left_nodes = {block for block, code in boundary_codes.items() if (code & BOTTOM) and (code & LEFT)}
        left_tail = self._left_tail_block(boundary_codes)
        bottom_tail = self._bottom_tail_block(boundary_codes)
        top_root = self._top_root_block(boundary_codes)
        top_tail = self._top_tail_block(boundary_codes)
        right_root = self._right_root_block(boundary_codes)

        penalty = 0.0
        if bottom_left_nodes and tree.root not in bottom_left_nodes:
            penalty += 1.0

        bottom_chain = set()
        node = tree.root
        while node != -1:
            bottom_chain.add(node)
            node = tree.left[node]

        left_chain = set()
        node = tree.root
        while node != -1:
            left_chain.add(node)
            node = tree.right[node]

        penalty += float(len(bottom_nodes - bottom_chain))
        penalty += float(len(left_nodes - left_chain))

        left_chain_list = []
        node = tree.root
        while node != -1:
            left_chain_list.append(node)
            node = tree.right[node]

        bottom_chain_list = []
        node = tree.root
        while node != -1:
            bottom_chain_list.append(node)
            node = tree.left[node]

        if left_tail is not None:
            if left_tail not in left_chain_list:
                penalty += 1.0
            else:
                tail_idx = left_chain_list.index(left_tail)
                penalty += float(sum(1 for block in left_chain_list[tail_idx + 1 :] if boundary_codes.get(block, 0) & LEFT))
        penalty += self._ordered_chain_violation(
            left_chain_list,
            self._ordered_left_chain_blocks(boundary_codes),
        )

        if bottom_tail is not None:
            if bottom_tail not in bottom_chain_list:
                penalty += 1.0
            else:
                tail_idx = bottom_chain_list.index(bottom_tail)
                penalty += float(sum(1 for block in bottom_chain_list[tail_idx + 1 :] if boundary_codes.get(block, 0) & BOTTOM))
        penalty += self._ordered_chain_violation(
            bottom_chain_list,
            self._ordered_bottom_chain_blocks(boundary_codes),
        )

        if top_root is not None:
            top_chain = set()
            node = top_root
            while node != -1:
                top_chain.add(node)
                node = tree.left[node]
            penalty += float(len(top_nodes - top_chain))

            top_chain_list = []
            node = top_root
            while node != -1:
                top_chain_list.append(node)
                node = tree.left[node]

            if top_tail is not None:
                if top_tail not in top_chain_list:
                    penalty += 1.0
                else:
                    tail_idx = top_chain_list.index(top_tail)
                    penalty += float(sum(1 for block in top_chain_list[tail_idx + 1 :] if boundary_codes.get(block, 0) & TOP))
            penalty += self._ordered_chain_violation(
                top_chain_list,
                self._ordered_top_chain_blocks(boundary_codes),
            )

            for block in top_nodes:
                if tree.right[block] != -1:
                    penalty += 1.0

        if right_root is not None:
            right_chain = set()
            node = tree.right[right_root]
            while node != -1:
                right_chain.add(node)
                node = tree.right[node]
            penalty += float(len(right_nodes - right_chain))

            right_chain_list = []
            node = tree.right[right_root]
            while node != -1:
                right_chain_list.append(node)
                node = tree.right[node]
            penalty += self._ordered_chain_violation(
                right_chain_list,
                self._ordered_right_chain_blocks(boundary_codes),
            )
        elif right_nodes:
            ordered_right = self._ordered_right_chain_blocks(boundary_codes)
            first = ordered_right[0] if ordered_right else next(iter(right_nodes))
            right_chain = set()
            node = first
            while node != -1:
                right_chain.add(node)
                node = tree.right[node]
            penalty += float(len(right_nodes - right_chain))
            penalty += self._ordered_chain_violation(
                ordered_right,
                self._ordered_right_chain_blocks(boundary_codes),
            )

        return penalty

    def _ordered_chain_violation(
        self,
        actual_chain: List[int],
        expected_chain: List[int],
    ) -> float:
        if not actual_chain or not expected_chain:
            return 0.0
        actual_positions = {block: idx for idx, block in enumerate(actual_chain)}
        expected_present = [block for block in expected_chain if block in actual_positions]
        penalty = 0.0
        for idx, block in enumerate(expected_present):
            for later in expected_present[idx + 1 :]:
                if actual_positions[block] > actual_positions[later]:
                    penalty += 1.0
        return penalty

    def _enforce_locked_geometry(
        self,
        tree: BStarTree,
        target_rects: Sequence[Rect],
        geometry_locked_blocks: Iterable[int],
    ) -> None:
        # Hard-constraint guard:
        # 1. fixed-shape / preplaced blocks always keep their target (w, h)
        # 2. if a MIB group contains one of those hard-geometry blocks, that
        #    block determines the whole group's aspect ratio.
        active_locked = self._active_geometry_locked_blocks(geometry_locked_blocks)
        for block in active_locked:
            if not (0 <= block < tree.n and 0 <= block < len(target_rects)):
                continue
            _, _, width, height = target_rects[block]
            if width > 0.0 and height > 0.0:
                tree.widths[block] = width
                tree.heights[block] = height
        for blocks, ratio, anchor in self._cached_mib_forced_ratio_groups:
            if anchor not in active_locked:
                continue
            valid_blocks = [block for block in blocks if 0 <= block < tree.n]
            if valid_blocks:
                tree.set_group_aspect(valid_blocks, ratio)
                for block in active_locked:
                    if 0 <= block < tree.n and 0 <= block < len(target_rects):
                        _, _, width, height = target_rects[block]
                        if width > 0.0 and height > 0.0:
                            tree.widths[block] = width
                            tree.heights[block] = height

    def _enforce_locked_rect_geometry(
        self,
        positions: List[Rect],
        target_rects: Sequence[Rect],
        geometry_locked_blocks: Iterable[int],
        immovable_blocks: Optional[Iterable[int]] = None,
    ) -> List[Rect]:
        # Placement-level hard-constraint guard:
        # - fixed-shape blocks keep target (w, h)
        # - preplaced/immovable blocks keep target (w, h), but their (x, y)
        #   should already come from anchors during pack/compaction.
        if not positions:
            return positions
        current = list(positions)
        locked = self._active_geometry_locked_blocks(geometry_locked_blocks)
        for block in locked:
            if not (0 <= block < len(current) and 0 <= block < len(target_rects)):
                continue
            _, _, tw, th = target_rects[block]
            x_pos, y_pos, width, height = current[block]
            if tw > 0.0 and th > 0.0:
                width = tw
                height = th
            current[block] = (x_pos, y_pos, width, height)
        return current

    # ------------------------------------------------------------------
    #  MIB constraint handling
    # ------------------------------------------------------------------
    def _build_mib_geometry_cache(
        self,
        mib_groups: Dict[int, List[int]],
        target_rects: Sequence[Rect],
        geometry_locked_blocks: Iterable[int],
    ) -> None:
        # If a MIB group contains a fixed-shape or preplaced block, that block
        # dictates the group's aspect ratio.
        self._cached_mib_forced_ratio_groups = []
        self._cached_mib_hard_shape_blocks = set()
        self._cached_mib_blocks = set()
        self._cached_mib_group_by_block = [None] * max(len(target_rects), 0)
        locked = set(geometry_locked_blocks)
        for blocks in mib_groups.values():
            self._cached_mib_blocks.update(blocks)
            group_list = list(blocks)
            for block in group_list:
                if 0 <= block < len(self._cached_mib_group_by_block):
                    self._cached_mib_group_by_block[block] = group_list
            anchor = next((block for block in blocks if block in locked), None)
            if anchor is None or anchor >= len(target_rects):
                continue
            _, _, width, height = target_rects[anchor]
            if width <= 0.0 or height <= 0.0:
                continue
            ratio = width / max(height, 1e-9)
            self._cached_mib_forced_ratio_groups.append((group_list, ratio, anchor))
            self._cached_mib_hard_shape_blocks.update(blocks)

    def _preplaced_overlap_penalty(
        self,
        positions: List[Rect],
        preplaced_blocks: Iterable[int],
    ) -> float:
        # Extra hard-style penalty requested by the user: if a preplaced block
        # overlaps any other block, add lambda * fixed-outline area.
        preplaced = [block for block in preplaced_blocks if 0 <= block < len(positions)]
        if not preplaced:
            return 0.0
        outline_width, outline_height = self._cached_terminal_outline
        outline_area = max(outline_width * outline_height, 1.0)
        for block in preplaced:
            x1, y1, w1, h1 = positions[block]
            for other, (x2, y2, w2, h2) in enumerate(positions):
                if other == block:
                    continue
                overlap_x = min(x1 + w1, x2 + w2) - max(x1, x2)
                overlap_y = min(y1 + h1, y2 + h2) - max(y1, y2)
                if overlap_x > 1e-9 and overlap_y > 1e-9:
                    return outline_area
        return 0.0

    def _anchor_penalty(
        self,
        positions: List[Rect],
        preplaced_anchors: Dict[int, Tuple[float, float]],
        dynamic_anchors: Dict[int, Tuple[float, float]],
    ) -> float:
        # Penalize deviations from explicit anchors.
        # In the current active flow this is usually zero because anchors are disabled.
        penalty = 0.0
        anchors = self._combine_anchors(preplaced_anchors, dynamic_anchors)
        for block, (ax, ay) in anchors.items():
            x_pos, y_pos, _, _ = positions[block]
            penalty += abs(x_pos - ax) + abs(y_pos - ay)
        return penalty

    def _estimate_preplaced_anchors(
        self,
        preplaced_blocks: Iterable[int],
        p2b_edges: List[Tuple[int, int, float]],
        pin_positions: List[Tuple[float, float]],
        widths: Sequence[float],
        heights: Sequence[float],
    ) -> Dict[int, Tuple[float, float]]:
        # Legacy anchor estimator from pin locations.
        # Currently not called because preplaced anchors are disabled.
        anchors: Dict[int, Tuple[float, float]] = {}
        preplaced = set(preplaced_blocks)
        if not preplaced:
            return anchors

        pin_weights: Dict[int, List[Tuple[float, float, float]]] = {block: [] for block in preplaced}
        for pin_idx, block_idx, weight in p2b_edges:
            if block_idx not in preplaced or pin_idx < 0:
                continue
            if pin_idx >= len(pin_positions):
                continue
            pin_x, pin_y = pin_positions[pin_idx]
            pin_weights[block_idx].append((pin_x, pin_y, max(weight, 1e-6)))

        for block in preplaced:
            if not pin_weights[block]:
                continue
            total_weight = sum(weight for _, _, weight in pin_weights[block])
            anchor_x = sum(px * weight for px, _, weight in pin_weights[block]) / total_weight
            anchor_y = sum(py * weight for _, py, weight in pin_weights[block]) / total_weight
            anchors[block] = (anchor_x - widths[block] / 2.0, anchor_y - heights[block] / 2.0)
        return anchors

    def _combine_anchors(
        self,
        preplaced_anchors: Dict[int, Tuple[float, float]],
        dynamic_anchors: Dict[int, Tuple[float, float]],
    ) -> Dict[int, Tuple[float, float]]:
        # Merge static anchors with dynamic repair anchors.
        combined = dict(dynamic_anchors)
        combined.update(preplaced_anchors)
        return combined

    # ------------------------------------------------------------------
    #  Input preprocessing helpers
    # ------------------------------------------------------------------
    def _group_indices(self, constraints_rows: List[List[float]], column: int) -> Dict[int, List[int]]:
        # Convert a constraint column into {group_id: [members]}.
        groups: Dict[int, List[int]] = {}
        for idx, row in enumerate(constraints_rows):
            if column >= len(row):
                continue
            group_id = int(row[column])
            if group_id <= 0:
                continue
            groups.setdefault(group_id, []).append(idx)
        return groups

    def _constraint_row(self, constraints, idx: int) -> List[float]:
        # Normalize one tensor row into plain Python floats.
        row = constraints[idx]
        if hasattr(row, "tolist"):
            values = row.tolist()
        else:
            values = list(row)
        return [float(value) for value in values]

    def _tensor_rows(self, tensor) -> Iterable[Sequence[float]]:
        # Iterate over either a tensor-like object or a plain nested list.
        if hasattr(tensor, "tolist"):
            return tensor.tolist()
        return tensor

    def _prepare_target_rects(self, target_positions, block_count: int) -> List[Rect]:
        # Parse evaluator-supplied target geometry once. Soft blocks keep -1s.
        target_rects: List[Rect] = [(-1.0, -1.0, -1.0, -1.0) for _ in range(block_count)]
        if target_positions is None:
            return target_rects
        for idx, row in enumerate(self._tensor_rows(target_positions)):
            if idx >= block_count:
                break
            values = list(row)
            while len(values) < 4:
                values.append(-1.0)
            target_rects[idx] = tuple(float(values[col]) for col in range(4))  # type: ignore[assignment]
        return target_rects

    def _prepare_edges(self, tensor, expected_len: int) -> List[Tuple[int, int, float]]:
        # Strip padding and convert connectivity tensors into Python tuples once.
        edges: List[Tuple[int, int, float]] = []
        for row in self._tensor_rows(tensor):
            if len(row) < expected_len:
                continue
            if float(row[0]) < 0:
                continue
            edges.append((int(row[0]), int(row[1]), float(row[2])))
        return edges

    def _prepare_points(self, tensor) -> List[Tuple[float, float]]:
        # Strip padding and convert pin tensors into Python tuples once.
        points: List[Tuple[float, float]] = []
        for row in self._tensor_rows(tensor):
            if len(row) < 2:
                continue
            if float(row[0]) < 0:
                continue
            points.append((float(row[0]), float(row[1])))
        return points

    def _scalar(self, value) -> float:
        # Read either a tensor scalar or a plain number.
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

    # ------------------------------------------------------------------
    #  Stretch-touching-boundary move helpers
    # ------------------------------------------------------------------
    def _find_touching_boundary_stretch(
        self,
        positions: List[Rect],
        boundary_codes: Dict[int, int],
        locked_blocks: Iterable[int],
    ) -> Tuple[Optional[int], bool]:
        """Pick a TOP or RIGHT boundary block that touches a non-boundary block.
        Returns (block_id, stretch_x) or (None, False)."""
        locked = locked_blocks if isinstance(locked_blocks, set) else set(locked_blocks)
        tol = 1e-6
        try_top_first = random.random() < 0.5

        def _touches(candidates, boundary_mask, check_top):
            for block in candidates:
                bx, by, bw, bh = positions[block]
                for other, (ox, oy, ow, oh) in enumerate(positions):
                    if other == block or other in locked:
                        continue
                    if boundary_codes.get(other, 0) & boundary_mask:
                        continue
                    if check_top:
                        if abs(by + bh - oy) <= tol and bx < ox + ow and bx + bw > ox:
                            return block
                    else:
                        if abs(bx - (ox + ow)) <= tol and by < oy + oh and by + bh > oy:
                            return block
            return None

        def _candidates(mask, exclude_other):
            out = []
            for b, code in boundary_codes.items():
                if (code & mask) and not (code & exclude_other):
                    if b in locked or b in self._cached_immovable_blocks:
                        continue
                    if (self._cached_geometry_locked_flags and b < len(self._cached_geometry_locked_flags)
                            and self._cached_geometry_locked_flags[b]):
                        continue
                    if self._cached_mib_blocks and b in self._cached_mib_blocks:
                        continue
                    out.append(b)
            return out

        top_candidates = _candidates(TOP, RIGHT)
        right_candidates = _candidates(RIGHT, TOP)

        if try_top_first and top_candidates:
            r = _touches(top_candidates, TOP, True)
            if r is not None:
                return r, True
        if right_candidates:
            r = _touches(right_candidates, RIGHT, False)
            if r is not None:
                return r, False
        if not try_top_first and top_candidates:
            r = _touches(top_candidates, TOP, True)
            if r is not None:
                return r, True
        return None, False

    def _apply_stretch_move(
        self,
        tree: BStarTree,
        block: int,
        stretch_x: bool,
        boundary_codes: Dict[int, int],
        mib_groups: Dict[int, List[int]],
    ) -> List[int]:
        """Apply an X-only or Y-only stretch to a single block (MIB already excluded)."""
        if self._touches_macro_cluster([block]):
            return []
        if 0 <= block < len(self._cached_geometry_locked_flags) and self._cached_geometry_locked_flags[block]:
            return []
        if self._cached_mib_blocks and block in self._cached_mib_blocks:
            return []
        low, high = self._default_sampling_window([block])
        ratio = self._sample_group_ratio([block], boundary_codes, low, high)
        tree.set_group_aspect([block], ratio)
        return [block]


# ---------------------------------------------------------------------------
# Module-level worker functions for parallel execution via ProcessPoolExecutor.
# These are defined outside MyOptimizer so they can be pickled by name.
# ---------------------------------------------------------------------------

def _probe_worker(args: Tuple) -> Tuple[float, float, float]:
    """Run one FAST-SA probe trial in a worker process.

    Receives pickled copies of (opt, tree, ...), makes one random move,
    evaluates, and returns (delta, bbox_area, total_wirelength).
    """
    (opt, tree, areas, locked_blocks, mib_groups, cluster_groups,
     boundary_codes, geometry_locked_blocks, immovable_blocks,
     target_rects, b2b_edges, p2b_edges, pin_positions,
     preplaced_anchors, dynamic_anchors, current_cost) = args

    trial_anchors = dict(dynamic_anchors)
    opt._apply_regular_move(
        tree, areas, locked_blocks, mib_groups, cluster_groups,
        boundary_codes, geometry_locked_blocks, stage2_mode=True,
    )
    opt._enforce_top_chain_constraints(tree, boundary_codes)
    anchors = opt._combine_anchors(preplaced_anchors, trial_anchors)
    trial_positions = tree.pack(anchors, boundary_codes)
    trial_positions = opt._pack_compact_with_boundary_repair(
        tree, anchors, boundary_codes, immovable_blocks=immovable_blocks,
    )
    trial_cost = opt._cost(
        trial_positions, b2b_edges, p2b_edges, pin_positions,
        cluster_groups, boundary_codes, areas, preplaced_anchors,
        trial_anchors, immovable_blocks, tree,
    )
    probe_area = contest_api.calculate_bbox_area(trial_positions)
    hpwl_b2b, hpwl_p2b = opt._wirelength_cost(
        trial_positions, b2b_edges, p2b_edges, pin_positions,
    )
    delta = trial_cost - current_cost
    return delta, probe_area, hpwl_b2b + hpwl_p2b


def _cluster_local_sa_worker(args: Tuple) -> Dict[int, Rect]:
    """Run _local_cluster_sa for one cluster in a worker process."""
    opt, tree, group, local_edges = args
    return opt._local_cluster_sa(tree, group, local_edges)



def _fork_worker_from_raw(opt_state: dict, restart_idx: int,
        block_count: int, area_targets, b2b_connectivity,
        p2b_connectivity, pins_pos, constraints, target_positions,
        start_time: float, time_limit: float) -> Dict[str, Any]:
    """Run one restart in a forked child from raw tensors (no shared data)."""
    opt = object.__new__(MyOptimizer)
    opt.__dict__.update(opt_state)
    opt.use_gpu = False
    opt._restart_workers = 1
    data = opt._build_restart_data(
        block_count, area_targets, b2b_connectivity,
        p2b_connectivity, pins_pos, constraints, target_positions,
        start_time, time_limit,
    )
    return opt._solve_one_restart(**data)


if _HAS_CYTHON:
    # Patch MyOptimizer._wirelength_cost to use Cython version
    def _cy_wl(self, positions, b2b_edges, p2b_edges, pin_positions):
        return bstar_core.cy_wirelength(positions, b2b_edges, p2b_edges, pin_positions)
    MyOptimizer._wirelength_cost = _cy_wl

    # Patch contour detection to use Cython O(n^2) version
    MyOptimizer._y_contour_blocks = lambda self, positions, tol=1e-6: bstar_core.cy_y_contour_blocks(positions, tol)
    MyOptimizer._x_contour_blocks = lambda self, positions, tol=1e-6: bstar_core.cy_x_contour_blocks(positions, tol)

    # Patch cluster graph traversal to use Cython
    def _cy_count_touching(self, positions, blocks):
        return bstar_core.cy_count_touching_components(positions, blocks)
    MyOptimizer._count_touching_components = _cy_count_touching

    def _cy_rect_touch(self, rect_a, rect_b):
        return bstar_core.cy_rectangles_touch(rect_a, rect_b)
    MyOptimizer._rectangles_touch = _cy_rect_touch

    # Patch boundary-cluster snap to use Cython
    def _cy_snap(self, positions, boundary_codes):
        return bstar_core.cy_snap_right_top_boundary_clusters(
            positions, boundary_codes,
            self._cached_ordered_top, self._cached_ordered_right,
            self._cached_immovable_blocks, self._cached_group_clusters,
        )
    MyOptimizer._snap_right_top_boundary_clusters = _cy_snap


# =============================================================================
#  Executable entry point (for PyInstaller and subprocess invocation)
# =============================================================================
def main() -> None:
    """Read problem data from stdin (JSON), write positions to stdout (JSON).

    JSON input schema (matches op_wrapper.py payload):
        block_count   int
        area_targets  list[list[float]]   shape (N,)
        b2b_connectivity  list[list[float]]   shape (N, N) or empty
        p2b_connectivity  list[list[float]]   shape (N, P) or empty
        pins_pos      list[list[float]]   shape (P, 2) or empty
        constraints   list[list[float]]   shape (N, C) or empty
        target_positions  list[list[float]] | null  shape (N, 4)

    JSON output schema:
        positions  list[list[float]]  shape (N, 4)  [[x, y, w, h], ...]
    """
    import json
    import torch

    raw = sys.stdin.read()
    if not raw.strip():
        return

    data = json.loads(raw)

    block_count = int(data["block_count"])
    area_targets = torch.tensor(data["area_targets"], dtype=torch.float)
    b2b_conn = torch.tensor(data["b2b_connectivity"], dtype=torch.float) if data["b2b_connectivity"] else torch.empty(0)
    p2b_conn = torch.tensor(data["p2b_connectivity"], dtype=torch.float) if data["p2b_connectivity"] else torch.empty(0)
    pins_pos = torch.tensor(data["pins_pos"], dtype=torch.float) if data["pins_pos"] else torch.empty(0, 2)
    constraints = torch.tensor(data["constraints"], dtype=torch.float) if data["constraints"] else torch.empty(0)
    target_positions = (
        torch.tensor(data["target_positions"], dtype=torch.float)
        if data.get("target_positions") is not None
        else None
    )

    # Redirect stdout to /dev/null during solve so only JSON lands on real stdout
    import io as _io
    _real_stdout = sys.stdout
    _real_stderr = sys.stderr
    sys.stdout = _io.StringIO()
    try:
        opt = MyOptimizer(verbose=False)
        positions = opt.solve(
            block_count, area_targets, b2b_conn, p2b_conn, pins_pos,
            constraints, target_positions,
        )
    finally:
        sys.stdout = _real_stdout

    out = {"positions": [list(p) for p in positions]}
    sys.stdout.write(json.dumps(out))


# ---------------------------------------------------------------------------
# Pickle / ProcessPoolExecutor support.
#
# The contest evaluator loads this file with
#     spec_from_file_location("optimizer_module", path)
# so __name__ is "optimizer_module" -- a name that cannot be imported inside a
# spawned child process on Windows, which is why parallel restarts used to die
# with "No module named 'optimizer_module'".  Re-point every module-level class
# and function at the real file name ("my_optimizer"), which sits next to this
# file and *is* importable by the child, and register both aliases.
# ---------------------------------------------------------------------------
import os as _mod_os
import sys as _mod_sys
import types as _mod_types

_mod_self = _mod_sys.modules.get(__name__)
if _mod_self is None:
    # Loaded through importlib.util.spec_from_file_location(), which does not
    # register the module in sys.modules.
    _mod_self = _mod_types.ModuleType(__name__)
    _mod_self.__dict__.update(globals())
    _mod_sys.modules[__name__] = _mod_self

if __name__ != "my_optimizer":
    _mod_here = _mod_os.path.dirname(_mod_os.path.abspath(__file__))
    if _mod_here and _mod_here not in _mod_sys.path:
        _mod_sys.path.insert(0, _mod_here)
    _mod_sys.modules.setdefault("my_optimizer", _mod_self)
    for _mod_obj in list(globals().values()):
        if getattr(_mod_obj, "__module__", None) == __name__:
            try:
                _mod_obj.__module__ = "my_optimizer"
            except (AttributeError, TypeError):
                pass

if __name__ == "__main__":
    main()
