# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython-optimized B*-tree pack and wirelength cost."""

from libc.math cimport fabs


cdef double _get_contour_y(list cx, list cy, double x_start, double x_end) noexcept:
    cdef int i
    cdef int n = len(cx)
    cdef double max_y = 0.0, cx_start, cx_end_val, cy_top
    for i in range(n):
        cx_end_val = cx[i]
        cy_top = cy[i]
        if i > 0:
            cx_start = cx[i - 1]
        else:
            cx_start = 0.0
        if x_start < cx_end_val and x_end > cx_start:
            if cy_top > max_y:
                max_y = cy_top
    return max_y


cdef void _update_contour(list cx, list cy, double x_start, double x_end, double y_top) noexcept:
    cdef int i, n, insert_pos, j, m
    cdef double cx_start, cx_end_val, cy_top, cx_val, cy_val, key_x, key_y, last_y
    cdef list new_cx = []
    cdef list new_cy = []

    n = len(cx)
    for i in range(n):
        cx_end_val = cx[i]
        cy_top = cy[i]
        if i > 0:
            cx_start = cx[i - 1]
        else:
            cx_start = 0.0
        if cx_end_val <= x_start or cx_start >= x_end:
            new_cx.append(cx_end_val)
            new_cy.append(cy_top)
            continue
        if cx_start < x_start:
            new_cx.append(x_start)
            new_cy.append(cy_top)
        if cx_end_val > x_end:
            new_cx.append(cx_end_val)
            new_cy.append(cy_top)

    insert_pos = 0
    for i in range(len(new_cx)):
        if new_cx[i] <= x_start:
            insert_pos = i + 1
    new_cx.insert(insert_pos, x_end)
    new_cy.insert(insert_pos, y_top)

    # Insertion sort
    m = len(new_cx)
    for i in range(1, m):
        key_x = new_cx[i]
        key_y = new_cy[i]
        j = i - 1
        while j >= 0 and new_cx[j] > key_x:
            new_cx[j+1] = new_cx[j]
            new_cy[j+1] = new_cy[j]
            j -= 1
        new_cx[j+1] = key_x
        new_cy[j+1] = key_y

    # Merge adjacent segments with same y
    cx.clear()
    cy.clear()
    for i in range(len(new_cx)):
        cx_val = new_cx[i]
        cy_val = new_cy[i]
        if cx:
            if fabs(cy[len(cy)-1] - cy_val) <= 1e-9:
                cx[len(cx)-1] = cx_val
                continue
        cx.append(cx_val)
        cy.append(cy_val)


cdef int _dfs(int node, int root, list parent, list left, list right,
               list widths, list heights,
               list positions,
               list placed_x, list placed_y, list placed_w, list placed_h,
               list cx, list cy, dict anchors) noexcept:
    cdef double x_pos, y_pos, width, height
    cdef double px, py, pw, ph, current_y, overlap_x, overlap_y, y_above
    cdef int p, i, n_placed
    cdef bint anchored, found_overlap
    cdef int guard = 0

    if node < 0:
        return 0
    width = widths[node]
    height = heights[node]

    anchored = node in anchors
    if anchored:
        _px, _py = <object>anchors[node]
        x_pos = _px
        y_pos = _py
        positions[node] = (x_pos, y_pos, width, height)
        placed_x.append(x_pos)
        placed_y.append(y_pos)
        placed_w.append(width)
        placed_h.append(height)
        _update_contour(cx, cy, x_pos, x_pos + width, y_pos + height)
        _dfs(left[node], root, parent, left, right, widths, heights,
             positions, placed_x, placed_y, placed_w, placed_h, cx, cy, anchors)
        _dfs(right[node], root, parent, left, right, widths, heights,
             positions, placed_x, placed_y, placed_w, placed_h, cx, cy, anchors)
        return 0

    if node == root:
        x_pos = 0.0
        y_pos = 0.0
    else:
        p = parent[node]
        _tup = <tuple>positions[p]
        px = _tup[0]
        py = _tup[1]
        pw = _tup[2]
        ph = _tup[3]
        if left[p] == node:
            x_pos = px + pw
            y_pos = 0.0
        else:
            x_pos = px
            y_pos = py + ph

    # legalize_y with guard
    current_y = _get_contour_y(cx, cy, x_pos, x_pos + width)
    if y_pos > current_y:
        current_y = y_pos
    while guard < 10000:
        guard += 1
        n_placed = len(placed_x)
        y_above = 0.0
        found_overlap = False
        for i in range(n_placed):
            px = placed_x[i]
            py = placed_y[i]
            pw = placed_w[i]
            ph = placed_h[i]
            overlap_x = (x_pos + width if x_pos + width < px + pw else px + pw) - (x_pos if x_pos > px else px)
            if overlap_x <= 1e-9:
                continue
            overlap_y = (current_y + height if current_y + height < py + ph else py + ph) - (current_y if current_y > py else py)
            if overlap_y > 1e-9:
                found_overlap = True
                if py + ph > y_above:
                    y_above = py + ph
        if not found_overlap:
            break
        current_y = y_above

    y_pos = current_y
    positions[node] = (x_pos, y_pos, width, height)
    placed_x.append(x_pos)
    placed_y.append(y_pos)
    placed_w.append(width)
    placed_h.append(height)
    _update_contour(cx, cy, x_pos, x_pos + width, y_pos + height)

    _dfs(left[node], root, parent, left, right, widths, heights,
         positions, placed_x, placed_y, placed_w, placed_h, cx, cy, anchors)
    _dfs(right[node], root, parent, left, right, widths, heights,
         positions, placed_x, placed_y, placed_w, placed_h, cx, cy, anchors)
    return 0


def cy_pack(int n, int root, list parent, list left, list right,
            list widths, list heights, list active, dict anchors=None):
    """Cython-optimized B*-tree pack. Returns list of (x, y, w, h)."""
    cdef int i
    cdef list positions = [(0.0, 0.0, widths[i], heights[i]) for i in range(n)]
    cdef list placed_x = []
    cdef list placed_y = []
    cdef list placed_w = []
    cdef list placed_h = []
    cdef list contour_x = [0.0]
    cdef list contour_y = [0.0]
    cdef dict _anchors = anchors or {}

    _dfs(root, root, parent, left, right, widths, heights,
         positions, placed_x, placed_y, placed_w, placed_h,
         contour_x, contour_y, _anchors)
    return positions


def cy_wirelength(list positions, list b2b_edges, list p2b_edges, list pin_positions):
    """Cython-optimized wirelength HPWL computation."""
    cdef int n_pos = len(positions)
    cdef int i, block_i, block_j, pin_idx, block_idx
    cdef double weight, x1, y1, x2, y2, px, py, bx, by
    cdef double hpwl_b2b = 0.0, hpwl_p2b = 0.0
    cdef list centers = [None] * n_pos

    for i in range(n_pos):
        _tup = <tuple>positions[i]
        x1 = _tup[0]
        y1 = _tup[1]
        cx = _tup[2]
        cy = _tup[3]
        centers[i] = (x1 + cx / 2.0, y1 + cy / 2.0)

    for i in range(len(b2b_edges)):
        _tup = <tuple>b2b_edges[i]
        block_i = <int>_tup[0]
        block_j = <int>_tup[1]
        weight = _tup[2]
        _ci = <tuple>centers[block_i]
        _cj = <tuple>centers[block_j]
        x1 = _ci[0]
        y1 = _ci[1]
        x2 = _cj[0]
        y2 = _cj[1]
        hpwl_b2b += weight * (fabs(x2 - x1) + fabs(y2 - y1))

    for i in range(len(p2b_edges)):
        _tup = <tuple>p2b_edges[i]
        pin_idx = <int>_tup[0]
        block_idx = <int>_tup[1]
        weight = _tup[2]
        if pin_idx >= len(pin_positions) or block_idx >= n_pos:
            continue
        _pin = <tuple>pin_positions[pin_idx]
        px = _pin[0]
        py = _pin[1]
        _cb = <tuple>centers[block_idx]
        bx = _cb[0]
        by = _cb[1]
        hpwl_p2b += weight * (fabs(px - bx) + fabs(py - by))

    return hpwl_b2b, hpwl_p2b


# ---------------------------------------------------------------------------
#  Contour detection — O(n^2) with pure C loops and C arrays.
# ---------------------------------------------------------------------------
from libc.stdlib cimport malloc, free

cdef bint _fully_exposed_c(double start, double end,
                            double *c_starts, double *c_ends, int c_count,
                            double tol) noexcept:
    cdef int i
    cdef double covered_len = 0.0
    cdef double left, right, m_last_end = -1e100

    if end - start <= tol:
        return False

    for i in range(c_count):
        left = c_starts[i] if c_starts[i] > start else start
        right = c_ends[i] if c_ends[i] < end else end
        if right - left <= tol:
            continue
        if left > m_last_end + tol:
            covered_len += right - left
            m_last_end = right
        elif right > m_last_end:
            covered_len += right - m_last_end
            m_last_end = right

    return covered_len <= tol


def cy_y_contour_blocks(list positions, double tol = 1e-6):
    """Y-contour: blocks whose top edge has some exposed x-interval (O(n^2), pure C)."""
    cdef int n = len(positions), i, j, c_count
    cdef double *xs = <double*>malloc(n * sizeof(double))
    cdef double *ys = <double*>malloc(n * sizeof(double))
    cdef double *ws = <double*>malloc(n * sizeof(double))
    cdef double *hs = <double*>malloc(n * sizeof(double))
    cdef double *c_starts = <double*>malloc(n * sizeof(double))
    cdef double *c_ends = <double*>malloc(n * sizeof(double))
    cdef set contour = set()
    cdef double top, ox, oy, ow, oh, overlap_start, overlap_end

    if not xs or not ys or not ws or not hs or not c_starts or not c_ends:
        free(xs); free(ys); free(ws); free(hs); free(c_starts); free(c_ends)
        raise MemoryError()

    for i in range(n):
        xs[i], ys[i], ws[i], hs[i] = <tuple>positions[i]

    for i in range(n):
        top = ys[i] + hs[i]
        c_count = 0
        for j in range(n):
            if i == j:
                continue
            oy = ys[j]; ow = ws[j]; oh = hs[j]; ox = xs[j]
            if oy > top + tol:
                continue
            if oy + oh <= top + tol:
                continue
            overlap_start = xs[i] if xs[i] > ox else ox
            overlap_end = (xs[i] + ws[i]) if (xs[i] + ws[i]) < (ox + ow) else (ox + ow)
            if overlap_end - overlap_start > tol:
                c_starts[c_count] = overlap_start
                c_ends[c_count] = overlap_end
                c_count += 1
        if _fully_exposed_c(xs[i], xs[i] + ws[i], c_starts, c_ends, c_count, tol):
            contour.add(i)

    free(xs); free(ys); free(ws); free(hs); free(c_starts); free(c_ends)
    return contour


def cy_x_contour_blocks(list positions, double tol = 1e-6):
    """Right-contour: blocks whose right edge has some exposed y-interval (O(n^2), pure C)."""
    cdef int n = len(positions), i, j, c_count
    cdef double *xs = <double*>malloc(n * sizeof(double))
    cdef double *ys = <double*>malloc(n * sizeof(double))
    cdef double *ws = <double*>malloc(n * sizeof(double))
    cdef double *hs = <double*>malloc(n * sizeof(double))
    cdef double *c_starts = <double*>malloc(n * sizeof(double))
    cdef double *c_ends = <double*>malloc(n * sizeof(double))
    cdef set contour = set()
    cdef double right, ox, oy, ow, oh, other_right, overlap_start, overlap_end

    if not xs or not ys or not ws or not hs or not c_starts or not c_ends:
        free(xs); free(ys); free(ws); free(hs); free(c_starts); free(c_ends)
        raise MemoryError()

    for i in range(n):
        xs[i], ys[i], ws[i], hs[i] = <tuple>positions[i]

    for i in range(n):
        right = xs[i] + ws[i]
        c_count = 0
        for j in range(n):
            if i == j:
                continue
            ox = xs[j]; ow = ws[j]; oy = ys[j]; oh = hs[j]
            other_right = ox + ow
            if other_right <= right + tol:
                continue
            if ox >= right - tol:
                continue
            overlap_start = ys[i] if ys[i] > oy else oy
            overlap_end = (ys[i] + hs[i]) if (ys[i] + hs[i]) < (oy + oh) else (oy + oh)
            if overlap_end - overlap_start > tol:
                c_starts[c_count] = overlap_start
                c_ends[c_count] = overlap_end
                c_count += 1
        if _fully_exposed_c(ys[i], ys[i] + hs[i], c_starts, c_ends, c_count, tol):
            contour.add(i)

    free(xs); free(ys); free(ws); free(hs); free(c_starts); free(c_ends)
    return contour


def cy_copy_tree(int n, list areas, list widths, list heights,
                 list parent, list left, list right, int root, list active):
    """Fast BStarTree copy."""
    return (areas[:], widths[:], heights[:], parent[:], left[:], right[:], root, active[:])


# ---------------------------------------------------------------------------
#  Compaction helpers — contour query/update, shift, bbox.
# ---------------------------------------------------------------------------

def cy_query_y_contour(list contour, double x_start, double x_end, double tol=1e-9):
    """Query y-contour for max y within [x_start, x_end]."""
    cdef int i, n = len(contour)
    cdef double max_y = 0.0, cx_start, cx_end_val, cy_top
    for i in range(n):
        _tup = <tuple>contour[i]
        cx_end_val = _tup[0]
        cy_top = _tup[1]
        if i > 0:
            _tup2 = <tuple>contour[i-1]
            cx_start = _tup2[0]
        else:
            cx_start = 0.0
        if x_start < cx_end_val - tol and x_end > cx_start + tol:
            if cy_top > max_y:
                max_y = cy_top
    return max_y


def cy_update_y_contour(list contour, double x_start, double x_end, double y_top, double tol=1e-9):
    """Update y-contour with a block's top edge. Returns new contour list."""
    cdef int i, n = len(contour)
    cdef double cx_start, cx_end_val, cy_top, replaced_max, x_val, y_val
    cdef list new_contour = []

    replaced_max = y_top
    for i in range(n):
        _tup = <tuple>contour[i]
        cx_end_val = _tup[0]
        cy_top = _tup[1]
        if i > 0:
            _tup2 = <tuple>contour[i-1]
            cx_start = _tup2[0]
        else:
            cx_start = 0.0
        if cx_end_val <= x_start + tol or cx_start >= x_end - tol:
            new_contour.append((cx_end_val, cy_top))
        else:
            if cy_top > replaced_max:
                replaced_max = cy_top
            if cx_start < x_start - tol:
                new_contour.append((x_start, cy_top))
            if cx_end_val > x_end + tol:
                new_contour.append((cx_end_val, cy_top))
    new_contour.append((x_end, replaced_max))
    new_contour.sort(key=lambda item: item[0])
    # Merge adjacent segments with same y
    cdef list merged = []
    for x_val, y_val in new_contour:
        if merged and fabs(merged[len(merged)-1][1] - y_val) <= tol:
            merged[len(merged)-1] = (x_val, y_val)
        else:
            merged.append((x_val, y_val))
    return merged


def cy_query_x_contour(list contour, double y_start, double y_end, double tol=1e-9):
    """Query x-contour for max x within [y_start, y_end]."""
    cdef int i, n = len(contour)
    cdef double max_x = 0.0, cy_start, cy_end_val, cx_right
    for i in range(n):
        _tup = <tuple>contour[i]
        cy_end_val = _tup[0]
        cx_right = _tup[1]
        if i > 0:
            _tup2 = <tuple>contour[i-1]
            cy_start = _tup2[0]
        else:
            cy_start = 0.0
        if y_start < cy_end_val - tol and y_end > cy_start + tol:
            if cx_right > max_x:
                max_x = cx_right
    return max_x


def cy_update_x_contour(list contour, double y_start, double y_end, double x_right, double tol=1e-9):
    """Update x-contour with a block's right edge. Returns new contour list."""
    cdef int i, n = len(contour)
    cdef double cy_start, cy_end_val, cx_top, replaced_max, y_val, x_val
    cdef list new_contour = []

    replaced_max = x_right
    for i in range(n):
        _tup = <tuple>contour[i]
        cy_end_val = _tup[0]
        cx_top = _tup[1]
        if i > 0:
            _tup2 = <tuple>contour[i-1]
            cy_start = _tup2[0]
        else:
            cy_start = 0.0
        if cy_end_val <= y_start + tol or cy_start >= y_end - tol:
            new_contour.append((cy_end_val, cx_top))
        else:
            if cx_top > replaced_max:
                replaced_max = cx_top
            if cy_start < y_start - tol:
                new_contour.append((y_start, cx_top))
            if cy_end_val > y_end + tol:
                new_contour.append((cy_end_val, cx_top))
    new_contour.append((y_end, replaced_max))
    new_contour.sort(key=lambda item: item[0])
    # Merge adjacent segments with same x
    cdef list merged = []
    for y_val, x_val in new_contour:
        if merged and fabs(merged[len(merged)-1][1] - x_val) <= tol:
            merged[len(merged)-1] = (y_val, x_val)
        else:
            merged.append((y_val, x_val))
    return merged


def cy_shift_right_of_preplaced(double x_pos, double y_pos, double width, double height,
                                 list rects, list support, double tol=1e-9):
    """Shift x_pos right of overlapping preplaced blocks. Returns new x."""
    cdef double current_x = x_pos, shifted_x, other_x, other_y, other_w, other_h
    cdef int i, n = len(support)
    cdef double overlap_x, overlap_y
    while True:
        shifted_x = current_x
        for i in range(n):
            other = support[i]
            if not (0 <= other < len(rects)):
                continue
            _tup = <tuple>rects[other]
            other_x = _tup[0]; other_y = _tup[1]; other_w = _tup[2]; other_h = _tup[3]
            # vertical overlap check
            if (current_x + width if current_x + width < other_x + other_w else other_x + other_w) - (current_x if current_x > other_x else other_x) <= tol:
                continue
            if (y_pos + height if y_pos + height < other_y + other_h else other_y + other_h) - (y_pos if y_pos > other_y else other_y) > tol:
                right = other_x + other_w
                if right > shifted_x:
                    shifted_x = right
        if shifted_x <= current_x + tol:
            return current_x
        current_x = shifted_x


def cy_shift_above_preplaced(double x_pos, double y_pos, double width, double height,
                              list rects, list support, double tol=1e-9):
    """Shift y_pos above overlapping preplaced blocks. Returns new y."""
    cdef double current_y = y_pos, shifted_y, other_x, other_y, other_w, other_h
    cdef int i, n = len(support)
    cdef double overlap_x, overlap_y
    while True:
        shifted_y = current_y
        for i in range(n):
            other = support[i]
            if not (0 <= other < len(rects)):
                continue
            _tup = <tuple>rects[other]
            other_x = _tup[0]; other_y = _tup[1]; other_w = _tup[2]; other_h = _tup[3]
            # horizontal overlap check
            if (x_pos + width if x_pos + width < other_x + other_w else other_x + other_w) - (x_pos if x_pos > other_x else other_x) <= tol:
                continue
            if (current_y + height if current_y + height < other_y + other_h else other_y + other_h) - (current_y if current_y > other_y else other_y) > tol:
                top = other_y + other_h
                if top > shifted_y:
                    shifted_y = top
        if shifted_y <= current_y + tol:
            return current_y
        current_y = shifted_y


def cy_bbox_dims(list rects):
    """Compute (area, width, height) of bounding box."""
    cdef int n = len(rects)
    cdef int i
    cdef double bbox_w = 0.0, bbox_h = 0.0, x, y, w, h
    for i in range(n):
        _tup = <tuple>rects[i]
        x = _tup[0]; y = _tup[1]; w = _tup[2]; h = _tup[3]
        r = x + w
        if r > bbox_w:
            bbox_w = r
        t = y + h
        if t > bbox_h:
            bbox_h = t
    return bbox_w * bbox_h, bbox_w, bbox_h


def cy_horizontal_overlap(rect_a, rect_b, double tol=1e-9):
    """Check if two rects horizontally overlap at given positions."""
    cdef double ax, aw, bx, bw
    if len(rect_a) < 4 or len(rect_b) < 4:
        return False
    _tup = <tuple>rect_a
    ax = _tup[0]; aw = _tup[2]
    _tup2 = <tuple>rect_b
    bx = _tup2[0]; bw = _tup2[2]
    return (ax + aw if ax + aw < bx + bw else bx + bw) - (ax if ax > bx else bx) > tol


# ---------------------------------------------------------------------------
#  Overlap / connectivity helpers
# ---------------------------------------------------------------------------

def cy_overlaps_any(list rects, int block, double tol=1e-6):
    """Check if block at index `block` overlaps any other rect in rects."""
    cdef int n = len(rects), other
    cdef double x1, y1, w1, h1, x2, y2, w2, h2, overlap_x, overlap_y
    _tup = <tuple>rects[block]
    x1 = _tup[0]; y1 = _tup[1]; w1 = _tup[2]; h1 = _tup[3]
    for other in range(n):
        if other == block:
            continue
        _tup2 = <tuple>rects[other]
        x2 = _tup2[0]; y2 = _tup2[1]; w2 = _tup2[2]; h2 = _tup2[3]
        overlap_x = (x1 + w1 if x1 + w1 < x2 + w2 else x2 + w2) - (x1 if x1 > x2 else x2)
        overlap_y = (y1 + h1 if y1 + h1 < y2 + h2 else y2 + h2) - (y1 if y1 > y2 else y2)
        if overlap_x > tol and overlap_y > tol:
            return True
    return False


def cy_count_touching_components(list positions, list blocks, double tol=1e-6):
    """Count connected components in the touching graph of one cluster."""
    cdef int comp = 0, remaining_count, n_blocks, i, start, current, other
    cdef list remaining = list(blocks)
    cdef list stack
    cdef double x1, y1, w1, h1, x2, y2, w2, h2, ox, oy
    remaining_count = len(remaining)
    while remaining_count > 0:
        comp += 1
        start = remaining.pop()
        remaining_count -= 1
        stack = [start]
        while stack:
            current = stack.pop()
            _tup = <tuple>positions[current]
            x1 = _tup[0]; y1 = _tup[1]; w1 = _tup[2]; h1 = _tup[3]
            i = 0
            while i < remaining_count:
                other = remaining[i]
                _tup2 = <tuple>positions[other]
                x2 = _tup2[0]; y2 = _tup2[1]; w2 = _tup2[2]; h2 = _tup2[3]
                ox = (x1 + w1 if x1 + w1 < x2 + w2 else x2 + w2) - (x1 if x1 > x2 else x2)
                oy = (y1 + h1 if y1 + h1 < y2 + h2 else y2 + h2) - (y1 if y1 > y2 else y2)
                if ox >= -tol and oy >= -tol:
                    # touching — remove and push
                    remaining.pop(i)
                    remaining_count -= 1
                    stack.append(other)
                else:
                    i += 1
    return comp


def cy_rectangles_touch(rect_a, rect_b, double tol=1e-6):
    """Check if two rects touch (overlap or edge contact)."""
    cdef double ax, ay, aw, ah, bx, by, bw, bh, ox, oy
    _tup = <tuple>rect_a; ax = _tup[0]; ay = _tup[1]; aw = _tup[2]; ah = _tup[3]
    _tup2 = <tuple>rect_b; bx = _tup2[0]; by = _tup2[1]; bw = _tup2[2]; bh = _tup2[3]
    ox = (ax + aw if ax + aw < bx + bw else bx + bw) - (ax if ax > bx else bx)
    oy = (ay + ah if ay + ah < by + bh else by + bh) - (ay if ay > by else by)
    return ox >= -tol and oy >= -tol


def cy_vertical_overlap(rect_a, rect_b, double tol=1e-9):
    """Check if two rects vertically overlap at given positions."""
    cdef double ay, ah, by, bh
    if len(rect_a) < 4 or len(rect_b) < 4:
        return False
    _tup = <tuple>rect_a
    ay = _tup[1]; ah = _tup[3]
    _tup2 = <tuple>rect_b
    by = _tup2[1]; bh = _tup2[3]
    return (ay + ah if ay + ah < by + bh else by + bh) - (ay if ay > by else by) > tol


# ---------------------------------------------------------------------------
#  Boundary-cluster snap toward TR corner
# ---------------------------------------------------------------------------

cdef int _find_block_index(list ordered, int block):
    """Return index of block in ordered list, or -1."""
    cdef int i, n = len(ordered)
    for i in range(n):
        if ordered[i] == block:
            return i
    return -1


cdef bint _check_overlap_snap(double bx, double by, double bw, double bh,
                                list current, int skip, double tol) noexcept:
    """Check if (bx, by, bw, bh) overlaps any rect in current except skip."""
    cdef int j, n = len(current)
    cdef double ox, oy, ow, oh, overlap_x, overlap_y
    for j in range(n):
        if j == skip:
            continue
        _tup = <tuple>current[j]
        ox = _tup[0]; oy = _tup[1]; ow = _tup[2]; oh = _tup[3]
        overlap_x = (bx + bw if bx + bw < ox + ow else ox + ow) - (bx if bx > ox else ox)
        overlap_y = (by + bh if by + bh < oy + oh else oy + oh) - (by if by > oy else oy)
        if overlap_x > tol and overlap_y > tol:
            return True
    return False


def cy_snap_right_top_boundary_clusters(
    list positions,
    dict boundary_codes,
    list ordered_top,
    list ordered_right,
    set immovable_blocks,
    dict group_clusters,
    double tol=1e-6,
):
    """Snap same-cluster TOP/RIGHT chain blocks toward the TR corner."""
    cdef int tr_idx, block, ref, group_id, i, other
    cdef double tx, ty, tw, th, rx, ry, rw, rh
    cdef double bx, by, bw, bh, new_bx, new_by
    cdef double snap_y, tbx, tby, tbw, tbh
    cdef bint overlap
    cdef list current = list(positions)
    cdef list tr_blocks = []

    # Build block-to-group lookup
    cdef dict block_to_group = {}
    for group_id, members in group_clusters.items():
        for m in members:
            block_to_group[m] = group_id

    # Find TR corner blocks (TOP & RIGHT)
    for block, code in boundary_codes.items():
        if (code & 3) == 3:  # TOP=1, RIGHT=2, TOP|RIGHT=3
            tr_blocks.append(block)

    if not tr_blocks:
        return current

    for tr_block in tr_blocks:
        group_id = block_to_group.get(tr_block, -1)
        if group_id < 0:
            continue
        _tup = <tuple>current[tr_block]
        tx = _tup[0]; ty = _tup[1]; tw = _tup[2]; th = _tup[3]

        # --- TOP chain: snap blocks rightward (toward TR) ---
        if ordered_top:
            tr_idx = _find_block_index(ordered_top, tr_block)
            if tr_idx >= 0:
                for i in range(tr_idx - 1, -1, -1):
                    block = ordered_top[i]
                    if block_to_group.get(block, -1) != group_id:
                        break
                    if block in immovable_blocks:
                        break
                    ref = ordered_top[i + 1]
                    _tup2 = <tuple>current[ref]
                    rx = _tup2[0]; ry = _tup2[1]; rw = _tup2[2]; rh = _tup2[3]
                    _tup3 = <tuple>current[block]
                    bx = _tup3[0]; by = _tup3[1]; bw = _tup3[2]; bh = _tup3[3]
                    new_bx = rx - bw
                    if (bx - new_bx if bx > new_bx else new_bx - bx) > tol:
                        current[block] = (new_bx, by, bw, bh)
            else:
                break

        # --- RIGHT chain: snap blocks upward (toward TR) ---
        if ordered_right:
            snap_y = ty
            for i in range(len(ordered_right) - 1, -1, -1):
                block = ordered_right[i]
                if block_to_group.get(block, -1) != group_id:
                    break
                if block in immovable_blocks:
                    continue
                _tup4 = <tuple>current[block]
                bx = _tup4[0]; by = _tup4[1]; bw = _tup4[2]; bh = _tup4[3]
                new_by = snap_y - bh
                if (by - new_by if by > new_by else new_by - by) <= tol:
                    snap_y = by
                    continue
                # Adjust downward if overlapping same-cluster TOP chain block
                for _tb in ordered_top:
                    if block_to_group.get(_tb, -1) == group_id:
                        _tup5 = <tuple>current[_tb]
                        tbx = _tup5[0]; tby = _tup5[1]; tbw = _tup5[2]; tbh = _tup5[3]
                        if (bx < tbx + tbw and bx + bw > tbx and
                            new_by < tby + tbh and new_by + bh > tby):
                            if tby + tbh > new_by:
                                new_by = tby + tbh
                if not _check_overlap_snap(bx, new_by, bw, bh, current, block, tol):
                    current[block] = (bx, new_by, bw, bh)
                    snap_y = new_by
                else:
                    continue

    return current

