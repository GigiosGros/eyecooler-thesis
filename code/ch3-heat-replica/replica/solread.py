"""Read the fields COMSOL stored in an .mph file: temperature at every mesh node, per domain.

Two files in the archive are involved:

  * ``solutionstatic*.mphbin`` holds a solution as one long vector in COMSOL's
    own numbering of the unknowns (preceded by the list of unknown numbers it
    contains);
  * ``xmesh*.mphbin`` defines that numbering.  For every domain it stores the
    list of the domain's mesh nodes and, for each variable solved there, the
    unknown number at each of those nodes.  Variables come in the order
    temperature, pressure, velocity x, y, z.

Because the map is stored per domain, a node shared by two domains appears in
both; where the model has a temperature jump (insulated interior faces) the two
domains hold different unknowns for the same node, so jumps are recovered too.
"""
import struct

import numpy as np


def read_vector(blob):
    """(numbers, values): the unknown numbers stored in a SolutionStatic file and their values."""
    start = None
    for p in range(100, 400):
        n = struct.unpack_from("<i", blob, p)[0]
        if 10000 < n < 50_000_000 and p + 4 + 4 * n < len(blob):
            a = np.frombuffer(blob, "<i4", 8, p + 4)
            if (np.diff(a) > 0).all() and a[0] < 10:
                start = p
                break
    if start is None:
        raise ValueError("solution header not recognised")
    big = np.frombuffer(blob, "<i4", (len(blob) - start - 4) // 4, start + 4)
    stop = np.where(np.diff(big.astype(np.int64)) <= 0)[0]
    n = int(stop[0]) + 1 if len(stop) else len(big)
    numbers = big[:n]
    q = start + 4 + 4 * n
    for skip in range(0, 64):                    # the float64 block follows a short header of varying length
        a = np.frombuffer(blob, "<f8", 4000, q + skip)
        if np.isfinite(a).all() and (np.abs(a) < 1e7).all() and ((a > 250) & (a < 330)).mean() > 0.05:
            vals = np.frombuffer(blob, "<f8", (len(blob) - q - skip) // 8, q + skip)
            return numbers, vals
    raise ValueError("solution values not found")


def domain_maps(xmesh, mesh):
    """For each domain of ``mesh``: (nodes, list of unknown-number arrays, one per variable).

    The domain's node list is located in the xmesh file by its content (it is
    the only run of that length holding exactly that set of node numbers); the
    unknown-number arrays are the equally long arrays stored just before it.
    """
    out = {}
    views = [np.frombuffer(xmesh[al:al + (len(xmesh) - al) // 4 * 4], "<i4") for al in range(4)]
    for d in np.unique(mesh["tet_dom"]):
        nodes = np.unique(mesh["tets"][mesh["tet_dom"] == d])
        n = len(nodes)
        found = None
        for a in views:
            for i in np.where(a == n)[0]:
                seg = a[i + 1:i + 1 + n]
                if len(seg) == n and seg.min() == nodes[0] and seg.max() == nodes[-1] and np.array_equal(np.sort(seg), nodes):
                    found = (a, int(i))
                    break
            if found:
                break
        if not found:
            continue
        a, i = found
        arrays = []
        j = i - 5                                # the node list has a 5-integer header ending with n
        while True:                              # walk back over arrays of n values, each with a header ending with n
            k = None
            for h in (0, 1, 2, 3, 4):            # header lengths vary by a few integers between arrays
                if j - h - n >= 1 and a[j - h - n] == n:
                    k = j - h - n
                    break
            if k is None:
                break
            arrays.append(a[k + 1:k + 1 + n])
            j = k - 1
            if len(arrays) >= 5:
                break
        out[int(d)] = (a[i + 1:i + 1 + n], arrays[::-1])
    return out


def temperature(numbers, values, maps, n_nodes):
    """{domain: (nodes, T)} from the stored vector, plus one nodal array (last domain wins at shared nodes)."""
    per, T = {}, np.full(n_nodes, np.nan)
    for d, (nodes, arrays) in maps.items():
        if not arrays:
            continue
        pos = np.searchsorted(numbers, arrays[0])
        ok = (pos < len(numbers)) & (numbers[np.minimum(pos, len(numbers) - 1)] == arrays[0])
        t = np.full(len(nodes), np.nan)
        t[ok] = values[pos[ok]]
        per[d] = (nodes, t)
        T[nodes] = t
    return per, T
