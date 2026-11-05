"""Read a COMSOL .mph archive without a COMSOL licence.

An .mph file is a zip archive.  This module pulls out of it everything the
replica needs to pose the same problem COMSOL solved:

  * the finite-element mesh of a component (nodes, tetrahedra with their domain
    number, boundary triangles with their boundary number and the two domains
    they separate) from ``mesh*.mphbin``;
  * named selections, global parameters, materials and their property
    functions, physics features and study settings from ``dmodel.xml`` and
    ``smodel.json``.

Entity numbers returned here are the 1-based numbers shown in the COMSOL GUI.
"""
import json
import re
import struct
import xml.etree.ElementTree as ET
import zipfile

import numpy as np


# --------------------------------------------------------------------- mesh
def _key(name):
    return struct.pack("<i", len(name)) + b"".join(struct.pack("<i", ord(c)) for c in name)


def parse_mesh(b):
    """Decode a serialized COMSOL Mesh object (version 7 binary layout).

    Layout (little-endian; strings are an int32 length then int32 characters):
      "Mesh", version, sdim, 6 bytes, n_nodes, lowest index, sdim,
      then for each coordinate: n_nodes, n_nodes doubles;
      n_types, then per element type: name, nodes per element,
      for each local node: n_elem, n_elem node indices;
      a parameter block, the geometric entity index of each element and, for
      boundary elements, the up/down domain numbers.
    """
    i32 = lambda o, n=1: np.frombuffer(b, "<i4", n, o)
    o = b.find(_key("Mesh")) + 20
    version, sdim = i32(o, 2)
    if sdim != 3:
        raise ValueError(f"expected a 3-D mesh, got sdim={sdim}")
    # The node block is (sdim, n_nodes, x..., n_nodes, y..., n_nodes, z...) and
    # ends right before the element-type count that precedes "vtx".  What
    # comes before it differs between file versions, so it is located by
    # that signature.  Node indices are 0-based.
    vtx = b.find(_key("vtx"), o)
    start = None
    lowest = 0
    for p in range(o + 8, min(vtx, o + 4_000_000)):
        rows, nv = struct.unpack_from("<2i", b, p)
        if rows == sdim and nv > 3 and p + 4 + rows * (4 + 8 * nv) + 4 == vtx:
            start = p
            break
    if start is None:
        raise ValueError("node block not found")
    o = start + 4
    coords = []
    for _ in range(rows):
        coords.append(np.frombuffer(b, "<f8", nv, o + 4))
        o += 4 + 8 * nv
    points = np.stack(coords, 1)

    starts = {}
    for name in ("vtx", "edg", "tri", "tet"):
        k = b.find(_key(name), o)
        if k < 0:
            raise ValueError(f"element type {name} not found")
        starts[name] = k
    for name in ("quad", "prism", "pyr", "hex"):
        if b.find(_key(name), o) >= 0:
            raise ValueError(f"mesh contains {name} elements; only tetrahedral meshes are supported")

    def conn(name, nper):
        p = starts[name] + 4 + 4 * len(name)
        if i32(p)[0] != nper:
            raise ValueError(f"{name}: unexpected nodes per element")
        p += 4
        n = int(i32(p)[0])
        cols = []
        for _ in range(nper):
            if i32(p)[0] != n:
                raise ValueError(f"{name}: unexpected connectivity layout")
            cols.append(i32(p + 4, n))
            p += 4 + 4 * n
        return np.stack(cols, 1) - lowest, n, p

    tets, nt, p = conn("tet", 4)
    hdr = i32(p, 4)
    if not (hdr[0] == 4 and hdr[1] == 0 and hdr[2] == nt and hdr[3] == nt):
        raise ValueError("tet: unexpected entity block layout")
    # Domain indices are stored 1-based (0 is the exterior, as in the up/down
    # arrays); boundary indices are stored 0-based.
    tet_dom = i32(p + 16, nt).copy()

    tris, ntri, p = conn("tri", 3)
    # The boundary number of each triangle is stored as (n, n, values).  Older
    # files append the up/down domain arrays after it, newer ones do not, so
    # the block is found by its signature and the two neighbouring domains
    # are worked out from the tetrahedra instead.
    tail = i32(p, (starts["tet"] - p) // 4)
    T = len(tail)
    ent = None
    for off in (T - ntri, T - 3 * ntri - 3):
        if off >= 2 and tail[off - 1] == ntri and tail[off - 2] == ntri:
            ent = tail[off:off + ntri]
            break
    if ent is None:
        raise ValueError("tri: unexpected entity block layout")
    up, down = _neighbours(tets, tet_dom, tris)
    # Edge elements and the geometry edge each one lies on (stored like the
    # boundary numbers: n, n, values, 0-based).  Only used for line results.
    edges = edge_ent = None
    try:
        edges, ne, pe = conn("edg", 2)
        tail = i32(pe, (starts["tri"] - pe) // 4)
        for off in np.where(tail[:len(tail) - ne] == ne)[0][::-1] + 1:      # after the edge parameters
            if off >= 2 and tail[off - 2] == ne:
                edge_ent = tail[off:off + ne] + 1
                break
    except ValueError:
        pass
    return dict(points=points, tets=tets, tet_dom=tet_dom, tris=tris,
                tri_bnd=ent + 1, tri_up=up, tri_down=down, version=int(version),
                edges=edges, edge_ent=edge_ent)


def _neighbours(tets, tet_dom, tris):
    """Domains on the two sides of each boundary triangle.

    "up" is the larger domain number and "down" the smaller; a triangle on the
    outer surface has up = 0 and down = its only domain.
    """
    n = int(tets.max()) + 1
    def key(f):
        f = np.sort(f.astype(np.int64), axis=1)
        return (f[:, 0] * n + f[:, 1]) * n + f[:, 2]
    fk = np.concatenate([key(tets[:, c]) for c in ([0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3])])
    fd = np.tile(tet_dom, 4)
    order = np.argsort(fk, kind="stable")
    fk, fd = fk[order], fd[order]
    tk = key(tris)
    lo = np.searchsorted(fk, tk, "left")
    hi = np.searchsorted(fk, tk, "right")
    if ((hi - lo) < 1).any() or ((hi - lo) > 2).any():
        raise ValueError("boundary triangle without a matching tetrahedron face")
    a = fd[lo]
    b = np.where(hi - lo == 2, fd[np.minimum(lo + 1, len(fd) - 1)], 0)
    up = np.where(b == 0, 0, np.maximum(a, b))
    down = np.where(b == 0, a, np.minimum(a, b))
    return up.astype(np.int32), down.astype(np.int32)


# ---------------------------------------------------------------- settings
def entities(text):
    """Entity list as stored: "18,1,2" (count first) or "-123,-1,8,10" -> [ints]."""
    v = [int(s) for s in (text or "").split(",") if s.strip()]
    if not v:
        return []
    return v[2:] if v[0] < 0 else v[1:]


def values(text):
    """COMSOL matrix string  9|1,'a'|1,'b'...  -> list of str."""
    return [m.replace("\\,", ",") for m in re.findall(r"'((?:[^'\\]|\\.)*)'", text or "")]


class Mph:
    def __init__(self, path):
        self.path = str(path)
        self.zip = zipfile.ZipFile(self.path)
        self.root = ET.fromstring(self.zip.read("dmodel.xml"))
        self.smodel = json.loads(self.zip.read("smodel.json"))
        self._sel = None

    # -- parameters ------------------------------------------------------
    def parameters(self):
        """Global parameters {name: SI value} as COMSOL evaluated them."""
        out = {}

        def walk(n):
            if n.get("apiClass") == "ModelParamGroup":
                for s in n.get("settings", []):
                    if "scalarReal" in s:
                        out[s["name"]] = float(s["scalarReal"])
            for c in n.get("nodes", []):
                walk(c)

        walk(self.smodel)
        return out

    # -- selections ------------------------------------------------------
    def selections(self):
        if self._sel is None:
            self._sel = {}
            for s in self.root.iter("SelectionFeature"):
                e = s.find("outputSelection/explicit")
                if e is None:
                    continue
                self._sel["/selection/" + s.get("tag")] = dict(
                    name=s.get("name"), dim=int(e.get("dim")), geom=e.get("geom").split("/")[-1],
                    entities=entities(e.get("entities")),
                    component=(s.findtext("storedParent") or "").split("/")[-1])
        return self._sel

    def selection_of(self, node):
        """(dim, entities, name) of a feature's selection; entities is None for the default selection."""
        s = node.find("selection")
        if s is None:
            return None, None, None
        for c in s:
            if c.tag == "named":
                d = self.selections().get(c.text)
                return (d["dim"], d["entities"], d["name"]) if d else (None, None, c.text)
            if c.tag == "explicit" and c.get("entities"):
                return int(c.get("dim")), entities(c.get("entities")), None
        return None, None, None

    # -- physics ---------------------------------------------------------
    def physics(self, tag):
        for ph in self.root.iter("Physics"):
            if ph.get("tag") == tag:
                return ph
        raise KeyError(tag)

    def features(self, physics_tag):
        """Active top-level features of a physics interface, in model-tree order.

        Later features override earlier ones on shared entities, as in COMSOL.
        """
        out = []
        for f in self.physics(physics_tag).find("PhysicsFeatureList").findall("PhysicsFeature"):
            if "DISABLED" in (f.findtext("entityFlags") or ""):
                continue
            dim, ents, selname = self.selection_of(f)
            par = {p.get("param"): values(p.get("value")) for p in f.findall("param")}
            sub = []
            fl = f.find("PhysicsFeatureList")
            if fl is not None:
                for g in fl.findall("PhysicsFeature"):
                    if "DISABLED" not in (g.findtext("entityFlags") or ""):
                        sub.append(dict(tag=g.get("tag"), op=g.get("op"), name=g.get("name"),
                                        par={p.get("param"): values(p.get("value")) for p in g.findall("param")}))
            out.append(dict(tag=f.get("tag"), op=f.get("op"), name=f.get("name"), dim=dim,
                            entities=ents, selection=selname, par=par, sub=sub))
        return out

    def flow_coupling(self, heat_tag):
        """Settings of the Nonisothermal Flow coupling attached to a heat interface, or None."""
        for c in self.root.iter("MultiphysicsCoupling"):
            par = {p.get("param"): values(p.get("value")) for p in c.findall("param")}
            if par.get("Heat_physics", [None])[0] == heat_tag:
                return dict(fluid=par["Fluid_physics"][0], boussinesq=par.get("BoussinesqApproximation", ["0"])[0] == "1")
        return None

    # -- materials -------------------------------------------------------
    def _material_nodes(self):
        top = next(l for l in self.root.iter("MaterialList") if l.get("tag") == "material")
        return {m.get("tag"): m for m in top.findall("Material")}

    def materials(self, component, switch_index=0):
        """Materials assigned in a component: list of dicts with the resolved source material.

        Each entry: name, dim, entities, props {property: expression},
        funcs {name: definition}.  A Material Switch resolves to its first
        member (what COMSOL uses outside a material sweep) unless
        ``switch_index`` picks another one (0-based).
        """
        nodes = self._material_nodes()
        out = []
        for m in nodes.values():
            if m.get("op") not in ("Link", "LayeredMaterialLink"):
                continue
            if (m.findtext("storedParent") or "").split("/")[-1] != component:
                continue
            ref = next((p.get("Reference") for p in m.findall("propertyValue") if p.get("name") == "p:link"), None)
            src = nodes.get((ref or "").split("/")[-1])
            switch = None
            if src is not None and src.get("op") == "Switch":
                switch = [s.get("name") for s in src.find("MaterialList").findall("Material")]
                src = src.find("MaterialList").findall("Material")[switch_index]
            dim, ents, selname = self.selection_of(m)
            props, funcs = {}, {}
            if src is not None:
                for mm in src.find("MaterialModelList").findall("MaterialModel"):
                    for p in mm.findall("param"):
                        if p.get("param").startswith("INFO"):
                            continue
                        v = values(p.get("value"))
                        if v and p.get("param") not in props:
                            props[p.get("param")] = v[0]
                    for f in mm.iter("FunctionFeature"):
                        pv = {q.get("name"): q for q in f.findall("propertyValue")}
                        name = pv["p:funcname"].get("value") if "p:funcname" in pv else f.get("tag")
                        if f.get("op") == "Piecewise":
                            v = values(pv["p:pieces"].get("valueMatrix"))
                            pieces = [(float(v[i]), float(v[i + 1]), v[i + 2]) for i in range(0, len(v), 3)]
                            funcs.setdefault(name, dict(kind="piecewise", arg=pv["p:arg"].get("value"), pieces=pieces))
                        elif f.get("op") == "Analytic":
                            funcs.setdefault(name, dict(kind="analytic", expr=pv["p:expr"].get("value"),
                                                        args=values(pv["p:args"].get("valueMatrix"))))
            out.append(dict(link=m.get("tag"), name=m.get("name"), dim=dim, entities=ents,
                            source=src.get("name") if src is not None else None,
                            switch_members=switch, props=props, funcs=funcs))
        return out

    # -- study -----------------------------------------------------------
    def study_steps(self, study_tag=None):
        """Steps of a study (the first one in the model unless a tag is given), in order.

        Each entry carries its settings plus "op": Stationary, Transient, Parametric, ...
        """
        out = {}
        for st in self.root.iter("Study"):
            if study_tag is not None and st.get("tag") != study_tag:
                continue
            if study_tag is None and out:
                break
            for f in st.iter("StudyFeature"):
                d = {"op": f.get("op")}
                for p in f.findall("propertyValue"):
                    if p.get("name"):
                        d[p.get("name")[2:]] = p.get("value") if p.get("value") is not None else values(p.get("valueMatrix"))
                out[f.get("tag")] = d
        return out

    # -- mesh ------------------------------------------------------------
    def mesh_resources(self):
        """Stored volume meshes: {geometry tag: mesh resource file}.

        The mesh object is matched to its geometry by bounding box, which the
        archive stores next to each mesh reference.
        """
        txt = self.zip.read("dmodel.xml").decode("utf-8", "replace")
        out = {}
        for g in re.finditer(r'<MeshSequence [^>]*tag="(\w+)"', txt):
            pass
        for m in re.finditer(r'<mesh T="52" class="MESH" resource="(mesh\d+\.mphbin)"', txt):
            res = m.group(1)
            if self.zip.getinfo(res).file_size < 10000:
                continue
            # the owning MeshSequence is the closest preceding one
            k = txt.rfind("<MeshSequence ", 0, m.start())
            head = txt[k:k + 4000]
            geom = re.search(r"<geom T=\"50\">/geom/(\w+)</geom>", head)
            tag = re.search(r'tag="(\w+)"', head)
            out[geom.group(1) if geom else res] = dict(resource=res, mesh_tag=tag.group(1) if tag else None)
        return out

    def mesh_sequences(self):
        """Every stored volume mesh: {mesh sequence tag: dict(resource, name, geom)}."""
        txt = self.zip.read("dmodel.xml").decode("utf-8", "replace")
        out = {}
        for m in re.finditer(r'<mesh T="52" class="MESH" resource="(mesh\d+\.mphbin)"', txt):
            res = m.group(1)
            if self.zip.getinfo(res).file_size < 10000:
                continue
            k = txt.rfind("<MeshSequence ", 0, m.start())
            head = txt[k:k + 4000]
            tag = re.search(r'tag="(\w+)"', head).group(1)
            name = re.search(r'name="([^"]*)"', head)
            geom = re.search(r"<geom T=\"50\">/geom/(\w+)</geom>", head)
            out[tag] = dict(resource=res, name=name.group(1) if name else tag, geom=geom.group(1) if geom else None)
        return out

    def mesh(self, resource):
        return parse_mesh(self.zip.read(resource))

    # -- components ------------------------------------------------------
    def components(self):
        """{component tag: dict(geom, heat, label)} for every heat-transfer interface."""
        comp_of = {d["geom"]: d["component"] for d in self.selections().values()}
        labels = {}

        def walk(n):
            if n.get("apiClass") == "ModelNode":
                labels[n.get("tag")] = n.get("label")
            for c in n.get("nodes", []):
                walk(c)

        walk(self.smodel)
        out = {}
        for ph in self.root.iter("Physics"):
            if "HeatTransfer" in (ph.get("op") or ""):
                geom = (ph.findtext("geom") or "").split("/")[-1]
                comp = comp_of.get(geom)
                if comp:
                    out[comp] = dict(geom=geom, heat=ph.get("tag"), label=labels.get(comp, comp))
        return out
