#!/usr/bin/env python3
"""
DRAFT (2026-09-29): PyLith tetrahedral mesh with a VERTICAL fault under the
hand-mapped creeping trace (Dani: "points at the top and the same points at
the bottom, and gmsh does the infilling").

Follows PyLith v5.0.2 examples/crustal-strikeslip-3d/generate_gmsh.py
(pylith.meshio.gmsh_utils.GenerateMesh: create_geometry, mark, generate_mesh).
Run inside the PyLith environment (its python has gmsh):
    python generate_gmsh.py --write [--gui]

Geometry (local transverse Mercator centred on Tule Pond, metres, z up):
- trace: Inputs/hayward_fault_profiles.csv (100 m hand-digitised points),
  52-67 km from Point Pinole (the CSI model's along-strike window), decimated
  to every 3rd point (300 m) so the spline does not follow digitising jitter
- fault: the spline extruded straight down to fault_depth_m (vertical)
- domain: a box domain_half_m either side of the site, domain_depth_m deep
- fault and box fragmented so the tets conform to the fault; the fault's top
  edge lies on the free surface, its two ends and bottom are buried edges
  (marked automatically by PyLith v5)
Sizes: dx_fault on the fault, dx_surface on the free surface within
data_half_m of the site, growing by `bias` to dx_far.
"""
import os
# PyLith build script (pylith/README.md). Ported from drafts/pylith (2026-09-30); every output path comes from
# TulePond_filepaths (2026-10-06 layout: meshes in paths_inputs['mesh'], SFCVM grids in paths_sfcvm['spatialdb']);
# the databases are written where the command line says, relative to the results directory.
import sys
import time

import numpy as np
import gmsh

from pylith.meshio.gmsh_utils import (BoundaryGroup, MaterialGroup, GenerateMesh)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from TulePond_filepaths import paths_inputs   # noqa: E402  (only `os` is imported there: safe in the PyLith python)
MESH_DIR = paths_inputs["mesh"]

site_lon, site_lat = -121.968, 37.557                # CSI frame origin (07_slip_model)
CRS = f"+proj=tmerc +datum=WGS84 +lon_0={site_lon} +lat_0={site_lat} +k=1 +units=m +type=crs"
km_window = (52.0, 67.0)                             # along-strike window of the CSI fault (km from Point Pinole)
decimate = 3                                         # keep every 3rd trace point (300 m)
fault_depth_m = 3000.0                               # CSI mesh_depth_km
domain_half_m = 50e3                                 # box +/- 50 km (several fault lengths)
domain_depth_m = 40e3
# mesh sizes (m); size grows linearly away from each target: size = dx + (bias - 1) * distance
PRESETS = {
    "coarse":  dict(dx_fine=200.0, dx_fault=400.0, dx_surface=400.0, dx_far=8000.0, bias=1.20),
    "default": dict(dx_fine=100.0, dx_fault=250.0, dx_surface=250.0, dx_far=6000.0, bias=1.15),
    "fine":    dict(dx_fine=60.0,  dx_fault=200.0, dx_surface=200.0, dx_far=6000.0, bias=1.12),
    # "shallow": default + a near-surface band (dx_band within band_m of the fault, top band_z_m, along the patch) to
    # resolve the SFCVM sediment layers (~200-550 m thick west of the trace) in the volume near the fault
    "shallow": dict(dx_fine=100.0, dx_fault=250.0, dx_surface=250.0, dx_far=6000.0, bias=1.15,
                    dx_band=60.0, band_m=1000.0, band_z_m=600.0),
}
patch_y_m = (-3100.0, 2800.0)       # Tule Pond section km 56-63 from Point Pinole (km 63 at y=-3084, km 56 at y=2776)
patch_z_m = 1500.0                  # fine patch: top 1.5 km of the fault
data_x_m = (-9300.0, 5300.0)        # extent of the 2386 quadtree samples (fig2_model_row.npz) in this CRS
data_y_m = (-6450.0, 6000.0)


def trace_xy():
    """The creeping trace in the mesh CRS, south to north (written by the TulePond
    environment: 52-67 km from Point Pinole, every 3rd 100 m point, round-trip checked)."""
    xy = np.loadtxt(os.path.join(MESH_DIR, "trace_tmerc.txt"))
    return xy[:, 0], xy[:, 1]


class App(GenerateMesh):
    def __init__(self):
        super().__init__()
        self.cell_choices = {"default": "tet", "choices": ["tet"]}
        self.filename = "mesh_vertical_tet.msh"
        self.preset = "default"

    def create_geometry(self):
        x, y = trace_xy()
        print(f"trace: {x.size} points, {np.hypot(np.diff(x), np.diff(y)).sum() / 1e3:.1f} km")
        occ = gmsh.model.occ
        pts = [occ.add_point(xi, yi, 0.0) for xi, yi in zip(x, y)]
        self.c_trace = occ.add_spline(pts)
        # the fault: the trace swept straight down (vertical)
        swept = occ.extrude([(1, self.c_trace)], 0.0, 0.0, -fault_depth_m)
        self.s_fault = [tag for dim, tag in swept if dim == 2][0]
        self.v_domain = occ.add_box(-domain_half_m, -domain_half_m, -domain_depth_m,
                                    2 * domain_half_m, 2 * domain_half_m, domain_depth_m)
        occ.fragment([(3, self.v_domain)], [(2, self.s_fault)], removeTool=True)
        occ.synchronize()

    def mark(self):
        # the domain's six faces by their bounding boxes; the fault is the surface inside the box
        eps = 1.0
        surfaces = gmsh.model.getEntities(2)
        def at(axis, value):
            out = []
            for dim, tag in surfaces:
                b = gmsh.model.getBoundingBox(dim, tag)
                lo, hi = b[axis], b[axis + 3]
                if abs(lo - value) < eps and abs(hi - value) < eps:
                    out.append(tag)
            return out
        boundaries = dict(boundary_xneg=at(0, -domain_half_m), boundary_xpos=at(0, domain_half_m),
                          boundary_yneg=at(1, -domain_half_m), boundary_ypos=at(1, domain_half_m),
                          boundary_zneg=at(2, -domain_depth_m), groundsurf=at(2, 0.0))
        box_faces = {t for v in boundaries.values() for t in v}
        fault = [tag for dim, tag in surfaces if tag not in box_faces]
        assert len(fault) >= 1, "fault surface not found after fragment"
        for name, tags in boundaries.items():
            assert len(tags) == 1, f"{name}: {tags}"
        self.fault_tags, self.top_tags = fault, boundaries["groundsurf"]
        groups = [BoundaryGroup(name=name, tag=10 + k, dim=2, entities=tags)
                  for k, (name, tags) in enumerate(boundaries.items())]
        groups.append(BoundaryGroup(name="fault", tag=20, dim=2, entities=fault))
        for g in groups:
            g.create_physical_group()
        MaterialGroup(tag=1, entities=[tag for dim, tag in gmsh.model.getEntities(3)]).create_physical_group()

    def generate_mesh(self, cell):
        """Cell size = the minimum of three linear-growth fields (gmsh_utils progression: size = dx + (bias-1) d):
        - fault_fine: dx_fine on the fault inside the Tule Pond patch (y in patch_y_m, top patch_z_m), where d is
          the larger of the distance to the fault and the distance outside the patch box;
        - fault: dx_fault on the whole fault (PyLith get_math_progression of the fault Distance field);
        - surface: dx_surface on the free surface over the data area (MathEval in x, y, z; replaces the gmsh "Box"
          field, which hung the 1D stage on 2026-09-29);
        each capped at dx_far inside its own MathEval and combined with a gmsh "Min" field. NOTE (gmsh 4.15.2,
        2026-09-29): a MathEval that references another MathEval (e.g. a separate cap field "Min(F3, 6000)") hangs
        the 1D stage, as did the "Box" field; a Min field over self-contained MathEvals meshes in 0.1 s."""
        p = PRESETS[self.preset]
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        field = gmsh.model.mesh.field
        f_fault = field.add("Distance")
        field.setNumbers(f_fault, "SurfacesList", self.fault_tags)
        field.setNumber(f_fault, "Sampling", 400)
        g = p["bias"] - 1.0
        size_fault = field.add("MathEval")
        field.setString(size_fault, "F", f"Min({self.get_math_progression(f_fault, min_dx=p['dx_fault'], bias=p['bias'])}, {p['dx_far']})")
        (y_lo, y_hi), z_patch = patch_y_m, patch_z_m
        size_fine = field.add("MathEval")
        field.setString(size_fine, "F", f"Min({p['dx_fine']} + {g}*Max(F{f_fault}, Max(Max({y_lo} - y, y - {y_hi}), -z - {z_patch})), "
                                        f"{p['dx_far']})")
        (dx_lo, dx_hi), (dy_lo, dy_hi) = data_x_m, data_y_m
        size_surface = field.add("MathEval")
        field.setString(size_surface, "F", f"Min({p['dx_surface']} + {g}*Max(Max(Max({dx_lo} - x, x - {dx_hi}), "
                                           f"Max({dy_lo} - y, y - {dy_hi})), Max(0, -z)), {p['dx_far']})")
        fields = [size_fault, size_fine, size_surface]
        if "dx_band" in p:
            size_band = field.add("MathEval")
            field.setString(size_band, "F", f"Min({p['dx_band']} + {g}*Max(0, Max(Max(F{f_fault} - {p['band_m']}, -z - {p['band_z_m']}), "
                                            f"Max({y_lo} - y, y - {y_hi}))), {p['dx_far']})")
            fields.append(size_band)
        f_min = field.add("Min")
        field.setNumbers(f_min, "FieldsList", fields)
        field.setAsBackgroundMesh(f_min)
        gmsh.option.setNumber("General.NumThreads", 8)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", 8)
        t0 = time.time()
        gmsh.model.mesh.generate(3)
        print(f"generate(3): {time.time() - t0:.1f} s")
        t0 = time.time()
        self.improve_quality()
        print(f"improve_quality: {time.time() - t0:.1f} s")
        self.report()

    def report(self):
        """Cell counts, fault-cell sizes and quality (gmsh getElementQualities, SICN), written to a sidecar."""
        types, tags, nodes = gmsh.model.mesh.getElements(3)
        assert len(types) == 1 and types[0] == 4, f"expected linear tets only, got {types}"
        n_tet = len(tags[0])
        sicn = gmsh.model.mesh.getElementQualities(tags[0], "minSICN")
        gamma = gmsh.model.mesh.getElementQualities(tags[0], "gamma")
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        n_nodes = len(node_tags)
        edges = []
        for tag in self.fault_tags:
            t2, e2, n2 = gmsh.model.mesh.getElements(2, tag)
            edges.append(np.asarray(n2[0]).reshape(-1, 3))
        fault_tri = np.vstack(edges)
        xyz = np.asarray(coords).reshape(-1, 3)
        lookup = np.zeros(int(np.max(node_tags)) + 1, dtype=int); lookup[np.asarray(node_tags, int)] = np.arange(n_nodes)
        tri_xyz = xyz[lookup[fault_tri.astype(int)]]
        edge_len = np.linalg.norm(tri_xyz[:, [1, 2, 0]] - tri_xyz, axis=2).mean(1)
        centre = tri_xyz.mean(1)
        in_patch = (centre[:, 1] >= patch_y_m[0]) & (centre[:, 1] <= patch_y_m[1]) & (centre[:, 2] >= -patch_z_m)
        fault_vertices = np.unique(fault_tri)
        lines = [f"preset {self.preset}: {PRESETS[self.preset]}",
                 f"tets {n_tet}, nodes {n_nodes}",
                 f"fault triangles {len(fault_tri)}, fault vertices {fault_vertices.size}",
                 f"fault edge length (m): patch median {np.median(edge_len[in_patch]):.0f} "
                 f"(min {edge_len[in_patch].min():.0f}, max {edge_len[in_patch].max():.0f}); "
                 f"outside median {np.median(edge_len[~in_patch]):.0f}",
                 f"tet minSICN: min {sicn.min():.3f}, 1st pct {np.percentile(sicn, 1):.3f}, median {np.median(sicn):.3f}, "
                 f"n<0.1 {np.sum(sicn < 0.1)}, n<=0 {np.sum(sicn <= 0)}",
                 f"tet gamma: min {gamma.min():.3f}, median {np.median(gamma):.3f}"]
        print("\n".join(lines))
        assert sicn.min() > 0, "inverted tets"
        with open(self.filename.replace(".msh", "_report.txt"), "w") as fh:
            fh.write("# " + __file__ + " " + time.strftime("%Y-%m-%d %H:%M") + "\n" + "\n".join(lines) + "\n")
        np.savez(self.filename.replace(".msh", "_fault.npz"), xyz=tri_xyz, edge_len=edge_len)

    @staticmethod
    def _add_arguments(parser):
        parser.add_argument("--preset", action="store", dest="preset", default="default", choices=sorted(PRESETS))

    def _parse_command_line(self):
        args = super()._parse_command_line()
        self.preset = args.preset
        if args.filename == self.filename:
            self.filename = os.path.join(MESH_DIR, f"mesh_vertical_tet_{self.preset}.msh")
            args.filename = self.filename
        return args


if __name__ == "__main__":
    os.makedirs(MESH_DIR, exist_ok=True)
    App().main()
