#!/usr/bin/env python3
"""
PyLith tetrahedral mesh with a DIPPING fault under the hand-mapped creeping trace (2026-09-30, dip sensitivity).
Same geometry, sizing, marking and quality steps as generate_gmsh_vertical.py (imported), with one change: the
trace spline is extruded along a dipping vector instead of straight down.

Dip convention: `--dip 80e` = 80 degrees, dipping EAST (hanging wall east); `70e`, `80w` likewise. The dip
direction is the horizontal unit normal to the MEAN strike of the trace (km 52-67), pointing to the given side,
so the fault is a ruled surface: top edge exactly on the trace, every depth level the same trace shifted by
z / tan(dip) along that normal (the construction of 07's wrap_fault for CSI). The fault depth is measured
VERTICALLY: fault_depth_m = 3000 m below the surface, so the down-dip length is 3000 / sin(dip) m and the bottom
edge lies 3000 / tan(dip) m (529 m for 80, 1092 m for 70) to the dipping side of the trace. The bottom and end
edges are buried (PyLith v5 marks them automatically); the domain is the same 100 x 100 x 40 km box.

Run inside the PyLith environment:  python generate_gmsh_dip.py --write --preset default --dip 80e
Output: mesh/mesh_dip<dip>_tet_<preset>.msh (+ _report.txt, _fault.npz as the vertical generator).
"""
import os
# PyLith build script (pylith/README.md). Ported from drafts/pylith (2026-09-30); every output path comes from
# TulePond_filepaths (2026-10-06 layout: meshes in paths_inputs['mesh'], SFCVM grids in paths_sfcvm['spatialdb']);
# the databases are written where the command line says, relative to the results directory.
import sys

import numpy as np
import gmsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_gmsh_vertical as gv
from generate_gmsh_vertical import App as VerticalApp, trace_xy, fault_depth_m, domain_half_m, domain_depth_m, PRESETS

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from TulePond_filepaths import paths_inputs   # noqa: E402  (only `os` is imported there: safe in the PyLith python)
MESH_DIR = paths_inputs["mesh"]


def dip_vector(dip_str):
    """(dx, dy, dz) of the extrusion, and the dip angle: horizontal offset fault_depth_m / tan(dip) along the mean
    normal of the trace, on the given side (e / w), vertical drop fault_depth_m."""
    assert dip_str[-1] in "ew" and dip_str[:-1].isdigit(), f"dip must be like 80e or 70w, got {dip_str!r}"
    dip_deg, side = float(dip_str[:-1]), dip_str[-1]
    assert 30.0 < dip_deg < 90.0, "use generate_gmsh_vertical.py for a vertical fault"
    x, y = trace_xy()
    strike = np.array([x[-1] - x[0], y[-1] - y[0]]); strike /= np.linalg.norm(strike)
    normal = np.array([strike[1], -strike[0]])
    if normal[0] < 0:
        normal = -normal                                             # + east
    if side == "w":
        normal = -normal
    h = fault_depth_m / np.tan(np.deg2rad(dip_deg))
    return h * normal[0], h * normal[1], -fault_depth_m, dip_deg, normal


class App(VerticalApp):
    def __init__(self):
        super().__init__()
        self.dip = "80e"

    def create_geometry(self):
        x, y = trace_xy()
        dx, dy, dz, dip_deg, normal = dip_vector(self.dip)
        print(f"trace: {x.size} points, {np.hypot(np.diff(x), np.diff(y)).sum() / 1e3:.1f} km; dip {dip_deg:g} deg, "
              f"extrusion ({dx:.0f}, {dy:.0f}, {dz:.0f}) m along the mean normal ({normal[0]:.3f}, {normal[1]:.3f})")
        assert abs(dx) + abs(dy) + fault_depth_m < domain_half_m - 5e3, "fault leaves the domain"
        occ = gmsh.model.occ
        pts = [occ.add_point(xi, yi, 0.0) for xi, yi in zip(x, y)]
        self.c_trace = occ.add_spline(pts)
        swept = occ.extrude([(1, self.c_trace)], dx, dy, dz)
        self.s_fault = [tag for dim, tag in swept if dim == 2][0]
        self.v_domain = occ.add_box(-domain_half_m, -domain_half_m, -domain_depth_m,
                                    2 * domain_half_m, 2 * domain_half_m, domain_depth_m)
        occ.fragment([(3, self.v_domain)], [(2, self.s_fault)], removeTool=True)
        occ.synchronize()

    def report(self):
        super().report()
        # dip check on the fault triangles: the mean unit normal's angle to the vertical
        xyz = np.load(self.filename.replace(".msh", "_fault.npz"))["xyz"]
        n = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
        n /= np.linalg.norm(n, axis=1)[:, None]
        n[n[:, 0] < 0] *= -1
        nz = float(np.median(n[:, 2]))                                # east-pointing normal: z > 0 means the plane dips east
        dip_meas = np.degrees(np.arccos(abs(nz)))                     # angle between the normal and the vertical = the dip
        dip_deg = float(self.dip[:-1])
        line = (f"measured dip from fault triangles: {dip_meas:.1f} deg, dipping {'east' if nz > 0 else 'west'} "
                f"(median east-normal z {nz:+.3f}; requested {dip_deg:g} {self.dip[-1]})")
        print(line)
        assert abs(dip_meas - dip_deg) < 1.0 and (nz > 0) == (self.dip[-1] == "e"), line
        with open(self.filename.replace(".msh", "_report.txt"), "a") as fh:
            fh.write(line + "\n")

    @staticmethod
    def _add_arguments(parser):
        VerticalApp._add_arguments(parser)
        parser.add_argument("--dip", action="store", dest="dip", default="80e", help="e.g. 80e, 70e, 80w")

    def _parse_command_line(self):
        args = super()._parse_command_line()
        self.dip = args.dip
        self.filename = os.path.join(MESH_DIR, f"mesh_dip{self.dip}_tet_{self.preset}.msh")
        args.filename = self.filename
        return args


if __name__ == "__main__":
    os.makedirs(MESH_DIR, exist_ok=True)
    App().main()
