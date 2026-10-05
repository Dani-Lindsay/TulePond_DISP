import sys, time, numpy as np, gmsh
xy = np.loadtxt("mesh/trace_tmerc.txt")
mode = sys.argv[1]
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
occ = gmsh.model.occ
pts = [occ.add_point(x, y, 0.0) for x, y in xy]
c = occ.add_spline(pts)
sw = occ.extrude([(1, c)], 0, 0, -3000.0)
f = [t for d, t in sw if d == 2][0]
v = occ.add_box(-50e3, -50e3, -40e3, 100e3, 100e3, 40e3)
occ.fragment([(3, v)], [(2, f)], removeTool=True); occ.synchronize()
print("surfaces", gmsh.model.getEntities(2))
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0); gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0); gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
if mode == "const":
    gmsh.option.setNumber("Mesh.MeshSizeMax", 2000.0)
elif mode == "dist":
    fl = gmsh.model.mesh.field
    d = fl.add("Distance"); fl.setNumbers(d, "SurfacesList", [int(sys.argv[2])]); fl.setNumber(d, "Sampling", 50)
    m = fl.add("MathEval"); fl.setString(m, "F", f"Min(500*1.3^(Log(1.0+F{d}/500*(1.3-1.0))/Log(1.3)), 5000)")
    fl.setAsBackgroundMesh(m)
elif mode.startswith("full"):
    fl = gmsh.model.mesh.field
    d = fl.add("Distance"); fl.setNumbers(d, "SurfacesList", [1]); fl.setNumber(d, "Sampling", 200)
    m = fl.add("MathEval"); fl.setString(m, "F", f"100.0*1.15^(Log(1.0+F{d}/100.0*(1.15-1.0))/Log(1.15))")
    b = fl.add("Box")
    for k, v in dict(VIn=150.0, VOut=5000.0, XMin=-6e3, XMax=6e3, YMin=-6e3, YMax=6e3, ZMin=-1000.0, ZMax=1.0, Thickness=2000.0).items():
        fl.setNumber(b, k, v)
    mn = fl.add("Min"); fl.setNumbers(mn, "FieldsList", [m, b])
    cap = fl.add("MathEval"); fl.setString(cap, "F", f"Min(F{mn}, 5000)")
    fl.setAsBackgroundMesh(cap if mode == "full" else m)
t = time.time(); gmsh.model.mesh.generate(1); print("1D", round(time.time() - t, 1), "s")
t = time.time(); gmsh.model.mesh.generate(2); print("2D", round(time.time() - t, 1), "s")
