"""Independent validation with Pixar OpenUSD, optional for native builds."""
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from pxr import Usd, UsdGeom, UsdVol, Sdf, Gf

with tempfile.TemporaryDirectory(prefix="photara-usdc-interop-") as directory:
    output = Path(directory) / "gaussians.usdc"
    subprocess.run([sys.argv[1], "--write", str(output)], check=True)
    assert output.read_bytes()[:8] == b"PXR-USDC"
    stage = Usd.Stage.Open(str(output))
    assert stage
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y
    prim = stage.GetPrimAtPath("/Gaussians")
    assert prim.GetTypeName() == "ParticleField3DGaussianSplat"
    assert stage.GetDefaultPrim() == prim
    splats = UsdVol.ParticleField3DGaussianSplat(prim)
    assert splats
    positions = splats.GetPositionsAttr().Get()
    assert list(positions) == [(1, -2, -3), (-2, -.5, -4), (0, -1, 1)]
    assert UsdGeom.Xformable(prim).GetLocalTransformation() == Gf.Matrix4d(1)
    scales = splats.GetScalesAttr().Get()
    assert len(scales) == 3
    for i in range(3):
        for k in range(3):
            assert math.isclose(scales[i][k], .1 * (3*i+k+1), abs_tol=1e-6)
    q = splats.GetOrientationsAttr().Get()
    expected = [(1, (0, 0, 0)), (math.sqrt(.5), (0, 0, -math.sqrt(.5))), (.5, (.5, -.5, -.5))]
    for quat, (real, imaginary) in zip(q, expected):
        assert math.isclose(quat.GetReal(), real, abs_tol=1e-6), (quat, real)
        assert all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(quat.GetImaginary(), imaginary))
    opacities = splats.GetOpacitiesAttr().Get()
    assert all(math.isclose(a, 1/(1+math.exp(-b)), abs_tol=1e-6) for a, b in zip(opacities, (-2, 0, 2)))
    assert splats.GetRadianceSphericalHarmonicsDegreeAttr().Get() == 3
    assert splats.GetRadianceSphericalHarmonicsDegreeAttr().GetVariability() == Sdf.VariabilityUniform
    attr = splats.GetRadianceSphericalHarmonicsCoefficientsAttr()
    coefficients = attr.Get()
    assert len(coefficients) == 48
    signs = [1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1]
    for i, value in enumerate(coefficients):
        for k in range(3):
            assert math.isclose(value[k], (((i*3+k)%19)-9)*.037 * signs[i % 16], abs_tol=1e-6)
    primvar = UsdGeom.Primvar(attr)
    assert primvar.GetElementSize() == 16
    assert primvar.GetInterpolation() == "vertex"
    extent = UsdGeom.Boundable(prim).GetExtentAttr().Get()
    assert len(extent) == 2
    assert all(extent[0][k] < extent[1][k] for k in range(3))
    # Check the full rotated ellipsoid covariance bounds, not just min < max.
    low, high = [math.inf]*3, [-math.inf]*3
    for pos, scale, quat in zip(positions, scales, q):
        w = quat.GetReal()
        x, y, z = quat.GetImaginary()
        rotation = [[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                    [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                    [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]]
        for k in range(3):
            radius = 3 * math.sqrt(sum((rotation[k][j]*scale[j])**2 for j in range(3)))
            low[k] = min(low[k], pos[k]-radius)
            high[k] = max(high[k], pos[k]+radius)
    for k in range(3):
        assert math.isclose(extent[0][k], low[k], abs_tol=2e-6)
        assert math.isclose(extent[1][k], high[k], abs_tol=2e-6)

    subprocess.run([sys.argv[1], "--read-fixture", str(output)], check=True)
    # A third-party Y-up file must also import into the native RDF frame.
    stage.GetRootLayer().comment = "Independent Y-up fixture"
    external = Path(directory) / "external.usdc"
    stage.GetRootLayer().Export(str(external))
    subprocess.run([sys.argv[1], "--read-fixture", str(external)], check=True)
    # Legacy Photara files have raw RDF arrays plus the old, incorrect Y-up
    # declaration. Preserve their old import behavior using the export marker.
    splats.GetPositionsAttr().Set([Gf.Vec3f(p[0], -p[1], -p[2]) for p in positions])
    splats.GetOrientationsAttr().Set([
        Gf.Quatf(v.GetReal(), Gf.Vec3f(v.GetImaginary()[0], -v.GetImaginary()[1], -v.GetImaginary()[2])) for v in q])
    attr.Set([Gf.Vec3f(*(component * signs[i % 16] for component in value)) for i, value in enumerate(coefficients)])
    UsdGeom.Boundable(prim).GetExtentAttr().Set([
        Gf.Vec3f(extent[0][0], -extent[1][1], -extent[1][2]),
        Gf.Vec3f(extent[1][0], -extent[0][1], -extent[0][2])])
    stage.GetRootLayer().comment = "Photara native C++ 3D Gaussian export"
    legacy = Path(directory) / "legacy.usdc"
    stage.GetRootLayer().Export(str(legacy))
    subprocess.run([sys.argv[1], "--read-fixture", str(legacy)], check=True)
    stage.GetRootLayer().comment = "Independent Z-up fixture"
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    z_up = Path(directory) / "z-up.usdc"
    stage.GetRootLayer().Export(str(z_up))
    rejected = subprocess.run([sys.argv[1], "--read-fixture", str(z_up)], capture_output=True, text=True)
    assert rejected.returncode != 0 and "Y-up" in rejected.stderr
    print("OpenUSD independently verified Y-up export, covariance bounds, SH signs, external import and legacy import")
