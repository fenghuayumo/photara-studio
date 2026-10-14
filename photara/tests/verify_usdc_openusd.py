"""Independent validation with Pixar OpenUSD, optional for native builds."""
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from pxr import Usd, UsdGeom, UsdVol, Sdf

with tempfile.TemporaryDirectory(prefix="photara-usdc-interop-") as directory:
    output = Path(directory) / "gaussians.usdc"
    subprocess.run([sys.argv[1], "--write", str(output)], check=True)
    assert output.read_bytes()[:8] == b"PXR-USDC"
    stage = Usd.Stage.Open(str(output))
    assert stage
    prim = stage.GetPrimAtPath("/Gaussians")
    assert prim.GetTypeName() == "ParticleField3DGaussianSplat"
    assert stage.GetDefaultPrim() == prim
    splats = UsdVol.ParticleField3DGaussianSplat(prim)
    assert splats
    positions = splats.GetPositionsAttr().Get()
    assert list(positions) == [(1, 2, 3), (-2, .5, 4), (0, 1, -1)]
    scales = splats.GetScalesAttr().Get()
    assert len(scales) == 3
    for i in range(3):
        for k in range(3):
            assert math.isclose(scales[i][k], .1 * (3*i+k+1), abs_tol=1e-6)
    q = splats.GetOrientationsAttr().Get()
    expected = [(1, (0, 0, 0)), (math.sqrt(.5), (0, 0, math.sqrt(.5))), (.5, (.5, .5, .5))]
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
    for i, value in enumerate(coefficients):
        for k in range(3):
            assert math.isclose(value[k], (((i*3+k)%19)-9)*.037, abs_tol=1e-6)
    primvar = UsdGeom.Primvar(attr)
    assert primvar.GetElementSize() == 16
    assert primvar.GetInterpolation() == "vertex"
    extent = UsdGeom.Boundable(prim).GetExtentAttr().Get()
    assert len(extent) == 2
    assert all(extent[0][k] < extent[1][k] for k in range(3))
    print("OpenUSD independently verified native USDC export")
