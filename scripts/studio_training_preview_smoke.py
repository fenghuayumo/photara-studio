#!/usr/bin/env python3
"""Exercise live Studio previews and retraining with an existing ASFM fixture.

Requires a desktop session, a Vulkan Studio build and photara-ctl. Inputs are
read-only: the ASFM working copy, cache, preferences and captures are isolated.
Example (set VK_ICD_FILENAMES for MoltenVK first):
  python3 scripts/studio_training_preview_smoke.py --asfm /data/scene.asfm
"""

import argparse
import json
import os
import sys
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time
import zlib


def has_scene_pixels(path):
    """Reject the uniform placeholder without a third-party image dependency."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    offset, compressed = 8, bytearray()
    while offset < len(data):
        size = struct.unpack_from(">I", data, offset)[0]
        if data[offset + 4:offset + 8] == b"IDAT":
            compressed.extend(data[offset + 8:offset + 8 + size])
        offset += size + 12
    return len(set(zlib.decompress(compressed))) > 16


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asfm", type=Path, required=True)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--studio", type=Path, default=repo / "build/photara/photara_studio")
    parser.add_argument("--ctl", type=Path, default=repo / "build/photara/photara-ctl")
    parser.add_argument("--iterations", type=int, default=600)
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="photara-preview-smoke-"))
    config = root / "config/Photara"
    config.mkdir(parents=True)
    (config / "editor.prefs").write_text("training=vulkan\ntraining_autosave_seconds=0\n")
    # ASFM import places its working scene in sfm.bin. Seed the same input
    # directly so this test can use the existing control API, which opens only
    # .ascan projects. A symlink keeps all generated files beside this fixture.
    images = root / "images"
    images.symlink_to((args.images or args.asfm.parent / "images").resolve(), target_is_directory=True)
    working = root / "images.cache/sfm.bin"
    working.parent.mkdir()
    shutil.copy2(args.asfm, working)
    env = dict(os.environ, XDG_CONFIG_HOME=str(root / "config"),
               PHOTARA_CONTROL_ALLOWED_ROOTS=str(root))
    with (root / "studio.log").open("w") as log:
        process = subprocess.Popen([str(args.studio.resolve()), "--enable-agent-control"],
                                   env=env, stdout=log, stderr=subprocess.STDOUT)

    def call(*command):
        result = subprocess.run([str(args.ctl.resolve()), "--pid", str(process.pid),
                                 *command, "--json"], env=env, capture_output=True,
                                text=True, timeout=15)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return json.loads(result.stdout)

    def wait_for(predicate, timeout=120):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if process.poll() is not None:
                raise RuntimeError(f"Studio exited with {process.returncode}")
            try:
                value = predicate()
                if value:
                    return value
            except (RuntimeError, subprocess.TimeoutExpired):
                pass
            time.sleep(0.2)
        raise RuntimeError("Timed out waiting for Studio; see " + str(root))

    try:
        wait_for(lambda: call("doctor"), timeout=30)
        call("project", "set-source", str(images))
        call("params", "set", f"iterations={args.iterations}", "preview_interval=10",
             "max_resolution=512", "progressive_resolution=false", "sh_degree=0",
             "densification_cap=100000")
        report = {"asfm": str(args.asfm.resolve()), "runs": []}
        for run in range(2):
            call("job", "start", "train")

            def live_frame():
                job = call("job", "status")["job"]
                if job["running"] and 0 < job["training"]["iteration"] < args.iterations:
                    capture = root / f"live-{run}.png"
                    call("capture", "--out", str(capture))
                    if has_scene_pixels(capture):
                        return job
                return None

            job = wait_for(live_frame)
            if sys.platform == "darwin":
                lines = call("inspect", "logs", "--limit", "2000")["lines"]
                assert any("splat_preview_transport=vulkan_iosurface" in line["message"]
                           for line in lines), "Expected GPU IOSurface preview transport"
                session = Path(tempfile.gettempdir()) / "Photara" / f"session-{process.pid}"
                assert not list(session.rglob("*.png")), "Preview must not write image files"
            baseline = call("state")["view"]
            yaw = baseline["camera"]["yaw"]
            previous_revision = baseline["preview_requested_camera_revision"]
            call("view", "set", f"yaw={yaw + 0.25}")
            if sys.platform == "darwin":
                # Training-only pixel changes cannot satisfy this check: the
                # camera revision comes from the raster actually submitted,
                # and is exposed only after its GPU copy fence completes.
                def requested_camera():
                    view = call("state")["view"]
                    return (view["preview_requested_camera_revision"]
                            if view["preview_requested_camera_revision"] > previous_revision else None)
                revision = wait_for(requested_camera)
                wait_for(lambda: call("state")["view"]["preview_displayed_camera_revision"] == revision)
                assert call("job", "status")["job"]["running"], "Camera must update during training"
            call("capture", "--out", str(root / f"camera-{run}.png"))
            final = wait_for(lambda: (lambda j: j if j["status"] in
                             ("succeeded", "failed", "cancelled") else None)(
                                 call("job", "status")["job"]))
            assert final["status"] == "succeeded", final
            call("capture", "--out", str(root / f"final-{run}.png"))
            assert has_scene_pixels(root / f"final-{run}.png")
            report["runs"].append({"live_iteration": job["training"]["iteration"],
                                   "camera_revision_verified": sys.platform == "darwin", "status": final["status"]})
        (root / "verification.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"passed": True, "artifacts": str(root), **report}, indent=2))
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)


if __name__ == "__main__":
    main()
