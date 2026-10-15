"""Run paired ADC-IGS experiments through the native Python API, from JSON."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def native_module(config):
    os.environ["PHOTARA_NATIVE_DIR"] = str(Path(config["native_directory"]).resolve())
    import photara
    return photara


def make_options(aes, config, dataset, variant):
    overrides = {**config["training"], **dataset.get("training", {}), **variant["training"]}
    options = aes.TrainingOptions()
    options.backend = aes.TrainingBackend.CUDA
    options.densification_strategy = aes.DensificationStrategy.ADC_IGS
    options.iterations = int(overrides["iterations"])
    options.input_is_dense = bool(overrides.get("input_is_dense", False))
    options.apply_strategy_defaults()
    # Set explicit experiment values after the strategy preset.
    for key, value in overrides.items():
        if not hasattr(options, key):
            raise ValueError(f"Unknown TrainingOptions field: {key}")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
            raise ValueError(f"Non-finite option: {key}")
        setattr(options, key, value)
    for key in ("scale_regularization_weight", "shape_erank_reg"):
        if getattr(options, key) < 0:
            raise ValueError(f"{key} must be non-negative")
    if options.input_is_dense:
        raise ValueError("This sparse ADC-IGS experiment requires input_is_dense=false")
    return options, overrides


def train(config, dataset_name, variant_name):
    aes = native_module(config)
    dataset = next(item for item in config["datasets"] if item["name"] == dataset_name)
    variant = next(item for item in config["variants"] if item["name"] == variant_name)
    root = Path(config["output_directory"]).resolve()
    output = root / dataset_name / variant_name
    output.mkdir(parents=True, exist_ok=True)
    if (output / "model.ply").exists():
        raise FileExistsError(f"Refusing to overwrite {output / 'model.ply'}")
    options, overrides = make_options(aes, config, dataset, variant)
    asfm = aes.AsfmOptions()
    asfm.path_base = Path(dataset["images"])
    sfm = aes.load_sfm(root / "inputs" / dataset_name / "sfm.bin", asfm)
    scene = aes.build_mvs_scene(sfm, aes.MvsOptions())
    scene.load_dense_cloud(root / "inputs" / dataset_name / "sparse.ply")
    del sfm
    gc.collect()
    if not scene.view_count or not scene.dense_point_count:
        raise RuntimeError("Empty training scene")
    metadata = {
        "backend": "CUDA", "strategy": "ADC_IGS", "training": overrides,
        "views": scene.view_count, "initial_cloud_points": scene.dense_point_count,
        "native_module": aes.__file__, "native_sha256": digest(Path(aes.__file__)),
        "state": "training",
    }
    write_json(output / "run.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    started = time.perf_counter()
    with (output / "progress.jsonl").open("w", encoding="utf-8", buffering=1) as stream:
        def progress(stats):
            row = {key: getattr(stats, key) for key in (
                "iteration", "total_iterations", "gaussian_count", "loss", "rgb_loss",
                "grown_count", "pruned_count", "image_width", "image_height")}
            row["elapsed_seconds"] = time.perf_counter() - started
            stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            return True

        model = aes.train_3dgs(scene, options, progress=progress)
    metadata.update(training_seconds=time.perf_counter() - started, gaussians=model.size)
    model.save(output / "model.ply")
    metadata["state"] = "complete"
    metadata["model_sha256"] = digest(output / "model.ply")
    write_json(output / "run.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


def run(config_path, config):
    root = Path(config["output_directory"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_native = Path(config["native_directory"]).resolve()
    frozen_native = root / "native"
    if source_native != frozen_native:
        extensions = sorted(source_native.glob("photara*.pyd"))
        if not extensions:
            raise FileNotFoundError(f"No built photara Python extension in {source_native}")
        frozen_native.mkdir(parents=True, exist_ok=True)
        for binary in [extensions[0], *source_native.glob("*.dll")]:
            destination = frozen_native / binary.name
            if not destination.exists():
                shutil.copy2(binary, destination)
        config = {**config, "source_native_directory": str(source_native),
                  "native_directory": str(frozen_native)}
        if config.get("evaluator"):
            evaluator = Path(config["evaluator"])
            destination = frozen_native / evaluator.name
            if not destination.exists():
                shutil.copy2(evaluator, destination)
            config["evaluator"] = str(destination)
    aes = native_module(config)
    # Reject misspelled/unbound settings before loading any dataset.
    for dataset in config["datasets"]:
        for variant in config["variants"]:
            make_options(aes, config, dataset, variant)
    frozen_config = root / "experiment.json"
    if frozen_config.exists() and json.loads(frozen_config.read_text(encoding="utf-8")) != config:
        raise ValueError("Output directory already contains a different experiment")
    write_json(frozen_config, config)
    state_path = root / "status.json"
    state = {"state": "preparing", "completed": []}
    write_json(state_path, state)
    for dataset in config["datasets"]:
        inputs = root / "inputs" / dataset["name"]
        inputs.mkdir(parents=True, exist_ok=True)
        snapshot = inputs / "sfm.bin"
        if not snapshot.exists():
            shutil.copy2(dataset["sfm"], snapshot)
        sparse = inputs / "sparse.ply"
        if not sparse.exists():
            asfm = aes.AsfmOptions()
            asfm.path_base = Path(dataset["images"])
            sfm = aes.load_sfm(snapshot, asfm)
            sfm.save_sparse_ply(sparse)
            del sfm
            gc.collect()
        write_json(inputs / "source.json", {**dataset, "sfm_sha256": digest(snapshot)})
    for dataset in config["datasets"]:
        for variant in config["variants"]:
            output = root / dataset["name"] / variant["name"]
            output.mkdir(parents=True, exist_ok=True)
            label = dataset["name"] + "/" + variant["name"]
            run_record = output / "run.json"
            completed = run_record.exists() and json.loads(run_record.read_text(encoding="utf-8")).get("state") == "complete"
            if not completed:
                state.update(state="training", current=label)
                write_json(state_path, state)
                print(f"Training {label}", flush=True)
                with (output / "train.log").open("w", encoding="utf-8") as log:
                    subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                                    "--config", str(frozen_config), "--train", dataset["name"], variant["name"]],
                                   cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
            evaluator = config.get("evaluator")
            if evaluator and not (output / "evaluation" / "complete.json").exists():
                state.update(state="evaluating", current=label)
                write_json(state_path, state)
                command = [evaluator, str(root / "inputs" / dataset["name"] / "sfm.bin"),
                           dataset["images"], str(output / "model.ply"), str(output / "evaluation"),
                           str(dataset["training"]["evaluation_split_every"]), "--projection-audit"]
                with (output / "eval.log").open("w", encoding="utf-8") as log:
                    subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
                write_json(output / "evaluation" / "complete.json", {"state": "complete", "model_sha256": digest(output / "model.ply")})
            state["completed"].append(label)
            state.update(state="running", current=None)
            write_json(state_path, state)
            print(f"Completed {label}", flush=True)
    state.update(state="complete", current=None)
    write_json(state_path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train", nargs=2, metavar=("DATASET", "VARIANT"))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if args.train:
        train(settings, *args.train)
    elif args.validate:
        aes = native_module(settings)
        for dataset in settings["datasets"]:
            for variant in settings["variants"]:
                options, _ = make_options(aes, settings, dataset, variant)
                print(dataset["name"], variant["name"], options.densification_strategy,
                      options.scale_regularization_weight, options.shape_erank_reg)
    else:
        try:
            run(args.config, settings)
        except Exception as error:
            state_path = Path(settings["output_directory"]) / "status.json"
            state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
            state.update(state="failed", error=str(error))
            write_json(state_path, state)
            raise
