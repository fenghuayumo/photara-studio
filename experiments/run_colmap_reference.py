"""Create a retained COLMAP reference with explicit camera initialization."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--camera-model', default='OPENCV_FISHEYE')
    parser.add_argument('--camera-params', required=True)
    parser.add_argument('--timeout-seconds', type=float, default=1800)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error('Timeout must be positive')
    root = Path(args.output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        parser.error('Output directory must be new or empty')
    root.mkdir(parents=True, exist_ok=True)
    sparse = root / 'sparse'
    sparse.mkdir()
    db = str(root / 'database.db')
    commands = [
        ('extract', ['feature_extractor', '--database_path', db,
          '--image_path', str(Path(args.images).resolve()), '--ImageReader.single_camera', '1',
          '--ImageReader.camera_model', args.camera_model, '--ImageReader.camera_params', args.camera_params,
          '--SiftExtraction.max_num_features', '6000', '--SiftExtraction.peak_threshold', '0.005',
          '--FeatureExtraction.use_gpu', '1']),
        ('match', ['exhaustive_matcher', '--database_path', db, '--FeatureMatching.use_gpu', '1',
          '--SiftMatching.max_ratio', '0.8', '--TwoViewGeometry.random_seed', '0']),
        ('map', ['mapper', '--database_path', db, '--image_path', str(Path(args.images).resolve()),
          '--output_path', str(sparse), '--Mapper.num_threads', '8', '--Mapper.random_seed', '0']),
    ]
    records = []
    for label, command in commands:
        command.insert(0, str(Path(args.exe).resolve()))
        started = time.perf_counter()
        with (root / (label+'.log')).open('w', encoding='utf-8') as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=args.timeout_seconds)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 'timeout'
        records.append(dict(stage=label, command=command, exit_code=code,
                            seconds=time.perf_counter()-started))
        (root / 'run.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
        print(label, code, records[-1]['seconds'], flush=True)
        if code != 0:
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
