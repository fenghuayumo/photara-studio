"""Exercise bake-only preprocessing with the built CLI, Vulkan, and CGAL.

Run: python photara/tests/texture_bake_cli_test.py path/to/photara.exe
"""

from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import math


def write_image(path):
    width = height = 64
    pixels = bytes((80, 150, 210)) * width * height
    header = struct.pack('<2sIHHI', b'BM', 54 + len(pixels), 0, 0, 54)
    info = struct.pack('<IiiHHIIiiII', 40, width, height, 1, 24, 0,
                       len(pixels), 2835, 2835, 0, 0)
    path.write_bytes(header + info + pixels)


def write_mesh(path, non_manifold=False):
    vertices = []
    for center in (-0.6, 0.6):
        vertices.extend(((center - 0.2, -0.2, 3),
                         (center + 0.2, -0.2, 3),
                         (center, 0.2, 3), (center, 0, 3.4)))
    faces = [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)]
    faces += [tuple(i + 4 for i in f) for f in faces]
    if non_manifold:
        vertices.append((-0.6, 0, 2.8))
        faces.append((0, 1, 8))  # Third face sharing an existing tetrahedron edge.
    text = (f'ply\nformat ascii 1.0\nelement vertex {len(vertices)}\n'
            'property float x\nproperty float y\nproperty float z\n'
            f'element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n')
    text += ''.join(' '.join(map(str, v)) + '\n' for v in vertices)
    text += ''.join('3 ' + ' '.join(map(str, f)) + '\n' for f in faces)
    path.write_text(text, encoding='ascii')


def read_x_coordinates(path):
    with path.open('rb') as stream:
        properties = []
        in_vertex = False
        count = 0
        while True:
            line = stream.readline().decode('ascii').strip()
            if line.startswith('element '):
                _, name, value = line.split()
                in_vertex = name == 'vertex'
                if in_vertex:
                    count = int(value)
            elif line.startswith('property ') and in_vertex:
                _, scalar, name = line.split()
                properties.append((scalar, name))
            elif line == 'end_header':
                break
            elif not line:
                raise AssertionError('Unexpected end of PLY header')
        codes = {'float': 'f', 'double': 'd', 'uchar': 'B'}
        record = struct.Struct('<' + ''.join(codes[t] for t, _ in properties))
        x_slot = [name for _, name in properties].index('x')
        return [record.unpack(stream.read(record.size))[x_slot] for _ in range(count)]


def write_sphere(path):
    vertices = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
             (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    for _ in range(2):
        midpoints = {}
        def midpoint(a, b):
            edge = tuple(sorted((a, b)))
            if edge not in midpoints:
                v = tuple(vertices[a][axis] + vertices[b][axis] for axis in range(3))
                length = math.sqrt(sum(c*c for c in v))
                midpoints[edge] = len(vertices)
                vertices.append(tuple(c/length for c in v))
            return midpoints[edge]
        refined = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            refined.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
        faces = refined
    text = (f'ply\nformat ascii 1.0\nelement vertex {len(vertices)}\n'
            'property float x\nproperty float y\nproperty float z\n'
            f'element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n')
    text += ''.join(f'{x*0.5} {y*0.5} {z*0.5+3}\n' for x, y, z in vertices)
    text += ''.join('3 ' + ' '.join(map(str, f)) + '\n' for f in faces)
    path.write_text(text, encoding='ascii')


def main():
    executable = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix='photara_bake_test_') as temporary:
        root = Path(temporary)
        images = root / 'images'
        sparse = root / 'sparse'
        images.mkdir()
        sparse.mkdir()
        for name in ('left.bmp', 'right.bmp'):
            write_image(images / name)
        (sparse / 'cameras.txt').write_text('1 SIMPLE_PINHOLE 64 64 40 32 32\n')
        (sparse / 'images.txt').write_text(
            '1 1 0 0 0 0 0 0 1 left.bmp\n\n'
            '2 1 0 0 0 -0.1 0 0 1 right.bmp\n\n')
        (sparse / 'points3D.txt').write_text('1 0 0 3 128 128 128 0\n')
        source = root / 'source.ply'
        write_mesh(source)
        original = source.read_bytes()
        base = [str(executable), '--images', str(images), '--splat-dataset', str(sparse),
                '--working-mesh', str(source), '--texture', '--texture-optimize=false',
                '--atlas-resolution', '128']
        for name, extra in (('processed', ['--mesh-target-faces', '4']),
                            ('default', []), ('disabled', ['--mesh-target-faces', '0']),
                            ('surface_fill', ['--texture-fill-unobserved=true',
                                              '--uv-max-stretch', '0.6'])):
            result = subprocess.run(base + extra + ['--output', str(root / (name + '.ply'))],
                                    capture_output=True, text=True, errors='replace', timeout=120)
            assert result.returncode == 0, result.stdout + result.stderr
            assert (root / (name + '_textured.obj')).exists(), result.stdout
            if name == 'processed':
                xs = read_x_coordinates(root / 'processed_bake_mesh.ply')
                assert min(xs) < -0.3 and max(xs) > 0.3, 'A disconnected subject was removed'
                assert 'mesh.photara_repair_decimate' in result.stdout, result.stdout
                assert 'mesh.instant_remesh' not in result.stdout, result.stdout
            elif name == 'default':
                assert 'mesh.photara_repair_decimate' not in result.stdout, result.stdout
                assert not (root / 'default_bake_mesh.ply').exists()
                assert sum(line.startswith('f ') for line in
                           (root / 'default_textured.obj').read_text().splitlines()) == 8
            elif name == 'disabled':
                assert 'mesh.photara_repair_decimate' not in result.stdout, result.stdout
                assert not (root / 'disabled_bake_mesh.ply').exists()
            else:
                assert 'texture surface fill:' in result.stdout, result.stdout
                assert 'max_stretch=0.6' in result.stdout, result.stdout
            assert source.read_bytes() == original, 'Baking modified the source mesh'
        for stretch in ('-0.1', '1.1', 'nan'):
            result = subprocess.run(base + ['--uv-max-stretch', stretch,
                                    '--output', str(root / 'invalid.ply')],
                                    capture_output=True, text=True, errors='replace', timeout=120)
            output = result.stdout + result.stderr
            assert result.returncode != 0 and (
                '--uv-max-stretch must be finite' in output or
                (stretch == 'nan' and "Argument 'nan' failed to parse" in output)), output
        for quality, target in (('fast', 500000), ('standard', 1000000), ('high', 2000000)):
            result = subprocess.run(base + ['--mesh-postprocess-quality', quality,
                                    '--output', str(root / ('preset_' + quality + '.ply'))],
                                    capture_output=True, text=True, errors='replace', timeout=120)
            output = result.stdout + result.stderr
            assert result.returncode == 0 and f'target_faces={target}' in output, output
            assert 'mesh.instant_remesh' not in output, output
        sphere = root / 'sphere.ply'
        write_sphere(sphere)
        sphere_source = sphere.read_bytes()
        sphere_command = base.copy()
        sphere_command[sphere_command.index('--working-mesh') + 1] = str(sphere)
        # Explicit remesh must run with simplification disabled, and even when
        # the reduction budget is greater than the input face count (128).
        for name, reduction in (('remesh_only', 0), ('remesh_below_cap', 1000)):
            result = subprocess.run(sphere_command + ['--mesh-target-faces', str(reduction),
                                    '--mesh-remesh=true', '--mesh-remesh-target-faces', '256',
                                    '--mesh-remesh-smooth-iterations', '0',
                                    '--output', str(root / (name + '.ply'))],
                                    capture_output=True, text=True, errors='replace', timeout=120)
            output = result.stdout + result.stderr
            assert result.returncode == 0, output
            assert 'instant mesh config: target_triangles=256 smooth_iterations=0' in output, output
            assert 'Instant Meshes failed' not in output, output
            assert (root / (name + '_bake_mesh.ply')).exists(), output
            assert sphere.read_bytes() == sphere_source
        for option, value in (('--mesh-remesh-target-faces', '3'),
                              ('--mesh-remesh-smooth-iterations', '-1'),
                              ('--mesh-remesh-smooth-iterations', '11'),
                              ('--mesh-postprocess-quality', 'invalid')):
            result = subprocess.run(base + [option, value, '--output', str(root / 'invalid_mesh.ply')],
                                    capture_output=True, text=True, errors='replace', timeout=120)
            assert result.returncode != 0 and option in (result.stdout + result.stderr)
        non_manifold = root / 'non_manifold.ply'
        write_mesh(non_manifold, non_manifold=True)
        non_manifold_original = non_manifold.read_bytes()
        repair_command = base.copy()
        repair_command[repair_command.index('--working-mesh') + 1] = str(non_manifold)
        result = subprocess.run(repair_command + ['--output', str(root / 'repaired.ply')],
                                capture_output=True, text=True, errors='replace', timeout=120)
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert 'texture.manifold_repair' in output, output
        assert 'mesh.photara_repair_decimate' not in output, output
        assert sum(line.startswith('f ') for line in
                   (root / 'repaired_textured.obj').read_text().splitlines()) == 9
        assert non_manifold.read_bytes() == non_manifold_original
        collision = root / 'collision_bake_mesh.ply'
        collision.write_bytes(original)
        collision_command = base.copy()
        collision_command[collision_command.index('--working-mesh') + 1] = str(collision)
        result = subprocess.run(collision_command + ['--mesh-target-faces', '4',
                                '--output', str(root / 'collision.ply')],
                                capture_output=True, text=True, errors='replace', timeout=120)
        output = result.stdout + result.stderr
        assert result.returncode != 0 and 'aliases the input mesh' in output, output
        assert collision.read_bytes() == original, 'An aliased source mesh was overwritten'
    print('Manifold repair without face reduction, optional simplification, and source preservation passed')


if __name__ == '__main__':
    main()
