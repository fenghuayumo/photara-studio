import csv
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sfm_acceptance import evaluate, reference_poses, rotation, similarity


class AcceptanceTests(unittest.TestCase):
    def test_similarity_and_world_to_camera_convention(self):
        x = np.random.default_rng(12).normal(size=(20, 3))
        r = rotation([.9, .1, -.2, .3])
        y = 2.7*x@r.T + [4, -2, 1]
        scale, estimated, translation = similarity(x, y)
        np.testing.assert_allclose(scale*x@estimated.T+translation, y, atol=1e-12)
        np.testing.assert_allclose(estimated, r, atol=1e-12)
        camera = rotation([.8, .2, .3, -.1])
        np.testing.assert_allclose((camera@r.T)@r, camera, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(estimated), 1)

    def test_unregistered_view_fails_even_with_perfect_reprojection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fields = ['name','registered','center_x','center_y','center_z',
                      'qw','qx','qy','qz','observations','reprojection_rms_px','reprojection_p95_px']
            xyz = np.random.default_rng(42).normal(size=(8,3))
            with (root/'images.txt').open('w') as reference, (root/'diagnostics.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(fields)
                for i, p in enumerate(xyz):
                    reference.write(f'{i+1} 1 0 0 0 {-p[0]} {-p[1]} {-p[2]} 1 image{i}.png\n\n')
                    writer.writerow([f'image{i}.png',int(i!=7),*p,1,0,0,0,100,.1,.2])
            result = evaluate(root/'diagnostics.csv', root)
            self.assertFalse(result['passed'])
            self.assertFalse(result['gates']['registration'])
            self.assertTrue(result['gates']['center_p95'])
            self.assertTrue(result['gates']['rotation_p95'])

    def test_binary_reference_skips_points_and_preserves_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            with (Path(tmp)/'images.bin').open('wb') as f:
                f.write(struct.pack('<Q',2))
                for i in range(2):
                    f.write(struct.pack('<i7di',i+1,1,0,0,0,2+i,3,4,1))
                    f.write(f'view {i}.png'.encode()+b'\0')
                    f.write(struct.pack('<Q',3))
                    f.write(struct.pack('<ddq',.5,.7,-1)*3)
            poses=reference_poses(tmp)
            np.testing.assert_allclose(poses['view 1.png'][0],[-3,-3,-4])
            np.testing.assert_allclose(poses['view 1.png'][1],np.eye(3))

    def test_evaluator_removes_similarity_but_not_camera_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            q=np.array([.9,.1,-.2,.3]); q/=np.linalg.norm(q)
            r=rotation(q)
            xyz=np.random.default_rng(4).normal(size=(20,3))
            y=2.7*xyz@r.T+[4,-2,1]
            fields=['name','registered','center_x','center_y','center_z',
                    'qw','qx','qy','qz','observations','reprojection_rms_px','reprojection_p95_px']
            with (root/'images.txt').open('w') as ref, (root/'diag.csv').open('w',newline='') as f:
                writer=csv.writer(f); writer.writerow(fields)
                for i,(x,c) in enumerate(zip(xyz,y)):
                    t=-r.T@c
                    qr=q*np.array([1,-1,-1,-1])
                    ref.write(f'{i+1} '+ ' '.join(map(str,[*qr,*t]))+f' 1 image{i}.png\n\n')
                    writer.writerow([f'image{i}.png',1,*x,1,0,0,0,100,.1,.2])
            result=evaluate(root/'diag.csv',root)
            self.assertTrue(result['passed'])
            self.assertLess(result['rotation_error_deg']['max'],1e-5)
            with (root/'diag.csv').open() as f:
                rows=list(csv.DictReader(f))
            rows[0]['center_x']=str(float(rows[0]['center_x'])+20)
            with (root/'diag.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields)
                writer.writeheader(); writer.writerows(rows)
            self.assertFalse(evaluate(root/'diag.csv',root)['gates']['center_p95'])


if __name__ == '__main__':
    unittest.main()
