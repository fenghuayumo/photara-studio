import unittest

from summarize_sfm_profile import render, summarize


class ProfileTests(unittest.TestCase):
    def test_nested_repeated_stages_and_legacy_logs(self):
        report = summarize('''
[info] stage finished: sfm.reconstruct elapsed_s=10.000
[info] stage finished: sfm.bundle_solve elapsed_s=2.000 backend=cuda
[info] stage finished: sfm.bundle_solve elapsed_s=1.000 backend=cpu
[info] stage finished: sfm.bundle_solve elapsed_s=3.000 backend=cuda
[info] stage finished: unrelated elapsed_s=20.000
[info] stage finished: sfm.triangulate_tracks elapsed_s=1e-3 backend=cpu
'''.splitlines())
        self.assertEqual(len(report['stages']), 4)
        cuda = next(r for r in report['stages'] if r['backend'] == 'cuda')
        self.assertEqual((cuda['calls'], cuda['total_s'], cuda['max_s']), (2, 5, 3))
        self.assertEqual(report['stages'][0]['backend'], 'unspecified')
        self.assertIn('Do not sum', render(report))

    def test_decisions_transfers_and_runtime_failure(self):
        report = summarize('''
[info] sfm backend selection: stage=bundle requested=automatic selected=cpu reason=partial_pose_locks
[info] sfm backend selection: stage=bundle requested=automatic selected=cpu reason=partial_pose_locks
[warning] CUDA bearing solve failed; using CPU: unusable solve
[info] checkpoint hit: features
[debug] CUDA match timing: upload_ms=1 compute_ms=2 download_ms=3
[debug] CUDA match timing: upload_ms=4 compute_ms=5 download_ms=6
[info] match/verify timing: pairs=8 match_wall_s=1 geometry_work_sum_s=4 wall_s=2
'''.splitlines())
        self.assertEqual(report['backend_decisions'][0]['calls'], 2)
        self.assertEqual(report['checkpoints'], {'features': 1})
        self.assertEqual(sum(report['fallback_messages'].values()), 1)
        self.assertEqual(report['matcher_timing']['download_ms'], 9)
        self.assertEqual(report['match_verify_pipeline']['geometry_work_sum_s'], 4)

    def test_empty_log(self):
        text = render(summarize([]))
        self.assertIn('No SfM stage timings', text)
        self.assertIn('unavailable', text)


if __name__ == '__main__':
    unittest.main()
