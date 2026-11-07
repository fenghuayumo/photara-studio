import unittest
from unittest.mock import patch
from types import SimpleNamespace
import tempfile, os
from photara_render import photara_bridge as bridge
class BridgeTests(unittest.TestCase):
 def test_json_only_handover_selects_file(self):
  props=SimpleNamespace(export_colmap=False, auto_start_training=True)
  with patch.object(bridge,'rpc',side_effect=[{'message':'loaded'},{'op_id':'train'}]) as rpc:
   bridge.handover_to_studio(props,'D:/dataset',{'session_id':'test'})
   self.assertEqual(rpc.call_args_list[0].args[2]['path'],os.path.join('D:/dataset','transforms.json'))
   self.assertEqual(rpc.call_args_list[1].args[1],'studio.job.start')
 def test_json_only_cli_selects_file(self):
  with tempfile.TemporaryDirectory() as root:
   props=SimpleNamespace(export_colmap=False,cli_exe='photara.exe',cli_args='',dataset_name='scene')
   with patch.object(bridge,'find_photara_executable',return_value='photara.exe'),patch.object(bridge,'_popen_detached') as launch:
    bridge.run_cli_training(props,root)
    self.assertEqual(launch.call_args.args[0][2],os.path.join(root,'transforms.json'))
if __name__=='__main__':unittest.main()
