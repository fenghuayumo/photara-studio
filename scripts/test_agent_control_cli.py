"""Exercise the native CLI against a deterministic local RPC fixture."""
import json
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CLI = sys.argv.pop(1)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.requests.append(request)
        result = self.server.respond(request)
        body = json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': result}).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class CliContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.server.requests = []

    def run_cli(self, args, respond):
        self.server.respond = respond
        process = subprocess.run(
            [CLI, '--url', f'http://127.0.0.1:{self.server.server_port}',
             '--token', 'fixture', '--json', *args],
            capture_output=True, text=True, encoding='utf-8', timeout=10)
        return process.returncode, json.loads(process.stdout)

    def test_terminal_states(self):
        for status, exit_code, expected in [('succeeded', 0, 0), ('failed', 1, 2), ('cancelled', 0, 2)]:
            with self.subTest(status=status):
                code, body = self.run_cli(['job', 'wait', '--op', 'op_7'], lambda _: {
                    'ok': True, 'job': {'op_id': 'op_7', 'running': False,
                    'status': status, 'exit_code': exit_code}})
                self.assertEqual(code, expected)
                self.assertEqual(body['status'], status)
                self.assertEqual(self.server.requests[-1]['params'], {'op_id': 'op_7'})

    def test_timeout_is_nonzero(self):
        code, body = self.run_cli(['job', 'wait', '--op', 'op_8', '--timeout', '1'],
            lambda _: {'ok': True, 'job': {'op_id': 'op_8', 'running': True}})
        self.assertEqual(code, 5)
        self.assertTrue(body['timed_out'])

    def test_wait_binds_first_observed_operation(self):
        def respond(request):
            if len(self.server.requests) == 1:
                return {'ok': True, 'job': {'op_id': 'op_9', 'running': True}}
            self.assertEqual(request['params'], {'op_id': 'op_9'})
            return {'ok': True, 'job': {'op_id': 'op_9', 'running': False, 'status': 'succeeded', 'exit_code': 0}}
        code, body = self.run_cli(['job', 'wait'], respond)
        self.assertEqual(code, 0)
        self.assertEqual(body['op_id'], 'op_9')

    def test_status_filters_operation(self):
        code, _ = self.run_cli(['job', 'status', '--op', 'op_10'],
            lambda _: {'ok': True, 'job': {'op_id': 'op_10'}})
        self.assertEqual(code, 0)
        self.assertEqual(self.server.requests[-1]['params'], {'op_id': 'op_10'})

    def test_unknown_operation(self):
        code, body = self.run_cli(['job', 'wait', '--op', 'missing'], lambda _: {
            'ok': False, 'error': {'code': 'NOT_FOUND', 'message': 'Unknown operation'}})
        self.assertEqual(code, 2)
        self.assertEqual(body['error']['code'], 'NOT_FOUND')

    def test_mutation_arguments(self):
        cases = [
            (['--request-id', 'save-1', 'project', 'save', 'C:/temp/demo.ascan', '--overwrite'],
             'studio.project.save', {'path': 'C:/temp/demo.ascan', 'overwrite': True}),
            (['view', 'set', 'yaw=1', 'target=[1,2,3]'],
             'studio.view.patch', {'patch': {'yaw': 1, 'target': [1, 2, 3]}}),
            (['export', 'model', '--out', 'C:/temp/model.ply', '--sh-degree', '0'],
             'studio.export.model', {'path': 'C:/temp/model.ply', 'sh_degree': 0}),
        ]
        for args, method, params in cases:
            with self.subTest(method=method):
                code, _ = self.run_cli(args, lambda _: {'ok': True})
                self.assertEqual(code, 0)
                request = self.server.requests[-1]
                self.assertEqual(request['method'], method)
                self.assertEqual(request['params'], params)
                if '--request-id' in args:
                    self.assertEqual(request['request_id'], 'save-1')


if __name__ == '__main__':
    unittest.main()
