import unittest
from copy import deepcopy
from recipes import ReconstructionRecipe


class FakeClient:
    def __init__(self):
        self.state = {'session_id': 'one', 'project': {'images': 'images', 'path': None}, 'job': {}}
        self.calls = []
        self.quality = 'passed'

    async def call(self, *args, **kwargs):
        self.calls.append(args)
        if args == ('state',):
            return deepcopy(self.state)
        if args[:2] == ('job', 'start'):
            return {'op_id': args[2]}
        if args == ('quality',):
            return {'checks': [{'status': self.quality}]}
        return {'ok': True}

    async def wait(self, op_id, timeout):
        return {'status': 'succeeded'}


class RecipeTests(unittest.IsolatedAsyncioTestCase):
    async def test_order_and_quality_gate(self):
        for quality, final in [('passed', 'succeeded'), ('unknown', 'needs_review'), ('failed', 'needs_review')]:
            client = FakeClient()
            client.quality = quality
            recipe = ReconstructionRecipe(client)
            run = await recipe.start(20)
            for _ in range(3):
                run = await recipe.advance(run['run_id'])
            self.assertEqual(run['status'], final)
            self.assertEqual([c[2] for c in client.calls if c[:2] == ('job', 'start')], ['align', 'train'])

    async def test_session_change_stops_plan(self):
        client = FakeClient()
        recipe = ReconstructionRecipe(client)
        run = await recipe.start()
        client.state['session_id'] = 'two'
        run = await recipe.advance(run['run_id'])
        self.assertEqual(run['status'], 'failed')
        self.assertFalse(any(c[:2] == ('job', 'start') for c in client.calls))

    async def test_duplicate_recipe_rejected(self):
        recipe = ReconstructionRecipe(FakeClient())
        await recipe.start()
        with self.assertRaises(RuntimeError):
            await recipe.start()

    async def test_failed_wait_stops_plan(self):
        client = FakeClient()
        async def failed(*args, **kwargs):
            raise RuntimeError('training failed')
        client.wait = failed
        recipe = ReconstructionRecipe(client)
        run = await recipe.start()
        await recipe.advance(run['run_id'])
        self.assertEqual((await recipe.advance(run['run_id']))['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
