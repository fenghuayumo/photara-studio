"""Deterministic, inspectable reconstruction plan with bounded step execution.

The host calls advance repeatedly. Each call starts or observes at most one job;
there is no background job lifecycle hidden in this Python process.
"""
import uuid


class ReconstructionRecipe:
    def __init__(self, client):
        self.client = client
        self.runs = {}

    async def start(self, iterations=30000):
        if not 1 <= iterations <= 1000000:
            raise ValueError('iterations must be in [1, 1000000]')
        state = await self.client.call('state')
        if state['job'].get('running') or state['job'].get('preparing'):
            raise RuntimeError('Studio is busy')
        if len(self.runs) >= 128:
            raise RuntimeError('Recipe history is full; restart the bridge after finishing active recipes')
        if any(r['status'] in ('ready', 'running') for r in self.runs.values()):
            raise RuntimeError('A recipe is already active')
        await self.client.call('params', 'set', f'iterations={iterations}', mutation=True)
        run_id = uuid.uuid4().hex
        self.runs[run_id] = {'run_id': run_id, 'session_id': state['session_id'],
            'project': state['project'], 'steps': ['align', 'train'], 'step': 0,
            'op_id': None, 'status': 'ready'}
        return dict(self.runs[run_id])

    async def advance(self, run_id):
        run = self.runs[run_id]
        if run['status'] in ('succeeded', 'failed', 'cancelled', 'needs_review'):
            return dict(run)
        state = await self.client.call('state')
        if state['session_id'] != run['session_id'] or state['project'] != run['project']:
            run['status'] = 'failed'
            run['reason'] = 'Studio session or input changed'
            return dict(run)
        if run['op_id']:
            try:
                result = await self.client.wait(run['op_id'], timeout=1)
            except RuntimeError as error:
                run['status'], run['reason'] = 'failed', str(error)
                return dict(run)
            if result.get('timed_out'):
                return dict(run)
            if result.get('status') != 'succeeded':
                run['status'] = 'failed'
                return dict(run)
            run['step'] += 1
            run['op_id'] = None
        if run['step'] == len(run['steps']):
            run['quality'] = await self.client.call('quality')
            statuses = [check['status'] for check in run['quality']['checks']]
            run['status'] = 'succeeded' if all(s == 'passed' for s in statuses) else 'needs_review'
        else:
            result = await self.client.call('job', 'start', run['steps'][run['step']], mutation=True)
            run['op_id'], run['status'] = result['op_id'], 'running'
        return dict(run)
