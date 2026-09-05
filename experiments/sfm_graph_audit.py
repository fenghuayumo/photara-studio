"""Report bridge-dependent camera branches without using reference poses.

Graph vulnerability is not proof of incorrect poses: landmark tracks can provide
additional constraints absent from the pair graph. This audit never prunes poses.
"""
import argparse
import csv
import json
from pathlib import Path


def audit(diagnostics, pairs):
    with open(diagnostics, encoding='utf-8-sig', newline='') as f:
        images = {int(r['image_id']): r['name'] for r in csv.DictReader(f)
                  if int(r['registered'])}
    adjacency = {i: set() for i in images}
    with open(pairs, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            a, b = int(row['image_id1']), int(row['image_id2'])
            if (int(row['active']) and float(row['composite_weight']) > 0
                    and a != b and a in images and b in images):
                adjacency[a].add(b)
                adjacency[b].add(a)
    # Iterative Tarjan traversal avoids Python recursion limits on long videos.
    discovery, low, parent, bridges = {}, {}, {}, set()
    components = []
    for root in sorted(images):
        if root in discovery:
            continue
        component = []
        discovery[root] = low[root] = len(discovery)
        parent[root] = None
        stack = [(root, iter(sorted(adjacency[root])))]
        while stack:
            node, neighbors = stack[-1]
            try:
                neighbor = next(neighbors)
            except StopIteration:
                stack.pop()
                component.append(node)
                p = parent[node]
                if p is not None:
                    low[p] = min(low[p], low[node])
                    if low[node] > discovery[p]:
                        bridges.add(tuple(sorted((node, p))))
                continue
            if neighbor == parent[node]:
                continue
            if neighbor in discovery:
                low[node] = min(low[node], discovery[neighbor])
            else:
                parent[neighbor] = node
                discovery[neighbor] = low[neighbor] = len(discovery)
                stack.append((neighbor, iter(sorted(adjacency[neighbor]))))
        components.append(component)
    blocks, seen = [], set()
    for root in sorted(images):
        if root in seen:
            continue
        pending, block = [root], []
        seen.add(root)
        while pending:
            node = pending.pop()
            block.append(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor not in seen and tuple(sorted((node, neighbor))) not in bridges:
                    seen.add(neighbor)
                    pending.append(neighbor)
        blocks.append(sorted(block))
    blocks.sort(key=lambda b: (-len(b), b[0]))
    return dict(registered=len(images), connected_components=len(components),
                bridge_count=len(bridges),
                bridges=[dict(first=images[a], second=images[b])
                         for a, b in sorted(bridges)],
                blocks=[dict(size=len(b), cameras=[images[i] for i in b]) for b in blocks],
                interpretation='Pair-graph vulnerability only; verify landmark support before pruning.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics', required=True)
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit(args.diagnostics, args.pairs)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('blocks', 'bridges')}, indent=2))
