import sys, time
sys.path.insert(0, 'src')
from dagflow.core.graph import ComputeGraph
from dagflow.memo.cache import MemoCache
from dagflow.propagation.eager import EagerPropagator

def build_wide_graph(width):
    g = ComputeGraph()
    g.add_input('root', value=1)
    for i in range(width):
        g.add_compute(f'n{i}', func=lambda d: d['root'] * 2, dependencies=['root'])
    return g

def bench(width=1000, iters=100):
    g = build_wide_graph(width)
    c = MemoCache()
    p = EagerPropagator(g, c)
    t = time.perf_counter()
    for _ in range(iters):
        p.propagate(['root'])
    e = time.perf_counter() - t
    print(f'Width={width}, Iters={iters}: {e:.3f}s ({e/iters*1000:.2f}ms/iter)')

if __name__ == '__main__':
    bench(100, 50)
    bench(1000, 20)
