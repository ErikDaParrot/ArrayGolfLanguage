import numpy as np
import main
import math

dedup = lambda x: [j for i, j in enumerate(x) if j not in x[:i]]
unique = lambda x: [int(j not in x[:i]) for i, j in enumerate(x)]
transpose = lambda x: [[[r[c] for r in x if c < len(r)] for c in range(0, max([len(r) for r in x]))]]
windows = lambda x, y: [x[i:i + y] for i in range(len(x) - y + 1)]
toBase = lambda x, y: sum([j * y ** i for i, j in enumerate(x[::-1])])
classify = lambda x: [dedup(x).index(i) for i in x]
group = lambda x, y: [i for i, j in zip(x, y) if j] if max(y) <= 1 else \
  [[x[n] for n in [j for j, k in enumerate(y) if k == i]] \
    for i in range(1, max(y) + 1)]
resize = lambda x, y: np.resize(list(x), tuple(y)).tolist()
ravel = lambda x: np.ravel(x).tolist()
parse = lambda x: [main.parseNestedBracks(i) for i in main.parseLine(x)]
trigonometry = lambda x: [0, math.sin(x), math.cos(x), math.tan(x), math.atan(x), math.acos(x), math.asin(x)]

def fromBase(x, y):
  if not x: return [0]
  r = []
  while x:
    x, m = divmod(x, y); r = [m] + r
  return r
  
def partition(x, y):
  ap, p = [], []   # all parts, part
  l = None         # last
  for i, j in zip(x, y):
    if j:
      if j != l:
        l = j; ap += [p] if p else []; p = [i]
      else: p += [i]
  ap += [p] if p else []
  return ap[0] if max(y) == 1 else ap
  
reshape = lambda x, y: (resize(ravel(x), y) if 0 not in y else \
  reshape(x, [len(ravel(x)) // math.prod([i for i in y if i != 0]) if i == 0 else i for i in y])) \
    if y.count(0) <= 1 else None