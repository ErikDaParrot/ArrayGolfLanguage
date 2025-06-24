import numpy as np
import main
from math import *
from skimage.util import *

tolist = lambda x: x if type(x) in [str, list] else [x]

dedup = lambda x: [j for i, j in enumerate(x) if j not in x[:i]]
unique = lambda x: [int(j not in x[:i]) for i, j in enumerate(x)]
transpose = lambda x: [[r[c] for r in x if c < len(r)] for c in range(0, max([len(r) for r in x]))]
smallWindows = lambda x, y: [x[i:i + y] for i in range(len(x) - y + 1)]
windows = lambda x, y: view_as_windows(np.array(x), tuple(tolist(y))).tolist()
toBase = lambda x, y: sum([j * y ** i for i, j in enumerate(x[::-1])])
classify = lambda x: [dedup(x).index(i) for i in x]
# keep = lambda x, y: [i for i, j in zip(x, y) if j] if max(y) <= 1 else []
keep = lambda x, y: sum([[0 if i < 0 else x[n]] * abs(i) for n, i in enumerate(y)], [])
group = lambda x, y: [[x[n] for n in [j for j, k in enumerate(y) if k == i]] \
  for i in range(max(y) + 1) if [x[n] for n in [j for j, k in enumerate(y) if k == i]]]
resize = lambda x, y: np.resize(list(x), tuple(y)).tolist()
select = lambda x, y: [np.array(x)[*tolist(i)].tolist() for i in y]
ravel = lambda x: np.ravel(x).tolist() if type(x) is list else x
parse = lambda x: [main.parseNestedBracks(i) for i in main.parseLine(x)]
# trigonometry = lambda x, y: [None, sin(x), cos(x), tan(x), atan(x), acos(x), asin(x)][y] if abs(x) < 4 else None
prefix = lambda x: [x[0] if i == 1 else x[:i] for i in range(1, len(x) + 1)]
suffix = lambda x: [x[-1] if i == 1 else x[-i:] for i in range(1, len(x) + 1)]
membership = lambda x, y: reshape([int(a in ravel(tolist(y))) for a in ravel(tolist(x))], np.array(tolist(x)).shape)
where = lambda x: [i if np.array(x).ndim == 1 else encode(x.shape, i) for i, j in enumerate(ravel(x)) if j]
# # reshape = lambda x, y: resize(ravel(x), [len(ravel(x)) // math.prod([i for i in y if i != 0]) \
# #  if not i else i for i in y]) if y.count(0) <= 1 else None;

printData = lambda s: '{\n' + '\n'.join([main.printData(i, 1) for i in s]) + '\n}'

def encode(x, y):
  if type(y) == list:
    r = [encode(x, i) for i in y]
    return np.array(r).T.tolist()
  else:
    r = []
    for a in x[::-1]:
      y, rem = divmod(y, a); r = [rem] + r
    return r
    
def decode(x, y):
  if type(y) == list:
    assert len(x) == len(y)
    return sum([y[i] * math.prod(x[i+1:]) for i in range(len(x))])
  else:
    return sum([j * len(x) ** i for i, j in enumerate(x[::-1])])
  

def partEnclose(x, y):
  r, l = [], None
  for i, a in enumerate(x):
    if a == 1:
      r += [y[l:i]] * (l != None); l = i
  return r + [y[l:len(x) + 1]]

def partition(x, y):
  # x, y = list(x), list(y)
  ap, p, l = [], [], None   # all parts, part, last
  # if set(y) == {0, 1}:
  #   for i, j in zip(x, y):
  #     if j == 1: p += [i]
  #     elif p:
  #       ap += [p]; p = []
  # else:
  for i, j in zip(x, y):
    if j:
      if j != l:
        l = j; ap += [p] if p else []; p = [i]
      else: p += [i]
    elif p:
      ap += [p]; p = []
  ap += [p] * (p != [])
  return ap

def partitions(x):
  if len(x) > 0:
    for i in range(1, len(x) + 1):
      first, rest = x[:i], x[i:]
      for p in partitions(rest): yield [first] + p
  else: yield []
 
def primeFactors(n):
  f = []
  while n % 2 == 0:
    f += [2]; n = n // 2
  for i in range(3, int(math.sqrt(n)) + 1, 2):
    while n % i == 0:
      f += [i]; n = n // i
  return f + ([n] * (n > 2))
 
def primesLess(n):
  prime, p = [True for i in range(n+1)], 2
  while (p * p <= n):
    if prime[p]: 
      for i in range(p * p, n + 1, p): prime[i] = False
    p += 1
  return [p for p in range(2, n + 1) if prime[p]]
  
def reshape(x, y):
  fill = None
  if -2 in y: fill, y = y[-1], y[:-1]
  if sum([i < 1 for i in y]) > 1: return None
  x = list(ravel(x)); dim = [i for i in y if i >= 1]
  dim = len(x) // math.prod(dim) if 0 in y else math.ceil(len(x) / math.prod(dim))
  dim = [dim if i < 1 else i for i in y]
  if -2 in y: x = x + [fill] * (math.prod(dim) - len(x)) if math.prod(dim) > len(x) else x[:math.prod(dim)]
  return np.resize(x, tuple(dim)).tolist()

def splitBy(x, y):
  # x, y = list(x), list(y)
  ap, p, i = [], [], 0 # all parts, part, index
  while i < len(x):
    if x[i:i + len(y)] != y: p += [x[i]]
    else: ap, p, i = ap + [p] * (p != []), [], i + len(y) - 1
    i += 1
  ap += [p] * (p != [])
  return ap
  
def trigonometry(x, y):
  try:
    match y:
      case 1: return sin(x)
      case 2: return cos(x)
      case 3: return tan(x)
      case -1: return asin(x)
      case -2: return acos(x)
      case -3: return atan(x)
      case _: return None
  except:
    return nan