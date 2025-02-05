import numpy as np
import random as rand
import math, random
import func
import itertools as it
import time, re
import datetime as dt
import calendar as cal

f, I, S, L, F = [complex, float, int], [int], [str], [list], [tuple]
s, A, i, N, a = [complex, float, int, str], [str, list], [int, list], [float, int, list], [complex, float, int, str, list]

findPop = lambda x, y, z, t: max(0, findSig(x, y + z)[0] - t)
findPush = lambda x, y, z, t: max(0, findSig(x, y + z)[1] - t)
VIV = lambda x, y: x if x else y # VALUE IF VALUE IS NOT 'EMPTY'

vars = {}

CONSTANTS = {
  'a': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
  'd': '0123456789',
  'p': func.primesLess(1000005),
  'P': math.pi,
  'E': math.e,
  'R': 180 / math.pi,
}

FUNCTIONS = {
  ## BASIC FUNCTIONS
  '+': [
    [f, f], lambda x, y: [x + y],
    [L, L], lambda x, y: [x + y],
    [L, s], lambda x, y: [x + [y]],
    [s, L], lambda x, y: [[x] + y],
    [S, S], lambda x, y: [x + y]
  ], '-': [
    [f, f], lambda x, y: [x - y],
    [A], lambda x: [func.classify(x)],
    [A, F], lambda x, y, z: [None, *run([y + (0, '=', '!'), '%', x, ':', '*'], z + [x])]
  ], '*': [
    [f, f], lambda x, y: [x * y],
    [A, I], lambda x, y: [x * y],
    [I, A], lambda x, y: [x * y],
    [A, L], lambda x, y: [func.group(x, y)],
    [F, I], lambda x, y, z: [None, *forLoop(x, y, z)],
    [F, F], lambda x, y, z: [None, *whileLoop(x, y, z)]
  ], '/': [
    [f, f], lambda x, y: [x / y],
    [A, I], lambda x, y: [func.windows(x, y)],
    [A, L], lambda x, y: [func.partition(x, y)],
    [A, F], lambda x, y, z: [fold(x, y, z, 0)]
  ], '%': [
    [f, f], lambda x, y: [x % y],
    [A], lambda x: [func.transpose(x)],
    [A, F], lambda x, y, z: [None, *map(x, y, z)],
  ], '^': [
    [f, f], lambda x, y: [x ** y],
    [A], lambda x: [func.dedup(x)]
  ], '!': [
    [f], lambda x: [1 - x],
    [A], lambda x: [func.unique(x)],
    [F], lambda x, y: [None, *run(x, y)]
  ], '~': [
    [f], lambda x: [-x],
    [L], lambda x: [func.ravel(x)],
    [S], lambda x, y: [None, *run(func.parse(x), y)]
  ], '|': [
    [f, f], lambda x, y: [x or y],
    [A, I], lambda x, y: [x[y:] + x[:y]],
    [A, L], lambda x, y: [func.reshape(x, y)],
    [A, A, F], lambda x, y, z, t: [None, *zipmap(x, y, z, t)]
  ], '<': [
    [f, f], lambda x, y: [x < y],
    [A, I], lambda x, y: [x[:y]],
    [A], lambda x: [x[0]], 
    [a, a, F], lambda x, y, z, t: [[x, y] + run(z, t)]
  ], '>': [
    [f, f], lambda x, y: [x > y],
    [A, I], lambda x, y: [x[y:]],
    [A], lambda x: [x[-1]],
    [a, a, F], lambda x, y, z, t: [run(z, t) + [x, y]]
  ], '=': [
    [a, a], lambda x, y: [x == y],
    [A, A, F], lambda x, y, z, t: [None, *table(x, y, z, t)]
  ], '\\': [
    [I], lambda x: [1 / x],
    [A], lambda x: [x[::-1]],
    [A, F], lambda x, y, z: [fold(x, y, z, 1)]
  ], '@': [
    [f, f], lambda x, y: [math.log(x, y)],
    [A, I], lambda x, y: [x[y]],
    [L, L], lambda x, y: [func.select(x, y)]
  ], '?': [
    [A, a], lambda x, y: [y in x],
    [I, L], lambda x, y, z: [None, *run(y[x], z)]
  ], '&': [
    [a, a], lambda x, y: [[x, y]],
    [a, a, F], lambda x, y, z, t: [run(z, t + [x]), run(z, t + [y])],
  ], '#': [
    [I], lambda x: [list(range(x))],
    [A], lambda x: [len(x)],
    [A, F], lambda x, y, z: [None, *zipmap(list(range(len(x))), x, y, z)]
  ], '_': [
    [a], lambda x: [[x]]
  ], '_~': [
    [f], lambda x: [(x > 0) - (x < 0)],
    [S], lambda x: [x.swapcase()],
  ], '_\\': [
    [f], lambda x: [math.sqrt(x.real**2 + x.imag**2)]
  ], '_<': [
    [f, f], lambda x, y: [min(x, y)],
    [S], lambda x: [x.lower() if len(x) == 1 else func.suffix(x)],
    [A], lambda x: [func.suffix(x)]
  ], '_>': [
    [f, f], lambda x, y: [max(x, y)],
    [S], lambda x: [x.upper() if len(x) == 1 else func.suffix(x)],
    [A], lambda x: [func.prefix(x)]
  ], '_#': [
    [I, I], lambda x, y: [list(range(x, y))],
    [L], lambda x: [list(np.array(x).shape)]
  ], '$': [
    [A], lambda x: [sorted(x)],
    [A, F], lambda x, y, z: [sorted(x, key = lambda i: run(y, z + [i])[-1])]
  ], '.': [
    [a], lambda x: [x, x]
  ], ',': [
    [a, a], lambda x, y: [x, y, x]
  ], ':': [
    [a, a], lambda x, y: [y, x]
  ], ';': [
    [a], lambda x: []
  ], '(': [
    [], lambda z: [None, *(z[1:] + z[:1])]
  ], ')': [
    [], lambda z: [None, *(z[-1:] + z[:-1])]
  ], '`': [
    [I], lambda x, y: [y[-x - 1]]
  ], 
  ## EXTERNAL FUNCTIONS
  # MATH FUNCTIONS
  'm[': [
    [f], lambda x: [math.floor(x)]
  ], 'm]': [
    [f], lambda x: [math.ceil(x)]
  ], 'm~': [
    [f], lambda x: [round(x)]
  ], 'mt': [
    [f, I], lambda x, y: [func.trigonometry(y)[x]]
  ], 'mE': [
    [I], lambda x: [math.exp(x)]
  ], 'm!': [
    [f], lambda x: [math.gamma(x)]
  ], 'mp': [
    [I], lambda x: [all(x % i > 0 for i in range(int(x ** 0.5) + 1)[2:])]
  ],
  # TUPLE FUNCTIONS
  'tp': [
    [f, f], lambda x, y: [math.perm(x, y)],
    [A, I], lambda x, y: [[list(i) for i in list(it.permutations(x, y))]]
  ], 'tc': [
    [f, f], lambda x, y: [math.comb(x, y)],
    [A, I], lambda x, y: [[list(i) for i in list(it.combinations(x, y))]]
  ], 'tC': [
    [A, I], lambda x, y: [[list(i) for i in list(it.combinations_with_replacement(x, y))]]
  ],
  # COMPLEX FUNCTIONS
  'c+': [
    [f[1:], f[1:]], lambda x, y: [complex(x, y)]
  ], 'c:': [
    [f[0]], lambda x: [x.real, x.imag]
  ], 'c-': [
    [f[0]], lambda x: [complex(x.real, -x.imag)]
  ],
  # STRING FUNCTIONS
  's#': [
    [I], lambda x: [chr(x)],
    [S], lambda x: [[ord(i) for i in x] if len(x) == 1 else ord(x)],
    [L], lambda x: [chr(i) for i in x],
  ], 's*': [
    [S, S], lambda x, y: [fold(x, (':', y, '+', ':', '+'), [], 0)],
    [L, a], lambda x, y: [fold(x, (':', y, '+', ':', '+'), [], 0)],
  ], 'sR': [
    [S, S], lambda x, y: [re.findall(y, x)]
  ], 'sr': [
    [S, S, S], lambda x, y, z: [x.replace(y, z)]
  ], 's!': [
    [S, S], lambda x, y: [x.replace(y, '')]
  ],
  # IN/OUT FUNCTIONS
  'p.': [
    [a], lambda x: ([], [print(x, end = '')])[0],
  ], 'p!': [
    [a], lambda x: ([], [print(x)])[0],
  ], 'p?': [
    [], lambda: [input()],
  ],
  # DATETIME FUNCTIONS
  'd>': [
    [f], lambda x: [list(time.gmtime(x))[:6]]
  ], 'd<': [
    [L], lambda x: [dt.datetime(*x, tzinfo = dt.timezone.utc).timestamp()]
  ], 'd.': [
    [], lambda: [time.time()]
  ], 'd|': [
    [], lambda: [time.localtime().tm_gmtoff / 3600]
  ],
  # RANDOM FUNCTIONS
  'r~': [
    [], lambda x: [random.random()],
  ], 'r-': [
    [I, I], lambda x, y: [random.randint(x, y)],
  ], 'r?': [
    [A], lambda x: [random.choice(x)],
  ]
}

def run(tokens, stack):
  for token in tokens:
    print(token, stack)
    if str(token)[0] + str(token)[-1] == '""': stack += [token[1:-1]]
    elif type(token) == list: stack += [run(token, [])]
    elif str(token)[:2] == '::': vars[token[2]] = stack[-1]; stack.pop()
    elif str(token).isupper() and str(token).isalpha(): stack += [vars[token]]
    elif token in FUNCTIONS.keys():
      stackValues, function = findFunc(token, stack)
      try:
        try: result = function[1](*[*stackValues, stack])
        except: result = function[1](*stackValues)
        if len(result) >= 1 and result[0] == None: 
          stack = []; result = result[1:]
        result = [correctType(i) for i in result]
      except (ZeroDivisionError, ValueError) as e:
        result = [math.nan]
      stack += result
    elif type(token) == str and token[0] == 'v':
      stack += [CONSTANTS[token[1]]]
    else: stack += [token]
  return stack

def findFunc(token, stack):
  function, stackpops = [], 0
  for i in range(0, len(FUNCTIONS[token]), 2):
    test = FUNCTIONS[token][i:i+2]; stackpops = len(test[0]);
    if stackpops > len(stack): continue
    vals = [stack[-_ - 1] for _ in range(stackpops)][::-1]
    types = [type(_) for _ in vals]
    types = [x in y for x, y in zip(types, test[0])]
    if all(types): 
      function = test; break
  if not function: raise ValueError(f"stack not compatible for '{token}'")
  return ([stack.pop() for _ in range(stackpops)][::-1], function)
  
def fold(x, y, z, s):
  a = [x[0]]
  while len(x) > 1:
    r = run(y, z + x[:2])[-1]; x = [r] + x[2:]; a += [r]
  return a if s else x[0]
  
def forLoop(x, y, z):
  for i in range(y): z = run(x, z)
  return z
  
def map(x, y, z):
  if not x: return z + [x]
  # print('map:\n', repr(x), '\n', repr(y), '\n', repr(z))
  pop = [findPop(y, z, [i], 1) for i in x]
  push = [findPush(y, z, [i], 1) for i in x]
  result = [run(y, z + [i])[-1 - min(push):] for i in x]
  result = func.transpose(result) if result else [[]]
  pop, push = VIV(pop, [0]), VIV(push, [0])
  return z[:VIV(-min(pop), len(z))] + result[0] if result else [[]]

def zipmap(x, y, z, t):
  if [] in [x, y]: raise ValueError('\'|\' requires two non-empty lists')
  if len(x) != len(y): raise ValueError('\'|\' requires two equal-length lists')
  pop = [findPop(z, t, [i, j], 2) for i, j in zip(x, y)]
  push = [findPush(z, t, [i, j], 1) for i, j in zip(x, y)]
  result = func.transpose([run(z, t + [i, j])[-1 - min(push):] for i, j in zip(x, y)])
  return t[:VIV(-min(pop), len(t))] + result[0]

def table(x, y, z, t):
  if [] in [x, y]: raise ValueError('\'=\' requires two non-empty lists')
  pop = [findPop(z, t, [i, j], 2) for j in y for i in x]
  push = [findPush(z, t, [i, j], 1) for j in y for i in x]
  result = func.transpose([func.transpose([run(z, t + [i, j])[-1 - min(push):] for j in y])[0] for i in x])
  return t[:VIV(-min(pop), len(t))] + result[0]
  
def whileLoop(x, y, z):
  z = run(y, z)
  while z[-1]: z = run(y, run(x, z[:-1]))
  return z[:-1]
  
def findSig(x, y):
  pop, push = 0, 0
  for i in x:
    # print(y, i)
    if i not in FUNCTIONS.keys():
      push += 1; y += run([i], []); continue
    vals, func = findFunc(i, y)
    try: output = func[1](*[*vals, y])
    except: output = func[1](*vals)
    if output and output[0] == None: continue
    output = [correctType(i) for i in output]
    pop += len(vals) - push
    push = max(0, push - len(vals)) + len(output)
    y = y[-len(vals):] + [*output]
  return pop, push
  
def correctType(x):
  if type(x) in f + [bool]:
    if math.isnan(x) or (type(x) is complex): return x
    elif type(x) is bool: return int(x)
    return int(x) if int(x) == x else x
  elif x and all([type(_) == str and len(_) == 1 for _ in x]):
    return ''.join(x)
  else: return x
  
# print(findSig(('+', 11), [1, 11]))
# print(map([1, 2, 3], ('+', 11), [11]))