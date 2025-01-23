import numpy as np
import random as rand
import math
import func
import itertools as it

f, I, S, L, F = [complex, float, int], [int], [str], [list], [tuple]
s, A, i, N, a = [complex, float, int, str], [str, list], [int, list], [float, int, list], [complex, float, int, str, list]

CONSTANTS = {
  'a': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
  'd': '0123456789',
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
    [A, F], lambda x, y, z: [[i for i in x if run(list(y), z + [i])[-1]]]
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
    [A, F], lambda x, y, z: [[run(list(y), z + [i])[-1] for i in x]],
  ], '^': [
    [f, f], lambda x, y: [x ** y],
    [A], lambda x: [func.dedup(x)]
  ], '!': [
    [f], lambda x: [1 - x],
    [A], lambda x: [func.unique(x)],
  ], '~': [
    [f], lambda x: [-x],
    [L], lambda x: [func.ravel(x)],
    [S], lambda x, y: [None, *run(func.parse(x), y)]
  ], '|': [
    [f, f], lambda x, y: [int(x or y)],
    [A, I], lambda x, y: [x[y:] + x[:y]],
    [A, L], lambda x, y: [func.reshape(x, y)],
    [A, A, F], lambda x, y, z, t: [[run(list(z), t + [i, j])[-1] for i, j in zip(x, y)]]
  ], '<': [
    [f, f], lambda x, y: [int(x < y)],
    [A, I], lambda x, y: [x[:y]],
    [A], lambda x: [x[0]], 
    [a, a, F], lambda x, y, z, t: [[x, y] + run(list(z), t)]
  ], '>': [
    [f, f], lambda x, y: [int(x > y)],
    [A, I], lambda x, y: [x[y:]],
    [A], lambda x: [x[-1]],
    [a, a, F], lambda x, y, z, t: [run(list(z), t) + [x, y]]
  ], '=': [
    [a, a], lambda x, y: [int(x == y)],
    [A, A, F], lambda x, y, z, t: [[[run(list(z), t + [i, j])[-1] for i in x] for j in y]]
  ], '\\': [
    [I], lambda x: [1 / x],
    [A], lambda x: [x[::-1]],
    [A, F], lambda x, y, z: [fold(x, y, z, 1)]
  ], '@': [
    [f, f], lambda x, y: [math.log(x, y)],
    [A, I], lambda x, y: [x[y]],
    [L, L], lambda x, y: [[np.array(x)[*i] for i in y]]
  ], '?': [
    [A, a], lambda x, y: [int(x in y)],
    [I, L], lambda x, y, z: [None, *run(list(y[x]), z)]
  ], '&': [
    [a, a], lambda x, y: [[x, y]],
    [a, a, F], lambda x, y, z, t: [run(list(z), t + [x]), run(list(z), t + [y])],
  ], '#': [
    [I], lambda x: [list(range(x))],
    [A], lambda x: [len(x)],
    [A, F], lambda x, y, z: [[run(list(y), z + [i, j])[-1] for i, j in enumerate(x)]]
  ], '_': [
    [a], lambda x: [[x]]
  ], '$': [
    [A], lambda x: [sorted(x)],
    [A, F], lambda x, y, z: [sorted(x, key = lambda i: run(list(y), z + [i])[-1])]
  ], '.': [
    [a], lambda x: [x, x]
  ], ',': [
    [a, a], lambda x, y: [x, y, x]
  ], ':': [
    [a, a], lambda x, y: [y, x]
  ], ';': [
    [a, a], lambda x, y: [x]
  ], '(': [
    [], lambda z: [None, *(z[1:] + z[:1])]
  ], ')': [
    [], lambda z: [None, *(z[-1:] + z[:-1])]
  ], '`': [
    [I], lambda x, y: [y[-x - 1]]
  ], 
  ### EXTERNAL FUNCTIONS
  ## MATH FUNCTIONS
  'm[': [
    [f], lambda x: [math.floor(x)]
  ], 'm]': [
    [f], lambda x: [math.ceil(x)]
  ], 'm~': [
    [f], lambda x: [round(x)]
  ], 'ms': [
    [f], lambda x: [math.sin(x)]
  ], 'mc': [
    [f], lambda x: [math.cos(x)]
  ], 'mt': [
    [f], lambda x: [math.tan(x)]
  ], 'mS': [
    [f], lambda x: [math.asin(x)]
  ], 'mC': [
    [f], lambda x: [math.acos(x)]
  ], 'mT': [
    [f], lambda x: [math.atan(x)]
  ], 'mG': [
    [f], lambda x: [math.gamma(x)]
  ], 't#': [
    [A], lambda x: [list(np.array(x).shape)]
  ], 'tp': [
    [f, f], lambda x, y: [math.perm(x, y)],
    [A, I], lambda x, y: [[list(i) for i in list(it.permutations(x, y))]]
  ], 'tc': [
    [f, f], lambda x, y: [math.comb(x, y)],
    [A, I], lambda x, y: [[list(i) for i in list(it.combinations(x, y))]]
  ], 'tC': [
    [A, I], lambda x, y: [[list(i) for i in list(it.combinations_with_replacement(x, y))]]
  ], 't>': [
    [A], lambda x: [[x[0] if i == 1 else x[:i] for i in range(1, len(x) + 1)]]
  ], 't<': [
    [A], lambda x: [[x[-1] if i == 1 else x[-i:] for i in range(1, len(x) + 1)]]
  ], 'c+': [
    [f, f], lambda x, y: [complex(x, y)]
  ], 'c|': [
    [f], lambda x: [math.sqrt(x.real**2 + x.imag**2)]
  ], 'c:': [
    [f], lambda x: [x.real, x.imag]
  ], 'c-': [
    [f], lambda x: [complex(x.real, -x.imag)]
  ], 'cT': [
    [f, f], lambda x, y: [math.atan2(x, y)]
  ], 
  ## STRING FUNCTIONS
  's#': [
    [I], lambda x: [chr(x)],
    [S], lambda x: [[ord(i) for i in x] if len(x) == 1 else ord(x)],
    [L], lambda x: [chr(i) for i in x],
  ], 'sU': [
    [S], lambda x: [x.upper()]
  ], 'sL': [
    [S], lambda x: [x.lower()]
  ], 's~': [
    [S], lambda x: [x.swapcase()]
  ], 's*': [
    [S, S], lambda x, y: [fold(x, (':', y, '+', ':', '+'), [], 0)],
    [L, a], lambda x, y: [fold(x, (':', y, '+', ':', '+'), [], 0)],
  ],
  ## IN/OUT FUNCTIONS
  'p.': [
    [a], lambda x: ([], [print(x, end = '')])[0],
  ], 'p!': [
    [a], lambda x: ([], [print(x)])[0],
  ], 'p?': [
    [], lambda: [input()],
  ],
  ### EXTERNAL FUNCTIONS
  ## MATH FUNCTIONS
  'm[': [
    [f], lambda x: [math.floor(x)]
  ], 'm]': [
    [f], lambda x: [math.ceil(x)]
  ], 'm~': [
    [f], lambda x: [round(x)]
  ], 'ms': [
    [f], lambda x: [math.sin(x)]
  ], 'mc': [
    [f], lambda x: [math.cos(x)]
  ], 'mt': [
    [f], lambda x: [math.tan(x)]
  ], 'mS': [
    [f], lambda x: [math.asin(x)]
  ], 'mC': [
    [f], lambda x: [math.acos(x)]
  ], 'mT': [
    [f], lambda x: [math.atan(x)]
  ], 'mG': [
    [f], lambda x: [math.gamma(x)]
  ], 't#': [
    [A], lambda x: [list(np.array(x).shape)]
  ], 'tp': [
    [f, f], lambda x, y: [math.perm(x, y)],
    [A, I], lambda x, y: [[list(i) for i in list(it.permutations(x, y))]]
  ], 'tc': [
    [f, f], lambda x, y: [math.comb(x, y)],
    [A, I], lambda x, y: [[list(i) for i in list(it.combinations(x, y))]]
  ], 'tC': [
    [A, I], lambda x, y: [[list(i) for i in list(it.combinations_with_replacement(x, y))]]
  ], 't>': [
    [A], lambda x: [[x[0] if i == 1 else x[:i] for i in range(1, len(x) + 1)]]
  ], 't<': [
    [A], lambda x: [[x[-1] if i == 1 else x[-i:] for i in range(1, len(x) + 1)]]
  ], 'c+': [
    [f, f], lambda x, y: [complex(x, y)]
  ], 'c|': [
    [f], lambda x: [math.sqrt(x.real**2 + x.imag**2)]
  ], 'c:': [
    [f], lambda x: [x.real, x.imag]
  ], 'c-': [
    [f], lambda x: [complex(x.real, -x.imag)]
  ], 'cT': [
    [f, f], lambda x, y: [math.atan2(x, y)]
  ], 
  ## STRING FUNCTIONS
  's#': [
    [I], lambda x: [chr(x)],
    [S], lambda x: [[ord(i) for i in x] if len(x) == 1 else ord(x)],
    [L], lambda x: [chr(i) for i in x],
  ], 'sU': [
    [S], lambda x: [x.upper()]
  ], 'sL': [
    [S], lambda x: [x.lower()]
  ], 's~': [
    [S], lambda x: [x.swapcase()]
  ], 's*': [
    [S, S], lambda x, y: [fold(x, (':', y, '+', ':', '+'), [], 0)],
    [L, a], lambda x, y: [fold(x, (':', y, '+', ':', '+'), [], 0)],
  ],
  ## IN/OUT FUNCTIONS
  'p.': [
    [a], lambda x: ([], [print(x, end = '')])[0],
  ], 'p!': [
    [a], lambda x: ([], [print(x)])[0],
  ], 'p?': [
    [], lambda: [input()],
  ],
}

def run(tokens, stack):
  for token in tokens:
    if str(token)[0] + str(token)[-1] == '""': stack += [token[1:-1]]
    elif type(token) == list: stack += [run(token, [])]
    elif token in FUNCTIONS.keys():
      stackB4 = stack
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
    test = FUNCTIONS[token][i:i+2];
    stackpops = len(test[0]);
    if stackpops > len(stack): continue
    stackValues = [stack[-_ - 1] for _ in range(stackpops)][::-1]
    stackTypes = [type(_) for _ in stackValues]
    stackTypes = [x in y for x, y in zip(stackTypes, test[0])]
    if all(stackTypes): 
      function = test; break
  if not function: raise ValueError(f"stack not compatible for '{token}'")
  return ([stack.pop() for _ in range(stackpops)][::-1], function)
  
def fold(x, y, z, s):
  x = list(x)
  a = [x[0]]
  while len(x) > 1:
    r = run(list(y), z + x[:2])[-1]; x = x[2:]
    a += [r]; x = [r] + x
  return a if s else x[0]
  
def forLoop(x, y, z):
  for i in range(y):
    z = run(list(x), z)
  return z
  
def whileLoop(x, y, z):
  z = run(list(y), z)
  while z[-1]:
    z = run(list(y), run(list(x), z[:-1]))
  z.pop()
  return z
  
def correctType(x):
  if type(x) in f:
    if math.isnan(x) or (type(x) is complex): return x
    return int(x) if int(x) == x else x
  elif all([type(_) == str and len(_) == 1 for _ in x]):
    return ''.join(x)
  else: return x