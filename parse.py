import numpy as np
import random as rand
import math, random, time, re
import func, sys
import itertools as it
import datetime as dt
import calendar as cal
from termcolor import colored, cprint

f, I, S, L, F = [complex, float, int], [int], [str], [list], [tuple]
s, A, i, N, a = [complex, float, int, str], [str, list], [int, list], [float, int, list], [complex, float, int, str, list]

findPop = lambda x, y: max(0, findSig(x, y)[0])
findPush = lambda x, y: max(0, findSig(x, y)[1])
VIV = lambda x, y: x if x else y # VALUE IF VALUE IS NOT 'EMPTY'

vars = {}

CONSTANTS = {
  'a': 'abcdefghijklmnopqrstuvwxyz',
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
    [A, F], lambda x, y, z: [None, *run([y + (0, '=', '!'), '%', x, ':', '~~'], z + [x])]
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
    [S], lambda x: [x.lower()],
    [A, F], lambda x, y, z: [fold(x, y, z, 0)],
    [A, I, F], lambda x, y, z, t: [None, *run(['/', ('\\', '_~',) * bool(z) + z, '%'], t + [x, y])]
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
  ], '~~': [
    [f], lambda x: [str(x)],
    [A, L], lambda x, y: [func.keep(x, y)],
  ], '~|': [
    
  ], '|': [
    [f, f], lambda x, y: [x or y],
    [A, I], lambda x, y: [x[y:] + x[:y]],
    [A, L], lambda x, y: [func.reshape(x, y)],
    [S], lambda x: [x.upper()],
    [A, A, F], lambda x, y, z, t: [None, *zipmap(x, y, z, t)]
  ], '|~': [
    
  ], '<': [
    [f, f], lambda x, y: [x < y],
    [A, I], lambda x, y: [x[:y]],
    [A], lambda x: [x[0]], 
    [F], lambda x, y: [None, *keepArgs(x, y[:-1], -1)]
  ], '>': [
    [f, f], lambda x, y: [x > y],
    [A, I], lambda x, y: [x[y:]],
    [A], lambda x: [x[-1]],
    [F], lambda x, y: [None, *keepArgs(x, y[:-1], 1)]
  ], '=': [
    [a, a], lambda x, y: [x == y],
    [A, A, F], lambda x, y, z, t: [None, *table(x, y, z, t)]
  ], '\\': [
    [f], lambda x: [1 / x],
    [A], lambda x: [x[::-1]],
    [A, F], lambda x, y, z: [fold(x, y, z, 1)]
  ], '@': [
    [f, f], lambda x, y: [math.log(x, y)],
    [S], lambda x: [x.swapcase()],
    [A, I], lambda x, y: [x[y]],
    [L, L], lambda x, y: [func.select(x, y)]
  ], '?': [
    [A, a], lambda x, y: [y in x],
    [I, F, F], lambda x, y, z, t: [None, *(run(y, t[:-3]) if x else run(z, t[:-3]))]
  ], '&': [
    [a, a], lambda x, y: [[x, y]],
    [a, a, F], lambda x, y, z, t: [run(z, t + [x]), run(z, t + [y])],
  ], '#': [
    [I], lambda x: [list(range(x))],
    [A], lambda x: [len(x)],
    [A, F], lambda x, y, z: [None, *zipmap(list(range(len(x))), x, y, z)]
  ], '_': [
    [a], lambda x: [[x]]
  ], '_^': [
  ], '_!': [
    [A, a], lambda x, y: [[i for i, j in enumerate(x) if j == y]],
  ], '_~': [
    [f], lambda x: [(x > 0) - (x < 0)],
    [A], lambda x: [*x]
  ], '_\\': [
    [A, I], lambda x, y: [[x[i:i + y] for i in range(0, len(x), y)]],
    [A, f], lambda x, y: [[x[i:i + int(len(x) * y)] for i in range(0, len(x), int(len(x) * y))]]
  ], '_<': [
    [f, f], lambda x, y: [min(x, y)],
    [A], lambda x: [func.suffix(x)]
  ], '_>': [
    [f, f], lambda x, y: [max(x, y)],
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
  ], 'm!': [
    [f], lambda x: [math.gamma(x)]
  ], 'mt': [
    [f, I], lambda x, y: [func.trigonometry(x, y)]
  ], 'mE': [
    [I], lambda x: [math.exp(x)]
  ], 'mp': [
    [I], lambda x: [func.primeFactors(x)]
  ],
  # TUPLE FUNCTIONS
  'tp': [
    [f, f], lambda x, y: [math.perm(x, y)],
    [A, I], lambda x, y: [[list(i) for i in list(it.permutations(x, y))]]
  ], 'tP': [
    [A], lambda x: [[i for i in func.partitions(x)]]
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
  ], 'c\\': [
    [f], lambda x: [math.sqrt(x.real**2 + x.imag**2)],
  ],
  # STRING FUNCTIONS
  's#': [
    [I], lambda x: [chr(x)],
    [S], lambda x: [[ord(i) for i in x] if len(x) > 1 else ord(x)],
    [L], lambda x: [chr(i) for i in x],
  ], 's*': [
    [A, a], lambda x, y: [fold(x, (':', f'"{y}"' if type(y) == str else y, '+', ':', '+'), [], 0)],
  ], 's@': [
    [S, I], lambda x, y: [[x.isalnum(), x.isalpha(), x.isdigit()][y] if 0 >= y >= 2 else None]
  ], 'sr!': [
    [S, S, S], lambda x, y, z: [re.sub(y, z, x)]
  ], 'sr-': [
    [S, S], lambda x, y: [re.sub(y, '', x)]
  ], 's!': [
    [S, S, S], lambda x, y, z: [x.replace(y, z)]
  ], 's-': [
    [S, S], lambda x, y: [x.replace(y, '')]
  ], 's?': [
    [S, S], lambda x, y: [re.findall(y, x)]
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

def run(tokens, stack, debug = False):
  # print(stack)
  if debug: print(colored('*Tokens:', 'yellow'), tokens)
  for token in tokens:
    if str(token)[0] + str(token)[-1] == '""': stack += [token[1:-1]]
    elif type(token) == list: 
      pop, push = findSig(token, stack[:])
      stack = stack[:-pop if pop else len(stack)] + [run(token, stack[:])[-push if push else len(stack):]]
    elif type(token) == tuple: stack += [token]
    elif str(token)[:2] == '::': vars[token[2]] = stack[-1]; stack.pop()
    elif str(token).isupper() and str(token).isalpha(): stack += [vars[token]]
    elif type(token) == str and token[0] == 'v': stack += [CONSTANTS[token[1]]]
    elif token in FUNCTIONS.keys():
      stackValues, function = findFunc(token, stack)
      try:
        try: result = function[1](*stackValues, stack[:])
        except: result = function[1](*stackValues)
          # except: 
          #   print(tmc.colored('FuncError: ', 'red') + tmc.colored(f'\'{token}\' is not compatible with the stack.'))
          #   print('{\n' + '\n'.join(['  ' + repr(i) for i in stack]) + '\n}')
          #   sys.exit(1)
        if len(result) >= 1 and result[0] == None: 
          stack = []; result = result[1:]
        result = [correctType(i) for i in result]
      except (ZeroDivisionError, ValueError) as e: result = [math.nan]
      stack += result
    else: stack += [token]
    if debug: print(colored('*Token&Stack:', 'yellow'), token, stack)
  if debug: print(colored('*Stack:', 'yellow'), stack)
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
  if not function:
    print(colored('FuncError: ', 'red', attrs = ['bold']) + colored(f'\'{token}\' is not compatible with the stack.', 'red'))
    print(colored('Debug:\n', 'yellow') + '{\n' + '\n'.join(['  ' + repr(i) for i in stack]) + '\n}')
    sys.exit(0)
  return ([stack.pop() for _ in range(stackpops)][::-1], function)
  
def fold(x, y, z, ac): # x: list, y: func, z: stack, s: accml?
  x = list(x)
  if x == []: return []
  a = [x[0]]
  while len(x) > 1:
    r = run(y, z + x[:2])[-1]; x = [r] + x[2:]; a += [r]
  return a if ac else x[0]
  
def keepArgs(x, y, p): # x: func, y: stack, p: place (1: above, 0: below) 
  pop, push = findSig(x, y[:])
  out = run(x, y[:])
  # print(x, y, out)
  return (y[:-pop] + out[-push:] + y[-pop:]) if p == 1 else (y + out[-push:])
  
def bothDo(x, y): # x: func, y: stack
  pass
  
def doBoth(x, y, z): # x: func, y: func, z: stack
  pass
  
def sign_map(x, y, z): # x: arg, y: func, z: stack
  pop = max([findPop(y, z + [i]) for i in x])
  push = max([findPush(y, z + [i]) for i in x])
  popat, pushat = -pop if pop else len(z) + 1, -push if push else len(z) + 1
  vals = (z + [x])[popat:]
  output = [run(y, z + [i])[pushat:] for i in x]
  # output = output if push else [[]]
  output = func.transpose(output)[0] if output else []
  return pop, push, vals, output
  
def sign_zipmap(x, y, z, t): # x: arg, y: arg, z: func, t: stack
  if [] in [x, y]: raise ValueError('\'|\' requires two non-empty lists')
  if len(x) != len(y): raise ValueError('\'|\' requires two equal-length lists')
  pop = max([findPop(z, t + [i, j]) for i, j in zip(x, y)])
  push = max([findPush(z, t + [i, j]) for i, j in zip(x, y)])
  popat, pushat = -pop if pop else len(t) + 2, -push if push else len(t) + 2
  vals = (t + [x, y])[popat:]
  output = [run(z, t + [i, j])[pushat:] for i, j in zip(x, y)]
  # output = output if push else [[]]
  output = func.transpose(output)[0] if output else []
  return pop, push, vals, output
  
def forLoop(x, y, z): # x: func, y: int, z: stack
  return run(x * y, z)
  
def map(x, y, z): # x: arg, y: func, z: stack
  pop, push, _, output = sign_map(x, y, z)
  # print([findPop(y, z + [i]) for i in x])
  # print([findPush(y, z + [i]) for i in x])
  z += [x]
  # print(pop, push, y, z, output)
  # print("endmap")
  return z[:-pop if pop else len(z)] + output
  
def zipmap(x, y, z, t): # x, y: arg, z: func, t: stack
  pop, _, _, output = sign_zipmap(x, y, z, t)
  t += [x, y]
  return t[:-pop if pop else len(t)] + output[0]
  
def table(x, y, z, t): # x, y: arg, z: func, t: stack
  if [] in [x, y]: raise ValueError('\'=\' requires two non-empty lists')
  pop = max([findPop(z, t + [i, j]) for j in y for i in x])
  push = max([findPush(z, t + [i, j]) for j in y for i in x])
  popat, pushat = -pop if pop else len(z), -push if push else len(z)
  result = func.transpose([func.transpose([run(z, t + [i, j])[pushat:] for j in y])[0] for i in x])
  return (t + [x, y])[:popat] + result[0] if result else [[]]
  
def whileLoop(x, y, z): # x: func, y: cond, z: stack
  z = run(y, z)
  while z[-1]: z = run(y, run(x, z[:-1]))
  return z[:-1]
  
def findSig(x, y): # x: func, y: stack
  pop, push = 0, 0
  variables = {}
  # print(x, y)
  for i in x:
    # print(colored('debug2:', 'magenta', attrs=['bold']), pop, push, y)
    if type(i) == list: 
      pop_, push_ = findSig(i, y[:])
      y = run(i, y)
      # y[:-pop_ if pop_ else len(y)] + [run(i, y[:])[-push_ if push_ else len(y):]]
      pop += max(0, -pop_ - push)
      push = max(0, push + pop_) + push_
      # print(pop, push, i, y)
      continue
    elif str(i)[:2] == '::': 
      vars[i[2]] = y[-1]; y.pop(); 
      # print(pop, push, i, y) 
      continue
    elif str(i).isupper() and str(i).isalpha(): 
      push += 1; y += [vars[i]]; 
      # print(pop, push, i, y)
      continue
    elif i not in FUNCTIONS.keys():
      push += 1; y += run([i], []); 
      # print(pop, push, i, y)
      continue
    vals, f = findFunc(i, y[:])
    # print(colored('debug:', 'green', attrs=['bold']), i, vals)
    containsFunc = [type(j) == tuple for j in vals]
    if any(containsFunc) and (i in ['%', '|', '=', '<', '>', '*', '?', '!']): # modifiers
      if i == '%':
        # print("hi")
        pop_, push_, vals, output = sign_map(y[-2], y[-1], y[:-2])
        # cprint(f'{pop_} {push_} {vals} {output}', 'green')
      elif i == '|':
        _, _, vals, output = sign_zipmap(y[-3], y[-2], y[-1], y[:-3])
      elif i == '*':
        mapf = y[-2]; arg = y[-1]; y = y[:-2]; z = y
        pop_ = findPop(mapf * arg, y[:])
        push_ = findPush(mapf * arg, y[:])
        popat, pushat = -pop_ if pop_ else len(y), -push_ if push_ else len(y)
        for _ in range(arg): 
          print(mapf, y); y = run(mapf, y, True)
        print(mapf, y)
        output, vals = y[pushat:], z[popat:]
      elif i == '!':
        mapf = y[-1]; y = y[:-1]; z = y
        pop_ = findPop(mapf, y[:])
        push_ = findPush(mapf, y[:])
        popat, pushat = -pop_ if pop_ else len(y), -push_ if push_ else len(y)
        # print(mapf, y)
        # print(pop_, push_)
        y = run(mapf, y[:])
        output, vals = y[pushat:], z[popat:]
      elif i == '?':
        pass
      push -= sum(containsFunc)
    elif str(i) in '()':
      y = f[1](y)[1:]; continue
    elif str(i)[0] == 'p':
      output = [] if i in ['p.', 'p!'] else ["0"]
    else:
      try: output = f[1](*vals, y)
      except: output = f[1](*vals)
    if output and output[0] == None: continue
    output = [correctType(i) for i in output]
    # print(pop, push, i, y)
    # print(colored('debug2:', 'yellow', attrs=['bold']), vals, output)
    pop += max(0, len(vals) - push)
    push = max(0, push - len(vals)) + len(output)
    y = y[:-len(vals)] + output
    # print(colored(y, 'yellow', attrs=['bold']))
  # print(colored('debug2:', 'magenta', attrs=['bold']), pop, push, y)
  # print(pop, push, i, y)
  return pop, push
  
def correctType(x):
  if type(x) in f + [bool]:
    if math.isnan(x) or (type(x) is complex): return x
    elif type(x) is bool: return int(x)
    return int(x) if int(x) == x else x
  elif x and all([type(_) == str and len(_) == 1 for _ in x]): return ''.join(x)
  elif type(x) == list: return [correctType(i) for i in x]
  else: return x
  
# print(findSig(((('+',), '%'), '%'), [3, [[1, 2], [11, 3]]]))
# print(map([1, 2, 3], ('+', 11), [11]))