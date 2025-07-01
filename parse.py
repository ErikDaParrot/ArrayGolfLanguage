import numpy as np
import random as rand
import math, random, time, re
import func, sys
import itertools as it
import datetime as dt
import calendar as cal
from termcolor import colored, cprint

f, I, S, L, F = [complex, float, int], [int], [str], [list], [tuple]
s, A, i, N, a, AA = f + S, S + L, I + L, [float] + I + L, f + S + L, f + S + L + F

# findPop = lambda x, y: max(0, findSig(x, y)[0])
# findPush = lambda x, y: max(0, findSig(x, y)[1])
# VIV = lambda x, y: x if x else y # VALUE IF VALUE IS NOT 'EMPTY'
rotU = lambda x: x[1:] + x[:1]; rotD = lambda x: x[-1:] + x[:-1]
fill, vars = None, {}

CONSTANTS = {
  'A': 'abcdefghijklmnopqrstuvwxyz',
  'D': '0123456789',
  'E': math.e,
  'F': 'FizzBuzz',
  'I': float('inf'),
  'P': math.pi,
}

FUNCTIONS = {
  ## BASIC FUNCTIONS
  '+': [
    [f, f], lambda x, y: [x + y],
    [L, L], lambda x, y: [x + y],
    [L, s], lambda x, y: [x + [y]],
    [s, L], lambda x, y: [[x] + y],
    [S, S], lambda x, y: [x + y],
    [S, I], lambda x, y: [chr(ord(x) + y)],
    # [F, F], lambda x, y: [x + y],
    [a, a, F], lambda x, y, z: table(x if type(x) in A else list(range(x)), y if type(y) in A else list(range(y)), z),
  ], '`+': [
    [f], lambda x: [abs(x)],
    [A, A], lambda x, y: [np.union1d(x, y).tolist()]
  ], '-': [
    [f, f], lambda x, y: [x - y],
    [S, I], lambda x, y: [chr(ord(x) - y)],
    [A, a], lambda x, y: [[i for i in x if i not in func.tolist(y)]],
    [A, F], lambda x, y, z: [list(filter(lambda i: run(y, [i])[-1], x))]
  ], '`-': [
    [f], lambda x: [(x > 0) - (x < 0)],
    [f[:1]], lambda x: [complex(x.real / (h := math.hypot(x.real, x.imag)), x.imag / h)],
    [A, A], lambda x, y: [np.intersect1d(x, y).tolist()]
  ], '*': [
    [f, f], lambda x, y: [x * y],
    [A, I], lambda x, y: [x * y],
    [I, A], lambda x, y: [x * y],
    [A, L], lambda x, y: [func.keep(x, y)],
    [F, I], lambda x, y, z: [None, *forLoop(x, y, z)],
    [F, F], lambda x, y, z: [None, *whileLoop(x, y, z)]
  ], '`*': [
    [f], lambda x: [math.exp(x)],
    [A, L], lambda x, y: [func.group(x, y)],
  ], '/': [
    [f, f], lambda x, y: [x / y if y != 0 else float('inf')],
    # [A, [I, L]], lambda x, y: [func.windows(x, y)],
    [A, I], lambda x, y: [func.smallWindows(x, y)],
    [L, L], lambda x, y: [func.windows(x, y)],
    [A, F], lambda x, y, z: [fold(x, y, z, 0)],
    [A, I, F], lambda x, y, z, t: [None, *run(['/', ('\\', '`',) * bool(z) + z, '%'], t + [x, y])]
  ], '`/': [
    [f, f], lambda x, y: [*divmod(x, y)],
    [A, L], lambda x, y: [func.partition(x, y) if len(x) == len(y) else func.splitBy(x, y)],
  ], '^': [
    [f, f], lambda x, y: [x ** y],
    [A], lambda x: [func.dedup(x)],
    [A, I], lambda x, y: [x[:y], x[y:]]
  ], '`^': [
    [A], lambda x: [func.unique(x)]
  ], '~': [
    [f], lambda x: [-x],
    [L], lambda x: [func.ravel(x)],
  ], '`~': [
    [f, I], lambda x, y: [func.trigonometry(x, y)],
    [f[:1]], lambda x: [complex(x.real, -x.imag)]
  ], '<': [
    [f, f], lambda x, y: [x < y],
    [A, I], lambda x, y: [x[:y]],
    [A], lambda x: [x[0]], 
    [a, a, a, F], lambda x, y, z, t: [None, sidedBoth(x, y, z, t, 0)]
  ], '`<': [
    [f, f], lambda x, y: [min(x, y)],
    [S, S], lambda x, y: [min(x, y)],
    [L, L], lambda x, y: [min(x, y)],
  ], '>': [
    [f, f], lambda x, y: [x > y],
    [A, I], lambda x, y: [x[y:]],
    [A], lambda x: [x[-1]],
    [a, a, a, F], lambda x, y, z, t: [None, sidedBoth(x, y, z, t, 1)]
  ], '`>': [
    [f, f], lambda x, y: [max(x, y)],
    [S, S], lambda x, y: [max(x, y)],
    [L, L], lambda x, y: [max(x, y)],
  ], '=': [
    [a, a], lambda x, y: [x == y],
    # [a, F, F], lambda x, y, z, t: [None, *fork(x, y, z, t)],
    # [A, A, F], lambda x, y, z, t: [None, *table(x, y, z, t)],
    [a, F, F], lambda x, y, z: fork(x, y, z),
  ], '!': [
    [a], lambda x: [int(not bool(x))],
    [F], lambda x, y: [None, run(x, y)]
  ], '`!': [
  ], '?': [
    [a, A], lambda x, y: [func.membership(x, y)],
    [A, a], lambda x, y: [func.membership(x, y)],
    [I, F, F], lambda x, y, z, t: [None, *(run(y, t[:-3]) if x else run(z, t[:-3]))]
  ], '`?': [
    [L], lambda x: [func.where(x)]
  ], '\\': [
    [f], lambda x: [1 / x],
    [A], lambda x: [x[::-1]],
    [A, F], lambda x, y, z: [fold(x, y, z, 1)]
  ], '_': [
    [a], lambda x: [[x]]
  ], '`_': [
    [f, f], lambda x, y: [complex(x, y)],
    [f[:1]], lambda x: [x.real, x.imag],
  ], '%': [
    [f, f], lambda x, y: [x % y],
    [A], lambda x: [np.transpose(x).tolist()],
    [A, I], lambda x, y: [func.cut(x, y)],
    # [A, F], lambda x, y, z: [None, *map(x, y, z)],
    # [A, F], lambda x, y: [[run(y, [i])[0] for i in x]],
    [A, F], lambda x, y: map(x, y)
  ], '`%': [
    [f, f], lambda x, y: [x % y == 0]
    # [f, f + L], lambda x, y: [func.encode(x, y)],
    # [L, f + L], lambda x, y: [func.decode(x, y)],
  ], '@': [
    [f, f], lambda x, y: [math.log(x, y)],
    [A, I], lambda x, y: [x[y]],
    [A, L], lambda x, y: [func.select(x, y)]
  ], '&': [
    [a, a], lambda x, y: [[x, y]],
    # [a, a, F], lambda x, y, z, t: [None, *both(x, y, z, t)]
    [a, a, F], lambda x, y, z: both(x, y, z),
  ], '`&': [
    [A], lambda x: [func.classify(x)]
  ], '$': [
    [A], lambda x: [sorted(x)],
    
    [A, F], lambda x, y, z: [sorted(x, key = lambda i: run(y, z + [i])[-1])]
  ], '`$': [
    [A], lambda x: [func.sortIdx(x)],
  ], '#': [
    [I], lambda x: [list(range(0, x, (x > 0) - (x < 0)))],
    [A], lambda x: [len(x)],
    
    [I, F], lambda x, y: map(range(0, x, (x > 0) - (x < 0)), y),
    [A, F], lambda x, y: zipmap(list(range(len(x))), x, y),
    [F, F], lambda x, y, z: [None, *tryCatch(x, y, z)]
  ], '`#': [
    [I], lambda x: [list(range((x > 0) - (x < 0), x + (x > 0) - (x < 0), (x > 0) - (x < 0)))],
    [L], lambda x: [list(np.array(x).shape)],
    
    [I, F], lambda x, y: map(range((x > 0) - (x < 0), x + (x > 0) - (x < 0), (x > 0) - (x < 0)), y),
    [A, F], lambda x, y: zipmap(list(range(1, len(x) + 1)), x, y),
  ], '|': [
    # [f, f], lambda x, y: [x or y],
    [f[:1]], lambda x: [math.hypot(x.real, x.imag)],
    [A, I], lambda x, y: [x[y:] + x[:y]],
    [A, L], lambda x, y: [func.reshape(x, y, fill)],
    
    [A, A, F], lambda x, y, z, t: zipmap(x, y, z),
    [a, A, F], lambda x, y, z: zipmap(func.tolist(x) * len(y), y, z),
    [A, a, F], lambda x, y, z: zipmap(x, func.tolist(y) * len(x), z),
  ], '`|': [
    [A, L], lambda x, y: [np.transpose(x, y).tolist()],
  ], '.': [
    [a + F], lambda x: [x, x]
  ], ',': [
    [a + F, a + F], lambda x, y: [x, y, x]
  ], ':': [
    [a + F, a + F], lambda x, y: [y, x]
  ], ';': [
    [a + F], lambda x: []
  ], 
  # '(': [
  #   [], lambda z: [None, *(z[1:] + z[:1])]
  # ], ')': [
  #   [], lambda z: [None, *(z[-1:] + z[:-1])]
  # ], 
  '(': [
    [], lambda x: [None, *rotU(x)]
  ], ')': [
    [], lambda x: [None, *rotD(x)]
  ],
  '`[': [
  ], '`]': [
  ],
  ## EXTERNAL FUNCTIONS
  # BITWISE, BASE FUNCTIONS
  'b&': [
    [I, I], lambda x, y: [x & y]
  ], 'b|': [
    [I, I], lambda x, y: [x | y]
  ], 'b^': [
    [I, I], lambda x, y: [x ^ y]
  ], 'b~': [
    [I], lambda x: [~x]
  ], 'b<': [
    [I, I], lambda x, y: [x << y]
  ], 'b>': [
    [I, I], lambda x, y: [x >> y]
  ], 'b=': [
    [f, L + f], lambda x, y: [func.encode(x, y)],
    [L, L + f], lambda x, y: [func.decode(x, y)],
  ],
  # ERROR FUNCTIONS
  'e!': [
    [I], lambda x: ([], exec('raise AssertionError()'))[0],
    [S], lambda x: ([], exec('raise RaisedError(x)'))[0],
  ], 'e*': [
    [], lambda: ([], sys.exit(0))[0]
  ],
  # GENERATE FNS
  'g': [
    [f], lambda x: [random.random() if x == 0 else random.randint(0, x)],
    [A], lambda x: [random.shuffle(x)]
  ],
  # TUPLE FNS
  # 'tp': [
  #   [f, f], lambda x, y: [math.perm(x, y)],
  #   [A, I], lambda x, y: [[list(i) for i in list(it.permutations(x, y))]]
  # ], 'tP': [
  #   [A], lambda x: [[i for i in func.partitions(x)]]
  # ], 'tc': [
  #   [f, f], lambda x, y: [math.comb(x, y)],
  #   [A, I], lambda x, y: [[list(i) for i in list(it.combinations(x, y))]]
  # ], 'tC': [
  #   [A, I], lambda x, y: [[list(i) for i in list(it.combinations_with_replacement(x, y))]]
  # ],
  # IN/OUT FNS
  'i?': [
    [], lambda: [input()],
  ], 'o.': [
    [a], lambda x: ([], [print(x, end = '')])[0],
  ], 'o!': [
    [a], lambda x: ([], [print(x)])[0],
  ],
  # STR FNS
  # 's#': [
  #   [I], lambda x: [chr(x)],
  #   [S], lambda x: [[ord(i) for i in x] if len(x) > 1 else ord(x)],
  #   [L], lambda x: [[chr(i) for i in x]],
  # ], 's@': [
  #   [S, I], lambda x, y: [[x.isalnum(), x.isalpha(), x.isdigit()][y] if 0 <= y <= 2 else None]
  # ], 
  's:': [
    [S, S, S], lambda x, y, z: [re.sub(y, z, x)]
  ], 's?': [
    [S, S], lambda x, y: [re.findall(y, x)]
  ],
  # TIME, TYPE FNS
  't=': [
    [f], lambda x: [list(time.gmtime(x))[:6]],
    [L], lambda x: [dt.datetime(*x, tzinfo = dt.timezone.utc).timestamp()]
  ], 't.': [
    [], lambda: [time.time()]
  ], 't|': [
    [], lambda: [time.localtime().tm_gmtoff / 3600]
  ], 't~': [
    [f], lambda x: [time.sleep(x)]
  ]
}

def run(tokens, stack, debug = False, catch = False):
  global vars, fill
  # print(stack)
  if debug: print(colored('*Tokens:', 'yellow'), tokens)
  for token in tokens:
    if str(token)[0] + str(token)[-1] == '""': stack += [token[1:-1]]
    elif type(token) == list: 
      # pop, push = findSig(token, stack[:])
      # stack = stack[:-pop if pop else len(stack)] + [run(token, stack[:])[-push if push else len(stack):]]
      stack += [run(token, [])]
    elif type(token) == tuple: stack += [token]
    elif str(token)[:2] == '`=': vars[token[2:]], stack = stack[-1], stack[:-1]
    elif str(token).isupper() and str(token).isalpha(): 
      if token in vars.keys(): 
        if type(vars[token]) != tuple: stack += [vars[token]]
        else: stack = run(vars[token], stack[:])
      else: vars[token], stack = stack[-1], stack[:-1]
    elif type(token) == str and token[0] == '`' and token[1].isalpha(): stack += [CONSTANTS[token[1]]]
    elif token == '`]': fill = stack.pop()
    elif token in FUNCTIONS.keys() or (type(token) == str and len(token) > 1 and all([i in FUNCTIONS.keys() for i in token])):
      try:
        stackValues, function = findFunc(token, stack)
        # print(token, stackValues, "pass")
        try:
          try: result = function(*stackValues, stack[:])
          except TypeError as e: result = function(*stackValues)
          if len(result) >= 1 and result[0] == None: stack, result = [], result[1:]
        except RaisedError as e: 
          if catch: raise e
          print(colored('RaisedError: ', 'red', attrs = ['bold']) + colored(e, 'red'))
          print(colored('Debug:\n', 'yellow') + func.printData(stack))
          sys.exit(0)
        except AssertionError as e:
          if catch: raise e
          print(colored('AssertionError: ', 'red', attrs = ['bold']))
          print(colored('Debug:\n', 'yellow') + func.printData(stack))
          sys.exit(0)
        except SystemExit as e:
          sys.exit(0)
        except Exception as e:
          if catch: raise e
          print(e)
          print(colored('FuncError: ', 'red', attrs = ['bold']) + colored(f'\'{token}\' reported an error.', 'red'))
          print(colored('Debug:\n', 'yellow') + func.printData(stack))
          sys.exit(0)
        stack += result
        fill = None
        #print(stack)
      except Func404: 
        if len(token) > 1: token = list(token)
        else: 
          print(colored('FuncError: ', 'red', attrs = ['bold']) + colored(f'\'{token}\' is not compatible with the stack.', 'red'))
          print(colored('Debug:\n', 'yellow') + func.printData(stack))
          sys.exit(0)
        stack = run(token, stack)
    else: stack += [token]
    stack = [correctType(i) for i in stack]
    if debug: print(colored('*Token&Stack:', 'yellow'), token, stack)
  if debug: print(colored('*Stack:', 'yellow'), stack)
  return stack

def findFunc(token, stack):
  function, stackpops = [], 0
  stack2 = stack[:]
  for i in range(0, len(FUNCTIONS[token]), 2):
    test = FUNCTIONS[token][i:i+2]; stackpops = len(test[0]);
    if stackpops > len(stack): continue
    vals = [stack[-_ - 1] for _ in range(stackpops)][::-1]
    types = [type(_) for _ in vals]
    types = [x in y for x, y in zip(types, test[0])]
    if all(types): 
      function = test[1]; break
  if not function:
    raise Func404()
  return ([stack.pop() for _ in range(stackpops)][::-1], function)
  
def fold(x, y, z, ac): # x: list, y: func, z: stack, s: accml?
  x = list(x)
  if x == []: return []

  # # optimizations
  # typeList = 0 if all([type(i) == int for i in x]) else "" if all([type(i) == str for i in x]) else None
  # print(typeList)
  
  # if not ac:
  #   if y == ('+',): 
  #     print(sum(x, typeList))
  #     return sum(x, typeList)
  #   elif y == ('*',): return math.prod(x) 
  #   elif y == ('_>',): return max(x)
  #   elif y == ('_<',): return min(x)
  
  a = [x[0]]
  while len(x) > 1:
    r = run(y, z + x[:2])[-1]; x = [r] + x[2:]; a += [r]
  return a if ac else x[0]
  
# def keepArgs(x, y, p): # x: func, y: stack, p: place (1: above, 0: below) 
#   pop, push = findSig(x, y[:])
#   out = run(x, y[:])
#   # print(x, y, out)
#   return (y[:-pop] + out[-push:] + y[-pop:]) if p == 1 else (y + out[-push:])
  
## BOTH ##
def both(x, y, z, t = []): # x: arg, y: arg, z: func, t: stack
  # pop, _, _, output = sign_both(x, y, z, t)
  # t += [x, y]
  # return t[:-pop if pop else len(t)] + output
  return run(z, t + [x]) + run(z, t + [y])
  
 
def sidedBoth(x, y, z, t, d, s = []): # x: arg, y: arg, z: arg, t: func, d: direction, s: stack
  # pop, _, _, output = sign_fork(x, y, z, t)
  # t += [x]
  # return t[:-pop if pop else len(t)] + output
  i1, i2 = run(t, s + [x, y, z][d:d + 2]), run(t, s + [x, z])
  return i2 + i1 if d else i1 + i2

# def sign_both(x, y, z, t): # x: arg, y: arg, z: func, t: stack
#   popx, popy = findPop(z, t + [x]), findPop(z, t + [y])
#   pushx, pushy = findPush(z, t + [x]), findPush(z, t + [y])
#   # print(pop, pushy, pushz)
#   pop, push = popx + popy, pushx + pushy
#   popat, pushxat, pushyat = -pop if pop else len(t) + 1, -pushx if pushx else len(t) + 1, -pushy if pushy else len(t) + 1
#   vals = (t + [x, y])[popat:]
#   output = run(z, t + [x])[pushxat:] + run(z, t + [y])[pushyat:]
#   # print(output)
#   return pop, pushx + pushy, vals, output
  
## FORK ##
def fork(x, y, z, t = []): # x: arg, y: arg, z: func, t: stack
  # pop, _, _, output = sign_fork(x, y, z, t)
  # t += [x]
  # return t[:-pop if pop else len(t)] + output
  return run(y, t + [x]) + run(z, t + [x])
  
# def sign_fork(x, y, z, t): # x: arg, y: func, z: func, t: stack
#   pop = max(findPop(y, t + [x]), findPop(z, t + [x]))
#   pushy, pushz = findPush(y, t + [x]), findPush(z, t + [x])
#   # print(pop, pushy, pushz)
#   popat, pushyat, pushzat = -pop if pop else len(t) + 1, -pushy if pushy else len(t) + 1, -pushz if pushz else len(t) + 1
#   vals = (t + [x])[popat:]
#   output = run(y, t + [x])[pushyat:] + run(z, t + [x])[pushzat:]
#   return pop, pushy + pushz, vals, output

## MAP ##
def map(x, y, z = []): # x: arg, y: func, z: stack
  # pop, _, _, output = sign_map(x, y, z)
  # # print([findPop(y, z + [i]) for i in x])
  # # print([findPush(y, z + [i]) for i in x])
  # z += [x]
  # # print(pop, push, y, z, output)
  # # print("endmap")
  # return z[:-pop if pop else len(z)] + output
  i = [a for i in x if (a := run(y, [i]))]
  return func.transpose(i) * (i != [])
  
# def sign_map(x, y, z): # x: arg, y: func, z: stack
#   if x == []: return 0, 0, [y], []
#   pop = max([findPop(y, z + [i]) for i in x])
#   push = max([findPush(y, z + [i]) for i in x])
#   print(pop, push)
#   popat, pushat = -pop if pop else len(z) + 1, -push if push else len(z) + 1
#   vals = (z + [x])[popat:]
#   output = [run(y, z + [i])[pushat:] for i in x]
#   # output = output if push else [[]]
#   output = func.transpose(output) if output else []
#   return pop, push, vals, output
  
## ZIPMAP ##
def zipmap(x, y, z, t = []): # x, y: arg, z: func, t: stack
  minimal = min(len(x), len(y))
  x, y = x[:minimal], y[:minimal]
  # pop, _, _, output = sign_zipmap(x, y, z, t)
  # t += [x, y]
  # return t[:-pop if pop else len(t)] + output[0]
  i = [a for i, j in zip(x, y) if (a := run(z, [i, j]))]
  return func.transpose(i) * (i != [])
  
# def sign_zipmap(x, y, z, t): # x: arg, y: arg, z: func, t: stack
#   if [] in [x, y]: raise ValueError('\'|\' requires two non-empty lists')
#   if len(x) != len(y): raise ValueError('\'|\' requires two equal-length lists')
#   pop = max([findPop(z, t + [i, j]) for i, j in zip(x, y)])
#   push = max([findPush(z, t + [i, j]) for i, j in zip(x, y)])
#   popat, pushat = -pop if pop else len(t) + 2, -push if push else len(t) + 2
#   vals = (t + [x, y])[popat:]
#   output = [run(z, t + [i, j])[pushat:] for i, j in zip(x, y)]
#   # output = output if push else [[]]
#   output = func.transpose(output) if output else []
#   return pop, push, vals, output
  
## TABLE ##
def table(x, y, z, t = 0): # x, y: arg, z: func, t: stack
  # if [] in [x, y]: raise ValueError('\'=\' requires two non-empty lists')
  # pop = max([findPop(z, t + [i, j]) for j in y for i in x])
  # push = max([findPush(z, t + [i, j]) for j in y for i in x])
  # popat, pushat = -pop if pop else len(z), -push if push else len(z)
  # result = func.transpose([func.transpose([run(z, t + [i, j])[pushat:] for j in y]) for i in x])
  # return (t + [x, y])[:popat] + result if result else [[]]
  i = [[a[-1] for i in x if (a := run(z, [i, j]))] for j in y]
  return [i] * (i != [])
  
def forLoop(x, y, z): # x: func, y: int, z: stack
  return run(x * y, z)
  
def whileLoop(x, y, z): # x: func, y: cond, z: stack
  z = run(y, z)
  while z[-1]: z = run(y, run(x, z[:-1]))
  return z[:-1]
  
def tryCatch(x, y, z): # x: func, y: func, z: stack
  try: return run(x, z[:], catch = True)
  except: return run(y, z[:])
  
# def findSig(x, y): # x: func, y: stack
#   pop, push = 0, 0
#   # print(x, y)
#   for i in x:
#     # print(colored('debug2:', 'magenta', attrs=['bold']), pop, push, y)
#     # print(i, y)
#     if type(i) == list: 
#       pop_, push_ = findSig(i, y[:])
#       y = run(i, y)
#       # y[:-pop_ if pop_ else len(y)] + [run(i, y[:])[-push_ if push_ else len(y):]]
#       pop += max(0, -pop_ - push)
#       push = max(0, push + pop_) + push_
#       # print(pop, push, i, y)
#       continue
#     elif str(i)[:2] == '::':
#       vars[i[2]] = y[-1]; y.pop(); 
#       # print(pop, push, i, y) 
#       continue
#     elif str(i).isupper() and str(i).isalpha(): 
#       push += 1; y += [vars[i]]; 
#       # print(pop, push, i, y)
#       continue
#     elif i not in FUNCTIONS.keys():
#       push += 1; y += run([i], []); 
#       # print(pop, push, i, y)
#       continue
#     try: vals, f = findFunc(i, y[:])
#     except:
#       pop_, push_ = findSig(list(i), y[:])
#       if pop_: 
#         pop += max(0, pop_ - push)
#         push = max(0, push - pop_) + push_
#       y = run(list(i), y)
#       continue
#     # if (vals, f) == (None, None):
#     #   print(colored('FuncError: ', 'red', attrs = ['bold']) + colored(f'\'{i}\' is not compatible with the stack.', 'red'))
#     #   print(colored('Debug:\n', 'yellow') + func.printData(stack))
#     #   sys.exit(0)
#     # print(colored('debug:', 'green', attrs=['bold']), i, vals)
#     # print(vals, f, i)
#     containsFunc = [type(j) == tuple for j in vals]
#     if any(containsFunc) and (i in ['%', '|', '=', '<', '>', '*', '?', '!']): # modifiers
#       if i == '%':
#         # print("hi")
#         pop_, push_, vals, output = sign_map(y[-2], y[-1], y[:-2])
#         # cprint(f'{pop_} {push_} {vals} {output}', 'green')
#       elif i == '|':
#         _, _, vals, output = sign_zipmap(y[-3], y[-2], y[-1], y[:-3])
#       elif i == '*':
#         if type(y[-1]) == int:
#           mapf = y[-2]; arg = y[-1]; y = y[:-2]; z = y
#           pop_ = findPop(mapf * arg, y[:])
#           push_ = findPush(mapf * arg, y[:])
#           popat, pushat = -pop_ if pop_ else len(y), -push_ if push_ else len(y)
#           for _ in range(arg): 
#             print(mapf, y); y = run(mapf, y, True)
#           print(mapf, y)
#           output, vals = y[pushat:], z[popat:]
#       elif i == '!':
#         mapf = y[-1]; y = y[:-1]; z = y
#         pop_ = findPop(mapf, y[:])
#         push_ = findPush(mapf, y[:])
#         popat, pushat = -pop_ if pop_ else len(y), -push_ if push_ else len(y)
#         # print(mapf, y)
#         # print(pop_, push_)
#         y = run(mapf, y[:])
#         output, vals = y[pushat:], z[popat:]
#       elif i == '?':
#         pass
#       push -= sum(containsFunc)
#     elif str(i) in '()':
#       # print("mod", y, pop, push)
#       y = f(y)[1:];
#       continue
#     elif str(i)[0] == 'p':
#       output = [] if i in ['p.', 'p!'] else ["0"]
#     else:
#       try: output = f(*vals, y)
#       except: output = f(*vals)
#     if output and output[0] == None: continue
#     output = [correctType(i) for i in output]
#     # print(pop, push, i, y)
#     # print(colored('debug2:', 'yellow', attrs=['bold']), vals, output)
#     if vals: 
#       pop += max(0, len(vals) - push)
#       push = max(0, push - len(vals)) + len(output)
#     y = y[:-len(vals) if len(vals) else len(y)] + output
#     # print(colored(y, 'yellow', attrs=['bold']))
#   # print(colored('debug2:', 'magenta', attrs=['bold']), pop, push, y)
#   # print(pop, push, i, y)
#   return pop, push
  
def correctType(x):
  if type(x) in f + [bool]:
    if math.isnan(x) or (type(x) is complex): return x
    elif type(x) is bool: return int(x)
    elif abs(x) == float('inf'): return x
    return int(x) if int(x) == x else x
  elif type(x) == list:
    if len(x) == 0: return x
    if all([type(_) == str and len(_) == 1 for _ in x]): return ''.join(x)
    return [correctType(i) for i in x]
  else: return x
  
class Func404(Exception): pass
class FuncError(Exception): pass
class RaisedError(Exception): pass
  
# print(findSig(((('+',), '%'), '%'), [3, [[1, 2], [11, 3]]]))
# print(map([1, 2, 3], ('+', 11), [11]))