import parse
import os, sys, re
import numpy as np
from termcolor import colored, cprint

def parseLine(line, reach = 0):
  tokens, token, idx = [], "", 0
  if line.isspace() or line == '': return [], 0
  while True:
    token = line[idx]
    if token.isdigit():
      while token.replace('.','',1).isdigit():
        idx += 1
        try: token += line[idx]
        except: token += ' '; break
      idx -= 1
      if token[-2] == '.': 
        idx -= 1; token = token[:-1]
      token = int(token[:-1]) if '.' not in token[:-1] else float(token[:-1])
    elif token == '"':
      idx += 1; token = ''
      while line[idx] != '"':
        token += eval(f"\"\\{line[idx + 1]}\"") if line[idx] == '\\' else line[idx]
        idx += (line[idx] == '\\') + 1
        if idx >= len(line): break
      token = f'"{token}"'
    elif token == '\'':
      idx += 1
      escape = { '0': 1, 'n': 1, 'r': 1, 't': 1, 'x': 3, 'u': 5, 'U': 9 }
      if line[idx] == '\\': 
        if line[idx + 1] in escape.keys():
          token = eval(f'"{line[idx:idx + 1 + escape[line[idx + 1]]]}"')
        else:
          print(colored('SyntaxError: ', 'red', attrs = ['bold']) + colored(f'"\\{line[idx + 1]}" is not a valid character', 'red')); sys.exit(0)
      else: token = line[idx]
      token = f'"{token}"'; idx += escape[line[idx + 1]] if (line[idx] == '\\') else 0
    elif token in '{[':
      bracks = [token];
      while bracks:
        # print(token, bracks, line)
        idx += 1;
        try: token += line[idx]
        except: 
          print(colored('SyntaxError: ', 'red', attrs = ['bold']) + colored(f'\'{bracks[-1]}\' was never closed', 'red')); sys.exit(0)
        if token[-1] == '│':
          token = token[:-1]; line = line[:idx] + '}{' + line[idx + 1:]; idx -= 1
        if token[-2:] not in list(parse.FUNCTIONS.keys()):
          if token[-1] in '{[': bracks += token[-1]
          elif token[-1] in ']}':
            if bracks[-1] + token[-1] == '{]':
              line = line[:idx] + '{' + line[idx + 1:]
              token = token[:-1] + '}'; idx -= 1; break;
            # print(bracks, repr(token))
            if bracks[-1] + token[-1] not in ['[]', '{}'] and token[-2:] not in list(parse.FUNCTIONS.keys()): 
              print(colored('SyntaxError: ', 'red', attrs = ['bold']) + colored(f'Unmatched \'{token[-1]}\'', 'red')); sys.exit(0)
            else: bracks.pop();
      if reach == 2: 
        tokens += [token]; idx += 1; break
      # print(token, bracks, idx)
    elif token in ']}':
      print(colored('SyntaxError: ', 'red', attrs = ['bold']) + colored(f'Unmatched \'{token[-1]}\'', 'red')); sys.exit(0)
    elif token in '‘“':
      idx += 1
      _, length = parseLine(line[idx:], ' ‘“'.index(token))
      token = '{' + line[idx:idx + length] + '}'
      # print(line[idx:idx + length], token)
      idx += length - 1
    elif token == '⌀':
      token = '{}'
    elif token.isupper(): 
      # while token.isupper() and token.isalpha():
      #   idx += 1
      #   try: token += line[idx]
      #   except: token += ' '; break
      # idx -= 1; token = token[:-1]
      pass
    elif line[idx:idx + 2] == '`=':
      idx += 2; 
      try: 
        assert line[idx].isupper(); token = '`=' + line[idx]
      except: print(colored('DefinitionError: ', 'red', attrs = ['bold']) + colored(f'Variable not found preceding "`="', 'red')); sys.exit(0)
      # while token.isupper() and token.isalpha():
      #   idx += 1
      #   try: token += line[idx]
      #   except: token += ' '; break
      # idx -= 1
      # token = "⸬" + token[:-1]
    elif token in [' ', '\n', '\t', '\r']:
      idx += 1; 
      try: line[idx]; continue
      except: break
    elif token == '`' and line[idx + 1].isalpha():
      token += line[idx + 1]; idx += 1
    elif token.islower():
      idx += 1; token += line[idx]
      if token not in list(parse.FUNCTIONS.keys()):
        print(colored('FuncError: ', 'red', attrs = ['bold']) + colored(f'"{token}" is not a valid function.', 'red')); sys.exit(0)
      elif token in list(parse.FUNCTIONS.keys()):
        if reach == 1: 
          tokens += [token]; idx += 1; break
        elif reach == 2: 
          print(colored('FuncError: ', 'red', attrs = ['bold']) + colored(f'"{token}" is not a constant.', 'red')); sys.exit(0)
    else:
      funcs = list(parse.FUNCTIONS.keys())
      contains = lambda x, y: [x in i for i in y]
      while any(contains(token, funcs)):
        idx += 1
        try: token += line[idx]
        except: token += ' '; break
      idx -= 1; token = token[:-1]
      if reach == 1: 
        tokens += [token]; idx += 1; break
      elif reach == 2: 
        print(colored('FuncError: ', 'red', attrs = ['bold']) + colored(f'"{token}" is not a constant.', 'red')); sys.exit(0)
    tokens += [token]; idx += 1
    # print(repr(token), idx, repr(line), repr(line[:idx]))
    try: line[idx]
    except: break
    if reach == 2: break
  return tokens, idx
  
def parseNestedBracks(line):
  if str(line)[0] + str(line)[-1] in ['[]', '{}']:
    result = []
    if line[0] in '[{': result = [parseNestedBracks(i) for i in parseLine(line[1:-1])[0]]
    return result if line[0] == '[' else tuple(result)
  else: return line
  
def depth(array, d = 1):
  if type(array) != list: return 0
  if all([type(i) != list for i in array]): return d + 1
  return max([depth(i, d + 1) for i in array])
  
def printData(value, d = 0):
  # if d == 0: maxd = depth(value)
  if type(value) != list: return '  ' * d + repr(value)
  if all([type(i) != list for i in value]): return f'{d * "  "}[{' '.join([repr(i) for i in value])}]'
  string = [printData(i, d + 1) for i in value]
  # if all([type(j) != list for i in value for j in i]):
  #   string = [d * '  '] * len(value)
  #   for i in range(len(value[0])):
  #     maximum = max([len(printData(j[i])) for j in value])
  #     print(maximum, value)
  #     data = [printData(j[i]) + ' ' * (maximum - len(printData(j[i]))) for j in value]
  #     string = [i + j + ' ' for i, j in zip(string, data)]
  #   return '\n'.join(string)
  # return d * '  ' + '╭\n' + ('\n' * (maxd - d - 2)).join(string) + '\n' + ' ' * (max([len(j) for i in string for j in i.split("\n")]) + 1) + '╯'
  return f'{d * "  "}[\n{"\n".join([printData(i, d + 1) for i in value])}\n{d * "  "}]' # + ('\n' * (d != 1))
  
FORMATS = {
  # '%%': '⁒',
  # '_>': '»',
  # '_<': '«',
  # '~~': '〜',
  # '_~': '≈',
  # '.;': '„',
  '`\'': '‘', # F-unction
  '`"': '“', # C-onstant
  '{}': '⌀',
  # '(:*)::(?!⸬)': '\\1⸬',
  # '\\\\': '＼',
  # '::': '≡',
  # '_!': '¡',
  # '_#': '↕',
  # '$$': '§'
  # '\n': ' '
}

def format(string, flip = False):
  for key, value in FORMATS.items():
    string = re.sub(value, key, string) if flip else re.sub(key, value, string)
  return string

if __name__ == '__main__':
  sys.tracebacklimit = None
  if sys.argv[1][-4:] == '.agl':
    with open(sys.argv[1], 'r') as contents:
      content = contents.readlines()
    with open(sys.argv[1], 'w') as contents:
      contents.writelines([format(i[:(a := i.index('``') if '``' in i else len(i))]) + i[a:] for i in content])
    with open(sys.argv[1], 'r') as contents: 
      contents.seek(0)
      file = ''.join(i[:i.index('``') if '``' in i else len(i)] for i in contents.readlines()).replace('\n', ' ')
    # print(file)
    os.system('cls' if os.name == 'nt' else 'clear')
    tokens = [parseNestedBracks(i) for i in parseLine(file)[0]]
    stack = parse.run(tokens, [])
    if stack: print('{\n' + '\n'.join([printData(i, 1) for i in stack]) + '\n}')