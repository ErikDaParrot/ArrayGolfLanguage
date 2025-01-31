import parse
import os
import sys

def parseLine(line):
  tokens, token, idx = [], "", 0;
  if line.isspace() or line == '': return []
  while True:
    token = line[idx]
    if token.isdigit():
      while token.replace('.','',1).isdigit():
        idx += 1
        try: token += line[idx]
        except: token += ' '; break
      idx -= 1
      token = int(token[:-1]) if '.' not in token[:-1] else float(token[:-1])
    elif token == '"':
      idx += 1; token = ''
      while line[idx] != '"':
        token += eval(f"\"\\{line[idx + 1]}\"") if line[idx] == '\\' else line[idx]
        idx += (line[idx] == '\\') + 1
      token = f'"{token}"'
    elif token == '\'':
      idx += 1
      if line[idx] == '\\': token = eval(f'"{line[idx:idx + 2]}"')
      else: token = line[idx]
      token = f'"{token}"'; idx += (line[idx] == '\\')
    elif token in '{[':
      bracks = [token];
      while bracks:
        idx += 1;
        try: token += line[idx]
        except: raise SyntaxError(f"'{bracks[-1]}' was never closed")
        if token[-1] in '{[': bracks += token[-1]
        elif token[-1] in ']}':
          if bracks[-1] + token[-1] not in ['[]', '{}']: raise SyntaxError(f"'{token[-1]}' != '{bracks[-1]}'")
          else: bracks.pop();
    elif token in ']}': raise SyntaxError(f"unmatched '{token}'")
    elif token in '()`': pass
    elif token == token.upper() and token.isalpha(): pass
    elif line[idx:idx + 2] == '::':
      token = line[idx:idx + 3]; idx += 2
    elif token in [' ', '\n']:
      idx += 1; 
      try: line[idx]; continue
      except: break
    elif token == 'v':
      idx += 1; token += line[idx]
    else:
      while token in ''.join(list(parse.FUNCTIONS.keys())):
        idx += 1
        try: token += line[idx]
        except: token += ' '; break
        if token[:-1] in list(parse.FUNCTIONS.keys()): break;
      idx -= 1; token = token[:-1]
    tokens += [token]; idx += 1
    try: line[idx]
    except: break
  return tokens
  
def parseNestedBracks(line):
  if str(line)[0] + str(line)[-1] in ['[]', '{}']:
    result = []
    if line[0] in '[{': result = [parseNestedBracks(i) for i in parseLine(line[1:-1])]
    return result if line[0] == '[' else tuple(result)
  else: return line
  
if __name__ == '__main__':
  if sys.argv[1][-3:] == '.ag':
    with open(sys.argv[1]) as contents:
      file = '\n'.join([i[:i.index('.;')] for i in contents.readlines()])
    os.system('cls' if os.name == 'nt' else 'clear')
    tokens = [parseNestedBracks(i) for i in parseLine(file)]
    stack = parse.run(tokens, [])
    if stack: print('{\n' + '\n'.join(['  ' + repr(i) for i in stack]) + '\n}')
