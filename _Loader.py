import time

class loading():
  def __init__(self, strMess, mod):
    self.statement = strMess
    self.i = 0
    self.mod = mod
    print('\n')
  def showLoad(self):
    print(self.statement+'.'*self.i, end='\r',flush=True)
    self.i += 1
    self.i = self.i%self.mod

def wait(x):
  print("Hold for {0} seconds ".format(x),end='\r',flush=True)
  while x > 0 :
    print("{0} seconds Remaining  ".format(x), end='\r',flush=True)
    time.sleep(1)
    x -= 1
