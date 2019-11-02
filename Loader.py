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