from pygransoStruct import Options

def makeStructWithFields(varargin):
    """
    makeStructWithFields.py
      Returns an empty struct with the fieldnames specified as input
      arguments but all set to [].
    """
    s = Options()
    for arg in varargin:
      setattr(s,arg,None)
    
    return s