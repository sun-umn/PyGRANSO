from private.isRealValued import isRealValued
from private.isARealNumber import isARealNumber

def isRestartData(data):
    """  
    isRestartData:
      Checks whether data is lbfgs restart data.
    """

    if isinstance(data,dict) == False:
      return False

    if  'S' in data and 'Y' in data and 'rho' in data and 'gamma' in data:
      return True
    
    return False

      # if  isARealNumber(data['gamma']):
      #   try:
      #     [n_S,cols_S]    = data['S'].shape
      #     [n_Y,cols_Y]    = data['Y'].shape
      #     [_,cols_rho]    = data['rho'].shape
      #     return True
      #   except Exception as e:
      #     return False

    