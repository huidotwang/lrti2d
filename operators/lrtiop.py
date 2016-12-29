from rsf.proj import *

lrti2d = '../tti/lrtiop.x'

# ========================================= #
class lrTIop2d:
    '''lowrank decompostion based wavefield simulation'''
    def __init__(self, v, ss, rr, par, custom=''):
        self.v = v
        self.custom = custom
        self.par = 'npk=%(npk)d seed=%(seed)d eps=%(eps)g dt=%(dt)g nb=%(nb)d atype=%(atype)s verb=%(verb)s '%par+custom
        self.lrti2d = lrti2d
        # if par['atype'] == 'tti':
            # self.septi2d = septti2d
        
        self.dep = [self.v]
        self.ss = ''
        if (ss != ''):
            self.ss = ' sou=' + ss + '.rsf '
            self.dep.append(ss)
        
        self.rr = ''
        if (rr != ''):
            self.rr = ' rec=' + rr + '.rsf '
            self.dep.append(rr)

    # ------------------------------------- #
    def FORW(self, m, d):
        Flow(d, [m]+self.dep,
            self.lrti2d 
            + ''' adj=n model=${SOURCES[1]} ''' 
            + self.ss + self.rr + self.par)

    # ------------------------------------- #
    def ADJT(self, m, d):
        Flow(m, [d]+self.dep,
            self.lrti2d 
            + ''' adj=y model=${SOURCES[1]} ''' 
            + self.ss + self.rr + self.par)

