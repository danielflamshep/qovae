import numpy as np
import sympy as sp
import random, time
from numpy.random import choice
from sympy.matrices import SparseMatrix
from sympy import collect, expand, Symbol,sqrt,pi,I
from itertools import combinations, chain
from sympy import *
a, b, c, d, e, f, FF1, FF2, FF3, FF4, FF5, FFn, HH, GG1, GG2, GG3, GG4, GG5 =map(sp.IndexedBase,['a','b','c','d','e', 'f', 'FF1','FF2','FF3','FF4','FF5','FFn','HH','GG1','GG2','GG3','GG4', 'GG5'])
l,l1, l2, l3, l4, l5, l6, l7, l8, x1, x2, x3, x4, x5, x6, coeff, powern =map(sp.Wild,['l','l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'coeff', 'powern'])

zero=Symbol('zero') ## using in 4-fold coincidence
psi=Symbol('psi')  ## represent a quantum state
sqr2=sqrt(2)/2  ## equals to 1/sqrt(2)
imagI=I ## equals to Imaginary
Pi=pi

def replace(expr, path, sub):
    return expr.replace(path, sub, map=False, simultaneous=True, exact=False)

## SPDC process
def DownConvOAM(lorder, p1, p2):  ## create the initial state, DC is a parameter
    initial_state = 0
    for ii in range(-lorder, lorder+1):
        initial_state = p1[ii] * p2[-ii] + initial_state
    return initial_state

def BS_fun(expr, p1, p2):
    if expr.base==p1:
        return replace(expr, p1[l], sqr2 * (p2[l] + imagI * p1[-l]))
    else:
        return replace(expr, p2[l], sqr2 * (p1[l] + imagI * p2[-l]))

def BS(psi,p1,p2):
    psi0=sp.expand(psi.replace(lambda expr: expr.base in [p1,p2],
                               lambda expr: BS_fun(expr,p1,p2)))
    log_print('state_calc','bs : '+  str(psi0))
    return psi0

def DownConv(state, lorder, p1, p2):
    for ii in range(-lorder, lorder+1):
        state = p1[ii]*p2[-ii] + state
    #log_print('state_calc','dc : '+  str(state))
    return state

def LI_fun(expr,p1,p2):
    if expr.base==p1:
        sub = (sp.cos(l1*Pi/2)**2)*p1[l1]+imagI*(sp.sin(l1*Pi/2)**2)*p2[-l1]
        return expr.replace(p1[l1], sub, map=False, simultaneous=True, exact=False)
    else:
        sub = -(sp.cos(l1*Pi/2)**2)*p2[l1]+imagI*(sp.sin(l1*Pi/2)**2)*p1[-l1]
        return expr.replace(p2[l1], sub, map=False, simultaneous=True, exact=False)

def LI(psi,p1,p2):
    psi0 = sp.expand(psi.replace(lambda expr: expr.base in [p1,p2],
                                 lambda expr: LI_fun(expr,p1,p2)))
    return psi0

def Reflection(expr, p):
    expr = replace(expr, p[l1], imagI * p[-l1])
    log_print('state_calc','ref : '+  str(expr))
    return expr

def OAMHolo(expr, p, n):
    expr = replace(expr, p[l1], p[l1+n])
    log_print('state_calc','h : '+ str(expr))
    return expr

def DP(expr, p):
    expr = replace(expr, p[l1], imagI * sp.exp(imagI * l1 * (Pi) ) * p[-l1])
    log_print('state_calc','dp : '+ str(expr))
    return expr

def setuptoinput(setup_list):
    """[device(), ...] --> device(device(...)) """
    YYY='XXX'
    for kk in range(len(setup_list)-1,-1,-1): # in the inverted sequence
        element = setup_list[kk]
        YYY = YYY.replace('XXX', element)
    return YYY

def toHH(expr):  ## calculate pseudo density matrix
    expr1 = expr.replace(coeff*FF2[l1]*FF3[l2]*FF4[l3]*FF5[l4],
                         sp.conjugate(coeff)*GG2[l1]*GG3[l2]*GG4[l3]*GG5[l4])
    rho0=sp.expand(expr*expr1)
    rho0=rho0.replace(coeff*FF2[l1]*FF3[l2]*FF4[l3]*FF5[l4]*GG2[l5]*GG3[l6]*GG4[l7]*GG5[l8],
                      coeff*HH[l1,l2,l3,l4,l5,l6,l7,l8])
    return rho0

def assess_rho(rho):
    """computes rank and entropy of density rho"""
    rho = np.array(rho).astype(np.complex64)
    rank = np.linalg.matrix_rank(rho)
    eig = np.linalg.eigvals(rho)
    eig = np.ma.masked_equal(eig, 0.0)
    entropy = -np.sum(eig*np.log(eig))
    return rank, entropy

def PartialTraceOne(expr,n): ##calculate the partial trace such as A|BCD

    dictadd=collect(expr, [HH[l1,l2,l3,l4,l5,l6,l7,l8]], evaluate=False)
    TermsCoeff=list(dictadd.items())

    ParticleOne=[]
    ParticleTwo=[]
    ParticleThree=[]
    ## get the size of the matrix
    for ii in range(len(TermsCoeff)):
        HHList=TermsCoeff[ii][0]
        if HHList.indices[n-1]==HHList.indices[n+3]:
           ll=[HHList.indices[0], HHList.indices[1],
               HHList.indices[2], HHList.indices[3],
               HHList.indices[4], HHList.indices[5],
               HHList.indices[6], HHList.indices[7]]
           del(ll[n-1],ll[n+2])  ## because cannot del all at the same time, thus do it one by one, the index is not n+2

           ParticleOne.append(ll[0])
           ParticleTwo.append(ll[1])
           ParticleThree.append(ll[2])
    # start from 0
    Upperone=max(ParticleOne)+1
    Lowerone=min(min(ParticleOne),0)
    Uppertwo=max(ParticleTwo)+1
    Lowertwo=min(min(ParticleTwo),0)
    Upperthree=max(ParticleThree)+1
    Lowerthree=min(min(ParticleThree),0)

    rangeP1=Upperone-Lowerone
    rangeP2=Uppertwo-Lowertwo
    rangeP3=Upperthree-Lowerthree

    Msize=(rangeP1*rangeP2*rangeP3)
    SMatrix=SparseMatrix(Msize, Msize, {(0, 0): 0})

    for ii in range(len(TermsCoeff)):
        HHList=TermsCoeff[ii][0]
        if HHList.indices[n-1]==HHList.indices[n+3]:
           ll=[HHList.indices[0],HHList.indices[1],HHList.indices[2],HHList.indices[3],HHList.indices[4],HHList.indices[5],HHList.indices[6],HHList.indices[7]]
           del(ll[n-1],ll[n+2]) ## because cannot del all at the same time, thus do it one by one, the index is not n+2
           Dimrow=(ll[0]-Lowerone)*rangeP3*rangeP2+(ll[1]-Lowertwo)*rangeP3+(ll[2]-Lowerthree)
           Dimcol=(ll[3]-Lowerone)*rangeP3*rangeP2+(ll[4]-Lowertwo)*rangeP3+(ll[5]-Lowerthree)
           SMatrix=SparseMatrix(Msize, Msize, {(Dimrow,Dimcol):TermsCoeff[ii][1]})+SMatrix

    return assess_rho(SMatrix)

def PartialTraceTwo(expr,m,n): ##calculate the partial trace such as AB|CD

    dictadd=collect(expr, [HH[l1,l2,l3,l4,l5,l6,l7,l8]], evaluate=False)
    TermsCoeff=list(dictadd.items())

    ParticleOne=[]
    ParticleTwo=[]
    ## get the size of the matrix
    for ii in range(len(TermsCoeff)):
        HHList=TermsCoeff[ii][0]
        if HHList.indices[m-1]==HHList.indices[m+3] and HHList.indices[n-1]==HHList.indices[n+3]:
           ll=[HHList.indices[0],HHList.indices[1],HHList.indices[2],HHList.indices[3],HHList.indices[4],HHList.indices[5],HHList.indices[6],HHList.indices[7]]
           del(ll[m-1])
           del(ll[m+2])  ## because cannot del all at the same time, thus do it one by one, the index is not n+2
           del(ll[n-2])
           del(ll[n])
           ParticleOne.append(ll[0])
           ParticleTwo.append(ll[1])

    # start from 0
    Upperone=max(ParticleOne)+1
    Lowerone=min(min(ParticleOne),0)
    Uppertwo=max(ParticleTwo)+1
    Lowertwo=min(min(ParticleTwo),0)
    rangeP1=Upperone-Lowerone
    rangeP2=Uppertwo-Lowertwo

    Msize=(rangeP1*rangeP2)
    SMatrix=SparseMatrix(Msize, Msize, {(0, 0): 0})

    for ii in range(len(TermsCoeff)):
        HHList=TermsCoeff[ii][0]
        if HHList.indices[m-1]==HHList.indices[m+3] and HHList.indices[n-1]==HHList.indices[n+3] :
           ll=[HHList.indices[0],HHList.indices[1],HHList.indices[2],HHList.indices[3],HHList.indices[4],HHList.indices[5],HHList.indices[6],HHList.indices[7]]
          ## because cannot del all at the same time, thus do it one by one, the index is not n+2
           del(ll[m-1])
           del(ll[m+2])  ## because cannot del all at the same time, thus do it one by one, the index is not n+2
           del(ll[n-2])
           del(ll[n])
           Dimrow=(ll[0]-Lowerone)*rangeP2+(ll[1]-Lowertwo)
           Dimcol=(ll[2]-Lowerone)*rangeP2+(ll[3]-Lowertwo)
           SMatrix=SparseMatrix(Msize, Msize, {(Dimrow,Dimcol):TermsCoeff[ii][1]})+SMatrix

    return assess_rho(SMatrix)

def replaceRule(expr, repls):
    for k, m in repls.items():
        expr = expr.replace(k, m, map=False, simultaneous=True, exact=False)
    return expr

def MakeFF2(expr):
    NFoldrepls = {coeff*a[l1]*a[l2] : 0, coeff*b[l1]*b[l2] : 0,
                  coeff*c[l1]*c[l2] : 0, coeff*d[l1]*d[l2] : 0}
    expr1=replaceRule(expr,NFoldrepls)
    NFoldrepls={coeff*a[l2]*b[l3]*c[l4]*d[l5]: 0}
    expr2=replaceRule(expr1, NFoldrepls)
    expr=expr1-expr2
    expr=expr.replace(coeff*a[l1]*b[l2]*c[l3]*d[l4],
                      coeff*FF2[l1]*FF3[l2]*FF4[l3]*FF5[l4])
    return expr

def NormalizedCoeff(expr):
    dictadd=collect(expr, [FF2[x1]*FF3[x2]*FF4[x3]*FF5[x4]], evaluate=False)
    TermsCoeff=list(dictadd.items())
    CoefOfEachTerm=[]
    for ii in range(len(TermsCoeff)):
        CoefOfEachTerm.append(np.abs(TermsCoeff[ii][1])**2)
    NormalCoeff=1/sqrt(sum(CoefOfEachTerm))
    return NormalCoeff

def list_coef(expr):
    dictadd=collect(expr, [FF2[x1]*FF3[x2]*FF4[x3]*FF5[x4]], evaluate=False)
    TermsCoeff=list(dictadd.items())
    coefs=[]
    for ii in range(len(TermsCoeff)):
        coefs.append(TermsCoeff[ii][1])
    return coefs

def compute(expr):
    rho=toHH(expr) #; print('RHO', rho) # matrix list with modes [-max_mode,max_mode]
    if rho==0 :return [0,0,0,0,0,0,0], 0
    else:
        srv1, e1 = PartialTraceOne(rho,1)
        op_srv = [srv1]
        tp_srv = []
        es = [e1]
        for i in [2,3,4]:
            srvi1, ei1 = PartialTraceOne(rho, i)
            srvi2, ei2 = PartialTraceTwo(rho, 1, i)
            op_srv += [srvi1]
            tp_srv += [srvi2]
            es += [ei1,ei2]
        return op_srv+tp_srv, [np.real(e) for e in es], np.real(np.sum(es))

def calculate(setup_list, save_name='state_calc'):
    """calcs entropy and srv of setup from list of devices"""
    setupStr = setuptoinput(setup_list)
    #log_print(save_name, setupStr) # Device(Device(...))
    DCState =(a[0]*b[0]+c[0]*d[0])
    FuncStr=setupStr.replace("XXX",str(DCState))
    output_state = expand(expand(eval(FuncStr))**2) # terms involving a[] b[] c[] d[]
    #log_print(save_name, str(output_state))
    outputState = MakeFF2(output_state)
    log_print(save_name, str(outputState)) # unnormalized FF2[]FF3[]FF4[]FF5[]
    normalized_output = expand(outputState*(NormalizedCoeff(outputState).evalf()))
    #log_print(save_name,str(normalized_output)) # normalized FF2[]FF3[]FF4[]FF5[]
    state, probs = state_to_latex(str(normalized_output), list_coef(normalized_output)) # state |0000>
    #log_print(save_name+'state', state)
    srv, es, entropy = compute(normalized_output)
    log_print(save_name+'entropy', 'SRV {} ENTROPY {} TOTAL {}'.format(srv, es, entropy))
    return state, probs, srv, es, entropy

def load_(num = 2):
    """loads data as list"""
    s = open('data/setups.smi','r')
    e = open('data/entropy.smi','r')
    s = np.array([line.strip("\r\n ").split('.') for line in s])
    e = np.array([float(line.strip("\r\n ")) for line in e])
    idx =  np.random.randint(0,len(s),num)
    #idx=102
    return s[idx], e[idx]

def get_num(dev, devices):
    return str(sum([1 for device in devices if dev in device]))

def state_to_latex(state_string, list_coef, add_coef=False):
    """ -sqrt(2)*I*FF2[-1]*FF3[-1]*FF4[0]*FF5[0]/2 + ... --> -0.71i|-1,-1,-,0,0> """
    state_list = state_string.split(' ')
    state_list = [term for term in state_list if term not in ['','+','-']]
    #print(state_list)
    state = ''
    vals = {}
    for term, val in zip(state_list, list_coef):
        ls = term.split('[')
        s = [c.split(']')[0]+',' for c in ls[1:]]
        v2 = np.absolute(N(val**2,2)) #; vals.append(v2)
        if add_coef:
            cf = str(N(val, 2)) ; cf = cf.replace('*I','i')
        else:
            cf = ''
        ket = ''.join(a for a in s)
        term = cf + '|'+ket+r"\rangle + "
        vals[tuple(s)] = v2
        state += term
#    print('kets and coefs', vals)
    print('state', state)
    return state, vals

def coef_to_latex(list_coef):
    """ """
    coefs = ""
    vals = []
    for i, val in enumerate(list_coef):
        v2 = np.absolute(N(val**2,2)) ; vals.append(v2)
        val = N(val, 4) #; print(val)
        c = 'a_' + str(i) + '=' + str(val) +' , '
        c = c.replace('*I','i')
        coefs += c
    print(coefs)
    return coefs

def log_print(fname, text, on=False):
    if on:
        with open(fname, 'a') as f:
            print(text)
            f.write('-'*50+'\n'+str(text) + '\n')

if __name__ == '__main__':
    example = ["OAMHolo(XXX,f,-2)","BS(XXX,a,d)","Reflection(XXX,f)","BS(XXX,b,c)","DownConv(XXX,1,a,d)","DownConv(XXX,1,b,e)","BS(XXX,a,d)", "OAMHolo(XXX,f,1)"]
    calculate(example)
