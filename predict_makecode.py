"""
used to plot estimation and prediction
different for minAone user because the small differences in equations.txt and specs.txt
please modify it carefully if you want to use it
written by Dawei Li, March 2019
"""
import matplotlib as mpl
mpl.use('pdf')
import numpy as np
import sys
import scipy as sp
from scipy.integrate import odeint
import sympy as sym
from sympy.solvers import solve
import matplotlib.pyplot as plt
import os

plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=8)
width = 6.974
height = width / 1.618

try:
    import myfunctions
except:
    print "Alert: no myfunctions.ppy file in directory"

if len(sys.argv) < 2:
    raise ValueError("Num time steps not specified")
elif len(sys.argv)<3:
    raise ValueError("Path not specified") 
    
predict_steps = int(sys.argv[1])
pathnum = sys.argv[2]


#nS=9###
#states=['V','n','h','e','rT','Cai','rf','rs','hp']

# replace with your own quations.txt path
# replace line index to make it consistent with minAone userguide
with open('/home/dal203/david/1makecode/bird/equations.txt') as eqfile:
    count = 0
    eqns = []
    eqnstrings = []
    states = []
    params = []
    inj = []
    funcs = []
    funcstrings = []
    measvars = []
    controls = []
    for i,line in enumerate(eqfile):
        if line[0] == '#': continue
        count += 1
        # skip first line, problem name
        if count == 1:  continue
        if count == 2: 
            nS, nP, nU, nI, nF = [int(x) for x in line.strip().split(',')]
            
        #equations start at line 3
        elif count <= 2 + nS:
           # print "eqns lines = ", line
            eqnstrings.append(line)
        # Variables names at line 4+nS
        elif count == 3 + nS:
            objfunc = line.rstrip('\n')
        elif 3+ nS< count < 4 + 2*nS:
            # print "states lines = ", line
            states.append(sym.symbols(line))
            exec('{0} = sym.symbols("{0}")'.format(line.strip()))
        # Parameters start at line 4+2*nS
        elif 3+2*nS < count < 4+2*nS+nP:
            # print "param lines = ", line
            params.append(sym.symbols(line))
        elif 4+2*nS+nP <= count < 4+2*nS+nP+nU:
            controls.append(sym.symbols(line))
            exec('{0} = sym.symbols("{0}")'.format(line.strip()))
        elif 4+2*nS+nP+nU <= count < 4+2*nS+nP+nU+nU:
            measvars.append(sym.symbols(line))
            exec('{0} = sym.symbols("{0}")'.format(line.strip()))
        # Injected current starts at 5+2*nS+nP+nU+nU
       
        elif 4+2*nS+nP+nU+nU<= count < 4+2*nS+nP+nU+nU+nI:
            #print "Iinj lines = ", line
            inj.append(sym.symbols(line))
            exec('{0} = sym.symbols("{0}")'.format(line.strip()))
        elif 4+2*nS+nP+nU+nU+nI <= count < 4+2*nS+nP+nU+nU+nI+nF:
            #print "Fcn lines = ", line
            fcnname, varnum = line.strip().split(',')

            funcstrings.append(fcnname)
            
            print fcnname
            
            try:
                fcntmp = eval('myfunctions.'+fcnname)
            except:
                print fcnname
                ValueError("Is function defined in myfunctions.py?")

                
            funcs.append(fcntmp)

#print(states)
            
#data_files = []
# replace with your own quations.txt path
# replace line index to make it consistent with minAone userguide
with open('specs.txt') as specsfile:
    count = 0
    for i,line in enumerate(specsfile):
        if line[0] == '#': continue
        count += 1
        # skip first line, problem name
        if count == 1:
            nT = 2*int(line.strip())+1
        elif count == 2:
            skip = int(line.strip())
        elif count == 3:
            dt = float(line.strip())/2.0
#        elif 4 < count < 4 + nU:
#            data_files.append(line.strip()[2:])
#        elif 4 + nU <= count < 4 + nU + nI:
#            Ifile=line.strip()[2:]
            
            
I = sp.loadtxt('i.dat')[skip:]
if len(I) < nT + predict_steps:
    raise ValueError("Current file too short. Only {} steps available for prediction".format(len(I)-nT-1))
    
print "loading Path file, saving to data.dat and params.dat"

pathfile = 'data{0}.dat'.format(pathnum)
costf = 'Obj{0}.dat'.format(pathnum)
data_times = np.linspace(0.0,nT*dt - dt,nT)

path_data = sp.loadtxt(pathfile)
cost_v = sp.loadtxt(costf)
cost_value = cost_v
init = path_data[-1,1:1+nS]
path = path_data[:,1:1+nS+1]
param_values = np.loadtxt("params{0}.dat".format(pathnum), dtype = str)[:,3].astype(float)

funcNameSpace = dict(zip(funcstrings,funcs))
paramNameSpace = dict(zip(params,param_values))

for control in controls:
    paramNameSpace[str(control)] = 0.0

print paramNameSpace

x = sym.symbols('x:'+str(nS+nI))


for eq in eqnstrings:
    eqfunc = sym.sympify(eq.rstrip('\n'),locals=funcNameSpace)
    
    eqfunc = eqfunc.subs(paramNameSpace)
    eqlamb = sym.lambdify(x, eqfunc.subs(zip(states+inj,x)))        
    eqns.append(eqlamb)
    
    
Vinj1 = np.loadtxt('v.dat')[skip:]

def eqns_wrapper(x,t,I,V,dt):
    deriv = []
    Vinj = V[int(t/dt)]
    Iinj = I[int(t/dt)]
    #if x[0] >= 50.0:
     #   x[0] = 49.9
    input = list(x)+[Iinj]
    for eq in eqns:
        deriv.append( eq(*input) )
    #print deriv     
    return deriv

print "Integrating..."
#+1 because of initial condition
time = sp.linspace(nT*dt -dt,(nT+predict_steps)*dt -dt, predict_steps+1)
#backwards method, keep 5 data points outside the prediction range
predict = odeint(eqns_wrapper, init, time, (I,Vinj1,dt), mxstep = 5000000)
np.savetxt('predict{0}.dat'.format(pathnum), predict)
data0 = Vinj1

predict=np.loadtxt('predict{0}.dat'.format(pathnum))#####

if len(data0) < nT+predict_steps:
    print "Warning: data file does not have enough data points for entire prediction range"
    data_length = len(data0)
else:
    data_length = nT+predict_steps

plt.figure(figsize=(width,height))
#plt.figure()

ax1=plt.subplot(311)
plt.plot(np.hstack((data_times,time[1:])), data0[0:data_length], color = 'purple', label="Voltage", alpha = 0.7, linewidth=0.8)
plt.plot(data_times,path[:,0], color = 'blue', label = "Estimate", linewidth=0.8)
plt.plot(time,predict[:,0],color = 'red', label = 'Prediction', linewidth=0.8)
plt.ylabel("voltage (mV)")
plt.title("estimation and prediction")
#plt.title("Cost Function = {0}".format(cost_value))
plt.legend(loc=1)

#ax2=plt.subplot(312,sharex=ax1)
#plt.plot(data_times, path[:,-1], color = 'blue', label = "end", linewidth=0.8)
#plt.plot(data_times, np.full(len(data_times),100) , color = 'red', label = "start", linewidth=0.8)
#plt.ylabel("u(t)")
#plt.legend(loc=1)

ax2=plt.subplot(212)
plt.plot(np.hstack((data_times,time[1:])), I[0:data_length], color = 'black', label = "Current", linewidth=0.8)
plt.xlabel("time (ms)")
plt.ylabel("current (pA)")
plt.legend(loc=1)
plt.subplots_adjust(hspace=0.25, wspace=0.85)
#plt.savefig('fig_pred{0}.png'.format(pathnum))
plt.savefig('fig_pred{0}.pdf'.format(pathnum))
plt.close()

plt.figure()
#plt.figure(figsize=(width,height))
for i in range(nS):
    plt.subplot(3,3,i+1)###
    plt.plot(data_times,path[:,i], label = "{0}".format(states[i]))
    plt.legend()
plt.xlabel("time (ms)")
plt.subplots_adjust(hspace=0.25, wspace=0.25)
#plt.savefig('gating_variables{0}.png'.format(pathnum))
plt.savefig('gating_variables{0}.pdf'.format(pathnum))
plt.close()
