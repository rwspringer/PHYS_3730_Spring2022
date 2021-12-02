import numpy as np
import pandas as pd
import pylab as plt
import scipy.integrate as integrate
import scipy.optimize as optimize
from scipy.ndimage.filters import uniform_filter1d as ufilt1d
import sys


state = "Utah"
#state = "Idaho"
#state = "Florida"
#state = "Missouri"

# choose a state at the command line?
if len(sys.argv)>1:
    state = ' '.join(sys.argv[1:])

# get population try US Census Bureau...

urlpop = "https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/national/totals/nst-est2019-alldata.csv"
df = pd.read_csv(urlpop)
df.index = df["NAME"]
# set up a series:
pop = pd.Series(df["POPESTIMATE2019"].values, index=df["NAME"]).astype(int)
print(f'{state} population (2019): {pop[state]}')

# get case counts...

urlcovid = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv"
df = pd.read_csv(urlcovid)

print('cols',df.columns)
df = df.loc[df["state"]==state].copy() # just keep this state's data

df['dcases'] = (df['cases']-df['cases'].shift(1,fill_value=0)) # daily cases


# smooth out some of the reporting/download irregularities...
df['dcases7dayavg'] = ufilt1d(df.dcases, size=7,mode='nearest')

# create a time coord in days since first case
df.index = np.arange(len(df.index))
df['dday'] = df['date'].to_numpy().astype(np.datetime64)
df['dday'] = (df.dday-df.dday[0]).astype('timedelta64[D]').astype(float)

# turn to model:

def dsir(t,Y, N, b0, g0):
    b, g = b0, g0
    S,I,R = Y
    dS, dI, dR = -b*I*S/N, b*I*S/N - g*I, g*I
    return dS, dI, dR

def sir(t,tinit,N,b0,g0):
    Iinit = 1.0; Sinit = N-Iinit; Rinit = 0.0
    S, I, R = np.ones(len(t))*Sinit, np.zeros(len(t)), np.zeros(len(t))
    tmsk = t>=tinit # start time is different from t[0], maybe
    sol = integrate.solve_ivp(dsir,[tinit,t[-1]],[Sinit,Iinit,Rinit],                              t_eval=t[tmsk],args=(N,b0,g0))
    S[tmsk],I[tmsk],R[tmsk] = sol.y
    return S,I,R

def dcasesmodel(t,tinit,N,b0,g0):
    S,I,R = sir(t,tinit,N,b0,g0)
    dS = -(S[1:]-S[:-1]) # change in suseptible from previous day
    return np.append(np.array([0]),dS)

def chi2ish(p,dday,dcases,tepi_start,tfit_start,tfit_end,N):
    b0,g0 = p
    model = dcasesmodel(dday.to_numpy(),tepi_start,N,b0,g0)
    tmsk = (dday>=tfit_start)&(dday<+tfit_end)
    return np.sum((model[tmsk]-dcases[tmsk])**2)

g0 = 1/3.; b0 = 1.1*g0
N = pop[state]
df['dmod'] = dcasesmodel(df.dday.to_numpy(),0,N,b0,g0)

bgguess = [b0,g0]
t0wild,tfbeg,tfend = 0,0,420 
args=(df.dday,df.dcases,t0wild,tfbeg,tfend,N)
sol = optimize.minimize(chi2ish,bgguess,args=args,method='Nelder-Mead')        
bgwild = sol.x
print('b,g fit:',sol.x)
df['dmod'] = dcasesmodel(df.dday.to_numpy(),0,N,*bgwild)


# show the world...
ax = df.plot('date','dcases',color='#aaaaff')
ax.plot(df['date'],df['dcases7dayavg'],'-k')
ax.plot(df.date,df.dmod,'k:')
ax.figure.autofmt_xdate()
ax.set_title(state+' case counts')
ax.figure.savefig('tmp.png')

# comment this out!?
if 1:
    import os
    os.system('convert tmp.png ~/www/tmp.jpg')

print('done')
quit()


