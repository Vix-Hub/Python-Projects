import numpy as np 
from matplotlib import pyplot as plt
import random as r
from scipy.linalg import eigh

r.seed(10)

N = 10

#Autofunzioni della buca simmetrica

def phi(x,n,L):
    if (n%2) == 0:
        f = np.cos((n+1)*np.pi*x/(2*L))
    else:
        f = np.sin((n+1)*np.pi*x/(2*L))
    f = f / np.sqrt(L)
    return f

#Numero di valori casuali provati per il parametro variazionale
n_a = 30

x = np.linspace(-20,20, 500)

## Derivata della funzione al quarto ordine in Taylor

def d_dx(ft, xt):
    dx = xt[1]-xt[0]
    ftp = np.zeros_like(ft)
    for i in range(2,len(ft)):
        if (i<(len(ft)-2)):
            rise = (-ft[i+2]+8*ft[i+1]-8*ft[i-1]+ft[i-2])
            ftp[i] = rise/(12*dx)
        else:
            rise = ft[i]-ft[i-1]
            ftp[i] = rise/dx
    ftp[0] = (ft[1]-ft[0])/dx
    ftp[1] = (ft[2]-ft[0])/dx
    return ftp

## Azione della parte cinetica T sulla funzione d'onda

def TPhi(ft,xt):
    ftp = d_dx(ft,xt)
    ftpp = d_dx(ftp,xt)
    return -0.5*ftpp

## Azione del potenziale V sulla funzione d'onda
    
def VPhi(ft,xt):
    return 0.5*xt*xt*ft

## Azione del funzionale T[Phi]
    
def T_Func(ft,xt):
    t = TPhi(ft,xt)
    num = np.trapz(t*ft,xt)
    den = np.trapz(ft*ft,xt)
    return num/den

## Azione del funzionale V[Phi]

def V_Func(ft,xt):
    vphi = VPhi(ft,xt)
    num = np.trapz(vphi*ft,xt)
    den = np.trapz(ft*ft, xt)
    return num/den

##Loop su vari valori del parametro per trovare gli L ottimali
L_best = []
E_best = []
L_alpha, E_alpha = [], []
var = np.zeros(shape=(n_a))
## se cambi funzione dovrebbero partire da i = 1 -> N+1, negli array metti [i-1]
for k in range(N):
    L = np.zeros(n_a)
    E_func = np.zeros(n_a)
    for i in range(len(L)):
        L[i] = 10* r.random()
    L.sort()
    if k==0:
        L_alpha= L
    for i in range(len(L)):
        y = np.linspace(-L[i], L[i], 500)
        Phi_Trial = phi(y,k,L[i])
        E_func[i] = T_Func(Phi_Trial,y)+V_Func(Phi_Trial,y)
        if k==0:
            var[i] = np.trapz(TPhi(TPhi(Phi_Trial,y),y),y)+np.trapz(VPhi(VPhi(Phi_Trial,y),y),y)+ np.trapz(TPhi(VPhi(Phi_Trial,y),y),y)+ np.trapz(VPhi(TPhi(Phi_Trial,y),y),y)
            var[i] = var[i] - E_func[i]**2
            var[i] = var[i] / np.trapz(Phi_Trial**2, y)
    E = list(E_func)
    if k==0:
        E_alpha = E
    ind = E.index(min(E_func))
    E_best.append(min(E_func))
    L_best.append(L[ind])

#Plottiamo il primo stato eccitato reale e quello approssimato
x = np.linspace(-L_best[1],L_best[1],500)
phi_1 = phi(x,1,L_best[1])
True_1 = np.sqrt(2)*x*np.exp(-x*x/2)/np.sqrt(np.sqrt(np.pi))
plt.plot(x,phi_1, label = 'Variational Excited State')
plt.plot(x,True_1, label = 'True State')
plt.legend(loc='upper left')
plt.xlabel('x')
plt.ylabel('Phi')
plt.show()

#plot autovalori
p, c = [], [] 
for i in range(N):
    p.append(i)
    c.append(i+0.5)

plt.plot(p,c, label= 'True Energy Levels')
plt.plot (p,E_best, label = 'Approximate Energy Levels')
plt.legend(loc='upper left')
plt.xlabel('n')
plt.ylabel('Energy')
plt.show()

#plot ground state
y = np.linspace(-L_best[0], L_best[0], 500)
phi_0 = phi(y,0,L_best[0])
True_0 = np.exp(-y*y/2)/np.sqrt(np.sqrt(np.pi))
plt.plot(y,phi_0, label='Variational Ground State')
plt.plot(y,True_0, label = 'True Ground State')
plt.legend(loc='lower center')
plt.xlabel('x')
plt.ylabel('Phi')
plt.show()


#plot dell'energia in funzione del parametro

Ls = L_alpha[1:]
Es = E_alpha[1:]

plt.plot(Ls,Es)
plt.xlabel('Lenght of the Box')
plt.ylabel('Ground State Energy')