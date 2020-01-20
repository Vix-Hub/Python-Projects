import numpy as np 
from scipy.linalg import eigh
from matplotlib import pyplot as plt

V_0 = 250
N = 50
a = 1.0
#b = 0.1
#d_1 = a/2 - b
#d_2 = a/2 + b
d_1 = 0.45   #posizione della barriera#
d_2 = 0.6
b = (d_2-d_1)/2


    
def k(n):
    return np.pi*n/a
    
def phi(i,x):
    s = np.sqrt(2/a)
    return s*np.sin(k(i)*x)

V = np.zeros(shape=(N,N))
T = np.zeros_like(V)

for n in range(1,N+1):
    for m in range(1,N+1):
        if n!=m:
            V[m-1][n-1] = (np.sin(k(n-m)*d_2)-np.sin(k(n-m)*d_1))/ (k(n-m)) - (np.sin(k(n+m)*d_2)-np.sin(k(n+m)*d_1)) / (k(n+m))
            V[m-1][n-1] = V_0*V[m-1][n-1] / a
        else:
            V[m-1][n-1] = d_2-d_1 - (np.sin(2*k(n)*d_2)-np.sin(2*k(n)*d_1)) / (2*k(n))
            V[m-1][n-1] = V_0*V[m-1][n-1] / a
            
for i in range(1,N+1):
    T[i-1][i-1] = 0.5*(k(i)**2)
    

H = T + V

#Diagonalizza la matrice H
eigvals, eigvecs = eigh(H)

#Calcola gli autostati approssimati
def eigstate(k,x):
    s = 0
    for i in range(N):
        s+= eigvecs[i][k]*phi(i+1,x)
    return s


x = np.linspace(0, a, 500)
f = eigstate(5,x)
plt.xlabel('x')
plt.plot(x,f*f,label='Square of the Wave Function, n=1')

#Per plottare il potenziale (non in scala)
def V(x):
    v = []
    for y in x:
        if y == 0:
            v.append(5)
        elif y == 1:
            v.append(5)
        elif (abs(y-d_1)<0.001):
            v.append(4.5)
        elif (d_1 <= abs(y)<= d_2):
            v.append(4.5)
        elif (abs(y-d_2)<0.001):
            v.append(4.5)
        else:
            v.append(0)
    return v

v = V(x)
plt.plot(x,v, label='Potential Well')
plt.legend(loc='upper right')
plt.show()

z = np.linspace(5, 7, 500)

def alpha(x):
    return np.sqrt(2*V_0 - x**2)

#Per la risoluzione grafica dell'equazione esatta
left = np.exp(-4*alpha(z)*b) * (alpha(z)-z/(np.tan(z*(0.5*a-b))))**2
right = (alpha(z)+z/(np.tan(z*(0.5*a-b))))**2

plt.plot(z,left, label='left')
plt.plot(z,right, label= 'right')
plt.legend(loc='upper left')
plt.show()


#Valori ottenuti eseguendo cambiando gli N messi nella lista ind
arr_1 = [16.98736, 16.92733, 16.90355, 16.89811, 16.89718]
ind = [5,10,15,30,50]

arr_2 = [21.88406, 21.68077, 21.67715, 21.67227, 21.67188]

plt.plot(ind,arr_1, marker='.', label='Ground State Energy')
plt.xlabel('N')
plt.ylabel('Energy')
plt.legend(loc='upper right')
plt.show()

plt.plot(ind,arr_2, marker='o', label='First Excited State')
plt.xlabel('N')
plt.ylabel('Energy')
plt.legend(loc='upper right')
plt.show()


