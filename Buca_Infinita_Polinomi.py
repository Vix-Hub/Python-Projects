import numpy as np
from scipy.linalg import eigh
from matplotlib import pyplot as plt

N = 10
H = np.zeros(shape=(N,N))
S = np.zeros(shape=(N,N))

for n in range(N):
    for m in range(N):
        if (m+n)%2 == 0:
            S[m][n]= 2/(n+m+5)-4/(n+m+3)+2/(n+m+1)
            H[m][n]=-8*((1-m-n-2*m*n)/((m+n+3)*(m+n+1)*(m+n-1)))

eigvals, eigvecs = eigh(H, S, eigvals_only=False)

def eig(k,x): 
    S=0
    for i in range(N):
        p=(x**i)*(x+1)*(x-1)
        S+= (eigvecs[i][k])*p
    return S


X = np.linspace(-1, 1, 200)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Autofunzioni della buca di potenziale infinita')
plt.plot(X,-eig(0,X), label="n=1")
plt.plot(X,-eig(1,X), label="n=2")
plt.plot(X,-eig(2,X), label="n=3")
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig('Ordine 1,2,3', dpi=300)
plt.show()

plt.xlabel('x')
plt.ylabel('y')
plt.title('Autofunzioni della buca di potenziale infinita')
plt.plot(X,-eig(3,X), label="n=4")
plt.plot(X,-eig(4,X), label="n=5")
plt.plot(X,-eig(5,X), label="n=6")
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig('Ordine 4,5,6', dpi=300)
plt.show()