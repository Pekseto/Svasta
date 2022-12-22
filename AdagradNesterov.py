######## UBRZANI GRADIJENT NESTEROVA ######
import numpy as np
 
def funkcija(x):
    # f(x,y) = 1.5x^2 + y^2 - 2xy + 2x^3 + 0.5x^4
    return 1.5*x[0]**2 + x[1]**2 - 2*x[0]*x[1] + 2*x[0]**3 + 0.5*x[0]**4
 
def gradijent(x):
    x = np.array(x).reshape(np.size(x))
    return np.asarray([[3*x[0] - 2*x[1] + 6*x[0]**2 + 2*x[0]**3], [2*x[1] - 2*x[0]]])
 
def nesterov(gradf, x0, gamma, epsilon, omega, N):
    x = np.array(x0).reshape(len(x0), 1)
    v = 0
 
    for k in range(N):
        xpre = x - omega*v
        g = gradf(xpre)
        v = omega*v + gamma*g
        x = x - v
 
        if np.linalg.norm(g) < epsilon:
            break
 
    return x
 
optimum = nesterov(gradijent, x0=[2, 2], gamma=0.05, epsilon=1e-4, omega=0.5, N=100)
vrednost_funkcije = funkcija(optimum)
print("Optimum funkcije se nalazi u tacki",optimum,",vrednost funkcije u toj tacki iznosi ", vrednost_funkcije)
 

######## ADAGRAD ######
import numpy as np
 
def funkcija(x):
    # f(x,y) = 1.5x^2 + y^2 - 2xy + 2x^3 + 0.5x^4
    return 1.5*x[0]**2 + x[1]**2 - 2*x[0]*x[1] + 2*x[0]**3 + 0.5*x[0]**4
 
def gradijent(x):
    x = np.array(x).reshape(np.size(x))
    return np.asarray([[3*x[0] - 2*x[1] + 6*x[0]**2 + 2*x[0]**3], [2*x[1] - 2*x[0]]])
 
def adagrad(gradf, x0, gamma, epsilon1, epsilon, N):
    x = np.array(x0).reshape(len(x0), 1)
    v = 0
    G = 0
 
    for k in range(N):
        g = gradf(x)
        G = G + np.multiply(g, g)
        v = (gamma * g)/np.sqrt(G + epsilon1)
        x = x - v
 
        if np.linalg.norm(g) < epsilon:
            break
 
    return x
 
optimum = adagrad(gradijent, x0=[2, 2], gamma=1, epsilon1=1e-6, epsilon=1e-6, N=100)
vrednost_funkcije = funkcija(optimum)
print("Optimum funkcije se nalazi u tacki",optimum,",vrednost funkcije u toj tacki iznosi ", vrednost_funkcije)