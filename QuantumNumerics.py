import matplotlib.pyplot as plt 
import numpy as np 
import numpy.linalg as npla
from scipy.sparse import diags


def Laplacian_Mat(size, dx, cyclic = True):
	#A discretized laplacian for a cyclic 1d lattice of length 'size'
	#lattice spacing dx
	if cyclic:
		vals = [1/(dx**2), 1/(dx**2), -2/(dx**2), 1/(dx**2), 1/(dx**2)]
		return diags(vals, [-size//2 -1, -1, 0, 1, size//2+1], shape=(size, size))
	else:
		vals = [ 1/(dx**2), -2/(dx**2), 1/(dx**2)]
		return diags(vals, [ -1, 0, 1], shape=(size, size))

def Potential_Mat(V):
	#V is the vector of potential values
	
	return diags([V], [0])

def Potential_Vectorizer(func, dx, minx, maxx):
	#takes a funtion func, and turns it into a vector according to dx, minx and maxx

	return func(np.range(minx, maxx, dx))

def Hamiltonian_Mat(dx, V, cyclic = True, m=1, hbar = 1):
	#returns hamiltonian matrix

	size = len(V)

	kinetic_term = (-hbar**2 /(2*m))*Laplacian_Mat(size, dx, cyclic)

	potential_term = Potential_Mat(V)

	return kinetic_term+potential_term

"""
Example 1: quantum harmonic oscillator
"""
run_example_1 = True

if run_example_1:
	dx = 0.1
	maxx = 100
	minx = -100
	grid = np.arange(minx, maxx, dx)
	V = lambda x: x**2

	vs = V(grid)

	ham = Hamiltonian_Mat(dx, vs) 

	ws, vecs = npla.eigh(ham.todense())

	plt.title("Wavefunctions Shifted by Energy")
	plt.ylabel("Energy (units of hbar omega)")
	plt.plot(grid, vs, color = (0,0,0), linewidth = 4)
	for i, v in enumerate(ws[:10]):
		
		plt.plot(grid, vecs[:,i]+v)

	plt.xlim(-5,5)
	plt.ylim(0, 5)
	
	plt.show()

"""
Play around with this
Try plotting the eigenvalues of a periodic potential (a dirac comb for example)
you could write something like


plt.title("Dirac Comb Bands")
for e in eigenvals:
	plt.axhline(y=e)
plt.show()

You could also plot the lowest n eigenvalues and change the 'interatomic distance' 
(varying a for a dirac comb, for instance) and see what kind of plot you create

Have fun!
"""