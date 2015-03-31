from potentials import V,kinetic
import numpy as np

# problem set 2
# Numerical matrix multiplication
# V=1/2 mw^2
# V2=b x^4
# V3=a x^2 + b x^4

#potential_flag='quartic'
#potential_flag='double_well'

# set some constants in atomic units
mass=1
hbar=1.
omega=1.


#for a harmonic oscillator, uncomment the line below and comment out the next. For a double well, comment the line below and uncomment the next
#potential_parameters={'type':'harmonic','mass':mass,'omega':omega}
potential_parameters={'type':'double_well','a':-.5,'b':.1}


# beta=1/kBT value
beta=5.
# number of beads P
P=64

# tau value
tau=beta/P
# maximum x value
xmax=10.

# number of grid points
size=101
# grid spacing
dx=2.*xmax/(size-1.)

print 'delta-x: ' + str(dx)
print 'x-min: ' + str(-xmax)
print 'P: ' + str(P)
print 'M: ' + str(size)

# create a vector to store the grid points
grid=np.zeros(size,float)
# populate the grid
for i in range(size):
     grid[i]=-xmax+i*dx
# allocate memory for the free particle density matrix
rho_free=np.zeros((size,size),float)
for i1,x1 in enumerate(grid): 
	for i2,x2 in enumerate(grid):
		rho_free[i1,i2]=np.sqrt(mass/(2.*np.pi*tau*hbar*hbar))*(np.exp(-(mass/(2.*hbar*hbar*tau))*(x1-x2)*(x1-x2)))
# allocate memory for the potenital density matrix (note the tau/2). The matrix diagonal
rho_potential=np.zeros((size),float)
potential=np.zeros((size),float)
for i1,x1 in enumerate(grid): 
	potential[i1]=V(x1,potential_parameters)
	rho_potential[i1]=np.exp(-(tau/2.)*potential[i1])

#output potential to a file
potential_out=open('V','w')
for i1,x1 in enumerate(grid): 
	potential_out.write(str(x1)+' '+str(potential[i1])+'\n')
potential_out.close()

# construct the high temperature density matrix
rho_tau=np.zeros((size,size),float)
for i1 in range(size): 
	for i2 in range(size): 
		rho_tau[i1,i2]=rho_potential[i1]*rho_free[i1,i2]*rho_potential[i2]

# form the density matrix via matrix multiplication
#set initial value of rho
#rho_beta=np.zeros((size,size),float)

rho_beta=rho_tau.copy()

for k in range(P-1):
	rho_beta=dx*np.dot(rho_beta,rho_tau)	

	

# Question: how could write the above loop more efficiently?

# Matrix products can be multiplied in O(logN) runtime instead of O(N)
#test = dx**(P-1)	
#for k in range(int(np.log2(P))):
     #rho_beta = np.dot(rho_beta,rho_beta)
#rho_beta = test*rho_beta



# calculate partition function from the trace of rho
Z=0.
Z_tau=0.
V_estim=0.
for i in range(size):
	Z_tau+=rho_tau[i,i]*dx
	Z+=rho_beta[i,i]*dx
	V_estim+=rho_beta[i,i]*dx*potential[i]
print 'Z(beta=',beta,',tau=',tau,')= ',Z
#print 'Z(analytical;harmonic)= ',1./(2.*np.sinh(beta*hbar*omega/2.))
print 'V(beta=',beta,',tau=',tau,')= ',V_estim/Z


print 'A(beta=',beta,',tau=',tau,')= ',-np.log(Z)/beta
#print 'A(analytical;harmonic)= ',-np.log(1./(2.*np.sinh(beta*hbar*omega/2.)))/beta


rho_V_out=open('rho_V','w')
rho_free_out=open('rho_free','w')
rho_tau_out=open('rho_tau','w')
rho_beta_out=open('rho_beta','w')
rho_beta_off_out=open('rho_beta_off','w')
for i1,x1 in enumerate(grid): 
	rho_V_out.write(str(x1)+' '+str(rho_potential[i1])+'\n')
	rho_tau_out.write(str(x1)+' '+str(rho_tau[i1,i1]/Z_tau)+'\n') # off diagonal
	rho_free_out.write(str(x1)+' '+str(rho_free[i1,size/2])+'\n')
	rho_beta_out.write(str(x1)+' '+str(rho_beta[i1,i1]/Z)+'\n')
	rho_beta_off_out.write(str(x1)+' '+str(rho_beta[i1,size/2]/Z)+'\n')


# calculate E,K,V from rho

# calculate properties from sum over states

# compare Z Trotter, Analytical, truncated sum over states
# Hamiltonian matrix
H=kinetic(size,mass,dx)
for i in range(size):	
	    H[i,i]+=potential[i]
eig_vals,eig_vecs=np.linalg.eig(H)

#eig_vals_sorted = np.sort(eig_vals)
#eig_vecs_sorted = eig_vecs[eig_vals.argsort()]
ix = np.argsort(eig_vals)
eig_vals_sorted = eig_vals[ix]
eig_vecs_sorted = eig_vecs[:,ix]

evalues_out=open('evalues','w')
for i in range(size):
	evalues_out.write(str(eig_vals_sorted[i])+'\n')

# sum over states
Z_sos=0.
for e in eig_vals_sorted:
	Z_sos+=np.exp(-beta*e)
print 'Z_sos= ',Z_sos
# average energy
E_sos=0.
for e in eig_vals_sorted:
	E_sos+=np.exp(-beta*e)*e
print '<E>_sos= ',E_sos/Z_sos
# average potential; requires matrix elements
V_sos=0.

V_elements=np.zeros(size,float)
# loop over eigenstates
for n in range(size):
	#loop over grid points
	for i in range(size):
		V_elements[n]+=potential[i]*eig_vecs_sorted[i,n]**2 

for n in range(size):
	V_sos+=np.exp(-beta*eig_vals_sorted[n])*V_elements[n]
print '<V>_sos= ',V_sos/Z_sos

print 'A_sos(beta=',beta,')= ',-np.log(Z_sos)/beta

rho_sos=np.zeros(size,float)
#loop over grid points
for i in range(size):
	for n in range(size):
		rho_sos[i]+=np.exp(-beta*eig_vals_sorted[n])*eig_vecs_sorted[i,n]**2 
rho_sos_out=open('rho_sos','w')
for i1,x1 in enumerate(grid): 
	rho_sos_out.write(str(x1)+' '+str(rho_sos[i1]/Z_sos/dx)+'\n')

