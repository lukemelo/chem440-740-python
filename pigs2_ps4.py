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
potential_parameters={'type':'harmonic','mass':mass,'omega':omega}
#potential_parameters={'type':'double_well','a':-.5,'b':.1}


# beta=1/kBT value
beta=25.
# number of beads P
P=800
# tau value
tau=beta/P
# maximum x value
xmax=10.
# number of imaginary time steps for corrlation: lamdba=nsteps*tau
nsteps=150

# numbe of grid points
size=101
# grid spacing
dx=2.*xmax/(size-1.)
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

#matrix for x
x=np.zeros((size,size),float)
for i1 in range(size): 
	x[i1,i1]=grid[i1]
	
# build C(tau)=<psi_T|exp(-beta H) x exp(tau H)xexp(-tau H)  exp(-beta H)|psi_T>
Ctau=np.zeros(nsteps,float)
xtau=x.copy()
rho_Ntau=np.zeros((size,size),float)
for i in range(size): rho_Ntau[i,i]=1.
rho_beta_m_tau=rho_beta.copy()
x_times_rho_beta=dx*np.dot(x,rho_beta)
Zpigs=0.
rho2beta=dx*np.dot(rho_beta,rho_beta)
for i in range(size):
	for ip in range(size):
		Zpigs+=dx*dx*rho2beta[i,ip]
		
# ground state estimation
E0=0.
for i in range(size):
	for ip in range(size):
		E0+=dx*dx*rho2beta[i,ip]*potential[ip]
		
print 'E0 (pigs) = ',E0/Zpigs
		
Ctau_out=open('Clambda','w')
for j in range(nsteps):
	xtau=dx*np.dot(x,rho_Ntau)
	a=dx*np.dot(rho_beta_m_tau,dx*np.dot(xtau,x_times_rho_beta))
	rho_beta_m_tau=rho_tau.copy()	
	for jp in range(P-j-1):
		rho_beta_m_tau=dx*np.dot(rho_beta_m_tau,rho_tau)

	rho_Ntau=dx*np.dot(rho_Ntau,rho_tau)
	for i in range(size):
		for ip in range(size):
			Ctau[j]+=dx*dx*a[i,ip]/Zpigs
		
	Ctau_out.write(str(j*tau)+' '+str(Ctau[j])+'\n')	
	
# calculate partition function from the trace of rho
Z=0.
Z_tau=0.
V_estim=0.
for i in range(size):
	Z+=rho_beta[i,i]*dx
	V_estim+=rho_beta[i,i]*dx*potential[i]

rho_beta_out=open('rho_beta','w')
rho_beta_off_out=open('rho_beta_off','w')
for i1,x1 in enumerate(grid): 
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

evalues0_out=open('evalues0','w')
evalues_out=open('evalues','w')
for i in range(size):
	evalues0_out.write(str(eig_vals_sorted[i]-eig_vals_sorted[0])+'\n')
	evalues_out.write(str(eig_vals_sorted[i])+'\n')
