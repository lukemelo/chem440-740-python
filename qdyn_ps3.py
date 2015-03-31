from potentials import V,kinetic
import numpy as np

# problem set 2
# Numerical matrix multiplication
# V=1/2 mw^2
# V2=b x^4
# V3=a x^2 + b x^4

#potential_flag='quartic'
#potential_flag='double_well'

# set some constants and parametersin atomic units
mass=1 # particle mass
hbar=1. # quantum of action
omega=1. # angular frequency of teh oscillators
# number of steps
nsteps=2000
# dt value
dt=0.1
# maximum x value
xmax=20.
# center of the initial Gaussian wavepacket
xmean=2.;
# width parameter of initial wavepacket
alpha=1.
window_bool = True
al = 0.00001

#for a harmonic oscillator, uncomment the line below and comment out the next. For a double well, comment the line below and uncomment the next
#potential_parameters={'type':'harmonic','mass':mass,'omega':omega}
potential_parameters={'type':'double_well','a':-.5,'b':.1}



# numbe of grid points
size=400
# grid spacing
dx=2.*xmax/(size-1.)
# create a vector to store the grid points
grid=np.zeros(size,float)

# populate the grid
for i in range(size):
     grid[i]=-xmax+i*dx
# allocate memory for the free particle propagator
U_free=np.zeros((size,size),complex)
U_free_diag=np.zeros((size,size),complex)
# kinetic energy matrix in th Discret variable Representation
K=kinetic(size,mass,dx)
Keig_vals,Keig_vecs=np.linalg.eig(K)
for i,kval in enumerate(Keig_vals):
	U_free_diag[i,i]=np.exp(-1j*kval*dt/hbar)

# transform from momentum back to DVR
U_free=np.dot(Keig_vecs,np.dot(U_free_diag,np.transpose(Keig_vecs)))

# allocate memory for the potenital density matrix (note the tau/2). The matrix diagonal
U_potential=np.zeros((size),complex)
potential=np.zeros((size),float)
for i1,x1 in enumerate(grid): 
	potential[i1]=V(x1,potential_parameters)
	U_potential[i1]=np.exp(-1j*(dt/2.*hbar)*potential[i1])

#output potential to a file
potential_out=open('V','w')
for i1,x1 in enumerate(grid): 
	potential_out.write(str(x1)+' '+str(potential[i1])+'\n')
potential_out.close()

# construct the propagator
U_dt=np.zeros((size,size),complex)
for i1 in range(size): 
	for i2 in range(size): 
		U_dt[i1,i2]=U_potential[i1]*U_free[i1,i2]*U_potential[i2]

# setup |psi_ini> as a normalized Gaussian
psi_ini=np.zeros(size,complex)
for i1,x1 in enumerate(grid): 
	psi_ini[i1]=complex(np.exp(-alpha*(x1-xmean)**2))
#normalize |psi_ini>
norm=np.dot(psi_ini,psi_ini)
psi_ini=psi_ini/np.sqrt(norm)

#output psi_ini to a file
psi0_out=open('psi0','w')
for i1,x1 in enumerate(grid): 
	psi0_out.write(str(x1)+' '+str(psi_ini[i1].real)+'\n')
psi0_out.close()


# construct survival amplitude
St=np.zeros(nsteps,complex)
psit=psi_ini.copy()
St_out=open('St','w')
for t in range(nsteps):
	St[t]=np.dot(psi_ini,psit)
	psit=np.dot(U_dt,psit)
	St_out.write(str(t*dt)+' '+str(St[t].real)+' '+str(St[t].imag)+'\n')
St_out.close()


if window_bool:
     for t in range(nsteps):
	  window = np.exp(-al*t**2)
	  St[t] = St[t]*window

## Using the DFT implementation outlined in the lecture notes:
# Initialize write out string and power spectrum array
write_power = ''
power = np.zeros(nsteps,float)
# Define the max frequency and freqency stepsize
wmax = (np.pi)/dt
dw = wmax/nsteps
for w in range(nsteps):
     for t in range(nsteps):
	  # Use definition of the power spectrum from lecture 13
	  power[w] += dt/(np.pi)*(np.cos(w*dw*t*dt)*St[t].real - \
	                          np.sin(w*dw*t*dt)*St[t].imag)
     # Generate the next output line
     write_power = write_power +str(w*dw)+','+str(power[w])+'\n'
# Write the power spectrum results to a csv file
fo = open('power.csv','w')
fo.writelines(write_power)
fo.close()

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
	evalues_out.write(str(eig_vals_sorted[i])+' 1'+'\n')
	
evec_out=open('evectors','w')
for i in range(size):
	evec_out.write(str(grid[i])+' '+str(eig_vecs_sorted[i,0])+' '+str(eig_vecs_sorted[i,1])+' '+str(eig_vecs_sorted[i,2])+' '+str(eig_vecs_sorted[i,3])+'\n')

# construct exact survival amplitude
StE=np.zeros(nsteps,complex)
StE_out=open('StE','w')
cn=np.zeros(size,complex)
for n in range(size):	
	cn[n]=np.dot(eig_vecs_sorted[:,n],psi_ini)
for t in range(nsteps):
	for n in range(size):
		StE[t]+=abs(cn[n])**2*np.exp(-1j*eig_vals_sorted[n]*t*dt/hbar)
	StE_out.write(str(t*dt)+' '+str(StE[t].real)+' '+str(StE[t].imag)+'\n')
StE_out.close()


