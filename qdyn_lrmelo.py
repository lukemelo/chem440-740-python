from potentials import V,kinetic
import numpy as np
import matplotlib.pyplot as plt

# set some constants and parametersin atomic units
mass=1836.153 # proton mass (mp/me)
hbar=1. # quantum of action
omega=1. # angular frequency of teh oscillators
# number of steps
nsteps=40000
# dt value
dt=.5
# maximum x value
xmax=10.

# potential parameters specified below
## for n_{MeOD}=3
potential_parameters={'type':'luke_wells','EMax':0.1,'xLeft':-5.0,\
    'E_R':0.0,'E_TS1':0.01,'E_BR':-0.008603,'E_TS2':-0.011504+0.01,'E_P':-0.011504,\
    'w_R':2.0,'w_TS1':2.0,'w_BR':2.0,'w_TS2':2.0,'w_P':2.0}
### for n_{MeOD}=5
#potential_parameters={'type':'luke_wells','EMax':0.1,'xLeft':-5.0,\
    #'E_R':0.0,'E_TS1':0.01,'E_BR':-0.011751,'E_TS2':-0.004034+0.01,'E_P':-0.004034,\
    #'w_R':2.0,'w_TS1':2.0,'w_BR':2.0,'w_TS2':2.0,'w_P':2.0}
### for n_{MeOD}=8
#potential_parameters={'type':'luke_wells','EMax':0.1,'xLeft':-5.0,\
    #'E_R':0.0,'E_TS1':0.01,'E_BR':-0.00255,'E_TS2':0.004062+0.01,'E_P':0.004062,\
    #'w_R':2.0,'w_TS1':2.0,'w_BR':2.0,'w_TS2':2.0,'w_P':2.0}



# 99.9% Confidence Interval
nstdev = 3

# center of the initial Gaussian wavepacket
xmeanL=0.;
# width parameter of initial wavepacket at 99.9% CI
alphaL=(2.*nstdev)**2/(2.*potential_parameters['w_R']**2)

# center of the initial Gaussian wavepacket
xmeanR=0.0;
# width parameter of initial wavepacket at 99.9% CI
alphaR=(2.*nstdev)**2/(2.*potential_parameters['w_P']**2)

#<PsiL|exp(-iHt/hbar)|PsiR>


# number of grid points
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
psi_iniL=np.zeros(size,complex)
for i1,x1 in enumerate(grid): 
    psi_iniL[i1]=complex(np.exp(-alphaL*(x1-xmeanL)**2))

#normalize |psi_ini>
normL=np.dot(psi_iniL,psi_iniL)
psi_iniL=psi_iniL/np.sqrt(normL)

psi_iniR=np.zeros(size,complex)
for i1,x1 in enumerate(grid): 
    psi_iniR[i1]=complex(np.exp(-alphaR*(x1-xmeanR)**2))
#normalize |psi_ini>
normR=np.dot(psi_iniR,psi_iniR)
psi_iniR=psi_iniR/np.sqrt(normR)


#output psi_ini to a file
psi0_out=open('psi0','w')
for i1,x1 in enumerate(grid): 
    psi0_out.write(str(x1)+' '+str(psi_iniL[i1].real)+' '+str(psi_iniR[i1].real)+'\n')
psi0_out.close()


# construct survival amplitude
#<psiL|U(t)|psiR>
St=np.zeros(nsteps,complex)
psit=psi_iniR.copy()
St_out=open('St','w')
for t in range(nsteps):
    St[t]=np.dot(psi_iniL,psit)
    psit=np.dot(U_dt,psit)
    St_out.write(str(t*dt)+' '+str(St[t].real)+' '+str(St[t].imag)+'\n')
St_out.close()

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
eig_vecs_scaled = np.zeros((size,size))
for i in range(size):
    evec_out.write(str(grid[i]))	
    for n in range(size):
	eig_vecs_scaled[i,n] += eig_vecs_sorted[i,n]+eig_vals_sorted[n]
	evec_out.write(' '+str(eig_vecs_sorted[i,n]+eig_vals_sorted[n]))
    evec_out.write('\n')

# construct exact survival amplitude
StE=np.zeros(nsteps,complex)
StE_out=open('StE','w')
cnR=np.zeros(size,complex)
cnL=np.zeros(size,complex)
for n in range(size):	
    cnR[n]=np.dot(eig_vecs_sorted[:,n],psi_iniR)
    cnL[n]=np.dot(eig_vecs_sorted[:,n],psi_iniL)
for t in range(nsteps):
    for n in range(size):
	StE[t]+=(np.conj(cnR[n])*cnL[n])*np.exp(-1j*eig_vals_sorted[n]*t*dt/hbar)
    StE_out.write(str(t*dt)+' '+str(StE[t].real)+' '+str(StE[t].imag)+'\n')
StE_out.close()


#power = np.fft.fft(StE)
#plt.figure(8000)
#plt.plot(power.real)
#plt.ylabel('Intensity'),plt.xlabel('Frequency')
#plt.show()
#plt.savefig('power.png', bbox_inches='tight')
#plt.close()

### Using the DFT implementation outlined in the lecture notes:
## Initialize write out string and power spectrum array
#write_power = ''
#power = np.zeros(nsteps,float)
#freqs = np.zeros(nsteps,float)
## Define the max frequency and freqency stepsize
#wmax = (np.pi)/dt
#dw = wmax/nsteps
#for w in range(nsteps):
    ##print str(w)
    #freqs[w] = w*dw
    #for t in range(nsteps):
	## Use definition of the power spectrum from lecture 13
	#power[w] += dt/(np.pi)*(np.cos(w*dw*t*dt)*St[t].real - \
	                          #np.sin(w*dw*t*dt)*St[t].imag)
    ## Generate the next output line
    #write_power = write_power +str(w*dw)+','+str(power[w])+'\n'
## Write the power spectrum results to a csv file
#fo = open('power.csv','w')
#fo.writelines(write_power)
#fo.close()

#plt.figure(1000)
#plt.plot(freqs,power)
#plt.ylabel('Intensity'),plt.xlabel('Frequency')
##plt.axis([-10,10,-0.02,0.12])
#plt.show()
#plt.savefig('Power.png', bbox_inches='tight')
#plt.close()



plt.figure(0)
plt.plot(grid,potential)
plt.ylabel('Energy [Hartrees]'),plt.xlabel('x [Bohr Radii]')
plt.axis([-10,10,-0.02,0.12])
plt.savefig('Potential.png', bbox_inches='tight')
plt.close()

plt.figure(1)
plt.plot(grid,potential,grid,psi_iniL.real,grid,psi_iniR.real)
plt.ylabel('Energy [Hartrees]'),plt.xlabel('x [Bohr Radii]')
plt.axis([-10,10,-0.2,0.4])
plt.savefig('Potential_wfn.png', bbox_inches='tight')
plt.close()

for i in range(30):
    plt.figure(2+i)
    plt.plot(grid,eig_vecs_scaled[:,i],grid,potential)
    plt.ylabel('Energy [Hartrees]'),plt.xlabel('x [Bohr Radii]')
    plt.axis([-10,10,-0.2,0.4])
    plt.savefig('Psi_vectors'+str(i)+'.png', bbox_inches='tight')
    plt.close()
    
plt.figure(9000)
plt.plot(eig_vals_sorted[:30])
plt.ylabel('Energy [Hartrees]'),plt.xlabel('n')
plt.savefig('evalues.png', bbox_inches='tight')
plt.close()

plt.figure(9001)
plt.plot(grid,eig_vecs_scaled[:,:9])
plt.ylabel('Energy [Hartrees]'),plt.xlabel('n')
plt.savefig('evectors.png', bbox_inches='tight')
plt.close()

plt.figure(9002)
plt.plot(range(len(np.abs(StE))),St.real)
plt.ylabel('Survival Amplitude'),plt.xlabel('Time')
#plt.axis([-10,10,-0.02,0.12])
plt.savefig('Survival Amplitude.png', bbox_inches='tight')
plt.close()

plt.figure(9003)
plt.plot(range(len(np.abs(StE))),StE.real)
plt.ylabel('Survival Amplitude Exact'),plt.xlabel('Time')
#plt.axis([-10,10,-0.02,0.12])
plt.savefig('Survival Amplitude Exact.png', bbox_inches='tight')
plt.close()
