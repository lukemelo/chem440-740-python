import numpy as np

def V(x1,potential_parameters):
	if potential_parameters['type'] == 'luke_wells':
		
		EMax=potential_parameters['EMax']
		xLeft=potential_parameters['xLeft']
		
		E_R=potential_parameters['E_R']
		E_TS1=potential_parameters['E_TS1']
		E_BR=potential_parameters['E_BR']
		E_TS2=potential_parameters['E_TS2']
		E_P=potential_parameters['E_P']
		
		w_R=potential_parameters['w_R']
		w_TS1=potential_parameters['w_TS1']
		w_BR=potential_parameters['w_BR']
		w_TS2=potential_parameters['w_TS2']
		w_P=potential_parameters['w_P']
		
		if x1<xLeft:
			value = EMax
		elif x1<(xLeft+ w_R):
			value = E_R
		elif x1<(xLeft+ w_R+ w_TS1):
			value = E_TS1
		elif x1<(xLeft+ w_R+ w_TS1+ w_BR):
			value = E_BR
		elif x1<(xLeft+ w_R+ w_TS1+ w_BR+ w_TS2):
			value = E_TS2
		elif x1<(xLeft+ w_R+ w_TS1+ w_BR+ w_TS2+ w_P):
			value = E_P
		else:
			value = EMax	
	return value

def kinetic(size,mass,dx):
	T=np.zeros((size,size),float)
	h_bar=1. 
	for i in range(size):	
		for ip in range(size):	
			T1 = h_bar*h_bar / (2.0*mass*dx*dx) * np.power(-1.,i-ip);
			if i==ip:
				T[i,ip] = T1 * (np.pi*np.pi/3.0);
			else:
				T[i,ip] = T1 * 2.0/((i-ip)*(i-ip))
	return T
