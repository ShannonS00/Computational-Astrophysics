"""
2D Finite Volume Hydrodynamics routines used for the numerics lab for astrophysicists at Vienna university
Written 2022 by Oliver Hahn
"""

import numpy as np
    

def primitive_to_conserved( Q, gamma ):
    """
    Return an array of conserved hydro variables given an array of primitive variables
    @param Q : primitive variables (NxNx4: rho, u, v, P)
    @return U: conserved variables (NxNx4: rho, rhou, rhov, E)
    """
    U = np.zeros_like( Q )
    U[...,0] = Q[...,0]
    U[...,1] = Q[...,0] * Q[...,1]
    U[...,2] = Q[...,0] * Q[...,2]
    U[...,3] = Q[...,3]/(gamma-1) + 0.5 * Q[...,0] * (Q[...,1]**2 + Q[...,2]**2)
    return U

def conserved_to_primitive( U, gamma ):
    """
    Return an array of primitive hydro variables given an array of conserved variables
    @param U : conserved variables (NxNx4: rho, rhou, rhov, E)
    @return Q: primitive variables (NxNx4: rho, u, v, P)
    """
    Q = np.zeros_like( U )
    Q[...,0] = U[...,0]
    Q[...,1] = U[...,1] / U[...,0]
    Q[...,2] = U[...,2] / U[...,0]
    Q[...,3] = (gamma-1) * (U[...,3] - 0.5 * (U[...,1]**2 + U[...,2]**2)/U[...,0])
    return Q
    
def get_timestep_from_U( U, *, dx, gamma, unsplit ):
    """
    Return the maximally allowed time step given the conserved hydro variables
    @param U  : conserved variables (NxNx4: rho, rhou, rhov, E)
    @param dx : grid spacing
    @param gamma : adiabatic exponent
    @param unsplit : flag whether unsplit (=True) or dimensionally split (=False)
    @ return dtmax : maximum CFL time step
    """
    vx = U[...,1] / U[...,0]
    vy = U[...,2] / U[...,0]
    
    cs = np.sqrt( gamma*(gamma-1)*(U[...,3]/U[...,0]-0.5*(vx**2 + vy**2) ) ) 
    cs[np.isnan(cs)] = 0.0
    
    if not unsplit:
        Smax = np.max( (np.max(np.abs(vx.flatten())+cs.flatten()),np.max(np.abs(vy.flatten())+cs.flatten())))
    else:
        Smax = np.max( np.abs(vx.flatten()) + np.abs(vy.flatten()) + 2 * cs.flatten()  )
        
    return dx / Smax
    

def get_limited_slope( U, *, dim ):
    """
    Return a limited slope between variables along a given dimension
    @param U  : variables (conserved or primitive) 
    @param dim: dimension along which to compute limited gradient
    @return dU: limited spatial derivative along dimension dim
    """
    dUl = U - np.roll(U,+1,axis=dim)
    dUr = np.roll(U,-1,axis=dim) - U
    dUc = (np.roll(U,-1,axis=dim) - np.roll(U,+1,axis=dim))/2

    # apply van Leer limiter
    dU = np.zeros_like(dUc)
    i  = np.logical_and((dUl+dUr)!=0,np.sign(dUl)==np.sign(dUr))
    dU[i] = 2*dUl[i]*dUr[i]/(dUl[i]+dUr[i])
    return dU

def predict_states( U, *, dx, dt, dim, gamma ):
    """
    Returns the conserved variables UL and UR, 
    where UR = U_{R,i-1/2}^{n+1/2}, UL = U_{L,i+1/2}^{n+1/2}
    
    @param U     : the vector of conserved variables (NxNx4)
    @param dx    : the grid spacing in the relevant dimension (scalar)
    @param dt    : the full timestep 
    @param dim   : the dimension currently active (0 or 1)
    @param gamma : the adiabatic exponent
    @return (UL,UR,v,cs) : tuple with the state extrapolated to 
      left and right and half time step, as well as v_dim and c_s
    """
    Q  = conserved_to_primitive( U, gamma )
    dQ = get_limited_slope( Q, dim=dim)
    # primitive equation is dQ/dt + A(Q).dQ/dx = 0
    # compute product of A with dQ/dx_dim
    AdQ = np.zeros_like(Q)
    AdQ[...,0] = Q[...,1+dim] * dQ[...,0] + Q[...,0] * dQ[...,1+dim]
    AdQ[...,1] = Q[...,1+dim] * dQ[...,1]
    AdQ[...,2] = Q[...,1+dim] * dQ[...,2]
    AdQ[...,1+dim] += 1/Q[...,0] * dQ[...,3]
    AdQ[...,3] = Q[...,1+dim] * dQ[...,3] + gamma*Q[...,3]*dQ[...,1+dim]
    # predict state extrapolated in space and time given limited slopes
    QR = Q - 0.5 * dQ - 0.5 * dt/dx * AdQ
    QL = Q + 0.5 * dQ - 0.5 * dt/dx * AdQ
    # convert to conserved
    UR = primitive_to_conserved( QR, gamma )
    UL = primitive_to_conserved( QL, gamma )
    # compute some additional quantities
    csL = np.sqrt( gamma * QL[...,3]/QL[...,0])
    csR = np.sqrt( gamma * QR[...,3]/QR[...,0])
    cs  = np.sqrt( gamma * Q[...,3]/Q[...,0])
    SL = np.abs(QL[...,1+dim])+csL
    SR = np.abs(QR[...,1+dim])+csR
    S  = np.abs(Q[...,1+dim])+cs
    # return UR = U_{R,i-1/2}^{n+1/2}, UL = U_{L,i+1/2}^{n+1/2}
    return UL,UR,SL,SR


def predict_states_unsplit( U, *, dx, dt, dim, gamma ):
    """
    Returns the conserved variables UL and UR, 
    where UR = U_{R,i-1/2}^{n+1/2}, UL = U_{L,i+1/2}^{n+1/2}
    
    @param U     : the vector of conserved variables (NxNx4)
    @param dx    : the grid spacing in the relevant dimension (scalar)
    @param dt    : the full timestep 
    @param dim   : the dimension currently active (0 or 1)
    @param gamma : the adiabatic exponent
    @return (UL,UR,v,cs) : tuple with the state extrapolated to 
      left and right and half time step, as well as v_dim and c_s
    """
    Q  = conserved_to_primitive( U, gamma )
    # get limited slope in normal direction
    dQ = get_limited_slope( Q, dim=dim)
    # get limited slope in transversal direction
    dimt = (dim+1)%2
    dQt = get_limited_slope( Q, dim=dimt)
    # primitive equation is 
    #    dQ/dt + A(Q).dQ/dx_normal + B(Q).dQ/dx_trans = 0
    # normal part: compute product of A with dQ/dx_dim = dQ/dx_normal
    AdQ = np.zeros_like(Q)
    AdQ[...,0] = Q[...,1+dim] * dQ[...,0] + Q[...,0] * dQ[...,1+dim]
    AdQ[...,1] = Q[...,1+dim] * dQ[...,1]
    AdQ[...,2] = Q[...,1+dim] * dQ[...,2]
    AdQ[...,1+dim] += 1/Q[...,0] * dQ[...,3]
    AdQ[...,3] = Q[...,1+dim] * dQ[...,3] + gamma*Q[...,3]*dQ[...,1+dim]
    # transversal part: compute product of B with dQ/dx_trans
    BdQt = np.zeros_like(Q)
    BdQt[...,0] = Q[...,1+dimt] * dQt[...,0] + Q[...,0] * dQt[...,1+dimt]
    BdQt[...,1] = Q[...,1+dimt] * dQt[...,1]
    BdQt[...,2] = Q[...,1+dimt] * dQt[...,2]
    BdQt[...,1+dimt] += 1/Q[...,0] * dQt[...,3]
    BdQt[...,3] = Q[...,1+dimt] * dQt[...,3] + gamma*Q[...,3]*dQt[...,1+dimt]
    # predict state extrapolated in space and time given limited slopes
    QR = Q - 0.5 * dQ - 0.5 * dt/dx * AdQ - 0.5 * dt/dx * BdQt
    QL = Q + 0.5 * dQ - 0.5 * dt/dx * AdQ - 0.5 * dt/dx * BdQt
    # convert to conserved
    UR = primitive_to_conserved( QR, gamma )
    UL = primitive_to_conserved( QL, gamma )
    # compute some additional quantities
    csL = np.sqrt( gamma * QL[...,3]/QL[...,0])
    csR = np.sqrt( gamma * QR[...,3]/QR[...,0])
    cs  = np.sqrt( gamma * Q[...,3]/Q[...,0])
    SL = np.abs(QL[...,1+dim])+csL
    SR = np.abs(QR[...,1+dim])+csR
    S  = np.abs(Q[...,1+dim])+cs
    # return UR = U_{R,i-1/2}^{n+1/2}, UL = U_{L,i+1/2}^{n+1/2}
    return UL,UR,S,S


def get_flux_from_U( U, *, dim, gamma ):
    """
    Return the value of the flux function F(U) given conserved hydro variables
    @param U    : field of conserved hydro variables (NxNx4)
    @param dim  : dimension along which to compute flux (0 or 1)
    @param gamma: the adiabatic exponent
    @return F(U): the flux along dimension dim
    """
    F = np.zeros_like(U)
    P = (gamma-1) * ( U[...,3] - 0.5 * (U[...,1]**2+U[...,2]**2) / U[...,0] )
    v = U[...,1+dim] / U[...,0]
    F[...,0] = U[...,0] * v
    F[...,1] = U[...,1] * v
    F[...,2] = U[...,2] * v
    F[...,1+dim] += P 
    F[...,3] = (U[...,3]+P) * v
    return F


def get_conservative_update( U, *, dx, dt, dim, gamma, unsplit ):
    """
    Compute an update to the conservative variables for a given dimension
    
    @param dx    : the grid spacing in the relevant dimension (scalar)
    @param dt    : the full timestep 
    @param dim   : the dimension currently active (0 or 1)
    @param gamma : the adiabatic exponent
    @return dU   : the update, to be multiplied by dt and added to U
    """

    ## TODO: apply boundary conditions to U?
    
    # get predicted states at boundaries 
    #    Uleft = U_{L,i+1/2} and Uright = U_{R,i-1/2}
    if not unsplit:
        Uleft,Uright,Sleft,Sright = predict_states( U, dx=dx, dt=dt, dim=dim, gamma=gamma )
    else:
        Uleft,Uright,Sleft,Sright = predict_states_unsplit( U, dx=dx, dt=dt, dim=dim, gamma=gamma )

    # shift right state so that we have the states for the same boundary
    #    i.e. Uleft = U_{L,i+1/2} and Uright = U_{r,i+1/2}
    Uright   = np.roll(Uright, -1,axis=dim)

    # compute left and right flux from left and right states at boundary i+1/2
    Fleft   = get_flux_from_U( Uleft,  dim=dim, gamma=gamma )
    Fright  = get_flux_from_U( Uright, dim=dim, gamma=gamma )
    
    # compute signal speeds, we have Sleft, need to shift Sright to have at i+1/2
    Sright = np.roll(Sright, -1, axis=dim)
    # get maximum signal speed
    Sstar  = np.max((Sleft,Sright))
    
    # compute HLL interface flux at i+1/2
    FHLL = np.zeros_like(U)
    for i in range(4):
        FHLL[...,i] = 0.5*(Fleft[...,i]+Fright[...,i]) - 0.5 * Sstar * (Uright[...,i]-Uleft[...,i])
    
    # compute net flux in and out of cell due to left and right boundary
    # as difference between flux at i+1/2 and i-1/2
    dU = -( FHLL - np.roll(FHLL,+1,axis=dim)) / dx

    ## TODO: apply boundary conditions to dU?
    
    return dU


def hydro_step_split( U, *, t, tmax, step, CFLfac, dx, gamma ):
    """
    Compute a hydro time step using the dimensionally split scheme
    @param U : conserved variables (NxNx4)
    @param t : current time
    @param tmax : maximal time (ends there if time step would be larger)
    @param step : curent time step number
    @param CFLfac : CFL safety factor
    @param dx : grid spacing
    @param gamma : adiabatic exponent
    @return (U, t, step) : updated conserved variables, time, and time step number
    """
    # compute time step as minimum along either dimension
    dt = get_timestep_from_U( U, dx=dx, gamma=gamma, unsplit=False )
    dt = np.min((CFLfac*dt,tmax-t))
    
    # alternate which dimension to treat first, 
    # gives even better results than just Strang split
    dim1 = step%2
    dim2 = (dim1+1)%2

    # do strang operator-split update
    dU = get_conservative_update( U, dx=dx, dt=dt/2, dim=dim1, gamma=gamma, unsplit=False )
    U = U + dt/2 * dU
    dU = get_conservative_update( U, dx=dx, dt=dt, dim=dim2, gamma=gamma, unsplit=False )
    U = U + dt * dU
    dU = get_conservative_update( U, dx=dx, dt=dt/2, dim=dim1, gamma=gamma, unsplit=False )
    U = U + dt/2 * dU

    t += dt
    step += 1    
    
    return U, t, step


def hydro_step_unsplit( U, *, t, tmax, step, CFLfac, dx, gamma ):
    """
    Compute a hydro time step using the dimensionally split scheme
    @param U : conserved variables (NxNx4)
    @param t : current time
    @param tmax : maximal time (ends there if time step would be larger)
    @param step : curent time step number
    @param CFLfac : CFL safety factor
    @param dx : grid spacing
    @param gamma : adiabatic exponent
    @return (U, t, step) : updated conserved variables, time, and time step number
    """
    # compute time step as minimum along either dimension
    dt = get_timestep_from_U( U, dx=dx, gamma=gamma, unsplit=True )
    dt = np.min((CFLfac*dt,tmax-t))
    
    # do unsplit update
    dU  = get_conservative_update( U, dx=dx, dt=dt, dim=0, gamma=gamma, unsplit=True )
    dU += get_conservative_update( U, dx=dx, dt=dt, dim=1, gamma=gamma, unsplit=True )
    
    U = U + dt * dU
    
    t += dt
    step += 1    
    
    return U, t, step