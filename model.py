import math
import warnings
from random import sample
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import networkx as nx
import matplotlib.pyplot as plt

def contact_precaution_remove(Rs, precautions, phi_ppe):
    
    T = len(Rs)
    new_Rs = []
    N = Rs[0].shape[0]

    for t in range(T):

        new_R = Rs[t].copy()
        idx = precautions[t]

        scale = np.ones(N)
        #col_scale = np.ones(N)
        
        scale[idx] = 0
        #col_scale[patients] = phi

        D = sp.spdiags(scale, 0, N, N)
        
        diag = new_R.diagonal()
        new_R = D * new_R * D

        newdiag = np.zeros(N)
        newdiag[idx] = diag[idx] * phi_ppe
        new_R = new_R + sp.spdiags(newdiag, 0, N, N)

        new_Rs.append(new_R)
    
    return new_Rs

def transfer_matrix(As, P, H, L, phi_p, phi_h, phi_l, taus):

    Rs = []
    N = P+H+L
    
    pat_idx = np.arange(P)
    hcw_idx = np.arange(P, P+H)
    loc_idx = np.arange(P+H, N)

    row_scale = np.ones(N)
    row_scale[:P] = phi_p
    row_scale[P:P+H] = phi_h
    row_scale[P+H:] = phi_l
    D = sp.spdiags(row_scale, 0, N, N)
    
    for A in As:

        R = fill_in_taus(A, P, H, L, taus)
            
        # make column-stochastic       
        diagterms = 1 - np.squeeze(np.asarray(R.sum(axis=0)))  
        R = R + sp.dia_matrix((diagterms,0), shape=(N,N))
        
        if np.any(diagterms < 0):
            print(np.where(diagterms < 0)[0])
            print(diagterms[diagterms < 0])
            warnings.warn("Negative transfer rate in R", RuntimeWarning)
        
        R = D * R

        Rs.append(R)
            
    return Rs

def fill_in_taus(A, P, H, L, taus):
    
    tau_p2p, tau_p2h, tau_p2l, tau_h2p, tau_h2h, tau_h2l, tau_l2p, tau_l2h = taus
    
    '''
    R = A.copy()
    P_idx = np.arange(P)
    H_idx = np.arange(P, P+H)
    L_idx = np.arange(P+H, P+H+L)

    R[P_idx[:,None], P_idx] *= tau_p2p
    R[P_idx[:,None], H_idx] *= tau_p2h
    R[P_idx[:,None], L_idx] *= tau_p2l

    R[H_idx[:,None], P_idx] *= tau_h2p
    R[H_idx[:,None], H_idx] *= tau_h2h
    R[H_idx[:,None], L_idx] *= tau_h2l

    R[L_idx[:,None], P_idx] *= tau_l2p
    R[L_idx[:,None], H_idx] *= tau_l2h
    R[L_idx[:,None], L_idx] *= 0

    return R
    '''

    R = A.tocoo()

    R.setdiag(0)
    R.sum_duplicates()
    R.eliminate_zeros()
    for idx, i, j in zip(range(R.nnz), R.row, R.col):
        if i < P and j < P:
            R.data[idx] *= tau_p2p
        elif i < P and j < P+H:
            R.data[idx] *= tau_h2p
        elif i < P and j < P+H+L:
            R.data[idx] *= tau_l2p
        elif i < P+H and j < P:
            R.data[idx] *= tau_p2h
        elif i < P+H and j < P+H:
            R.data[idx] *= tau_h2h
        elif i < P+H and j < P+H+L:
            R.data[idx] *= tau_l2h
        elif i < P+H+L and j < P:
            R.data[idx] *= tau_p2l
        elif i < P+H+L and j < P+H:
            R.data[idx] *= tau_h2l
        else:
            R.data[idx] *= 0

    return R.tocsr()
            

def simulate(Rs, P, H, L, seeds, init_loads, alpha, beta, delta, T=1):
    

    states = np.zeros(P)
    loads = np.zeros(P+H+L)
    loads[:P] = init_loads
    states[seeds] = 1

    counts = []
    loads_patients = []
    loads_other = []
    cum_new_cases = []
    inc_new_cases = []
    ever_infected = set(seeds)

    counts.append(np.sum(states))
    loads_patients.append(np.sum(loads[:P]))
    loads_other.append(np.sum(loads[P:]))
    cum_new_cases.append(0)
    inc_new_cases.append(0)   

    for _ in range(T):
        for t in range(len(Rs)):

            #print(loads)

            rands = np.random.rand(P)

            if np.any(beta*loads[:P] > 1):
                warnings.warn("Beta too large!", RuntimeWarning)

            s2i = np.where((rands < beta*loads[:P]) & (states == 0))[0]
            i2s = np.where((rands < delta) & (states == 1))[0]

            if s2i.size != 0:
                states[s2i] = 1
                ever_infected.update(s2i)
            if i2s.size != 0:
                states[i2s] = 0


            loads = (Rs[t].dot(loads))
            loads[:P] += alpha*states

            #print(loads)

            counts.append(np.sum(states))
            loads_patients.append(np.sum(loads[:P]))
            loads_other.append(np.sum(loads[P:]))
            cum_new_cases.append(len(ever_infected) - len(seeds))
            inc_new_cases.append(s2i.size)    

    return counts, cum_new_cases, inc_new_cases, loads_patients, loads_other


def system_matrix(Rs, P, H, L, alpha, beta, delta):
    
    N = 2*P+H+L
    Ss = []

    for R in Rs:

        # Dense formulation
        #Su = np.concatenate([(1-delta)*np.identity(P), beta*np.identity(P), np.zeros((P,L))], axis=1)
        #Sl = np.concatenate([np.concatenate([alpha*np.identity(P), np.zeros((L,P))], axis=0), (1-gamma)*A], axis=1)
        #S = np.concatenate([Su, Sl], axis=0)

        Su = sp.hstack([(1-delta)*sp.identity(P), beta*sp.identity(P), sp.csr_matrix((P,H+L))])
        Sl = sp.hstack([sp.vstack([alpha*sp.identity(P), sp.csr_matrix((H+L,P))]), R])
        S = sp.vstack([Su, Sl])

        Ss.append(S.tocsr())

    def mv(v):

        for S in Ss:
            v = S.dot(v)

        return v

    '''
    def mv_mod(v):
        for R in Rs:
            tmp = R.dot(v[P:])
            tmp[:P] += alpha*v[:P]
            v = np.append((1-delta)*v[:P]+beta*v[P:2*P], tmp, axis=0)
        return v
    '''
            
    return spla.LinearOperator((N,N), matvec=mv)

def system_matrixRR(Rs, P, H, L, alpha, beta, delta):

    v = np.identity(2*P+H+L)

    for R in Rs:
        v = R.dot(v)

    return v

def system_matrixR(Rs, P, H, L, alpha, beta, delta):

    N = 2*P+H+L

    def mv(v):

        for R in Rs:
            v = R.dot(v)

        return v

    '''
    def mv_mod(v):
        for R in Rs:
            tmp = R.dot(v[P:])
            tmp[:P] += alpha*v[:P]
            v = np.append((1-delta)*v[:P]+beta*v[P:2*P], tmp, axis=0)
        return v
    '''
            
    return spla.LinearOperator((N,N), matvec=mv)

def average_system_matrix(Rs, P, H, L, alpha, beta, delta):
    
    N = 2*P+H+L
    Ss = []
    T = len(Rs)

    for R in Rs:

        # Dense formulation
        #Su = np.concatenate([(1-delta)*np.identity(P), beta*np.identity(P), np.zeros((P,L))], axis=1)
        #Sl = np.concatenate([np.concatenate([alpha*np.identity(P), np.zeros((L,P))], axis=0), (1-gamma)*A], axis=1)
        #S = np.concatenate([Su, Sl], axis=0)

        Su = sp.hstack([(1-delta)*sp.identity(P), beta*sp.identity(P), sp.csr_matrix((P,H+L))])
        Sl = sp.hstack([sp.vstack([alpha*sp.identity(P), sp.csr_matrix((H+L,P))]), R])
        S = sp.vstack([Su, Sl])

        Ss.append(S.tocsr())

    def mv(v):

        result = np.zeros_like(v)
        for S in Ss:
            result += S.dot(v)

        return result/T
            
    return spla.LinearOperator((N,N), matvec=mv)

def compute_rho(S):
    
    #T = len(As)
    #P = S.shape[0] - As[0].shape[0]
    #N = As[0].shape[0]
    
    # Compare the actual characteristic
    #actual = np.max(np.abs(np.linalg.eigvals(S)))**(1/T)
    rho = abs(spla.eigs(S, k=1, which='LM', return_eigenvectors=False)[0])

    return rho

    '''
    # Bound 1: using row-sums, max diagonal/off-diagonal entries
    d = np.max(np.diagonal(S))
    Stmp = np.copy(S)
    np.fill_diagonal(Stmp, -np.inf)
    g = Stmp.max()

    bound1 = np.inf
    row_sums = np.sum(S, axis=1)
    sorted_idx = np.argsort(row_sums)[::-1]
    for i, idx in enumerate(sorted_idx):
        r = row_sums[idx]
        t = r + d - g
        bound1 = np.minimum(bound1, (t + np.sqrt(t**2 + 4*g*np.sum(row_sums[sorted_idx[0:i]] - r)))/2) 
    '''

    '''
    bound2 = 1
    for i, A in enumerate(As):
        rho = abs(max(np.linalg.eigvals(A)))
        t = 1+rho-delta
        bound2 = bound2 * (t+np.sqrt(t**2 - 4*(rho-delta*rho-alpha*beta)))/2
    '''

    '''
    def mvAs_avg(v):

        result = np.zeros(v.shape)
        for A in As:
            result += A.dot(v)/T

        return result
            
    As_avg = spla.LinearOperator((N,N), matvec=mvAs_avg)
    rho = abs(spla.eigs(As_avg, k=1, which='LM', return_eigenvectors=False)[0])
    approx_avg = 0.5*(1-delta+(1-gamma)*rho+np.sqrt((1-delta+(1-gamma)*rho)**2 - 4*((1-delta)*(1-gamma)*rho - alpha*beta)))
    if verbose:
        print('Actual rho(S) = ', actual)
        print('Approximation using rho(average) = ', approx_avg**T)

    return actual, approx_avg
    '''
    
    '''
    # Bound using 2x2 block-structure
    smallS1 = np.identity(2)
    smallS2 = np.identity(2)
    smallSinf = np.identity(2)
    smallSfro = np.identity(2)
    if sp.isspmatrix(As[0]):
        for A in reversed(As):
            smallS1 = smallS1 @ np.array([[1-delta, beta],[alpha, (1-gamma)*spla.norm(A, ord=1)]])
            smallS2 = smallS2 @ np.array([[1-delta, beta],[alpha, (1-gamma)*spla.norm(A, ord=2)]])
            smallSinf = smallSinf @ np.array([[1-delta, beta],[alpha, (1-gamma)*spla.norm(A, ord=np.inf)]])
            smallSfro = smallSfro @ np.array([[np.sqrt(P*(1-delta)**2), np.sqrt(P*(beta**2))],[np.sqrt(P*(alpha**2)), (1-gamma)*spla.norm(A, ord='fro')]])
    else:
        for A in reversed(As):
            smallS1 = smallS1 @ np.array([[1-delta, beta],[alpha, (1-gamma)*np.linalg.norm(A, ord=1)]])
            smallS2 = smallS2 @ np.array([[1-delta, beta],[alpha, (1-gamma)*np.linalg.norm(A, ord=2)]])
            smallSinf = smallSinf @ np.array([[1-delta, beta],[alpha, (1-gamma)*np.linalg.norm(A, ord=np.inf)]])    
            smallSfro = smallSfro @ np.array([[np.sqrt(P*((1-delta)**2)), np.sqrt(P*(beta**2))],[np.sqrt(P*(alpha**2)), (1-gamma)*np.linalg.norm(A, ord='fro')]])   

    bound1 = np.max(np.abs(np.linalg.eigvals(smallS1)))
    bound2 = np.max(np.abs(np.linalg.eigvals(smallS2)))
    boundinf = np.max(np.abs(np.linalg.eigvals(smallSinf)))
    boundfro = np.max(np.abs(np.linalg.eigvals(smallSfro)))
    bound = np.min([bound1, bound2, boundinf, boundfro])
    
    if verbose:
        print('Actual rho(S) = ', actual)
        print('Upper bound rho(B)_2x2 using 1-norm = ', bound1)
        print('Upper bound rho(B)_2x2 using 2-norm = ', bound2)
        print('Upper bound rho(B)_2x2 using inf-norm = ', boundinf)
        print('Upper bound rho(B)_2x2 using fro-norm = ', boundfro)

    return actual, approx_avg, bound
    '''


    # Bound using 3x3 block-structure
    '''
    ### 3x3 rho(B) estimate
    smallS1 = np.identity(3)
    smallS2 = np.identity(3)
    smallSinf = np.identity(3)
    smallSfro = np.identity(3)
    if sp.isspmatrix(As[0]):
        for A in reversed(As):
            smallS1 = smallS1 @ np.array([[1-delta, beta, 0],[alpha, (1-gamma)*spla.norm(A[:P,:P], ord=1), (1-gamma)*spla.norm(A[:P,P:], ord=1)],[0, (1-gamma)*spla.norm(A[P:,:P], ord=1), (1-gamma)*spla.norm(A[P:,P:], ord=1)]])
            smallS2 = smallS2 @ np.array([[1-delta, beta, 0],[alpha, (1-gamma)*spla.norm(A[:P,:P], ord=2), (1-gamma)*spla.norm(A[:P,P:], ord=2)],[0, (1-gamma)*spla.norm(A[P:,:P], ord=2), (1-gamma)*spla.norm(A[P:,P:], ord=2)]])
            smallSinf = smallSinf @ np.array([[1-delta, beta, 0],[alpha, (1-gamma)*spla.norm(A[:P,:P], ord=np.inf), (1-gamma)*spla.norm(A[:P,P:], ord=np.inf)],[0, (1-gamma)*spla.norm(A[P:,:P], ord=np.inf), (1-gamma)*spla.norm(A[P:,P:], ord=np.inf)]])
            #smallSfro = smallSfro @ np.array([[np.sqrt(P*(1-delta)**2), np.sqrt(P*(beta**2))],[np.sqrt(P*(alpha**2)), (1-gamma)*spla.norm(A, ord='fro')]])
    else:
        for A in reversed(As):
            smallS1 = smallS1 @ np.array([[1-delta, beta, 0],[alpha, (1-gamma)*np.linalg.norm(A[:P,:P], ord=1), (1-gamma)*np.linalg.norm(A[:P,P:], ord=1)],[0, (1-gamma)*np.linalg.norm(A[P:,:P], ord=1), (1-gamma)*np.linalg.norm(A[P:,P:], ord=1)]])
            smallS2 = smallS2 @ np.array([[1-delta, beta, 0],[alpha, (1-gamma)*np.linalg.norm(A[:P,:P], ord=2), (1-gamma)*np.linalg.norm(A[:P,P:], ord=2)],[0, (1-gamma)*np.linalg.norm(A[P:,:P], ord=2), (1-gamma)*np.linalg.norm(A[P:,P:], ord=2)]])
            smallSinf = smallSinf @ np.array([[1-delta, beta, 0],[alpha, (1-gamma)*np.linalg.norm(A[:P,:P], ord=np.inf), (1-gamma)*np.linalg.norm(A[:P,P:], ord=np.inf)],[0, (1-gamma)*np.linalg.norm(A[P:,:P], ord=np.inf), (1-gamma)*np.linalg.norm(A[P:,P:], ord=np.inf)]])
            #smallSfro = smallSfro @ np.array([[np.sqrt(P*((1-delta)**2)), np.sqrt(P*(beta**2))],[np.sqrt(P*(alpha**2)), (1-gamma)*np.linalg.norm(A, ord='fro')]])   

    bound1 = np.max(np.abs(np.linalg.eigvals(smallS1)))
    bound2 = np.max(np.abs(np.linalg.eigvals(smallS2)))
    boundinf = np.max(np.abs(np.linalg.eigvals(smallSinf)))
    #boundfro = np.max(np.abs(np.linalg.eigvals(smallSfro)))
    #bound = np.min([bound1, bound2, boundinf])
    bound = np.min([bound1, boundinf])
    
    if verbose:
        print('Actual rho(S) = ', actual)
        print('Upper bound rho(B)_3x3 using 1-norm = ', bound1)
        print('Upper bound rho(B)_3x3 using 2-norm = ', bound2)
        print('Upper bound rho(B)_3x3 using inf-norm = ', boundinf)
        print('Returned estimate = ', bound)
        #print('Upper bound rho(B)_2x2 using fro-norm = ', boundfro)
    
    
    return actual, bound
    '''
