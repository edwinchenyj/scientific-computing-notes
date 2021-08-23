import numpy as np
import scipy.io as sio
import matplotlib.pyplot as pyplt
from scipy.sparse import csr_matrix, linalg
from numpy.random import default_rng
rng = default_rng(1)

class ElasticTriObj:
    G = np.array([[1,0], [0, 1], [-1,  -1]])
    Iv = np.eye(2)
    Im = np.eye(4)
    
    Kmm = np.array([[1. , 0., 0., 0.], [0., 0., 1., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]])
            
    def __init__(self, node, elem):
        # index conversion
        if np.amin(elem) == 1:
            elem = elem - 1
        self.nodeM = node
        self.elem = elem
        self.__elem_map__()
        self.__sparse_indices__()
        self.set_current_state(np.zeros(self.x.shape))
        self.set_current_velocity(np.zeros(self.x.shape))
        
        
    def __elem_map__(self):
        self.Dm = np.zeros((2*self.elem.shape[0],2))
        self.DmINV = np.zeros((2*self.elem.shape[0],2))
        self.W = np.zeros((self.elem.shape[0],))
        self.T = np.zeros((4*self.elem.shape[0], 6))
        
        self.X = np.reshape(self.nodeM.T,(2*self.nodeM.shape[0],1),order='F').flatten('F')
        self.x = self.X.copy()
        self.Ds = np.zeros((2*self.elem.shape[0],2))
        self.F = np.zeros((2*self.elem.shape[0],2))
        self.FINV = np.zeros((2*self.elem.shape[0],2))
        

        for i in range(self.elem.shape[0]):
            T_node = self.nodeM[self.elem[i,:],:]
            self.Dm[2*i:2*(i+1),:] = T_node.T.dot(self.G)
            self.DmINV[2*i:2*(i+1),:] = np.linalg.inv(T_node.T.dot(self.G))
            vol = np.linalg.det(T_node.T.dot(self.G))/2
            assert vol > 0, "need to fix mesh orientation"
            self.W[i] = vol
            self.T[4*i:4*(i+1),:] = np.kron((self.G.dot(self.DmINV[2*i:2*(i+1),:]).T), self.Iv) # definition: vec(F) = T * vec(x), or vec(dF) = T * vec(dx)
    
    def __sparse_indices__(self):
        self.ii = np.zeros((36*self.elem.shape[0],))
        self.jj = np.zeros((36*self.elem.shape[0],))

        index = 0
        for i in range(self.elem.shape[0]):
            for ti in range(3):
                for tj in range(3):
                    self.ii[index:index+4] = np.tile(np.r_[2*self.elem[i,ti]:2*(self.elem[i,ti]+1)].T,(2,1)).flatten()
                    self.jj[index:index+4] = np.tile(np.r_[2*self.elem[i,tj]:2*(self.elem[i,tj]+1)],(2,1)).reshape(4,1,order='F').flatten()
                    index = index + 4
        
    def set_current_state(self, Dx):
        # set the deformation using Dx, the displacement field on the nodes, and v, 
        # the velocity of the nodes. Notice Dx is not nodal position
        Dx = Dx.flatten()
        self.Dx = Dx
        self.x = self.X + Dx
        self.node = self.x.reshape(2, self.nodeM.shape[0], order='F').T.copy() # update nodal positions
        for i in range(self.elem.shape[0]):
            T_node = self.node[self.elem[i,:],:] # element nodal position in the world space
            self.Ds[2*i:2*(i+1),:] = T_node.T.dot(self.G)
            self.F[2*i:2*(i+1),:] = T_node.T.dot(self.G).dot(self.DmINV[2*i:2*(i+1),:])
            self.FINV[2*i:2*(i+1),:] = np.linalg.inv(self.F[2*i:2*(i+1),:])
            
    def set_current_velocity(self,v):
        v = v.flatten()
        self.v = v
    
    def stiffness_matrix(self):
        # construct and return the stiffness matrix under the current deformation,
        # this can be calculated using current deformation state as the only input
        index = 0
        sA = np.zeros((self.ii.shape))
        if(self.material_type == 'neo-hookean'):        
            for t in range(self.elem.shape[0]):
                mu = self.mu
                material_lambda = self.material_lambda
                tT = self.T[4*t:4*(t+1),:]
                W = self.W[t]
                tF = self.F[2*t:2*(t+1),:]
                tFINV = self.FINV[2*t:2*(t+1),:]
                
                # the fourth order tensor
                # for Neo-hookean
                C = (mu * self.Im + material_lambda * tFINV.T.flatten('F').dot(tFINV.T.flatten('F').T)  - (material_lambda * np.log(tF[0,0]*tF[1,1] - tF[0,1]*tF[1,0]) - mu) * np.kron(tFINV,tFINV.T).dot(self.Kmm))
                
                # element stiffness matrix
                
                Kt = W * (tT.T).dot(C).dot(tT)
                Kt = 1/2 * (Kt + Kt.T)
                
                for ti in range(3):
                    for tj in range(3):
                        # sA(index:index+3) = Kt(obj.IndK(3*(ti-1) + tj,:))
                        # same as:
                        sA[index:index+4] = [Kt[2*ti,2*tj], Kt[2*ti+1,2*tj], Kt[2*ti,2*tj+1], Kt[2*ti+1,2*tj+1]]
                        index = index + 4
                   
        # global stiffness matrix
        K = csr_matrix((sA, (self.ii, self.jj)), shape=(2*self.nodeM.shape[0], 2*self.nodeM.shape[0]))
        return K
    
    def elastic_force(self):
        f = np.zeros(self.x.shape)
        if(self.material_type == 'neo-hookean'):
            for t in range(self.elem.shape[0]):
            
                f_new = np.zeros(f.shape)
                tF = self.F[2*t:2*(t+1),:]
                tFINV = self.FINV[2*t:2*(t+1),:]

                mu = self.mu
                material_lambda = self.material_lambda
                
                J = np.linalg.det(tF)
                P = mu *(tF - tFINV.T) + material_lambda * np.log(J) * tFINV.T
                
                H = -self.W[t] * P.dot((self.DmINV[2*t:2*(t+1),:].T))
                i = self.elem[t, 0]
                j = self.elem[t, 1] 
                k = self.elem[t, 2]
                
                f_new[2*i:2*(i+1)] = f_new[2*i:2*(i+1)]+H[:,0]
                f_new[2*j:2*(j+1)] = f_new[2*j:2*(j+1)]+H[:,1]
                f_new[2*k:2*(k+1)] = f_new[2*k:2*(k+1)]-H[:,0] - H[:,1]
                f = f + f_new
        return f
                    
    def simple_vis(self):
        pyplt.triplot(self.node[:,0],self.node[:,1],self.elem)
        
    def set_material(self, Y, P, Rho, type, a, b):
        self.a = a
        self.b = b
        self.M = csr_matrix((self.X.shape[0],self.X.shape[0]))
        self.mu = Y/ ( 2 * (1 + P) )
        self.material_lambda = ( Y * P ) / ( (1 + P) * (1 - 2 * P) )
        self.Y = Y
        self.P = P
        self.Rho = Rho
        self.material_type = type
        # simple mass lumping
        for t in range(self.elem.shape[0]):
            for e in self.elem[t,:]:
                for mi in np.r_[e*2:(e+1)*2]:
                    self.M[mi,mi] = self.M[mi,mi]+self.W[t]/3 * Rho
    
    def step(self, dt=0.005, constraints=None, f_external=None):
        if(constraints==None):
            constraints = np.zeros(self.Dx.shape)
        if(f_external==None):
            f_external = np.zeros(constraints.shape)
        
        u = np.concatenate((self.Dx,self.v))
        K = self.stiffness_matrix()
        Mass = self.M
        Eforce = self.elastic_force()
        
        Mass = Mass[:,constraints==0]
        Mass = Mass[constraints==0,:]
        K = K[:,constraints==0]
        K = K[constraints==0,:]
        B = -self.a * Mass - self.b * K
        
        Eforce = Eforce[constraints==0]
        
       
        vold = u[int(u.shape[0]/2):]
        v = vold
        f = Eforce + f_external + B.dot(v[constraints==0])
        
        ## version 1
        A = (Mass - dt * B + (dt**2) * K)
        rhs = dt * (f - dt * K.dot(v[constraints==0]))
        # rhs.shape = 134,1
        dv_free = linalg.spsolve(A,rhs)
        
        v[constraints==0] = v[constraints==0] + dv_free
        
        u[int(u.shape[0]/2):] = v
        u[:int(u.shape[0]/2)] = u[:int(u.shape[0]/2)] + dt * v
        self.x = self.X +  u[:int(u.shape[0]/2)]
        
        self.set_current_velocity(v)
        self.set_current_state(self.x - self.X)
        u_new = u
        
        
        Eforce = self.elastic_force()
        Eforce = Eforce[constraints==0]
        f = Eforce + f_external + B.dot(v[constraints==0])
        residual = np.linalg.norm((v[constraints==0] - vold[constraints==0] - dt * (linalg.spsolve(Mass,f))))
        step_length = np.linalg.norm(dv_free)

        return u_new


triangle = sio.loadmat('triangle el1e-01.mat')
obj = ElasticTriObj(triangle['nodeM'], triangle['elem'])
obj.set_material(Y=1e5, P=0.45, Rho=1e3, type='neo-hookean', a=0, b=0)
obj.set_current_velocity(rng.random(obj.v.shape)*1)
obj.step()
obj.simple_vis()
