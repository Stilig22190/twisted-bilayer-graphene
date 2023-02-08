'''
 This is for continuum model moire bands of Twisted bilayer graphene
'''

from numpy import *
import matplotlib.pyplot as plt
import numpy as np

#define constant
theta  = 1.05/180.0*np.pi          #degree
omega  = 110.7          #mev
a      = 2.46          #angstrom(埃米)(lattice constant)
hv     = 1.5*a/sqrt(3)*2970     #meV*angstrom, Fermi velocity for SLG
N      = 4            #truncate range
valley = +1              #+1 for K, -1 for K'

I      = complex(0, 1) #复数
ei120  = cos(2*pi/3) + valley*I*sin(2*pi/3)
ei240  = cos(2*pi/3) - valley*I*sin(2*pi/3)
bm=8*np.pi*sin(theta/2)/(a*3) #

#图可见https://journals.aps.org/prb/abstract/10.1103/PhysRevB.101.195425 的图1(a)
G1     = 8*np.pi*sin(theta/2)/(a*sqrt(3))*np.array([-0.5, -np.sqrt(3)/2]) #The reciprocal lattice vectors of superlattice basis vector 二维向量
'''superlattice constant is L and L=a/(2*sin(theta/2)).
    So the magnitude of lattice vectors of superlattice is |t1|=|t2|=L.
    then the area of the superlattice is S=sqrt(3)/2*L**2.
    then the magnitude of the reciprocal lattice vectors of superlattice is |G1|=|G2|=8*np.pi*sin(theta/2)/(a*sqrt(3))
     '''
G2     = 8*np.pi*sin(theta/2)/(a*sqrt(3))*np.array([1, 0]) 
K1     = 8*np.pi*sin(theta/2)/(a*3)*array([sqrt(3)/2,-0.5]) 
K2     = 8*np.pi*sin(theta/2)/(a*3)*array([sqrt(3)/2,0.5])

T1    = omega*np.array([[1,1], [1,1]], dtype=complex) #即https://link.aps.org/doi/10.1103/PhysRevB.86.155449 文献中表1，G=0时的情况
T2   = omega*np.array([[ei120, 1], [ei240, ei120]], dtype=complex)
T3   = omega*np.array([[ei240, 1], [ei120, ei240]], dtype=complex)

waven=(2*N+1)**2
k=0;
L=np.array(zeros((waven, 2))) #waven行2列

for i in np.arange(2*N+1): #[0,..,2*N]
  for j in np.arange(2*N+1):
      L[k,0]=i-N
      L[k,1]=j-N
      k=k+1
      
def Hamiltonian(kx,ky):
    H = array(zeros((4*waven, 4*waven)), dtype=complex)
    for i in range(0,waven):
       
        n1 = L[i, 0]
        n2 = L[i, 1]
        
        qx1 = kx -K1[0]+ n1*G1[0] + n2*G2[0]
        qy1 = ky -K1[1]+ n1*G1[1] + n2*G2[1]
        qx2 = kx -K2[0]+ n1*G1[0] + n2*G2[0] 
        qy2 = ky -K2[1]+ n1*G1[1] + n2*G2[1] 

        #相当于将各分量上的泡利矩阵与qx1和qy1相乘
        H[2*i, 2*i+1] = -hv*(valley*qx1 - I*qy1) 
        H[2*i+1, 2*i] = -hv*(valley*qx1 + I*qy1)
        H[2*i+2*waven, 2*i+2*waven+1] =  -hv*(valley*qx2-I*qy2)
        H[2*i+2*waven+1, 2*i+2*waven] =  -hv*(valley*qx2+I*qy2)
        for j in np.arange(0,waven):
            m1 = L[j, 0]
            m2 = L[j, 1]
            if (i == j):
                H[2*i, 2*j+2*waven]     = T1[0, 0]
                H[2*i, 2*j+2*waven+1]   = T1[0, 1]
                H[2*i+1, 2*j+2*waven]   = T1[1, 0]
                H[2*i+1, 2*j+2*waven+1] = T1[1, 1]

                H[2*j+2*waven, 2*i]     = T1[0, 0].conjugate()
                H[2*j+2*waven+1,2*i]    = T1[0, 1].conjugate()
                H[2*j+2*waven,2*i+1]    = T1[1, 0].conjugate()
                H[2*j+2*waven+1,2*i+1]  = T1[1, 1].conjugate()

            if (m1-n1==-valley and m2==n2):
                H[2*i, 2*j+2*waven]    = T2[0, 0]
                H[2*i, 2*j+2*waven+1]   = T2[0, 1]
                H[2*i+1, 2*j+2*waven]   = T2[1, 0]
                H[2*i+1, 2*j+2*waven+1] = T2[1, 1]

                H[2*j+2*waven, 2*i]     = T2[0, 0].conjugate()
                H[2*j+2*waven+1,2*i]    = T2[0, 1].conjugate()
                H[2*j+2*waven,2*i+1]    = T2[1, 0].conjugate()
                H[2*j+2*waven+1,2*i+1]  = T2[1, 1].conjugate()
            if (m1-n1==-valley and m2-n2==-valley):
                H[2*i, 2*j+2*waven]     = T3[0, 0]
                H[2*i, 2*j+2*waven+1]   = T3[0, 1]
                H[2*i+1, 2*j+2*waven]   = T3[1, 0]
                H[2*i+1, 2*j+2*waven+1] = T3[1, 1]

                H[2*j+2*waven, 2*i]     = T3[0, 0].conjugate()
                H[2*j+2*waven+1,2*i]    = T3[0, 1].conjugate()
                H[2*j+2*waven,2*i+1]    = T3[1, 0].conjugate()
                H[2*j+2*waven+1,2*i+1]  = T3[1, 1].conjugate()


                
    eigenvalue,featurevector=np.linalg.eig(H) #返回特征值，特征向量
    eig_vals_sorted = np.sort(eigenvalue) #数组按行排序
    #eig_vecs_sorted = featurevector[:,eigenvalue.argsort()] #将eigenvalue中的元素从小到大排列，提取其对应的index(索引)，然后返回index数组
    e=eig_vals_sorted
    return e
#plot bands
M_1 = arange(0, 101, 1)
M_2 = arange(100, 201, 1)
M_3 = arange(200, 301, 1)
G_1=array(zeros((len(M_1), 4*waven))) #指定长度数组(M_1lenth,4*waven)matrix,且这里的G_1与上G1无关
G_2=array(zeros((len(M_2), 4*waven)))
G_3=array(zeros((len(M_3), 4*waven)))
for i in range(0,len(M_1)):
    G_1[i]=real(Hamiltonian(-bm*sqrt(3)*(M_1[i]-100)/200, -bm*(M_1[i]-100)/200))
    G_2[i]=real(Hamiltonian(bm*(M_2[i]-100)*sqrt(3)/200, 0))
    G_3[i]=real(Hamiltonian(bm*sqrt(3)/2, -bm*(M_3[i]-200)/200))

for j in range(0,4*waven):
    plt.plot(M_1,G_1[:,j],linestyle="-",color="b", linewidth=0.6) #即 m[:,j]相当于 m[0:len(M_1),j],matrix的第j列
    plt.plot(M_2,G_2[:,j],linestyle="-",color="b", linewidth=0.6)
    plt.plot(M_3,G_3[:,j],linestyle="-",color="b", linewidth=0.6)

plt.xlim(0,301)
plt.ylim(-50,50)
plt.yticks(np.arange(-50, 75, step=25))
positions = (0,100,200,300)
labels = ("$K_+^m$","$\Gamma^m$","$M^m$","$K_-^m$")
plt.xticks(positions, labels)
plt.ylabel("E(meV)")
plt.axvline(x=101,color='gray',linestyle='--',linewidth=0.5)
plt.axvline(x=201,color='gray',linestyle='--',linewidth=0.5)
plt.axvline(x=301,color='gray',linestyle='--',linewidth=0.5)
plt.show()


