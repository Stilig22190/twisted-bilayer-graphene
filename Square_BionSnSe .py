'''
 This is for continuum model moire bands of Transition metal dichalcogenide.

'''

from numpy import *
import matplotlib.pyplot as plt
import numpy as np

#define constant
theta   = 4/180.0*np.pi          #degree
W       = 20*10**(-3)         #ev
a_b     = 3.575*10**(-10)      #m
a_t     =  3.32*10**(-10)
Delta   =(a_b-a_t)/a_t
a_m     =a_b/(sqrt(Delta**2+2*(1-cos(theta))))
E_g     =36.8*10**(-3)         #ev
m0      =0.51099895 * 10**(6)  
m_b     =0.65*m0
m_t     =0.35*m0
hbar    = 4.135667 * 10**(-15)/2/pi
N       = 4            #truncate range
valley  = +1      
V_b     =4.1*10**(-3)          #ev
psi_b   =14/180.0*np.pi  
kin_b = -hbar**2/2/m_b *9*10**(16)
kin_t = -hbar**2/2/m_t *9*10**(16)

                                             

I      = complex(0, 1) #复数
ei120  = cos(2*pi/3) + I*sin(2*pi/3)
ei240  = cos(2*pi/3) - I*sin(2*pi/3)


G1     = 2*np.pi/(a_m)*np.array([1, 0]) #The reciprocal lattice vectors of superlattice basis vector
G2     = 2*np.pi/(a_m)*np.array([0, 1]) 
bm     = 2*np.pi/(a_m)
K1     = np.pi/(a_m)*array([1,1]) 
K2     = np.pi/(a_m)*array([1,-1]) 

T1D   = W
T2D   = W*ei240
T3D   = W*ei120 
T4D   =W
T1   =T1D.conjugate()  #共轭
T2   = T2D.conjugate() 
T3   = T3D.conjugate()
T4   = T4D.conjugate()

waven=(2*N+1)**2
k=0;
L=np.array(zeros((waven, 2))) #waven行2列

for i in np.arange(2*N+1): #[0,..,2*N]
  for j in np.arange(2*N+1):
      L[k,0]=i-N
      L[k,1]=j-N
      k=k+1
      
def Hamiltonian(kx,ky):
    H = array(zeros((2*waven, 2*waven)), dtype=complex)
    for i in range(0,waven):
       
        n1 = L[i, 0]
        n2 = L[i, 1]
        
        qx1 = kx -K1[0]+ n1*G1[0] + n2*G2[0]
        qy1 = ky -K1[1]+ n1*G1[1] + n2*G2[1]
        qx2 = kx -K2[0]+ n1*G1[0] + n2*G2[0] 
        qy2 = ky -K2[1]+ n1*G1[1] + n2*G2[1] 

        H[i, i] =kin_b *(qx1**2+qy1**2)
        H[i+waven, i+waven] = kin_t*(qx2**2+qy2**2)-E_g 
        for j in np.arange(0,waven):
            m1 = L[j, 0]
            m2 = L[j, 1]
            if (i == j):
                H[i, j+waven]     = T1
                H[j+waven, i]     = T1D
            if (m1-n1==-valley and m2==n2):
                H[i, j+waven]    = T2
                H[j+waven, i]     = T2D
                H[i,j]=V_b*exp(-I*psi_b)
                H[j,i]=V_b*exp(I*psi_b)
            if (m1==n1 and m2-n2==-valley):
                H[i, j+waven]     = T3
                H[j+waven, i]     = T3D
                H[i,j]=V_b*exp(-I*psi_b)
                H[j,i]=V_b*exp(I*psi_b)    
            if (m1-n1==-valley and m2-n2==-valley):
                H[i, j+waven]     = T4
                H[j+waven, i]     = T4D
                H[i,j]=V_b*exp(-I*psi_b)
                H[j,i]=V_b*exp(I*psi_b)
            
  


                
    eigenvalue,featurevector=np.linalg.eig(H) #返回特征值，特征向量
    eig_vals_sorted = np.sort(eigenvalue) #数组按行排序
    #eig_vecs_sorted = featurevector[:,eigenvalue.argsort()] #将eigenvalue中的元素从小到大排列，提取其对应的index(索引)，然后返回index数组
    e=eig_vals_sorted
    return e
#plot bands
M_1 = arange(0, 101, 1) #分成三个部分
M_2 = arange(100, 201, 1)
M_3 = arange(200, 301, 1)
G_1=array(zeros((len(M_1), 2*waven))) #指定长度数组(M_1lenth,4*waven)matrix,且这里的G_1与上G1无关
G_2=array(zeros((len(M_2), 2*waven)))     
G_3=array(zeros((len(M_3), 2*waven)))

for i in range(0,len(M_1)):    
    #算能带结构,就是算高对称点上的连线上的能级结构(即:不可约布里渊区的边界）

    G_1[i]=real(Hamiltonian(bm/2*(M_1[i])/100,  bm/2*(M_1[i])/100))
    G_2[i]=real(Hamiltonian(bm/2,  bm/2*(200-M_2[i])/100))
    G_3[i]=real(Hamiltonian(bm/2*(300-M_3[i])/100,0))

for j in range(0,2*waven):
    if(j%4==1):
        plt.plot(M_1,G_1[:,j],linestyle="-",color="b", linewidth=0.6) #即 G_1[:,j]相当于G_1[0:len(M_1),j],即G_1 matrix的第j列
        plt.plot(M_2,G_2[:,j],linestyle="-",color="b", linewidth=0.6)
        plt.plot(M_3,G_3[:,j],linestyle="-",color="b", linewidth=0.6)
    elif(j%4==2):
        plt.plot(M_1,G_1[:,j],linestyle="-",color="r", linewidth=0.6) 
        plt.plot(M_2,G_2[:,j],linestyle="-",color="r", linewidth=0.6)
        plt.plot(M_3,G_3[:,j],linestyle="-",color="r", linewidth=0.6)
    elif(j%4==3):
        plt.plot(M_1,G_1[:,j],linestyle="-",color="black", linewidth=0.6)
        plt.plot(M_2,G_2[:,j],linestyle="-",color="black", linewidth=0.6)
        plt.plot(M_3,G_3[:,j],linestyle="-",color="black", linewidth=0.6)
    elif(j%4==0):
        plt.plot(M_1,G_1[:,j],linestyle="-",color="green", linewidth=0.6) 
        plt.plot(M_2,G_2[:,j],linestyle="-",color="green", linewidth=0.6)
        plt.plot(M_3,G_3[:,j],linestyle="-",color="green", linewidth=0.6)


plt.xlim(0,301)
plt.ylim(-0.08,0.02)
#plt.yticks(np.arange(-50, 75, step=25))
positions = (0,100,200,300)
labels = ("$\Gamma$","$K$","$M$","$\Gamma$")
plt.xticks(positions, labels)
plt.ylabel("E(eV)")
plt.axvline(x=101,color='gray',linestyle='--',linewidth=0.5)
plt.axvline(x=201,color='gray',linestyle='--',linewidth=0.5)
plt.axvline(x=301,color='gray',linestyle='--',linewidth=0.5)
plt.show()


