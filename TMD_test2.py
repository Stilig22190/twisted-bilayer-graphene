'''
 This is for continuum model moire bands of Transition metal dichalcogenide.(MoSe2/MoS2)
 And the parameters come form https://www.pnas.org/doi/abs/10.1073/pnas.2112673118
 Relational knowledge comes form https://www.guanjihuan.com/
'''

from numpy import *
import matplotlib.pyplot as plt
import numpy as np

#define constant
theta   = 1/180.0*np.pi          #degree
t_c     = 26*10**(-3)         #ev
t_v     =2*t_c
a_b     = 3.157*10**(-10)      #m
a_t     =  3.289*10**(-10)
Delta   =(a_b-a_t)/a_t
a_m     =a_b/(sqrt(Delta**2+theta**2))
m0      =0.51099895 * 10**(6)  
m_b_v     =0.43*m0
m_t_v     =0.44*m0
m_b_c     =0.35*m0
m_t_c     =0.38*m0
hbar    = 4.135667 * 10**(-15)/2/pi
N       = 5            #truncate range
valley  = +1      
kin_b_c = hbar**2/2/m_b_c *9*10**(16)
kin_t_c = hbar**2/2/m_t_c *9*10**(16)
kin_b_v = -hbar**2/2/m_b_v *9*10**(16)
kin_t_v = -hbar**2/2/m_t_v *9*10**(16)  
E_g=0.96        
delta_v=0.63
delta_c=0.370                    
dK0=4*np.pi/(a_m*3)*np.array([0, -1])
dK1=4*np.pi/(a_m*3)*np.array([sqrt(3)/2, 0.5])
dK2=4*np.pi/(a_m*3)*np.array([-sqrt(3)/2, 0.5])               

I      = complex(0, 1) #复数
ei120  = cos(2*pi/3) + I*sin(2*pi/3)
ei240  = cos(2*pi/3) - I*sin(2*pi/3)


b1     = 4*np.pi/(a_m*sqrt(3))*np.array([0.5, -sqrt(3)/2]) #The reciprocal lattice vectors of superlattice basis vector
b2     = 4*np.pi/(a_m*sqrt(3))*np.array([0.5, sqrt(3)/2])
b3     = 4*np.pi/(a_m*sqrt(3))*np.array([-1, 0]) 
bm     = 4*np.pi/(a_m*3)
K1     = 4*np.pi/(a_m*3)*array([sqrt(3)/2,0.5]) 
K2     = 4*np.pi/(a_m*3)*array([sqrt(3)/2,-0.5]) 

T1D    = np.array([[t_c,0], [0,t_v]], dtype=complex) #It is Table 1 in https://link.aps.org/doi/10.1103/PhysRevB.86.155449 reference, when G=0
T2D   = np.array([[t_c*ei240, 0], [0, t_v*ei240]], dtype=complex)
T3D   = np.array([[t_c*ei120, 0], [0,  t_v*ei120 ]], dtype=complex)
T1   = np.array(np.matrix(T1D).H) #共轭转置
T2   = np.array(np.matrix(T2D).H) 
T3   = np.array(np.matrix(T3D).H)

waven=(2*N+1)**2
k=0;
L=np.array(zeros((waven, 2))) #waven行2列

for i in np.arange(2*N+1): #[0,..,2*N]
  for j in np.arange(2*N+1):
      L[k,0]=i-N
      L[k,1]=j-N
      k=k+1
      
def E(layer,band,kx,ky):
    if (layer=='t' and band=='c'):
        return kin_t_c*(kx**2+ky**2)+E_g+delta_c
    if (layer=='t' and band=='v'):
        return kin_t_v*(kx**2+ky**2)
    if (layer=='b' and band=='c'):
        return kin_b_c*(kx**2+ky**2)+E_g 
    if (layer=='b' and band=='v'):
        return  kin_b_v*(kx**2+ky**2)-delta_v
    
      
def Hamiltonian(kx,ky):
    H = array(zeros((4*waven, 4*waven)), dtype=complex)
    for i in range(0,waven):
       
        n1 = L[i, 0]
        n2 = L[i, 1]
        
        qx1 = kx -K1[0]+n1*b1[0] + n2*b2[0]
        qy1 = ky -K1[1]+ n1*b1[1] + n2*b2[1]
        qx2 = kx -K2[0]+ n1*b1[0] + n2*b2[0] 
        qy2 = ky -K2[1]+ n1*b1[1] + n2*b2[1] 

        
        H[2*i, 2*i] = E('t','c',qx1,qy1)+t_c**2*(1/(E('t','c',qx1,qy1)-E('b','c',qx1-dK0[0],qy1-dK0[1]))+1/(E('t','c',qx1,qy1)-E('b','c',qx1-dK1[0],qy1-dK1[1]))+1/(E('t','c',qx1,qy1)-E('b','c',qx1-dK2[0],qy1-dK2[1])))
        H[2*i+1, 2*i+1] = E('t','v',qx1,qy1)+t_v**2*(1/(E('t','v',qx1,qy1)-E('b','v',qx1+dK0[0],qy1+dK0[1]))+1/(E('t','v',qx1,qy1)-E('b','v',qx1+dK1[0],qy1+dK1[1]))+1/(E('t','v',qx1,qy1)-E('b','v',qx1+dK2[0],qy1+dK2[1])))
        H[2*i+2*waven, 2*i+2*waven] = E('b','c',qx2,qy2)-t_c**2*(1/(E('t','c',qx2+dK0[0],qy2+dK0[1])-E('b','c',qx2,qy2))+1/(E('t','c',qx2+dK1[0],qy2+dK1[1])-E('b','c',qx2,qy2))+1/(E('t','c',qx2+dK2[0],qy2+dK2[1])-E('b','c',qx2,qy2)))
        H[2*i+2*waven+1, 2*i+2*waven+1] =  E('b','v',qx2,qy2)-t_v**2*(1/(E('t','v',qx2-dK0[0],qy2-dK0[1])-E('b','v',qx2,qy2))+1/(E('t','v',qx2-dK1[0],qy2-dK1[1])-E('b','v',qx2,qy2))+1/(E('t','v',qx2-dK2[0],qy2-dK2[1])-E('b','v',qx2,qy2)))
        for j in np.arange(0,waven):
            m1 = L[j, 0]
            m2 = L[j, 1]
            jx1 = kx -K1[0]+ m1*b1[0] + m2*b2[0]
            jy1 = ky -K1[1]+ m1*b1[1] + m2*b2[1]
            jx2 = kx -K2[0]+ m1*b1[0] + m2*b2[0] 
            jy2 = ky -K2[1]+ m1*b1[1] + m2*b2[1] 
            if (i == j):
                H[2*i, 2*j+2*waven]     = T1[0, 0]
                H[2*i, 2*j+2*waven+1]   = T1[0, 1]
                H[2*i+1, 2*j+2*waven]   = T1[1, 0]
                H[2*i+1, 2*j+2*waven+1] = T1[1, 1]

                H[2*j+2*waven, 2*i]     = T1D[0, 0]
                H[2*j+2*waven,2*i+1]    = T1D[0, 1]
                H[2*j+2*waven+1,2*i]    = T1D[1, 0]
                H[2*j+2*waven+1,2*i+1]  = T1D[1, 1]


            if (m1-n1==-valley and m2==n2):
                H[2*i, 2*j+2*waven]    = T2[0, 0]
                H[2*i, 2*j+2*waven+1]   = T2[0, 1]
                H[2*i+1, 2*j+2*waven]   = T2[1, 0]
                H[2*i+1, 2*j+2*waven+1] = T2[1, 1]

                H[2*j+2*waven, 2*i]     = T2D[0, 0]
                H[2*j+2*waven,2*i+1]    = T2D[0, 1]
                H[2*j+2*waven+1,2*i]    = T2D[1, 0]
                H[2*j+2*waven+1,2*i+1]  = T2D[1, 1]
                #n=+1
                 #G1*r0=2*pi/3,G2*r0=2*pi/3,G3*r0=-4*pi/3
                H[2*i,2*j]=t_c**2/2*ei120/(E('t','c',jx1,jy1)-E('b','c',jx1-dK2[0],jy1-dK2[1]))
                H[2*i+1,2*j+1]=t_v**2/2*1/(E('b','v',jx1-dK2[0],jy1-dK2[1])-E('t','v',jx1,jy1))
                H[2*i+2*waven,2*j+2*waven]=-t_c**2/2*ei120/(E('t','c',jx2+dK0[0],jy2+dK0[1])-E('b','c',jx2,jy2))
                H[2*j+2*waven+1,2*i+2*waven+1]=-t_v**2/2*1/(E('b','v',jx2,jy2)-E('t','v',jx2-dK0[0],jy2-dK0[1]))
                H[2*j,2*i]=H[2*i,2*j].conjugate()
                H[2*j+1,2*i+1]=H[2*i+1,2*j+1].conjugate()
                H[2*j+2*waven,2*i+2*waven]=H[2*i+2*waven,2*j+2*waven].conjugate()
                H[2*j+2*waven+1,2*i+2*waven+1]=H[2*i+2*waven+1,2*j+2*waven+1].conjugate()

            if (m1-n1==valley and m2==n2):
                #n=-1
                 #G1*r0=2*pi/3,G2*r0=2*pi/3,G3*r0=-4*pi/3
                H[2*i,2*j]=t_c**2/2*ei240/(E('t','c',jx1,jy1)-E('b','c',jx1-dK0[0],jy1-dK0[1]))
                H[2*i+1,2*j+1]=t_v**2/2*1/(E('b','v',jx1-dK0[0],jy1-dK0[1])-E('t','v',jx1,jy1))
                H[2*i+2*waven,2*j+2*waven]=-t_c**2/2*ei240/(E('t','c',jx2+dK2[0],jy2+dK2[1])-E('b','c',jx2,jy2))
                H[2*j+2*waven+1,2*i+2*waven+1]=-t_v**2/2*1/(E('b','v',jx2,jy2)-E('t','v',jx2-dK2[0],jy2-dK2[1]))
                H[2*j,2*i]=H[2*i,2*j].conjugate()
                H[2*j+1,2*i+1]=H[2*i+1,2*j+1].conjugate()
                H[2*j+2*waven,2*i+2*waven]=H[2*i+2*waven,2*j+2*waven].conjugate()
                H[2*j+2*waven+1,2*i+2*waven+1]=H[2*i+2*waven+1,2*j+2*waven+1].conjugate()

            if (m2-n2==-valley and m1==n1):
                #n=+2
                 #G1*r0=2*pi/3,G2*r0=2*pi/3,G3*r0=-4*pi/3
                H[2*i,2*j]=t_c**2/2*ei120/(E('t','c',jx1,jy1)-E('b','c',jx1-dK0[0],jy1-dK0[1]))
                H[2*i+1,2*j+1]=t_v**2/2*ei120/(E('b','v',jx1-dK0[0],jy1-dK0[1])-E('t','v',jx1,jy1))
                H[2*i+2*waven,2*j+2*waven]=-t_c**2/2*ei120/(E('t','c',jx2+dK1[0],jy2+dK1[1])-E('b','c',jx2,jy2))
                H[2*j+2*waven+1,2*i+2*waven+1]=-t_v**2/2*ei120/(E('b','v',jx2,jy2)-E('t','v',jx2-dK1[0],jy2-dK1[1]))
                H[2*j,2*i]=H[2*i,2*j].conjugate()
                H[2*j+1,2*i+1]=H[2*i+1,2*j+1].conjugate()
                H[2*j+2*waven,2*i+2*waven]=H[2*i+2*waven,2*j+2*waven].conjugate()
                H[2*j+2*waven+1,2*i+2*waven+1]=H[2*i+2*waven+1,2*j+2*waven+1].conjugate()
            if (m2-n2==valley and m1==n1):
                #n=-2
                 #G1*r0=2*pi/3,G2*r0=2*pi/3,G3*r0=-4*pi/3
                H[2*i,2*j]=t_c**2/2*ei240/(E('t','c',jx1,jy1)-E('b','c',jx1-dK1[0],jy1-dK1[1]))
                H[2*i+1,2*j+1]=t_v**2/2*ei240/(E('b','v',jx1-dK1[0],jy1-dK1[1])-E('t','v',jx1,jy1))
                H[2*i+2*waven,2*j+2*waven]=-t_c**2/2*ei240/(E('t','c',jx2+dK0[0],jy2+dK0[1])-E('b','c',jx2,jy2))
                H[2*j+2*waven+1,2*i+2*waven+1]=-t_v**2/2*ei240/(E('b','v',jx2,jy2)-E('t','v',jx2-dK0[0],jy2-dK0[1]))
                H[2*j,2*i]=H[2*i,2*j].conjugate()
                H[2*j+1,2*i+1]=H[2*i+1,2*j+1].conjugate()
                H[2*j+2*waven,2*i+2*waven]=H[2*i+2*waven,2*j+2*waven].conjugate()
                H[2*j+2*waven+1,2*i+2*waven+1]=H[2*i+2*waven+1,2*j+2*waven+1].conjugate()

            if (m1-n1==-valley and m2-n2==-valley):
                H[2*i, 2*j+2*waven]     = T3[0, 0]
                H[2*i, 2*j+2*waven+1]   = T3[0, 1]
                H[2*i+1, 2*j+2*waven]   = T3[1, 0]
                H[2*i+1, 2*j+2*waven+1] = T3[1, 1]

                H[2*j+2*waven, 2*i]     = T3D[0, 0]
                H[2*j+2*waven,2*i+1]    = T3D[0, 1]
                H[2*j+2*waven+1,2*i]    = T3D[1, 0]
                H[2*j+2*waven+1,2*i+1]  = T3D[1, 1]
                #n=-3
                H[2*i,2*j]=t_c**2/2*ei240/(E('t','c',jx1,jy1)-E('b','c',jx1-dK2[0],jy1-dK2[1]))
                H[2*i+1,2*j+1]=t_v**2/2*ei120/(E('b','v',jx1-dK2[0],jy1-dK2[1])-E('t','v',jx1,jy1))
                H[2*i+2*waven,2*j+2*waven]=-t_c**2/2*ei240/(E('t','c',jx2+dK1[0],jy2+dK1[1])-E('b','c',jx2,jy2))
                H[2*j+2*waven+1,2*i+2*waven+1]=-t_v**2/2*ei120/(E('b','v',jx2,jy2)-E('t','v',jx2-dK1[0],jy2-dK1[1]))
                H[2*j,2*i]=H[2*i,2*j].conjugate()
                H[2*j+1,2*i+1]=H[2*i+1,2*j+1].conjugate()
                H[2*j+2*waven,2*i+2*waven]=H[2*i+2*waven,2*j+2*waven].conjugate()
                H[2*j+2*waven+1,2*i+2*waven+1]=H[2*i+2*waven+1,2*j+2*waven+1].conjugate()
            if (m1-n1==valley and m2-n2==valley):
                #n=3
                H[2*i,2*j]=t_c**2/2*ei120/(E('t','c',jx1,jy1)-E('b','c',jx1-dK1[0],jy1-dK1[1]))
                H[2*i+1,2*j+1]=t_v**2/2*ei240/(E('b','v',jx1-dK1[0],jy1-dK1[1])-E('t','v',jx1,jy1))
                H[2*i+2*waven,2*j+2*waven]=-t_c**2/2*ei120/(E('t','c',jx2+dK2[0],jy2+dK2[1])-E('b','c',jx2,jy2))
                H[2*j+2*waven+1,2*i+2*waven+1]=-t_v**2/2*ei240/(E('b','v',jx2,jy2)-E('t','v',jx2-dK2[0],jy2-dK2[1]))
                H[2*j,2*i]=H[2*i,2*j].conjugate()
                H[2*j+1,2*i+1]=H[2*i+1,2*j+1].conjugate()
                H[2*j+2*waven,2*i+2*waven]=H[2*i+2*waven,2*j+2*waven].conjugate()
                H[2*j+2*waven+1,2*i+2*waven+1]=H[2*i+2*waven+1,2*j+2*waven+1].conjugate()

                
    eigenvalue,featurevector=np.linalg.eig(H) #返回特征值，特征向量
    eig_vals_sorted = np.sort(eigenvalue) #数组按行排序
    #eig_vecs_sorted = featurevector[:,eigenvalue.argsort()] #将eigenvalue中的元素从小到大排列，提取其对应的index(索引)，然后返回index数组
    e=eig_vals_sorted
    return e
#plot bands
M_1 = arange(0, 101, 1) #分成三个部分
M_2 = arange(100, 201, 1)
M_3 = arange(200, 301, 1)
G_1=array(zeros((len(M_1), 4*waven))) #指定长度数组(M_1lenth,4*waven)matrix,且这里的G_1与上G1无关
G_2=array(zeros((len(M_2), 4*waven)))     
G_3=array(zeros((len(M_3), 4*waven)))

for i in range(0,len(M_1)):    
    #算能带结构,就是算高对称点上的连线上的能级结构(即:不可约布里渊区的边界）
    G_1[i]=real(Hamiltonian(bm*sqrt(3)/2*M_1[i]/100, bm/2*M_1[i]/100)) 
    G_2[i]=real(Hamiltonian(bm*sqrt(3)/2, -bm/2*(M_2[i]-200)/100)) 
    G_3[i]=real(Hamiltonian(-bm*sqrt(3)/2*(M_3[i]-300)/100, 0)) 

for j in range(0,4*waven):
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
plt.ylim(-0.5,0.5)
#plt.yticks(np.arange(-50, 75, step=25))
positions = (0,100,200,300)
labels = ("$\Gamma$","$K$","$M$","$\Gamma$")
plt.xticks(positions, labels)
plt.ylabel("E(eV)")
plt.axvline(x=101,color='gray',linestyle='--',linewidth=0.5)
plt.axvline(x=201,color='gray',linestyle='--',linewidth=0.5)
plt.axvline(x=301,color='gray',linestyle='--',linewidth=0.5)
plt.show()

