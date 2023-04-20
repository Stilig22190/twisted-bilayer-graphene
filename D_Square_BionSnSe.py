'''
 This is for continuum model moire bands of Bi/SnSe.
 Figure comes form https://doi.org/10.1103/PhysRevB.105.165422

'''

from numpy import *
import matplotlib.pyplot as plt
import numpy as np

#define constant
theta   =0/180.0*np.pi          #degree
a_t     = 4.5
a_b     = 4.2   #angstrom(埃米)(lattice constant)
I      = complex(0, 1) #复数
Delta   =(a_b-a_t)/a_t
a_m     =a_b/(sqrt(Delta**2+2*(1-cos(theta))))
N       = 4            #truncate range
valley  =1      
t_t     =800#mev
t_b     =900
hv_t    =a_t*t_t
hv_b    =a_b*t_b
delta_t =0     #mev
delta_b =900/2
w1      =80  #mev
w2      =80 
w2D     =w2.conjugate() 

                                             


ei90  = cos(pi/2) + valley*I*sin(pi/2)
ei180  = cos(pi) + valley*I*sin(pi)
ei270 = cos(3*pi/2) + valley*I*sin(3*pi/2)
ei_90  = cos(-pi/2) + valley*I*sin(-pi/2)
ei_180  = cos(-pi) + valley*I*sin(-pi)
ei_270  = cos(-3*pi/2) + valley*I*sin(-3*pi/2)


G1     = 2*np.pi/(a_m)*np.array([sqrt(2)/2, sqrt(2)/2]) #The reciprocal lattice vectors of superlattice basis vector
G2     = 2*np.pi/(a_m)*np.array([-sqrt(2)/2,sqrt(2)/2]) 
bm     = 2*np.pi/(a_m)
K1     = valley*np.pi/(a_m)*array([sqrt(2)/2, -sqrt(2)/2]) 
K2     = valley*np.pi/(a_m)*array([sqrt(2)/2, sqrt(2)/2]) 

T1    = np.array([[w1,w2D], [w2,w1]], dtype=complex)  
T2   = np.array([[w1, w2D*ei_90], [w2*ei90, w1]], dtype=complex)
T3   = np.array([[w1, w2D*ei_180], [w2*ei180, w1]], dtype=complex)
T4   = np.array([[w1, w2D*ei_270], [w2*ei270, w1]], dtype=complex)
T1D   = np.array(np.matrix(T1).H) #共轭转置
T2D   = np.array(np.matrix(T2).H) 
T3D   = np.array(np.matrix(T3).H)
T4D   = np.array(np.matrix(T4).H)

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
        H[2*i, 2*i+1] = hv_t*(valley*qx1 - I*qy1) 
        H[2*i+1, 2*i] = hv_t*(valley*qx1 + I*qy1)
        H[2*i,2*i]=delta_t
        H[2*i+1,2*i+1]=-delta_t
        H[2*i+2*waven, 2*i+2*waven+1] =  hv_b*(valley*qx2-I*qy2)
        H[2*i+2*waven+1, 2*i+2*waven] =  hv_b*(valley*qx2+I*qy2)
        H[2*i+2*waven,2*i+2*waven]=delta_b
        H[2*i+2*waven+1,2*i+2*waven+1]=-delta_b

        for j in np.arange(0,waven):
            m1 = L[j, 0]
            m2 = L[j, 1]
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
            
            if (m1-n1==-valley and m2-n2==-valley):
                H[2*i, 2*j+2*waven]    = T3[0, 0]
                H[2*i, 2*j+2*waven+1]   = T3[0, 1]
                H[2*i+1, 2*j+2*waven]   = T3[1, 0]
                H[2*i+1, 2*j+2*waven+1] = T3[1, 1]

                H[2*j+2*waven, 2*i]     = T3D[0, 0]
                H[2*j+2*waven,2*i+1]    = T3D[0, 1]
                H[2*j+2*waven+1,2*i]    = T3D[1, 0]
                H[2*j+2*waven+1,2*i+1]  = T3D[1, 1]

            if (m1==n1 and m2-n2==-valley):
                H[2*i, 2*j+2*waven]     = T4[0, 0]
                H[2*i, 2*j+2*waven+1]   = T4[0, 1]
                H[2*i+1, 2*j+2*waven]   = T4[1, 0]
                H[2*i+1, 2*j+2*waven+1] = T4[1, 1]

                H[2*j+2*waven, 2*i]     = T4D[0, 0]
                H[2*j+2*waven,2*i+1]    = T4D[0, 1]
                H[2*j+2*waven+1,2*i]    = T4D[1, 0]
                H[2*j+2*waven+1,2*i+1]  = T4D[1, 1]


                
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

    G_1[i]=real(Hamiltonian(bm/2*sqrt(2)/2*(M_1[i])/100,  bm/2*sqrt(2)/2*(M_1[i])/100))
    G_2[i]=real(Hamiltonian(bm/2*sqrt(2)/2*(M_2[i])/100,  bm/2*sqrt(2)/2*(200-M_2[i])/100))
    G_3[i]=real(Hamiltonian(bm/2*sqrt(2)*(300-M_3[i])/100,0))

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
plt.ylim(-500,500)
#plt.yticks(np.arange(-50, 75, step=25))
positions = (0,100,200,300)
labels = ("$\Gamma$","$X$","$M$","$\Gamma$")
plt.xticks(positions, labels)
plt.ylabel("E(meV)")
plt.axvline(x=101,color='gray',linestyle='--',linewidth=0.5)
plt.axvline(x=201,color='gray',linestyle='--',linewidth=0.5)
plt.axvline(x=301,color='gray',linestyle='--',linewidth=0.5)
plt.show()


