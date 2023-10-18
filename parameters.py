import numpy as np





def Richards2021(T):
    T[T<-30] = -30
    iotaD = 0.0259733*T + 1.95268104
    iotaS = np.zeros_like(T)
    lambtilde = (0.00251776*T + 0.41244777)/np.sqrt(2) #correction to agree with SpecCAF
    betatilde = 5*(0.35182521*T + 12.17066493)/np.sqrt(2)

    Ecc = np.ones_like(T)
    Eca = np.ones_like(T)
    power = np.ones_like(T)
    

    x = np.array([iotaD,iotaS,lambtilde,betatilde,Ecc,Eca,power])
    
    return x.T


def Richards2021Reduced(T,reduce=0.25):
    T[T<-30] = -30
    iotaD = 0.0259733*T + 1.95268104
    iotaS = np.zeros_like(T)
    lambtilde = reduce*(0.00251776*T + 0.41244777)/np.sqrt(2) #correction to agree with SpecCAF
    betatilde = 5*(0.35182521*T + 12.17066493)/np.sqrt(2)

    Ecc = np.ones_like(T)
    Eca = np.ones_like(T)
    power = np.ones_like(T)
    

    x = np.array([iotaD,iotaS,lambtilde,betatilde,Ecc,Eca,power])
    
    return x.T

def Elmer(T):
    iotaD = 0.94*np.ones_like(T)
    iotaS = 0.6*np.ones_like(T)
    lambtilde = np.sqrt(2)*2e-3 * np.exp(np.log(10)*T/10)
    betatilde = np.zeros_like(T)
    Ecc = np.ones_like(T)
    Eca = 25*np.ones_like(T)
    power = np.ones_like(T)

    x = np.array([iotaD,iotaS,lambtilde,betatilde,Ecc,Eca,power])
    
    return x.T
