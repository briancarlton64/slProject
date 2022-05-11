import numpy as np
import phasedWave
def gradDesc(c,n,k,d,lam,regType):
    w = np.zeros(n)
    h = np.zeros(np.size(c))    
    gr = ((np.sqrt(5) + 1) / 2)**-1
    tau = gr
    alpha =1
    lossVec =[]
    sigma = 10**-3
    lossDiff=10
    times = 0
    while lossDiff>sigma:
        times +=1
        match regType:
            case 'None':
                regLoss = 0
                reg = 0
            case 'Lasso':
                regLoss = lam/2*np.sum(w**2)
                reg = lam*w
            case 'Ridge':
                regLoss = lam/2*np.sum(np.abs(w))
                reg = lam*np.sign(w)
            case 'Elastic':
                regLoss = 1/2*(lam*np.sum(np.abs(w))+(1-lam)*np.sum(w**2))
                reg = lam*np.sign(w)+(1-lam)*w
        loss =  1/(2*n)*np.sum((h-c)**2)+regLoss
        lossVec.append(loss)
        delta = np.zeros(n)
        i= 0
        for a in c:
            x = phasedWave.phasedWave(a,n,k,d)
            h[i] = np.sum(w*x) 
            delta += ((h[i]-c[i])*x+reg)
            i += 1
        newLoss =  1/(2*n)*np.sum((h-c)**2)+regLoss
        lossDiff = loss-newLoss
        lossDiffPrev = lossDiff
        steps = 0
        while (lossDiff>=lossDiffPrev or lossDiff<0) and steps<20:
            i = 0
            lossDiffPrev = lossDiff
            alpha = tau*alpha
            for a in c:
                x = phasedWave.phasedWave(a,n,k,d)
                h[i] = np.sum((w-alpha*delta)*x) 
                i += 1
            newLoss =  1/(2*n)*np.sum((h-c)**2)+regLoss
            lossDiff = loss-newLoss
            steps +=1
        alpha = alpha/tau;
        w-=alpha*delta
    return [lossVec,w]
        