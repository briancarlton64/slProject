import numpy as np
import matplotlib.pyplot as plt
import gradDesc
import phasedWave

n = 128
l = 1
d = l/2
k = 2*np.pi/l;
lam = 0.01
snrTimes = 100
c = np.random.uniform(-np.pi/2,np.pi/2,4*n);

[lossVecNoReg,wNone] = gradDesc(c,n,k,d,lam,'None')
[lossVecLasso,wLasso] = gradDesc(c,n,k,d,lam,'Lasso')
[lossVecRidge,wRidge] = gradDesc(c,n,k,d,lam,'Ridge')
[lossVecElastic,wElastic] = gradDesc(c,n,k,d,lam,'Elastic')

hNone = np.zeros(np.size(c))
hLasso = np.zeros(np.size(c))
hRidge = np.zeros(np.size(c))
hElastic = np.zeros(np.size(c))
riskNone = np.zeros(snrTimes)
riskLasso = np.zeros(snrTimes)
riskRidge = np.zeros(snrTimes)
riskElastic = np.zeros(snrTimes)
rssNone = np.zeros(snrTimes)
rssLasso = np.zeros(snrTimes)
rssRidge = np.zeros(snrTimes)
rssElastic = np.zeros(snrTimes)
tss =  1/(2*n)*np.sum((np.mean(c)-c)**2)

j=0
for snr in np.linspace(0,20,snrTimes):
    i = 0
    for a in c:
            x = phasedWave(a,n,k,d)
            xPower = np.mean(x**2)
            noisePower = xPower/(10**(snr/10))
            x+=np.random.normal(0,noisePower,n)
            hNone[i] = np.sum(wNone*x)
            hLasso[i] = np.sum(wLasso*x)
            hRidge[i] = np.sum(wRidge*x)
            hElastic[i] = np.sum(wElastic*x)
            i+=1
    rssNone[j] =  1/(2*n)*np.sum((hNone-c)**2)
    rssLasso[j] =  1/(2*n)*np.sum((hLasso-c)**2)
    rssRidge[j] =  1/(2*n)*np.sum((hRidge-c)**2)
    rssElastic[j] =  1/(2*n)*np.sum((hElastic-c)**2)
    
    riskNone[j] =  (tss-rssNone[j])/(tss)
    riskLasso[j] =  (tss-rssLasso[j])/(tss)
    riskRidge[j] =  (tss-rssRidge[j])/(tss)
    riskElastic[j] =  (tss-rssElastic[j])/(tss)
    j +=1

plt.figure()
plt.title("Empirical Risk versus Iteration")
plt.ylabel("Empirical Risk")
plt.xlabel("Iteration")
plt.plot(lossVecNoReg)
plt.plot(lossVecRidge)
plt.plot(lossVecLasso)
plt.plot(lossVecElastic)
plt.legend(["No Regularization","Ridge L2","Lasso L1","Elastic L1,2"])
plt.savefig("risk.png")
plt.figure()
plt.title("R2 versus SNR")
plt.ylabel("R2")
plt.xlabel("SNR (dB)")
plt.plot(np.linspace(0,20,snrTimes),riskNone)
plt.plot(np.linspace(0,20,snrTimes),riskLasso)
plt.plot(np.linspace(0,20,snrTimes),riskRidge)
plt.plot(np.linspace(0,20,snrTimes),riskElastic)
plt.legend(["No Regularization","Ridge L2","Lasso L1","Elastic L1,2"])
plt.savefig("r2.png")
plt.show()