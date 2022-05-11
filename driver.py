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
cMod = np.linspace(-np.pi/2,np.pi/2,4*n)
Xmod = np.zeros((n,4*n))
i = 0
for a in cMod:
    Xmod[:,i] = phasedWave.phasedWave(a,n,k,d)
    i += 1
Xmod = np.transpose(Xmod)
beta = np.linalg.pinv(Xmod)@cMod
c = np.random.uniform(-np.pi/2,np.pi/2,4*n);

tss =  1/(2*n)*np.sum((np.mean(c)-c)**2)
fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()

ax1 = fig1.subplots()
ax2 = fig2.subplots()
ax3 = fig3.subplots()
#Ordinary Least Squares
w = beta;
h = np.zeros(np.size(c))
risk = np.zeros(snrTimes)
rss = np.zeros(snrTimes)
j=0
for snr in np.linspace(0,20,snrTimes):
    i = 0
    for a in c:
            x = phasedWave.phasedWave(a,n,k,d)
            xPower = np.mean(x**2)
            noisePower = xPower/(10**(snr/10))
            x+=np.random.normal(0,noisePower,n)
            h[i] = np.sum(w*x)
            i+=1
    rss[j] =  1/(2*n)*np.sum((h-c)**2)
    risk[j] =  (tss-rss[j])/(tss)
    j +=1
ax3.plot(np.linspace(0,20,snrTimes),risk)

#Generalized Least Squares

h = np.zeros(np.size(c))
risk = np.zeros(snrTimes)
rss = np.zeros(snrTimes)
j=0
noisePowerInd = np.zeros(4*n)
for snr in np.linspace(0,20,snrTimes):
    i = 0
    z=0
    for x in Xmod:
        xPowerMod = np.mean(x**2)
        noisePowerInd[z] = xPowerMod/(10**(snr/10))
        z +=1
    omega = np.mean(noisePowerInd)**2*np.identity(4*n);
    w = np.linalg.pinv(np.transpose(Xmod)@np.linalg.inv(omega)@Xmod)@np.transpose(Xmod)@np.linalg.inv(omega)@cMod
    for a in c:
            x = phasedWave.phasedWave(a,n,k,d)
            xPower = np.mean(x**2)
            noisePower = xPower/(10**(snr/10))
            
            x+=np.random.normal(0,noisePower,n)
            h[i] = np.sum(w*x)
            i+=1
    rss[j] =  1/(2*n)*np.sum((h-c)**2)
    risk[j] =  (tss-rss[j])/(tss)
    j +=1
ax3.plot(np.linspace(0,20,snrTimes),risk)


#Gradient Descent
regTypes = ["None","Lasso","Ridge","Elastic"]
for type in regTypes:
    [lossVec,w] = gradDesc.gradDesc(c,n,k,d,lam,type)
    h = np.zeros(np.size(c))
    risk = np.zeros(snrTimes)
    rss = np.zeros(snrTimes)
    j=0
    for snr in np.linspace(0,20,snrTimes):
        i = 0
        for a in c:
                x = phasedWave.phasedWave(a,n,k,d)
                xPower = np.mean(x**2)
                noisePower = xPower/(10**(snr/10))
                x+=np.random.normal(0,noisePower,n)
                h[i] = np.sum(w*x)
                i+=1
        rss[j] =  1/(2*n)*np.sum((h-c)**2)
        risk[j] =  (tss-rss[j])/(tss)
        j +=1
    ax1.plot(lossVec)
    ax2.plot(np.linspace(0,20,snrTimes),risk)
ax1.legend(regTypes)
ax2.legend(regTypes)
ax3.legend(["OLS","GLS"])

ax1.set_title("Empirical Risk versus Iteration")
ax2.set_title("R2 versus SNR Gradient Descent")
ax3.set_title("R2 versus SNR Least Squares")

ax1.set_ylabel("Empirical Risk")
ax2.set_ylabel("R2")
ax3.set_ylabel("R2")


ax1.set_xlabel("Iteration")
ax2.set_xlabel("SNR (dB)")
ax3.set_xlabel("SNR (dB)")


fig1.savefig("riskGrad.png")
fig2.savefig("r2Grad.png")
fig3.savefig("r2OLS.png")
plt.show()