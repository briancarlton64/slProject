import numpy as np
import matplotlib.pyplot as plt
from phasedWave import phasedWave

# Developed from our solutions to the NN regression homework problems

def kNN(data,toClassify,numNeighbors):
    for tup in data:
        tup[len(tup)-1] = np.linalg.norm(toClassify-tup[0:len(tup)-2])

    data = sorted(data, key=lambda tup: tup[len(tup)-1])

    numerator = 0

    denominator = 0

    for i in range(0,numNeighbors):
        c = data[i][len(data[i])-2]
        w = 1/data[i][len(data[i])-1]

        numerator += w*c

        denominator += w

    

    return numerator/denominator

n = 30
l = 1
d = l/2
k = 2*np.pi/l;
lam = 0.01
snrTimes = 100
c = np.random.uniform(-np.pi/2,np.pi/2,4*n);

xs = []
for a in c:
    x = phasedWave(a,n,k,d)
    x = np.append(x, [a])
    x = np.append(x, [0])
    xs.append(x)

h2NN = np.zeros(np.size(c))
h3NN = np.zeros(np.size(c))
h10NN = np.zeros(np.size(c))
h30NN = np.zeros(np.size(c))
risk2NN = np.zeros(snrTimes)
risk3NN = np.zeros(snrTimes)
risk10NN = np.zeros(snrTimes)
risk30NN = np.zeros(snrTimes)
rss2NN = np.zeros(snrTimes)
rss3NN = np.zeros(snrTimes)
rss10NN = np.zeros(snrTimes)
rss30NN = np.zeros(snrTimes)
tss =  1/(2*n)*np.sum((np.mean(c)-c)**2)

np.random.seed(1)
j=0
for snr in np.linspace(0,20,snrTimes):
    i = 0
    for a in c:
        x = phasedWave(a,n,k,d)
        xPower = np.mean(x**2)
        noisePower = xPower/(10**(snr/10))
        x+=np.random.normal(0,noisePower,n)
        h2NN[i] = kNN(xs, x, 2)
        h3NN[i] = kNN(xs, x, 3)
        h10NN[i] = kNN(xs, x, 10)
        h30NN[i] = kNN(xs, x, 30)
        i+=1
        
    rss2NN[j] = 1/(2*n)*np.sum((h2NN-c)**2)
    rss3NN[j] = 1/(2*n)*np.sum((h3NN-c)**2)
    rss10NN[j] = 1/(2*n)*np.sum((h10NN-c)**2)
    rss30NN[j] = 1/(2*n)*np.sum((h30NN-c)**2)

    risk2NN[j] = (tss-rss2NN[j])/(tss)
    risk3NN[j] = (tss-rss3NN[j])/(tss)
    risk10NN[j] = (tss-rss10NN[j])/(tss)
    risk30NN[j] = (tss-rss30NN[j])/(tss)

    j +=1

plt.figure()
plt.title("Nearest Neighbors R2 versus SNR")
plt.ylabel("R2")
plt.xlabel("SNR (dB)")
plt.ylim(0.4, 1)
plt.plot(np.linspace(0,20,snrTimes),risk2NN)
plt.plot(np.linspace(0,20,snrTimes),risk3NN)
plt.plot(np.linspace(0,20,snrTimes),risk10NN)
plt.plot(np.linspace(0,20,snrTimes),risk30NN)
plt.legend(["2 NN", "3 NN", "10 NN", "30 NN"])
plt.savefig("r2NN.png")
plt.show()
