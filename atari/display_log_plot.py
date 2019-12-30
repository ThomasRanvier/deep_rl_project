import matplotlib.pyplot as plt

filepath = 'logs/big_run.out'
f = open(filepath)
line = f.readline()
i = 1
rew = []
eps = []
los = []
ep = []
while line:
    if i > 2:
        words = line.split()
        rew.append(float(words[7]))
        eps.append(float(words[10]))
        los.append(min(.4, float(words[13])))
        ep.append(i - 2)
    line = f.readline()
    i += 1

fig, (ax1, ax3) = plt.subplots(2)
#ax2 = ax1.twinx()

ax1.set_xlabel('episodes')
ax1.set_ylabel('loss', color='red')
ax1.plot(ep, los, color='red')
ax1.tick_params(axis='y', labelcolor='red')
#ax2.set_ylabel('reward per episode', color='blue')
#ax2.plot(ep, rew, color='blue')
#ax2.tick_params(axis='y', labelcolor='blue')
ax3.set_ylabel('epsilon', color='orange')
ax3.plot(ep, eps, color='orange')
plt.show()
