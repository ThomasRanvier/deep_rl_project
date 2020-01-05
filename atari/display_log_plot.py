import matplotlib.pyplot as plt
import sys

#filepath = 'logs/dueling.out'
#filepath = 'logs/b32_ed1._.1_.1_g.99.out'
filepath = sys.argv[1]
max_loss_value = float(sys.argv[2])
f = open(filepath)
line = f.readline()
i = 1
mean_k = 10
rew = []
rew_mean = []
rew_sum = 0
eps = []
los = []
los_mean = []
los_sum = 0
it = []
it_mean = []
while line:
    if i > 2:
        words = line.split()
        rew.append(float(words[7]))
        rew_sum += float(words[7])
        eps.append(float(words[10]))
        los.append(min(max_loss_value, float(words[13])))
        los_sum += min(max_loss_value, float(words[13]))
        iteration = words[4].split('/')[0]
        it.append(int(iteration))
        if (i - 3) % mean_k == mean_k - 1:
            rew_mean.append(rew_sum / mean_k)
            los_mean.append(los_sum / mean_k)
            it_mean.append(int(iteration))
            rew_sum = 0
            los_sum = 0
    line = f.readline()
    i += 1

fig, (ax1, ax2, ax3) = plt.subplots(3)

ax1.set_xlabel('episodes')
ax1.set_ylabel('loss', color='red')
ax1.plot(it, los, color='orange', alpha=.35)
ax1.plot(it_mean, los_mean, color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax2.set_ylabel('reward per episode', color='blue')
ax2.plot(it, rew, color='orange', alpha=.35)
ax2.plot(it_mean, rew_mean, color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax3.set_ylabel('epsilon', color='orange')
ax3.plot(it, eps, color='orange')
plt.show()
