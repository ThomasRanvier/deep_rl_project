import matplotlib.pyplot as plt
import torch
from atari import Atari

N_EPISODES = 1
MEAN_K = 10
DISPLAY = False
DYNAMIC = False
SAVE_GIF = True

# New lr0000625
# 57
#MODEL_PATH = '../../backup_models/norm_dbl_duel_g99_bs32_lr0000625/policy_net_lr6.25e-05_bs32_tu2500_it4500000_g0.99_ed0.2777777777777778_c1350000.pt'
# 50 régulier
#MODEL_PATH = '../../backup_models/norm_dbl_duel_g99_bs32_lr0000625/policy_net_lr6.25e-05_bs32_tu2500_it4500000_g0.99_ed0.2777777777777778_c1400000.pt'
# 46, puis 38
#MODEL_PATH = '../../backup_models/norm_dbl_duel_g99_bs32_lr0000625/policy_net_lr6.25e-05_bs32_tu2500_it4500000_g0.99_ed0.2777777777777778_c1500000.pt'
# 44
#MODEL_PATH = '../../backup_models/norm_dbl_duel_g99_bs32_lr0000625/policy_net_lr6.25e-05_bs32_tu2500_it4500000_g0.99_ed0.2777777777777778_c1550000.pt'
# 85 régulier !!!!!!!!!!!!!!!
#MODEL_PATH = '../../backup_models/norm_dbl_duel_g99_bs32_lr0000625/policy_net_lr6.25e-05_bs32_tu2500_it4500000_g0.99_ed0.2777777777777778_c1600000.pt'
# 43
#MODEL_PATH = '../../backup_models/norm_dbl_duel_g99_bs32_lr0000625/policy_net_lr6.25e-05_bs32_tu2500_it4500000_g0.99_ed0.2777777777777778_c1750000.pt'
# 35
#MODEL_PATH = '../../backup_models/norm_dbl_duel_g99_bs32_lr0000625/policy_net_lr6.25e-05_bs32_tu2500_it4500000_g0.99_ed0.2777777777777778_c1950000.pt'
# 64
#MODEL_PATH = '../../backup_models/norm_dbl_duel_g99_bs32_lr0000625/policy_net_lr6.25e-05_bs32_tu2500_it4500000_g0.99_ed0.2777777777777778_c2000000.pt'
# 39
#MODEL_PATH = '../../backup_models/norm_dbl_duel_g99_bs32_lr0000625/policy_net_lr6.25e-05_bs32_tu2500_it4500000_g0.99_ed0.2777777777777778_c2050000.pt'
# 46
#MODEL_PATH = '../../backup_models/norm_dbl_duel_g99_bs32_lr0000625/policy_net_lr6.25e-05_bs32_tu2500_it4500000_g0.99_ed0.2777777777777778_c2450000.pt'
# 46
#MODEL_PATH = '../../backup_models/norm_dbl_duel_g99_bs32_lr0000625/policy_net_lr6.25e-05_bs32_tu2500_it4500000_g0.99_ed0.2777777777777778_c3100000.pt'


# New lr00001
# 23
#MODEL_PATH = '../../backup_models/norm_dbl_duel_g99_bs32_lr00001/policy_net_lr1e-05_bs32_tu2500_it4500000_g0.99_ed0.2777777777777778_c2500000.pt'

# No reward clip bs32 g99
MODEL_PATH = '../../backup_models/norewardclip_norm_dbl_duel_g99_bs32_lr0000625/policy_net_lr6.25e-05_bs32_tu2500_it4500000_g0.99_ed0.2777777777777778_c1350000.pt'

def run_episode(net, env):
    env.reset()
    episode_reward = 0
    terminal = False
    terminal_life_loss = False
    while not terminal:
        if DISPLAY:
            env.render(mode='human')
        if not terminal_life_loss:
            output = net(env.get_state())
            action_index = int(torch.argmax(output)) + 1
        else:
            action_index = 1
        _, reward, terminal, terminal_life_loss = env.step(action_index)
        episode_reward += reward
    return episode_reward

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_net = torch.load(MODEL_PATH).double()
    policy_net.eval()
    env = Atari(device, save_gif=SAVE_GIF)
    ep = []
    rew = []
    ep_mean = []
    rew_mean = []
    iterations = 0
    reward_sum = 0
    total = 0
    if DYNAMIC:
        plt.ion()
    fig, ax = plt.subplots()
    for episode in range(N_EPISODES):
        episode_reward = run_episode(policy_net, env)
        reward_sum += episode_reward
        total += episode_reward
        rew.append(episode_reward)
        ep.append(episode + 1)
        if episode % MEAN_K == MEAN_K - 1:
            print('Episode', episode + 1)
            ep_mean.append(episode + 1)
            rew_mean.append(reward_sum / MEAN_K)
            reward_sum = 0
            if DYNAMIC:
                ax.set_xlabel('Episodes')
                ax.set_ylabel('Reward', color='red')
                ax.plot(ep, rew, color='orange', alpha=.35)
                ax.plot(ep_mean, rew_mean, color='red')
                ax.tick_params(axis='y', labelcolor='red')
    print('Reward mean:', total / N_EPISODES)
    if DYNAMIC:
        plt.ioff()
    else:
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Reward', color='red')
        ax.plot(ep, rew, color='orange', alpha=.35)
        ax.plot(ep_mean, rew_mean, color='red')
        ax.tick_params(axis='y', labelcolor='red')
    plt.show()
