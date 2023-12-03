# The code is referenced from this github address (https://github.com/seungeunrho/minimalRL/blob/master/ppo.py)

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import time
import os

# Environment(Cartpole) related parameters
NUM_ACTION = 2
episode_length = 100

# (Learning) Hyperparameters
learning_rate = 0.0005     # 0.0005
num_episodes  = int(1e4)   # 10000
fixed_batch = False
batch_size = 128

# PPO Hyperparameters
K_epoch  = 3
gamma    = 0.98  # discount factor
lmbda    = 0.95
eps_clip = 0.1
c1       = 1
c2       = 0.00001

# Interval
print_interval = 20
render_interval = 20
save_interval = 20

def entropy(prob):
    return -torch.sum(prob * torch.log(prob))

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.transition_lst = []

        self.fc1 = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU()
        )
        self.fc_pi = nn.Linear(256, NUM_ACTION)
        self.fc_v  = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, softmax_dim=0, return_value=True):
        output = {}        
        h = self.fc1(x)
        h_pi = self.fc_pi(h)
        prob = F.softmax(h_pi, dim=softmax_dim)
        output.update({"prob": prob})

        if return_value:
            v = self.fc_v(h)
            output.update({"v": v})

        return output

    def put_transition(self, transition):
        self.transition_lst.append(transition)

    def make_batch(self):
        if fixed_batch:
            if len(self.transition_lst) < batch_size:
                return

        s_lst, a_lst, r_lst, next_s_lst, prob_a_old_lst, done_lst = [], [], [], [], [], []
        transition_lst = self.transition_lst[:batch_size] if fixed_batch else self.transition_lst
        for transition in transition_lst:
            s_lst.append(transition['s'])
            a_lst.append([transition['a']])
            r_lst.append([transition['r']])
            next_s_lst.append(transition['next_s'])
            prob_a_old_lst.append([transition['prob_a_old']])
            done_mask = 0 if transition['done'] else 1  # 0 if done
            done_lst.append([done_mask])

        batch = {
            's': torch.tensor(s_lst, dtype=torch.float),
            'a': torch.tensor(a_lst),
            'r': torch.tensor(r_lst),
            'next_s': torch.tensor(next_s_lst, dtype=torch.float),
            'prob_a_old': torch.tensor(prob_a_old_lst),
            'done_mask': torch.tensor(done_lst, dtype=torch.float)
        }
        
        self.transition_lst = self.transition_lst[batch_size:] if fixed_batch else []
        return batch

    def train_net(self):
        batch = self.make_batch()
        if not batch:
            return
        
        s = batch['s']
        a = batch['a']
        r = batch['r']
        next_s = batch['next_s']
        prob_a_old = batch['prob_a_old']
        done_mask = batch['done_mask']

        for _ in range(K_epoch):
            td_target = r + gamma * self.forward(next_s)['v'] * done_mask  # 0 if done
            delta = td_target - self.forward(s)['v']
            delta = delta.detach().numpy()

            adv_lst = []
            adv = 0.0
            for delta_t in delta[::-1]:
                adv = gamma * lmbda * adv + delta_t[0]
                adv_lst.append([adv])
            adv_lst.reverse()
            adv = torch.tensor(adv_lst, dtype=torch.float)

            output = self.forward(s, softmax_dim=1)
            prob = output['prob']
            prob_a = torch.gather(prob, dim=1, index=a)
            ratio = prob_a / prob_a_old

            # maximize objective: E [CLIP obj - c1 * VF obj + c2 * entropy]
            # minimize loss     : E [- CLIP obj + c1 * VF obj - c2 * entropy]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * adv
            clip_obj = torch.min(surr1, surr2)
            vf_obj = F.mse_loss(self.forward(s)['v'], td_target.detach())
            loss = torch.mean(- clip_obj + c1 * vf_obj - c2 * entropy(prob))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
def main():
    env = gym.make('CartPole-v1')
    model = PPO()
    score = 0
    os.makedirs('output', exist_ok=True)

    for epi_i in range(num_episodes):
        if epi_i % render_interval == 0:
            env.close()
            env = gym.make('CartPole-v1', render_mode='human')

        s, _ = env.reset()
        done = False
        while not done:
            # Collect
            for t in range(episode_length):
                # Get action from state
                output = model.forward(torch.from_numpy(s).float(), return_value=False)
                a = Categorical(output['prob']).sample().item()
                # Get next state
                next_s, r, done, truncated, info = env.step(a)

                # Put transition
                transition = {
                    "s": s,
                    "a": a,
                    "r": r/100.0,
                    "next_s": next_s,
                    "prob_a_old": output['prob'][a].item(),
                    "done": done
                }
                model.put_transition(transition)
                s = next_s
                
                score += r
                if done:
                    break

            # Train
            model.train_net()

        if epi_i % print_interval == 0 and epi_i != 0:
            print(f"# of episode : {epi_i}, avg_score : {score/print_interval:.1f}")
            # print(f"avg collect time : {sum(collect_time) / len(collect_time)}")
            # print(f"avg train time : {sum(train_time) / len(train_time)}")
            score = 0

        if epi_i % save_interval == 0:
            torch.save(model, f'output/checkpoint_{epi_i}.pt')
            model = torch.load(f'output/checkpoint_{epi_i}.pt')

        if epi_i % render_interval == 0:
            env.close()
            env = gym.make('CartPole-v1')


    env.close()

if __name__ == '__main__':
    main()