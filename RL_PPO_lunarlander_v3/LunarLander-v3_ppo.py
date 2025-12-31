# PPOAgent
#  ├── ActorNet      → πθ(a|s)
#  ├── CriticNet     → V(s)
#  ├── rollout()     → 采样 episode
#  ├── compute_adv() → 计算 advantage
#  ├── update()      → PPO clipped loss
#  └── train()

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.distributions import Categorical      # 提供的一个离散概率分布对象
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# DQN 的 Q 网络 = actor + critic 的合体
                # 能更好地处理状态空间中的微小变化


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Actor 网络（策略 πθ）
        # 学的是：π(a|s)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            # nn.ReLU(),
            nn.Mish(),  # 平滑过渡，通常收敛更快，但计算开销更大
            nn.Linear(128, action_dim)
            # nn.Softmax(dim=-1)        -> logits 原始输出
        )

        
        # Critic 网络（V(s)）
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def act(self, state):
        logits = self.actor(state)
        dist = Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        value = self.critic(state).squeeze(-1)

        return action, log_prob, value

    def evaluate(self, state, action):
        logits = self.actor(state)
        dist = Categorical(logits=logits)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        value = self.critic(state).squeeze(-1)

        return log_prob, entropy, value



# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim):    
        self.model = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.clip_eps = 0.2

        self.gae_lambda = 0.95  # 经过大量实验得出的“黄金分割点”

        # Entropy Bonus（防止策略塌缩）
        self.entropy_coef = 0.01
        self.value_coef = 0.5

        self.rollout_steps = 2048 # 依然是经验主义得到的“黄金标准”
        self.ppo_epochs = 10
        self.batch_size = 128

    # Step-based
    # rollout 期间：用 tensor，不用 .item()
    def collect_rollout(self, env):
        states, actions, log_probs = [], [], []
        rewards, dones, values = [], [], []

        state, _ = env.reset()

        for _ in range(self.rollout_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)

            with torch.no_grad():
                action, log_prob, value = self.model.act(state_tensor)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            states.append(state_tensor)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(torch.tensor(reward, dtype=torch.float32))
            dones.append(torch.tensor(done, dtype=torch.float32))
            values.append(value) 


            state = next_state
            if done:
                state, _ = env.reset()
            
        # ?????????????????????
        with torch.no_grad():
            next_state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            next_value = self.model.critic(next_state_tensor).squeeze()

        
        values.append(next_value)   # ⭐ 这是 V(s_T)
        return states, actions, log_probs, rewards, dones, values


    # GAE（Generalized Advantage Estimation）
    # A_t = r_t + γ V(s_{t+1}) - V(s_t)
    def compute_gae(self, rewards, values, dones):
        # 统一转 tensor + device
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones   = torch.tensor(dones,   dtype=torch.float32).to(device)
        values  = torch.tensor(values,  dtype=torch.float32).to(device)

        advantages = torch.zeros_like(rewards, device=device)
        gae = 0.0

        # for r, v, nv, d in zip(rewards, values, next_values, dones):
        #     td_target = r + self.gamma * nv * (1 - d)
        #     advantages.append(td_target - v)

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t] # 几何分布首项
            # 无穷步优势估计器  -> Monte Carlo 估计
            delta = rewards[t] + self.gamma * values[t+1] * mask - values[t]    # values[t+1]越界问题
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            
            
        returns = advantages + values[:-1]
        return advantages, returns


    def update(self, states, actions, old_log_probs, advantages, returns):      
        # states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        # actions = torch.tensor(np.array(actions)).to(device)
        # old_log_probs = torch.stack(old_log_probs).detach().to(device)
        # returns = returns.to(device)   # 已经是 device 上的 Tensor
        
        # 标准化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        dataset_size = len(states)
        idx = torch.randperm(dataset_size)      # 生成一个从 0 到 n-1 的随机打乱的整数序列

        for _ in range(self.ppo_epochs):

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                minibatch_idx = idx[start:end]

                minibatch_states = torch.stack([states[i] for i in minibatch_idx])
                minibatch_actions = torch.stack([actions[i] for i in minibatch_idx])
                minibatch_old_log_probs = torch.stack([old_log_probs[i] for i in minibatch_idx])
                minibatch_returns = returns[minibatch_idx]
                minibatch_advantages = advantages[minibatch_idx]

                # 更新 Actor & Critic
                # probs = self.actor(states)
                # dist = Categorical(probs)
               
                new_minibatch_log_probs, entropy, values = self.model.evaluate(minibatch_states, minibatch_actions)
                
                # 替换 KL 散度
                ratio = torch.exp(new_minibatch_log_probs - minibatch_old_log_probs)
                surr1 = ratio * minibatch_advantages
                # 限制 ratio (新旧策略概率比) 在 [0.8, 1.2] 之间 
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * minibatch_advantages   # clamp() 强行截断
                # Maximize J(θ) = Minimize -J(θ)
                policy_loss = -torch.min(surr1, surr2).mean()     # 优化器（如 Adam）默认是用来“最小化”一个目标的，而 PPO 的目标是“最大化”回报
                
                values_loss = (values - minibatch_returns).pow(2).mean()
                entropy_loss = entropy.mean()

                loss = (
                    policy_loss 
                        + values_loss * self.value_coef
                        - entropy_loss * self.entropy_coef
                    )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()




# On-policy（同策略） 和 Off-policy（异策略） 的区别是 PPO 和 DQN 最核心、最本质的差异
# 因此这里用 steps 替换 episodes（否则模型不稳定）
# step-based PPO:
#   rollout_steps = 2048
#   └── 中间可能跑完很多 episode

# episode-based logging:
#   每当 done == True
#   └── 记录一个 episode_return (float)


# for iteration:
#     rollout N steps using current policy
#     compute GAE + returns
#     for epoch in K:
#         for minibatch:
#             PPO update


# 训练
def train():
    # 初始化环境
    env = gym.make("LunarLander-v3")  # , render_mode="human"
    # env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5)
    state_dim = env.observation_space.shape[0]   
    action_dim = env.action_space.n
    
    ppoAgent = PPOAgent(state_dim, action_dim)
    
    iterations = 1000

    score_hist = []
    best_score = -np.inf
    episode_reward = 0.0

    for iteration in range(iterations):
        states, actions, log_probs, rewards, dones, values = ppoAgent.collect_rollout(env)
        
        # ⭐ episode logging（关键）
        for r, d in zip(rewards, dones):
            episode_reward += r
            if d:
                score_hist.append(episode_reward)
                episode_reward = 0.0


        # 统计 reward
        advantages, returns = ppoAgent.compute_gae(rewards, values, dones)
       
        # update
        ppoAgent.update(states, actions, log_probs, advantages, returns)
        
        # ===== 打印 & 保存 =====
        if len(score_hist) >= 10:
            avg_score = np.mean(score_hist[-10:])

            if avg_score > best_score:
                best_score = avg_score
                torch.save(ppoAgent.model.state_dict(), "best_ppo_lunarlander.pth")

        if iteration % 10 == 0 and len(score_hist) > 0:
            print(
                f"Iter {iteration:4d} | "
                f"Last score {score_hist[-1]:7.1f} | "
                f"Avg(10) {np.mean(score_hist[-10:]):7.1f}"
            )

    env.close()
    torch.save(ppoAgent.model.state_dict(), "ppo_lunarlander.pth")
    print("训练完成，模型已保存：ppo_lunarlander.pth")


    # plt ==========================================================
    target_score = 200  
    window_size = 100

    # 滑动平均
    avg_scores = [
        np.mean(score_hist[max(0, i - window_size + 1): i + 1])
        for i in range(len(score_hist))
    ]

    plt.figure(figsize=(12, 6))

    plt.plot(
        score_hist,
        alpha=0.4,
        linewidth=0.8,
        label="Episode Reward"
    )

    plt.plot(
        avg_scores,
        linewidth=2,
        label=f"Moving Average ({window_size})"
    )

    plt.axhline(
        y=target_score,
        linestyle="--",
        linewidth=1,
        label="Target Score (200)"
    )

    plt.title("LunarLander PPO Training Progress", fontsize=14)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig("training_progress.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    train()

    
# 因为 T=2048
# 导致 每次iteration  2048 / 400 ≈ 5 个 episode
# 所以 1000 iteration × 4–5 episode ≈ 4000–5000 episode


#                 定义                            类比 (以学生学习为例)
# Step          一次动作交互                     做出一道题
# Episode       一次完整的任务(从开始到结束)      完成一张考卷
# Iteration     一次采样并训练的完整周期          一个学期 (先上课搜集知识，再期末考试总结)
# Epoch         对同一批数据的重复训练次数        复习 (这本卷子做一遍不够，反复看 10 遍)