import random
import torch
import gymnasium as gym
import numpy as np
from torch import nn
from torch import optim

from dqn import DQN, ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# 固定随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # PyTorch 相关的种子设置
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

SEED = 42 # 或者其他你选择的数字
set_seed(SEED)

# 初始化环境
env = gym.make("LunarLander-v3", render_mode="human") 
# env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5)
state_dim = env.observation_space.shape[0]   
action_dim = env.action_space.n

# 初始化网络并移动到 GPU
main_net = DQN(state_dim, action_dim, hidden=128).to(device)
target_net = DQN(state_dim, action_dim, hidden=128).to(device)
target_net.load_state_dict(main_net.state_dict())

score_hist = []
lr = 1e-3
optimizer = optim.Adam(main_net.parameters(), lr)
buffer = ReplayBuffer(capacity=10000)

batch_size = 64
episodes = 200
total_steps = 0

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 20000

gamma = 0.99    # discount factor
when_update_target = 500
best_score = -float('inf')  # 保存最好模型


# 训练
for episode in range(episodes):
    # state, _ = env.reset(seed=SEED)
    state, _ = env.reset()
    episode_reward = 0          # gamma = 1
    done = False
    
    while not done:
        total_steps += 1
        # ε-greedy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                q_net = main_net(state_tensor)
                action = q_net.argmax().item()
        # step
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        # store experience
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        # epsilon decay
        epsilon = max(epsilon_min, 1.0 - total_steps / epsilon_decay)
        
        # Training (Learning)
        if len(buffer) >= batch_size:
            s, a, r, ns, d = buffer.sample(batch_size)

            # 移动 tensor 到 GPU
            s = torch.tensor(s, dtype=torch.float32, device=device)
            a = torch.tensor(a, dtype=torch.int64, device=device).unsqueeze(1)
            r = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
            ns = torch.tensor(ns, dtype=torch.float32, device=device)
            d = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)
        
            # TD Update
            q_values = main_net(s).gather(1, a)
            with torch.no_grad():
                next_q = target_net(ns).max(dim=1, keepdim=True)[0]
                # [0] Maximum Q value
                # [1] index(action)

                # Bellman equation
                target = r + gamma * next_q * (1 - d)           
            loss = nn.MSELoss()(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 更新 target net
        if total_steps % when_update_target == 0:
            target_net.load_state_dict(main_net.state_dict())
    
    score_hist.append(episode_reward)
    print(f"Episode {episode}, Reward = {episode_reward:.1f}, Epsilon = {epsilon:.3f}")
    
    if len(score_hist) >= 10:
        avg_score_recent = np.mean(score_hist[-10:])
        if avg_score_recent > best_score and avg_score_recent > 50:
            best_score = avg_score_recent
            torch.save(main_net.state_dict(), "best_dqn_lunarlander.pth")
            print(f"  --> [SAVE] 发现更好模型，当前10轮均分: {avg_score_recent:.1f}")

env.close()
torch.save(main_net.state_dict(), "dqn_lunarlander.pth")
print("训练完成，模型已保存：dqn_lunarlander.pth")




# save plt
import matplotlib.pyplot as plt

target_score = 200     # LunarLander 经典成功线
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

plt.title("LunarLander DQN Training Progress", fontsize=14)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("training_progress.png", dpi=300)
plt.show()

