class RewardCounter:
    def __init__(self, discount_factor=1):
        self.reset(discount_factor)

    def reset(self, discount_factor=1):
        self.rewards = []
        self.rewards_sum = 0.
        self.rewards_count = 0.
        self.last_reward_index = 0
        self.gamma = discount_factor

    def reward(self, rew):
        self.rewards.append(rew)
        if rew:
            self.rewards_count += 1
            self.rewards_sum += rew
            for i in reversed(range(self.last_reward_index + 1, len(self.rewards))):
                self.rewards[i - 1] = self.rewards[i] * self.gamma
            self.last_reward_index = len(self.rewards)

    def loss(self):
        loss = (self.rewards_count - self.rewards_sum) / (self.rewards_count * 2)
        return loss * loss
