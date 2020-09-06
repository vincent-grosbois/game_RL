class Reward:
    def __init__(self, gamma):
        self.rewardPlayer0 = [0]
        self.rewardPlayer1 = [0]
        self.totalRewardP0 = 0
        self.totalRewardP1 = 0
        self.gamma = gamma

    def add_reward(self, player_id, value):
        if player_id == 0:
            self.rewardPlayer0[0] += value
            self.totalRewardP0 += value
        elif player_id == 1:
            self.rewardPlayer1[0] += value
            self.totalRewardP1 += value
        else:
            assert False

    def get_reward(self, player_id):
        if player_id == 0:
            return [(a-b) for (a, b) in zip(self.rewardPlayer0, self.rewardPlayer1)]
        elif player_id == 1:
            return [(b-a) for (a, b) in zip(self.rewardPlayer0, self.rewardPlayer1)]
        else:
            assert False

    def new_turn(self):
        self.rewardPlayer0.insert(0, 0)
        self.rewardPlayer1.insert(0, 0)
        del self.rewardPlayer0[10:]
        del self.rewardPlayer1[10:]
        self.totalRewardP0 *= self.gamma
        self.totalRewardP1 *= self.gamma

    def __str__(self):
        return f"Player0: {self.rewardPlayer0}, Player1: {self.rewardPlayer1}\nTotal player0: {self.totalRewardP0}, Total player1: {self.totalRewardP1}"
