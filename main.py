import torch
import math
from monsters import *
from reward import *
from troop import *
from board import *
from game import *

import random

from DQN1 import *


def make_batch(batch_size, model, epsilon, verbose=False, infer=False):

    game = 0
    turns = 0
    data = []
    nb_wins_p0 = 0
    nb_wins_p1 = 0
    while turns < batch_size or batch_size == -1:
        game = game + 1
        if verbose:
            print(f"#### Starting game {game} #####")
        troop1 = Troop(PikemenMonster, 0, 4)
        troop2 = Troop(PikemenMonster, 1, random.randint(3, 5))
        rewards = Reward(gamma=0.9)

        board = Board().placeTroopAt(troop1, 0, 0).placeTroopAt(troop2, random.randint(1,4), random.randint(1,4))

        policies = {}

        class HookRandomPolicy:
            def __init__(self, seed=1337):
                self.policy_done = False
                self.rnd = random.Random()#seed)
                self.storage = []
                self.current_data = None

            def policy(self, board, troop, actions):
                assert not self.policy_done
                self.policy_done = True
                state = board_to_state_extended(troop, board)
                action = actions[self.rnd.randrange(0, len(actions))]
                self.current_data = state, actions, action
                return action

            def capture_new_state(self, board, actions, troop, reward):
                assert self.policy_done
                assert self.current_data is not None
                self.policy_done = False
                new_state = board_to_state_extended(troop, board)
                new_actions = actions
                self.storage.append((self.current_data[0], self.current_data[1], self.current_data[2], new_state, new_actions, float(reward)))
                self.current_data = None

        class HookModelPolicy:
            def __init__(self, epsilon, verbose=False, seed=1337, infer=False):
                self.policy_done = False
                self.rnd = random.Random()#seed)
                self.storage = []
                self.current_data = None
                self.verbose = verbose
                self.epsilon = epsilon
                self.infer = infer

            def policy(self, board, troop, actions):
                assert not self.policy_done
                self.policy_done = True
                state = board_to_state_extended(troop, board)
                if epsilon > 0 and self.rnd.random() < epsilon:
                    action = actions[self.rnd.randrange(0, len(actions))]
                    if verbose:
                        print(f"Picked action {action} at random")
                else:
                    action = model_infer(model, state, actions, verbose=self.verbose, full_debug=self.infer)
                self.current_data = state, actions, action
                return action

            def capture_new_state(self, board, actions, troop, reward):
                assert self.policy_done
                assert self.current_data is not None
                self.policy_done = False
                new_state = board_to_state_extended(troop, board)
                new_actions = actions
                self.storage.append((self.current_data[0], self.current_data[1], self.current_data[2], new_state, new_actions, float(reward)))
                self.current_data = None

        policies[troop1] = HookModelPolicy(epsilon, verbose=verbose, infer=infer)
        policies[troop2] = HookRandomPolicy()

        turn_id = 0

        while True:
            if verbose:
                print(f"#### Turn {turn_id}, {turns} ####")
            turn_id += 1
            turns += 1
            run_result = run_turn(board, rewards, policies, verbose=verbose)
            if run_result != 0 or turns == batch_size:
                data.extend(policies[troop1].storage)
                if run_result == 2:
                    nb_wins_p0 += 1
                if run_result == 3:
                    nb_wins_p1 += 1
                break

        if batch_size ==-1:
            break

    states = torch.stack([d[0] for d in data], 0)

    def make_action_mask(actions):
        nb_actions = 51
        a = torch.zeros(nb_actions)
        a.scatter_(0, torch.tensor(actions), torch.ones(nb_actions))
        return a

    actions_available = torch.stack([make_action_mask(d[1]) for d in data], 0)

    actions_taken = torch.tensor([d[2] for d in data]).view([-1, 1])

    next_states = torch.stack([d[3] for d in data], 0)
    next_actions_available = torch.stack([make_action_mask(d[4]) for d in data], 0)

    rewards = torch.tensor([d[5] for d in data]).view([-1, 1])

    print(f"P0 wins: {nb_wins_p0}.  P1 wins: {nb_wins_p1}")

    return states, actions_available, actions_taken, next_states, next_actions_available, rewards


def model_infer(model, state, actions_available, verbose = False, full_debug=False):
    with torch.no_grad():
        state2 = torch.unsqueeze(state, 0)
        result = model.forward(state2).squeeze()
        mask = torch.zeros([51]).scatter(0, torch.tensor(actions_available), 1)

        result_masked = result * mask
        max_val, max_action = result_masked.max(0)
        max_action = max_action.item()
        if max_action not in actions_available:
            max_action = 0

        if full_debug:
            print(f"all rewards {result_masked}")
        if verbose:
            print(f"Choosing action {max_action} with reward {max_val}")
        return max_action

if __name__ == '__main__':
    dqn1 = DQN1(9, 51)
    print(dqn1)

    batch_size = 100
    gamma = 0.9
    N_epochs = 200

    for epoch in range(N_epochs):
        print(epoch)

        epsilon = min(max(0.05, 1.0/(epoch+1)), 0.5)
        print(f"epsilon = {epsilon}")

        states, actions_available, actions_taken, next_states, next_actions_available, rewards = \
            make_batch(batch_size, dqn1, epsilon, verbose=False)

        with torch.no_grad():
            all_next_Q_star = (dqn1.forward(next_states) * next_actions_available)
            next_Q_star = torch.max(all_next_Q_star, 1)[0].view([-1, 1])
            labels = next_Q_star*gamma + rewards

        my_Q_values = dqn1.forward(states).gather(1, actions_taken)

        loss = torch.pow(my_Q_values-labels, 2).sum()
        print(loss)

        optimizer = optim.RMSprop(dqn1.parameters())
        optimizer.zero_grad()
        loss.backward()
        for param in dqn1.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        #my_Q_values = dqn1.forward(states).gather(1, actions_taken)
        #loss = (my_Q_values - labels).pow(2).sum()
        #print(loss)

    make_batch(-1, dqn1, epsilon=0, verbose=True, infer=True)
    make_batch(-1, dqn1, epsilon=0, verbose=True, infer=True)
    make_batch(-1, dqn1, epsilon=0, verbose=True, infer=True)
    for param in dqn1.parameters():
        continue
        print(param.data)




