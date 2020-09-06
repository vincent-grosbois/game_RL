import math
from monsters import *
from reward import *
from troop import *
from board import *


def run_action(board, troop, action_id, rewards, verbose=False):
    assert troop in board.troops
    if verbose:
        print(f"Selected action {action_id}")
    if action_id == 0:  # defend
        troop.is_defending = True
        if verbose:
            print(f"Troop {troop} is defending")
        return []
    elif action_id > 0 and action_id <= 25:  # just move
        old_posX = troop.posX
        old_posY = troop.posY
        old_pos = troop.posX*board.BOARD_SIZE + troop.posY
        posX, posY = (action_id - 1)//5, (action_id - 1) % 5
        troop.posX = posX
        troop.posY = posY
        assert board.content[old_pos] is troop
        board.content[old_pos] = None
        assert board.content[posX*board.BOARD_SIZE + posY] is None
        board.content[posX*board.BOARD_SIZE + posY] = troop
        if verbose:
            print(f"Troop {troop} is moving from ({old_posX}, {old_posY}) to ({posX}, {posY})")
        return []
    elif action_id > 25:  # move and attack
        old_posX = troop.posX
        old_posY = troop.posY
        old_pos = troop.posX * board.BOARD_SIZE + troop.posY
        posX, posY = (action_id - 1 - 25)//5, (action_id - 1 - 25) % 5
        troop.posX = posX
        troop.posY = posY
        assert board.content[old_pos] is troop
        board.content[old_pos] = None
        assert board.content[posX * board.BOARD_SIZE + posY] is None
        board.content[posX * board.BOARD_SIZE + posY] = troop
        enemy = enemy_in_range_at_pos(posX, posY, troop.player_id, board)
        assert enemy is not None
        if verbose:
            print(f"Troop {troop} is moving from ({old_posX}, {old_posY}) to ({posX}, {posY}). Attacks enemy {enemy} at ({enemy.posX}, {enemy.posY})")
        dmg = troop.damage_on_other_troop(enemy, retal = False)
        if verbose:
            print(f"{troop} deals {dmg} damages to {enemy}")
        rewards.add_reward(troop.player_id, dmg)
        killed_troops = []
        killed_enemy, nb_killed = enemy.take_damage(dmg)
        rewards.add_reward(troop.player_id, nb_killed*enemy.monster_hp)
        if verbose:
            print(f", now {enemy}")
        if killed_enemy:
            if verbose:
                print(f", and killed it")
            rewards.add_reward(troop.player_id, enemy.initial_hp)
            board.content[enemy.posX * board.BOARD_SIZE + enemy.posY] = None
            board.troops.remove(enemy)
            killed_troops.append(killed_enemy)
        elif enemy.can_retaliate:
            dmg_me = enemy.damage_on_other_troop(troop, retal=True)
            if verbose:
                print(f"\n{enemy} retaliates {dmg_me} damages to {troop}")
            enemy.can_retaliate = False
            rewards.add_reward(enemy.player_id, dmg_me)
            killed_me, nb_killed = troop.take_damage(dmg_me)
            rewards.add_reward(enemy.player_id, nb_killed*troop.monster_hp)
            if verbose:
                print(f", now {troop}")
            if killed_me:
                if verbose:
                    print(f", and killed it")
                rewards.add_reward(enemy.player_id, troop.initial_hp)
                board.content[troop.posX * board.BOARD_SIZE + troop.posY] = None
                board.troops.remove(troop)
                killed_troops.append(troop)
        return killed_troops


def run_turn(board: Board, rewards, hooks_map, verbose=False):
    rewards.new_turn()

    assert len(board.troops) > 0
    troops_sorted = [(t, True) for t in sorted(board.troops, key=lambda x: -x.speed)]

    for t, is_alive in troops_sorted:
        if not is_alive:
            continue

        if verbose:
            print(f"Turn of {t}")
        t.new_turn()
        allowed_actions = actions_for_troop(board, t)
        if verbose:
            print(allowed_actions)
        hooks = hooks_map[t]
        action = hooks.policy(board, t, allowed_actions)
        assert action in allowed_actions
        killed_troops = run_action(board, t, action, rewards, verbose=verbose)

        for k in killed_troops:
            for idx, (tr, _) in enumerate(troops_sorted):
                if tr is k:
                    troops_sorted[idx] = (tr, False)
                    break

        result = 0  # battle continues

        if len(board.troops) == 0:
            rewards.add_reward(0, board.rewards_per_player[1])
            rewards.add_reward(1, board.rewards_per_player[0])
            if verbose:
                print("Draw")
            result = 1
        elif all([t.player_id == 0 for t in board.troops]):
            rewards.add_reward(0, board.rewards_per_player[1])
            if verbose:
                print("Player 0 won")
            result = 2
        elif all([t.player_id == 1 for t in board.troops]):
            rewards.add_reward(1, board.rewards_per_player[0])
            if verbose:
                print("Player 1 won")
            result = 3

        if verbose:
            print(rewards)

        next_allowed_actions = actions_for_troop(board, t) if result == 0 else [0]
        hooks.capture_new_state(board, next_allowed_actions, t, rewards.get_reward(t.player_id)[0])

        if result != 0:
            return result

    return result