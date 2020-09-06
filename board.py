from typing import List, Optional
from troop import *


class Board:
    def __init__(self):
        self.BOARD_SIZE = 5
        self.content = [None] * self.BOARD_SIZE ** 2
        self.troops: List[Troop] = []
        self.rewards_per_player = {0: 0, 1: 0}

    def placeTroopAt(self, troop, posX, posY):
        idx = posX*self.BOARD_SIZE + posY
        assert self.content[idx] is None
        self.content[idx] = troop
        troop.posX = posX
        troop.posY = posY
        self.troops.append(troop)
        self.rewards_per_player[troop.player_id] += troop.initial_hp
        return self


def enemy_in_range_at_pos(posX, posY, player_id, board) -> Optional[Troop]:
    for deltaX in [-1, 0, 1]:
        for deltaY in [-1, 0, 1]:
            if deltaX != 0 or deltaY != 0:
                posX_ = posX + deltaX
                posY_ = posY + deltaY
                if 0 <= posX_ < board.BOARD_SIZE and 0 <= posY_ < board.BOARD_SIZE:
                    if board.content[posX_*board.BOARD_SIZE + posY_] is not None:
                        if board.content[posX_ * board.BOARD_SIZE + posY_].player_id != player_id:
                            return board.content[posX_*board.BOARD_SIZE + posY_]
    return None


def actions_for_troop(board: Board, troop: Troop):
    assert troop in board.troops
    result = [0]  # 'defend' action
    for posX in range(board.BOARD_SIZE):
        for posY in range(board.BOARD_SIZE):
            if abs(posX - troop.posX) + abs(posY - troop.posY) <= troop.speed:
                if board.content[posX*board.BOARD_SIZE + posY] is None:  # empty tile, can move
                    result.append(posX*board.BOARD_SIZE + posY + 1)
                    if enemy_in_range_at_pos(posX, posY, troop.player_id, board) is not None:  # can attack
                        result.append(posX*board.BOARD_SIZE + posY + 1 + 25)
                elif posX == troop.posX and posY == troop.posY:  # don't move, just attack
                    if enemy_in_range_at_pos(posX, posY, troop.player_id, board) is not None:  # can attack
                        result.append(posX*board.BOARD_SIZE + posY + 1 + 25)

    return sorted(result)


import torch
def board_to_state_extended(troop: Troop, board: Board) -> torch.Tensor:
    my_id = troop.player_id
    total_cells = board.BOARD_SIZE ** 2
    state = []
    for i in range(total_cells):
        if board.content[i] is troop:
            state.append(1)  # this unit
        else:
            state.append(0)

    for i in range(total_cells):
        if board.content[i] is not None and board.content[i].player_id != my_id:
            state.append(1)  # enemy unit
        else:
            state.append(0)

    lambdas = [lambda x: x.current_count,
               lambda x: x.current_hp,
               lambda x: x.dmg,
               lambda x: x.attack,
               lambda x: x.defense,
               lambda x: 1 if x.can_retaliate else 0,
               lambda x: 1 if x.is_defending else 0]

    for f in lambdas:
        for i in range(total_cells):
            if board.content[i] is None:
                state.append(0)
            else:
                state.append(f(board.content[i]))

    s2 = torch.Tensor(state).view([-1, board.BOARD_SIZE, board.BOARD_SIZE])
    return s2
