import math
from typing import Tuple


class Monster:
    def __init__(self, name, min_dmg, max_dmg, hp, speed, attack, defense):
        self.name = name
        self.min_dmg = min_dmg
        self.max_dmg = max_dmg
        self.hp = hp
        self.speed = speed
        self.attack = attack
        self.defense = defense


class Troop:
    def __init__(self, monster, player_id, count):
        assert count > 0
        self.monster_hp = monster.hp
        self.initial_count = count
        self.initial_hp = count * self.monster_hp
        self.name = monster.name
        self.player_id = player_id
        self.dmg = (monster.min_dmg + monster.max_dmg) / 2
        self.speed = monster.speed
        self.attack = monster.attack
        self.defense = monster.defense
        self.current_count = count
        self.current_hp = self.initial_hp
        self.can_retaliate = True
        self.is_defending = False
        self.posX = -1
        self.posY = -1

    def __str__(self):
        return f"{self.name}(player={self.player_id}, count={self.current_count}, hp={self.current_hp}, loc={self.posX},{self.posY})"

    def new_turn(self):
        assert self.current_count > 0
        self.can_retaliate = True
        self.is_defending = False

    def damage_on_other_troop(self, other: 'Troop', retal) -> int:
        """"
        cf https://heroes.thelazy.net/index.php/Damage
        """""
        assert self.current_count > 0
        base_dmg = self.current_count * self.dmg
        other_defense = other.defense * (1.20 if other.is_defending else 1.0)
        AmD = self.attack - other_defense
        I1 = 0.05*AmD if AmD >= 0 else 0
        R1 = -0.025*AmD if AmD <= 0 else 0
        dmg = int(base_dmg*(1 + I1)*(1 - R1))
        if retal:
            dmg = int(dmg*0.5)
        return max(1, dmg)

    def take_damage(self, dmg) -> Tuple[bool, int]:
        assert self.current_count > 0
        self.current_hp = max(0, self.current_hp - dmg)
        old_count = self.current_count
        self.current_count = int(math.ceil(self.current_hp / self.monster_hp))
        if self.current_count == 0:
            return True, old_count - self.current_count
        else:
            return False, old_count - self.current_count
