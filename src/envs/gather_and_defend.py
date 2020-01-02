from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
import math
from smac.env.multiagentenv import MultiAgentEnv
import random
import numpy as np


class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class Pos:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Unit:
    def __init__(self, x, y, health_max, n_resources):
        self.pos = Pos(x, y)
        self.health_max = health_max
        self.health = health_max
        self.resources_loaded = np.array([False for _ in range(n_resources)])
        self.loaded = False
        self.resources_gathered = np.array([0.0 for _ in range(n_resources)])
        self.attack_num = 0.0

        self.info={}
        self.info.update({"pick_{}".format(i):0 for i in range(n_resources)})
        self.info.update({"drop_{}".format(i):0 for i in range(n_resources)})
        self.info.update({"attack":0})

class Resource:
    def __init__(self, x, y):
        self.pos = Pos(x, y)


class Building:
    def __init__(self, x, y, health, n_resources):
        self.pos = Pos(x, y)
        self.health = health
        self.max_health = health

        self.resources_amount = [0. for _ in range(n_resources)]


resources_pos = [Pos(1, 5),
                 Pos(5, 1),
                 Pos(1, 1)]


class GatherDefendEnv(MultiAgentEnv):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """

    def __init__(
            self,
            n_agents=10,
            n_enemies=2,
            episode_limit=200,
            move_amount=1,
            continuing_episode=False,
            obs_all_health=True,
            obs_enemy_health=True,
            obs_own_health=True,
            obs_last_action=False,
            obs_pathing_grid=False,
            obs_terrain_height=False,
            obs_instead_of_state=False,
            obs_timestep_number=False,
            state_last_action=True,
            state_timestep_number=False,
            reward_sparse=False,
            reward_only_positive=True,
            reward_death_value=0.5,
            reward_win=5,
            reward_defeat=0.01,
            reward_integrate=20,
            reward_pick=2,
            reward_drop=2,
            reward_negative_scale=0.5,
            reward_scale=True,
            reward_scale_rate=40,
            debug=False,
            use_p_pick=False,
            use_p_attack=False,
            sight_range=9,
            shoot_range=1,
            map_x=20,
            map_y=20,
            base_x=10,
            base_y=10,
            agent_health=10,
            enemy_health=10,
            agent_attack=5,
            enemy_attack=2,
            base_health=150,
            n_resources=2,
            seed=None
    ):
        # Map arguments
        self.sight_range = sight_range
        self.shoot_range = shoot_range

        self.n_agents = n_agents
        self.episode_limit = episode_limit
        self._move_amount = move_amount
        self.n_enemies = n_enemies
        self.n_resources = n_resources

        # Observations and state
        self.obs_own_health = obs_own_health
        self.obs_all_health = obs_all_health
        self.obs_enemy_health = obs_enemy_health
        self.obs_instead_of_state = obs_instead_of_state
        self.obs_last_action = obs_last_action
        self.obs_pathing_grid = obs_pathing_grid
        self.obs_terrain_height = obs_terrain_height
        self.state_last_action = state_last_action
        if self.obs_all_health:
            self.obs_own_health = True

        # Rewards args
        self.reward_sparse = reward_sparse
        self.reward_only_positive = reward_only_positive
        self.reward_negative_scale = reward_negative_scale
        self.reward_death_value = reward_death_value
        self.reward_integrate = reward_integrate
        self.reward_win = reward_win
        self.reward_defeat = reward_defeat
        self.reward_pick = reward_pick
        self.reward_drop = reward_drop
        self.reward_scale = reward_scale
        self.reward_scale_rate = reward_scale_rate

        # Other
        self.continuing_episode = continuing_episode
        # self._seed = seed
        self._seed = random.randint(0, 9999)
        np.random.seed(self._seed)
        self.debug = debug
        self.use_p_pick=use_p_pick
        self.use_p_attack=use_p_attack
        self.base_diff=2

        # Actions
        self.n_actions_no_attack = 6
        self.n_actions_move = 4
        self.n_actions_resources = 2 * n_resources
        self.n_actions_no_resources = self.n_actions_no_attack + self.n_enemies
        self.n_actions = self.n_actions_no_resources + self.n_actions_resources

        # Property
        self.agent_health = agent_health
        self.enemy_health = enemy_health
        self.agent_attack = agent_attack
        self.enemy_attack = enemy_attack
        self.base_health = base_health

        # Resources
        self.base_x = base_x
        self.base_y = base_y
        self.resources = dict()
        for resources_id in range(self.n_resources):
            resource_pos = resources_pos[resources_id]
            self.resources[resources_id] = Resource(resource_pos.x, resource_pos.y) # TODO: observe base, resources
        self.base = Building(self.base_x, self.base_y, self.base_health, self.n_resources) # TODO: Initialize
        self.integrated = 0
        self.kill_number = 0

        # Map info
        #max_kill = self.episode_limit // (self.enemy_health // self.agent_attack) * self.n_enemies
        max_kill = self.episode_limit*self.n_enemies
        max_integrate = self.episode_limit//((self.base_x + self.base_y)*(self.n_resources*(self.n_resources+1)/2))
        self.max_reward = (max_kill * (self.reward_death_value + self.enemy_health * self.reward_defeat)
                           + self.reward_win
                           + max_integrate * ((self.reward_pick + self.reward_drop)*(self.n_resources*(self.n_resources+1)/2) + self.reward_integrate))

        self.GATHER_MAX = self.episode_limit // (self.base_x + self.base_y) #per agent
        self.ATTACK_MAX = self.episode_limit # per agent


        self.agents = {}
        self.enemies = {}
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        # self._obs = None
        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0
        self.force_restarts = 0
        self.last_stats = None
        # self.death_tracker_ally = np.zeros(self.n_agents)
        # self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        self.map_x = map_x
        self.map_y = map_y
        self.reset()

    def reset(self):
        """Reset the environment. Required after each full episode.
        Returns initial observations and states.
        """
        self._episode_steps = 0
        self.reset_resources_and_base()
        self.kill_number = 0

        # Information kept for counting the reward
        # self.death_tracker_ally = np.zeros(self.n_agents)
        # self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_ally_units = None
        self.previous_enemy_units = None

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        # self._obs = self._controller.observe()
        self.init_units()

        return self.get_obs(), self.get_state()

    def reset_resources_and_base(self):
        for resources_id in range(self.n_resources):
            resource_pos = resources_pos[resources_id]
            self.resources[resources_id] = Resource(resource_pos.x, resource_pos.y)

        self.base = Building(self.base_x, self.base_y, self.base_health, self.n_resources)
        self.integrated = 0

    def init_units(self):
        self.agents = {}
        for agent_id in range(self.n_agents):
            self.agents[agent_id] = Unit(random.randint(self.base_x, self.base_x+self.base_diff), #self.base_x
                                         random.randint(self.base_y, self.base_y+self.base_diff), #self.base_y
                                         self.agent_health,
                                         self.n_resources)

        self.enemies = {}
        for enemy_id in range(self.n_enemies):
            self.enemies[enemy_id] = Unit(10,#random.randint(self.map_x-self.base_diff, self.map_x), #self.base_x + 2
                                          10,#random.randint(self.map_y-self.base_diff, self.map_y), #self.base_y + 2
                                          self.enemy_health,
                                          self.n_resources)

    def ally_step(self, actions):
        attack_reward = 0
        attack_value = [0 for _ in range(self.n_enemies)]
        for agent_id, action in enumerate(actions):
            avail_actions = self.get_avail_agent_actions(agent_id)
            if avail_actions[action] ==0:
                avail_actions = self.get_avail_agent_actions(agent_id)

            assert avail_actions[action] == 1, \
                "Agent {} cannot perform action {}".format(agent_id, action)

            unit = self.get_unit_by_id(agent_id)
            if action == 2:
                unit.pos.y += self._move_amount
            elif action == 3:
                unit.pos.y -= self._move_amount
            elif action == 4:
                unit.pos.x += self._move_amount
            elif action == 5:
                unit.pos.x -= self._move_amount
            elif self.n_actions_no_attack <= action < self.n_actions_no_attack + self.n_enemies:
                if not self.use_p_attack or np.random.binomial(1,1.0/(1+np.exp(-unit.attack_num)))==1:
                    target_id = action - self.n_actions_no_attack
                    attack_value[target_id] += self.agent_attack
                    unit.attack_num+=1
                    unit.info["attack"]+=1

            elif action >= self.n_actions_no_resources:
                res_i = (action - self.n_actions_no_resources) // 2
                gather_down = (action - self.n_actions_no_attack - self.n_enemies) % 2

                if gather_down: #unit.loaded==True
                    # try:
                    assert unit.resources_loaded[res_i], "Agent {} does not have resource {}".format(agent_id, res_i)
                    # except:
                    #     a=1
                    self.base.resources_amount[res_i] += 1
                    unit.resources_loaded[res_i] = False
                    unit.loaded = False
                    unit.info["drop_{}".format(res_i)]+=1
                    attack_reward += self.reward_drop
                else:
                    # try:
                    assert unit.loaded is False, "Agent {} is loaded when trying to gather resource {}".format(agent_id, res_i)
                    # except:
                    #     a=1
                    if not self.use_p_pick or np.random.binomial(1,1.0/(1+np.exp(-unit.resources_gathered[res_i])))==1:
                        unit.resources_loaded[res_i] = True
                        unit.loaded = True
                        unit.resources_gathered[res_i] +=1
                        unit.info["pick_{}".format(res_i)]+=1
                        attack_reward += self.reward_pick

        # Attack
        for enemy_id in range(self.n_enemies):
            if self.enemies[enemy_id].health - attack_value[enemy_id] <= 0:
                attack_reward += self.reward_death_value
                attack_reward += self.reward_defeat * self.enemies[enemy_id].health

                self.enemies[enemy_id] = Unit( 10,#random.randint(self.map_x-self.base_diff, self.map_x), #self.base_x + 2
                                               10,#random.randint(self.map_y-self.base_diff, self.map_y), #self.base_y + 2
                                               self.enemy_health,
                                               self.n_resources)
                self.kill_number += 1
            else:
                attack_reward += self.reward_defeat * attack_value[enemy_id]
                self.enemies[enemy_id].health -= attack_value[enemy_id]

        return attack_reward

    def enemy_step(self):
        game_end_code = None
        for enemy_id, enemy in self.enemies.items():
            if self.can_reach(enemy.pos, Pos(self.base.pos.x+self.base_diff,self.base.pos.y+self.base_diff)):
                self.base.health -= self.enemy_attack
            else:
                if enemy.pos.x > self.base_x+self.base_diff:
                    enemy.pos.x -= 1

                elif enemy.pos.y > self.base_y+self.base_diff:
                    enemy.pos.y -= 1

                #elif enemy.pos.x < self.base_x+self.base_diff:
                #    enemy.pos.x += 1

                #elif enemy.pos.y < self.base_y+self.base_diff:
                #    enemy.pos.y += 1

        if self.base.health <= 0:
            game_end_code = -1

        return game_end_code

    def update_units(self, actions):
        """Update units after an environment step.
        This function assumes that self._obs is up-to-date.
        """
        attack_reward = self.ally_step(actions)
        game_end_code = self.enemy_step()

        return attack_reward, game_end_code

    def base_integrate(self):
        number = 0
        while True:
            can_integrate = True
            for res_i in range(self.n_resources):
                if self.base.resources_amount[res_i] <= res_i:
                    can_integrate = False
                    break

            if can_integrate:
                for res_i in range(self.n_resources):
                    self.base.resources_amount[res_i] -= (res_i + 1)
                number += 1
                self.integrated += 1
                # print('!!!!!!!!!!!!!')
            else:
                break

        return number * self.reward_integrate

    def step(self, actions):
        """A single environment step. Returns reward, terminated, info."""
        actions = [int(a) for a in actions]

        self.last_action = np.eye(self.n_actions)[np.array(actions)]

        # Collect individual actions
        # self._obs = self._controller.observe()

        self._total_steps += 1
        self._episode_steps += 1

        # Update units
        reward, game_end_code = self.update_units(actions)
        # Update base
        resource_reward = self.base_integrate()
        reward += resource_reward

        terminated = False
        info = {"battle_won": False}

        if game_end_code is not None:
            # Battle is over
            terminated = True
            self.battles_game += 1
            if game_end_code == 1:
                self.battles_won += 1
                info["battle_won"] = True
                if not self.reward_sparse:
                    reward += self.reward_win
                else:
                    reward = 1
            elif game_end_code == -1:
                if not self.reward_sparse:
                    reward += self.reward_defeat
                else:
                    reward = -1

        elif self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True
            self.battles_won += 1
            info["battle_won"] = True
            if self.continuing_episode:
                info["episode_limit"] = True
            self.battles_game += 1

        if self.debug:
            positions=[]
            positions.append((self.base.health,))
            for agent_id in range(self.n_agents):
                unit = self.agents[agent_id]
                positions.append((agent_id,unit.pos.x, unit.pos.y, unit.resources_loaded))
            for enemy_id in range(self.n_enemies):
                unit = self.enemies[enemy_id]
                positions.append((enemy_id,unit.pos.x,unit.pos.y,unit.health))

            print("[ ",self._episode_steps, ", ",positions,"], ")

        if terminated:
            self._episode_count += 1
            info["integrated"] = self.integrated
            for resource_i in range(self.n_resources):
                info["remaining_{}".format(resource_i)] = self.base.resources_amount[resource_i]
            info["kill"] = self.kill_number

            if self.debug:
                for agent_id in range(self.n_agents):
                    unit=self.agents[agent_id]
                    print(unit.info)

        if self.reward_scale:
            reward /= self.max_reward / self.reward_scale_rate



        return reward, terminated, info

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    @staticmethod
    def distance(x1, y1, x2, y2):
        """Distance between two points."""
        return math.hypot(x2 - x1, y2 - y1)

    def unit_shoot_range(self, agent_id):
        """Returns the shooting range for an agent."""
        return self.shoot_range

    def unit_sight_range(self, agent_id):
        """Returns the sight range for an agent."""
        return self.sight_range

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        unit = self.get_unit_by_id(agent_id)

        nf_al = 4
        nf_en = 4

        if self.obs_all_health:
            nf_al += 1
            nf_en += 1

        if self.obs_enemy_health:
            nf_en += 1

        if self.obs_last_action:
            nf_al += self.n_actions

        nf_own = 0
        if self.obs_own_health:
            nf_own += 1
        nf_own += 1 + self.n_resources*2+1 #specialization



        move_feats_len = self.n_actions_move

        move_feats = np.zeros(move_feats_len, dtype=np.float32)
        enemy_feats = np.zeros((self.n_enemies, nf_en), dtype=np.float32)
        ally_feats = np.zeros((self.n_agents - 1, nf_al), dtype=np.float32)
        own_feats = np.zeros(nf_own, dtype=np.float32)
        resources_feats = np.zeros(2*self.n_resources, np.float32)
        base_feats = np.zeros(3 + self.n_resources + self.base.max_health+1, np.float32)

        if unit.health > 0:  # otherwise dead, return all zeros
            x = unit.pos.x
            y = unit.pos.y
            sight_range = self.unit_sight_range(agent_id)

            # Movement features
            avail_actions = self.get_avail_agent_actions(agent_id)
            for m in range(self.n_actions_move):
                move_feats[m] = avail_actions[m + 2]

            ind = self.n_actions_move

            # Enemy features
            for e_id, e_unit in self.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)

                if (
                        dist < sight_range and e_unit.health > 0
                ):  # visible and alive
                    # Sight range > shoot range
                    enemy_feats[e_id, 0] = avail_actions[
                        self.n_actions_no_attack + e_id
                        ]  # available
                    enemy_feats[e_id, 1] = dist / sight_range  # distance
                    enemy_feats[e_id, 2] = (e_x - x) / sight_range  # relative X
                    enemy_feats[e_id, 3] = (e_y - y) / sight_range  # relative Y

                    ind = 4
                    if self.obs_all_health or self.obs_enemy_health:
                        enemy_feats[e_id, ind] = (
                                e_unit.health / e_unit.health_max
                        )  # health
                        ind += 1

            # Ally features
            al_ids = [
                al_id for al_id in range(self.n_agents) if al_id != agent_id
            ]
            for i, al_id in enumerate(al_ids):

                al_unit = self.get_unit_by_id(al_id)
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                dist = self.distance(x, y, al_x, al_y)

                if (dist < sight_range and al_unit.health > 0):  # visible and alive
                    ally_feats[i, 0] = 1  # visible
                    ally_feats[i, 1] = dist / sight_range  # distance
                    ally_feats[i, 2] = (al_x - x) / sight_range  # relative X
                    ally_feats[i, 3] = (al_y - y) / sight_range  # relative Y

                    ind = 4
                    if self.obs_all_health:
                        ally_feats[i, ind] = (
                                al_unit.health / al_unit.health_max
                        )  # health
                        ind += 1

                    if self.obs_last_action:
                        ally_feats[i, ind:] = self.last_action[al_id]

            # Own features
            ind = 0
            if self.obs_own_health:
                own_feats[ind] = unit.health / unit.health_max
                ind += 1
            own_feats[ind] = float(unit.loaded)
            ind += 1
            for resource_i in range(self.n_resources):
                own_feats[ind] = float(unit.resources_loaded[resource_i]/self.GATHER_MAX)
                ind += 1
            for resource_i in range(self.n_resources):
                own_feats[ind] = float(unit.resources_gathered[resource_i]/self.GATHER_MAX)
                ind +=1
            own_feats[ind] = float(unit.attack_num/self.ATTACK_MAX)

        x = unit.pos.x
        y = unit.pos.y
        sight_range = self.unit_sight_range(agent_id)

        for res_i in range(self.n_resources):
            resources_feats[res_i*2] = (self.resources[res_i].pos.x - x) / sight_range
            resources_feats[res_i*2+1] = (self.resources[res_i].pos.y - y) / sight_range

        base_feats[0] = (self.base_x - x) / sight_range
        base_feats[1] = (self.base_y - y) / sight_range
        base_feats[2] = self.base.health / self.base.max_health
        for res_i in range(self.n_resources):
            base_feats[3 + res_i] = self.base.resources_amount[res_i] / self.episode_limit
        base_feats[3+self.n_resources+self.base.health] = 1.0

        agent_obs = np.concatenate(
            (
                move_feats.flatten(),
                enemy_feats.flatten(),
                ally_feats.flatten(),
                own_feats.flatten(),
                resources_feats.flatten(),
                base_feats.flatten()
            )
        )

        return agent_obs

    def get_obs(self):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_state(self):
        """Returns the global state.
        NOTE: This functon should not be used during decentralised execution.
        """
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
                np.float32
            )
            return obs_concat

        nf_al = 3 + 1 + self.n_resources*2+1 #specialization
        nf_en = 3

        ally_state = np.zeros((self.n_agents, nf_al))
        enemy_state = np.zeros((self.n_enemies, nf_en))

        center_x = self.map_x / 2
        center_y = self.map_y / 2

        for al_id, al_unit in self.agents.items():
            if al_unit.health > 0:
                x = al_unit.pos.x
                y = al_unit.pos.y

                ally_state[al_id, 0] = (al_unit.health / al_unit.health_max)  # health
                ally_state[al_id, 1] = (x - center_x) / self.map_x  # relative X
                ally_state[al_id, 2] = (y - center_y) / self.map_y  # relative Y
                ally_state[al_id, 3] = float(al_unit.loaded)
                ind = 4
                for resource_i in range(self.n_resources):
                    ally_state[al_id, ind] = float(al_unit.resources_loaded[resource_i]/self.GATHER_MAX)
                    ind += 1
                for resource_i in range(self.n_resources):
                    ally_state[al_id, ind] = float(al_unit.resources_gathered[resource_i]/self.GATHER_MAX)
                    ind +=1
                ally_state[al_id, ind] = float(al_unit.attack_num/self.ATTACK_MAX)

        for e_id, e_unit in self.enemies.items():
            if e_unit.health > 0:
                x = e_unit.pos.x
                y = e_unit.pos.y

                enemy_state[e_id, 0] = (e_unit.health / e_unit.health_max)  # health
                enemy_state[e_id, 1] = (x - center_x) / self.map_x  # relative X
                enemy_state[e_id, 2] = (y - center_y) / self.map_y  # relative Y

                ind = 3

        state = np.append(ally_state.flatten(), enemy_state.flatten())
        if self.state_last_action:
            state = np.append(state, self.last_action.flatten())

        for resource_i in range(self.n_resources):
            state = np.append(state, np.array([(self.resources[resource_i].pos.x-center_x) / self.map_x,
                                               (self.resources[resource_i].pos.y-center_y) / self.map_y]))

        base_health_onehot=[0.0 for _ in range(self.base.max_health+1)]
        base_health_onehot[self.base.health]=1.0
        state = np.append(state, np.array([(self.base_x - center_x) / self.map_x,
                                           (self.base_y - center_y) / self.map_y,
                                           self.base.health / self.base.max_health] +
                                          [ras / self.episode_limit for ras in self.base.resources_amount]+
                                          base_health_onehot
                                          ))
        state = state.astype(dtype=np.float32)

        return state

    def get_obs_size(self):
        """Returns the size of the observation."""
        nf_al = 4
        nf_en = 4

        if self.obs_all_health:
            nf_al += 1
            nf_en += 1

        if self.obs_enemy_health:
            nf_en += 1

        own_feats = 1 + self.n_resources*2+1 #specialization
        if self.obs_own_health:
            own_feats += 1

        if self.obs_last_action:
            nf_al += self.n_actions

        move_feats = self.n_actions_move

        enemy_feats = self.n_enemies * nf_en
        ally_feats = (self.n_agents - 1) * nf_al

        base_feats = self.n_resources + 3  + self.base.max_health+1
        resources_feats = 2 * self.n_resources

        return move_feats + enemy_feats + ally_feats + own_feats + base_feats + resources_feats

    def get_state_size(self):
        """Returns the size of the global state."""
        if self.obs_instead_of_state:
            return self.get_obs_size() * self.n_agents

        nf_al = 3 + 1 + self.n_resources*2+1 #specialization
        nf_en = 3

        enemy_state = self.n_enemies * nf_en
        ally_state = self.n_agents * nf_al

        size = enemy_state + ally_state

        if self.state_last_action:
            size += self.n_agents * self.n_actions

        size += 3 + self.n_resources + 2 * self.n_resources + self.base.max_health+1

        return size

    def check_bounds(self, x, y):
        """Whether a point is within the map bounds."""
        return (1 <= x <= self.map_x and 1 <= y <= self.map_y)

    def can_move(self, unit, direction):
        """Whether a unit can move in a given direction."""
        m = self._move_amount

        if direction == Direction.NORTH:
            x, y = int(unit.pos.x), int(unit.pos.y + m)
        elif direction == Direction.SOUTH:
            x, y = int(unit.pos.x), int(unit.pos.y - m)
        elif direction == Direction.EAST:
            x, y = int(unit.pos.x + m), int(unit.pos.y)
        else:
            x, y = int(unit.pos.x - m), int(unit.pos.y)

        if self.check_bounds(x, y):
            return True

        return False

    def can_reach(self, pos1, pos2):
        #return ((abs(pos1.x - pos2.x) <= 1) and (abs(pos1.y - pos2.y) <= 1))
        return (abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y) <= 1)

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        unit = self.get_unit_by_id(agent_id)
        if unit.health > 0:
            # cannot choose no-op when alive
            avail_actions = [0] * self.n_actions

            # stop should be allowed
            avail_actions[1] = 1

            # see if we can move
            if self.can_move(unit, Direction.NORTH):
                avail_actions[2] = 1
            if self.can_move(unit, Direction.SOUTH):
                avail_actions[3] = 1
            if self.can_move(unit, Direction.EAST):
                avail_actions[4] = 1
            if self.can_move(unit, Direction.WEST):
                avail_actions[5] = 1

            # Can attack only alive units that are alive in the shooting range
            shoot_range = self.unit_shoot_range(agent_id)

            target_items = self.enemies.items()

            for t_id, t_unit in target_items:
                if t_unit.health > 0:
                    if self.can_reach(unit.pos, t_unit.pos):
                        avail_actions[t_id + self.n_actions_no_attack] = 1

            index = self.n_actions_no_attack + self.n_enemies

            for res_i in range(self.n_resources):
                if unit.resources_loaded[res_i] and unit.loaded:  # Put Down
                    if self.can_reach(unit.pos, self.base.pos):
                        avail_actions[index + res_i * 2 + 1] = 1

                if not unit.loaded:  # Gather
                    if self.can_reach(unit.pos, self.resources[res_i].pos):
                        avail_actions[index + res_i * 2] = 1

            return avail_actions
        else:
            # only no-op allowed
            return [1] + [0] * (self.n_actions - 1)

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def seed(self):
        """Returns the random seed used by the environment."""
        return self._seed

    def render(self):
        """Not implemented."""
        pass

    def save_replay(self):
        pass

    def close(self):
        pass

    def get_unit_by_id(self, a_id):
        """Get unit by ID."""
        return self.agents[a_id]

    def get_stats(self):
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "battles_draw": self.timeouts,
            "win_rate": self.battles_won / self.battles_game,
            "timeouts": self.timeouts,
            "restarts": self.force_restarts,
        }
        return stats

    def get_own_feature_size(self):
        return self.get_obs_size()


if __name__ == '__main__':
    env = GatherDefendEnv()
    env.reset()
    print(env.get_obs_size())
    print(env.get_state_size())

    for t in range(150):
        actions = []
        avail_actions = env.get_avail_actions()
        for agent_i in range(10):
            action = 0
            while True:
                action = random.randint(0, 11)

                if avail_actions[agent_i][action]:
                    break

            actions.append(action)

        reward, terminate, info = env.step(actions)

        print(">>>", t)
        print("state size:", env.get_state().shape)
        print("obs size:", env.get_obs_agent(0).shape)
        print("reward:", reward)
        print(env.base.pos.x, env.base.pos.y, env.resources[0].pos.x, env.resources[0].pos.y)
        for i in range(10):
            print(env.agents[i].pos.x, env.agents[i].pos.y, env.agents[i].resources_loaded)

        for i in range(2):
            print(env.enemies[i].pos.x, env.enemies[i].pos.y, env.enemies[i].health)

        print("base health:", env.base.health)
        print(env.base.resources_amount)

        print('\n\n\n\n')

        if terminate:
            break