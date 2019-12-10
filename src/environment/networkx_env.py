import gym
from gym import spaces
import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
from environment.core import *
import sys
import copy
from collections import defaultdict
from PIL import Image


def get_services_position(state):
    services = np.where(state == ENTITY_AGENT)[0]  # in diagonal
    return services


def get_workload_position(state):
    services = np.where(state == ENTITY_WL)[0]  # in diagonal
    return services


class NetworkEnv(gym.Env):

    def __init__(self, graph, services, workloads, diameter):
        super(NetworkEnv, self).__init__()

        self.fig, self.ax = plt.subplots(figsize=(9.0, 6.0))
        left, bottom, width, height = self.ax.get_position().bounds
        self.right = left + width
        self.top = bottom + height

        self.G = graph
        self.pos = nx.kamada_kawai_layout(self.G)  # el layout podria ser una entrada?
        self.max_diameter = diameter
        self.matrix = np.array(nx.to_numpy_matrix(self.G))

        self.dim = len(self.G.nodes)
        self.shape_size = (self.dim * self.dim)
        self.shape = (self.dim, self.dim)

        self.services = services
        self.workloads = workloads
        self.n = None
        self.original_space = None

        self.observation_space = spaces.Box(low=-1.0, high=3.0, shape=self.matrix.shape, dtype=np.int)  # from 0 to 3
        self.action_space = np.zeros((self.dim, self.dim, 2), dtype=int)
        self.action_size = self.action_space.shape  # actions are independent in the trainer

    def reset(self):
        self.observation_space = self.matrix
        for wl in self.workloads:  # WL-AG order matter
            self.observation_space[wl.pos][wl.pos] = ENTITY_WL
        for agent in self.services:
            self.observation_space[agent.pos][agent.pos] = ENTITY_AGENT
        self.original_space = copy.copy(self.observation_space)
        self.n = len(self.services)
        return self.observation_space

    def get_utilization(self, new_space):
        """
            returns a dictionary with Key - Node and Value ->[ [User  Reward] ]
        :param new_space:
        :return: dictionary
        """
        requests = defaultdict(list)
        assitance = {}
        new_services = get_services_position(new_space)

        for wl in self.workloads:
            minpath = math.inf
            best_service = None
            for pos in new_services:
                path = nx.shortest_path(self.G, source=wl.pos, target=pos)
                if len(path) < minpath:
                    minpath = len(path)
                    best_service = pos
            if minpath == math.inf:
                assitance[wl.pos] = []
            else:
                distance = self.max_diameter - minpath + 1
                requests[best_service].append([wl.pos, distance])
                assitance[wl.pos] = [best_service, distance]

        # available_services = set(requests.keys())
        # total_services = set(new_services)
        # non_used=total_services.difference(available_services)
        # for nused in non_used:
        #     requests[nused] = []

        return requests, assitance

    def step(self, action):
        pos = get_services_position(self.observation_space)  # where the services are
        new_space = copy.copy(self.observation_space)
        new_service_created_by = defaultdict(list)
        for p in pos:
            act = action[p]
            s = self.observation_space[p]
            for i in range(len(act)):
                if p == i and act[i] == 0:
                    if self.original_space[i, i] == ENTITY_WL:
                        new_space[i, i] = self.original_space[i,i]
                    else:
                        new_space[i, i] = 0
                if act[i] == 1:
                    if s[i] >= 1:
                        new_space[i, i] = ENTITY_AGENT
                        new_service_created_by[i].append(p)

        # print("NEW SPACE")
        # print(new_space)

        old_traffic, old_assit = self.get_utilization(self.observation_space)
        new_traffic, new_assit = self.get_utilization(new_space)
        pos_new = get_services_position(new_space)
        reward_space = np.zeros(shape=self.action_size)

        # print("OLD traffic")
        # print(old_traffic)
        # print("OLD assit")
        # print(old_assit)
        # print("NEw traffic")
        # print(new_traffic)
        # print("NEw new_assit")
        # print(new_assit)
        # print("POS")
        # print(pos)
        # print("POS NEW")
        # print(pos_new)

        # TODO Check if active agent (pos_new)?
        # and do this:
        if len(pos_new)>=1:
            pos = set(pos)
            pos_new = set(pos_new)

            removed_services = pos.union(pos_new).difference(pos_new)
            new_services = pos.union(pos_new).difference(pos)

            for rs in removed_services:
                print("Removed service: ",rs)
                #el mismo agente decidio eliminarse por lo tanto: pos_agent = rs
                agent = rs
                if agent in old_traffic: #if the service dealed with requests
                    # si el agente trataba peticiones entonces
                    # analizamos el reward actual de los usuarios que trataba
                    # el reward es la aceleración de ambas mejoras
                    users = np.array(old_traffic[agent])[:, 0]
                    previous_rew = np.mean(np.array(old_traffic[agent])[:, 1])
                    current_rew = 0
                    # print("USERS: ",users)
                    for user in users:
                        current_rew += new_assit[user][1]
                    current_rew = current_rew / len(users)
                    inc_rew = ((current_rew/previous_rew)-1.0)*100.0
                    reward_space[agent,agent,0] = inc_rew #agent p, action:i, type:0 (act[i]==0)
                else:
                    reward_space[agent, agent, 0] =  200.0 #sino se utiliza es mejor borrarlo
                    reward_space[agent, agent, 1] = -200.0

            """
            Version 2. Se calcula el reward de los nuevos servicios en funcion de los usuarios al nodo actual
            """
            for ns in new_services:
                # saber quien hizo este servicio es más complicado
                # necesario crear una variable de control (aunque se tenga un poco más de overhead)
                # print("New service: ", ns)

                for agent in new_service_created_by[ns]:
                    if ns in new_traffic:
                        inc_rew = np.mean(np.array(new_traffic[ns])[:, 1]) * 100.0
                        reward_space[agent, ns, 1] = inc_rew
                    else:
                        # el nuevo servicio no tiene trafico
                        reward_space[agent, ns, 0] = 200.0  # era mejor no hacerlo
                        reward_space[agent, ns, 1] = -200.0  # es malo crearlo


            # """
            # Version 1. Se calcula el reward en funcion de la aceleraci´´n de los usuarios
            # """
            # for ns in new_services:
            #     # saber quien hizo este servicio es más complicado
            #     # necesario crear una variable de control (aunque se tenga un poco más de overhead)
            #     print("New service: ",ns)
            #     inside = False
            #     for agent in new_service_created_by[ns]: #solo uno o varios? #TODO el max?
            #         print ("\t created by agent: ",agent)
            #         if agent in old_traffic:
            #             users = np.array(old_traffic[agent])[:, 0]
            #             previous_rew = np.mean(np.array(old_traffic[agent])[:, 1])
            #             current_rew = 0
            #             print("USERS: ", users)
            #             for user in users:
            #                 current_rew += new_assit[user][1]
            #             current_rew = current_rew / len(users)
            #             inc_rew = ((current_rew / previous_rew) - 1.0) * 100.0
            #             reward_space[agent, ns, 1] = inc_rew  # agent p, action:i, type:0 (act[i]==0)
            #         else:
            #             #el agente no tenia usuarios
            #             # ns tiene usuarios ? han mejorado con respecto a su anterior ubicacion?
            #             if ns in new_traffic:
            #                 users = np.array(new_traffic[agent])[:, 0]
            #                 current_rew = 0
            #                 previous_rew = 0
            #                 for user in users:
            #                     current_rew += new_assit[user][1]
            #                     previous_rew += old_assit[user][1]
            #                 current_rew = (current_rew / len(users))
            #                 previous_rew = (previous_rew / len(users))
            #                 inc_rew = ((current_rew / previous_rew) - 1.0) * 100.0
            #                 reward_space[agent, ns, 1] = inc_rew
            #             else:
            #                 #el nuevo servicio no tiene trafico
            #                 reward_space[agent, ns, 0] = 200.0 #era mejor no hacerlo
            #                 reward_space[agent, ns, 1] = -200.0 #es malo crearlo

        # print(reward_space[0])
        # sys.exit()
        # for p in pos:  # for active agent
        #     act = action[p]
        #     s = self.observation_space[p]
        #     sn = new_space[p]
        #     for i in range(len(act)):
        #         # se elimino un servicio
        #         if act[i] == 0 and p == i:
        #             # en el nuevo trafico los servicios que atendian han mejorado?
        #             agent = p
        #             users = np.array(old_traffic[agent])[:,0]
        #             previous_rew = np.mean(np.array(old_traffic[agent])[:,1])
        #             current_rew = 0
        #             for user in users:
        #                 current_rew += new_assit[user][1]
        #             current_rew = current_rew / len(users)
        #             inc_rew = ((current_rew/previous_rew)-1.0)*100.0
        #             print("Previous rew: ",previous_rew)
        #             print("Current  rew: ",current_rew)
        #             print("inc_rew: ",inc_rew)
        #             reward_space[p,i,0] = inc_rew #agent p, action:i, type:0 (act[i]==0)
        #
        #         # se creo uno nuevo
        #         if act[i] == 1 and new_space[i, i] == ENTITY_AGENT:
        #             #se creo uno nuevo


        ## PREVIOUS NOT WORKING VERSION
        # for i,act in enumerate(action[pos]):
        #     index_obs = pos[i]
        #     state_row = self.observation_space[index_obs]
        #     for j, a in enumerate(act):
        #         if a == 1:
        #             if state_row[j] == 1: # There is a link among act[j] and state_row[j]  added a new agent in [j,j]
        #
        #                 if j in new_traffic: # is used?
        #                     reward_acc = np.sum(np.array(new_traffic[j])[:,1])
        #                     reward_space[index_obs][j][a] = reward_acc
        #                     # reward_space[index_obs, j, a] = reward_acc
        #                 else:
        #                     reward_space[index_obs][j][a]  = -100
        #                     # reward_space[index_obs][j][0]  = 10 #v1
        #             elif state_row[j] == ENTITY_AGENT:
        #                 # se añadio un agente donde habia otro agente
        #                 # la acción no sirvio
        #                 # reward_space[index_obs, j, a] = -10
        #                 pass
        #             elif state_row[j] == 0:
        #                 # la accion no sirvio para nada
        #                 reward_space[index_obs, j, a] = -1
        #                 pass
        #         if a == 0:
        #             if state_row[j] == ENTITY_AGENT:
        #                 # se elimino una entidad en J
        #                 # if not j in new_traffic:
        #                 #     reward_space[index_obs, j, a] = 5
        #                 if (j in old_traffic):
        #                     # el servicio se elimino, con la eliminación se ha mejorado algo?
        #                     old_users_from_that_service = np.array(old_traffic[j])[:,0]
        #                     # print("OLD_USERS:",old_users_from_that_service)
        #
        #                     improve = 0
        #                     for user in old_users_from_that_service:
        #                         old_ser_rew = old_assit[user]
        #                         new_ser_rew = new_assit[user]
        #                         if new_ser_rew == []:
        #                             improve = -100
        #                             break
        #                         improve = new_ser_rew[1]-old_ser_rew[1]
        #                         print("IMPROVE - >>>>: ",improve)
        #
        #                     reward_space[index_obs][j][a]  = improve
        #                 pass
        #
        #             elif state_row[j] == 0:
        #                 pass

        done = np.array_equal(np.where(self.original_space == ENTITY_WL),
                              np.where(new_space == ENTITY_BOTH))  # podia usar self.workloads.pos

        if len(pos_new) == 0:
            done = True

        self.observation_space = new_space

        return new_space, reward_space, done, None

    def get_acc_reward(self, rew, pos_services):
        r = rew.reshape(1, self.dim * self.dim * 2)
        idx = r[0].argsort()[-len(pos_services):][::-1]
        return np.sum(r[0][idx])

    def render(self, **kwargs):
        label = kwargs["label"]
        path = kwargs["path"]

        nx.draw(self.G, self.pos, with_labels=True, node_size=200, node_color="#1260A0", edge_color="gray",
                node_shape="o", font_size=7, font_color="white", ax=self.ax)

        for wl in self.workloads:
            self.ax.scatter(self.pos[wl.pos][0], self.pos[wl.pos][1], s=750.0, marker='H', color="red")

        for pos in get_services_position(self.observation_space):
            self.ax.scatter(self.pos[pos][0], self.pos[pos][1], s=500.0, marker='o', color="orange")

        self.ax.text(self.right, self.top, label, horizontalalignment='right', verticalalignment='top',
                     transform=self.ax.transAxes, fontsize=16)
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        pil_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        pil_image.save(path)
        self.ax.clear()

        return True
