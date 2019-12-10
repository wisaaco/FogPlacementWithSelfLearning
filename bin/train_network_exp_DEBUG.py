import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
from environment.networkx_env import NetworkEnv, get_services_position
from environment.core import *

import networkx as nx
import time
from DQNxAgent import DQNAgent
import pickle
from pathlib import Path
import subprocess
from matplotlib import pyplot as plt


experiments = [

    {"name": "debug",
     "generator": nx.barabasi_albert_graph,
     "episodes": 1,
     "conf": [{
         "name": "n10",
         "args": {"n": 5, "m": 2, "seed": 4},
         "services": [0,1],
         "wl": [3, 2],
         "render": True
     },
     ]}

    # {"name": "barabasi",
    #  "generator": nx.barabasi_albert_graph,
    #  "episodes": 1,
    #  "conf": [{
    #      "name": "n10",
    #      "args": {"n": 10, "m": 2, "seed": 10},
    #      "services": [0,3,2,6,4,9],
    #      "wl": [8, 9, 5],
    #      "render": False
    #  },
    #  ]}
]

if __name__ == "__main__":
    for exp in experiments:
        print("Preparing experiment: %s"%exp["name"])
        for conf in exp["conf"]:
            G = exp["generator"](**conf["args"])
            if exp["name"]=="grid": #relabilling node.id (x,y) -> xy
                mapping = {}
                dim = conf["dim"]
                for i in range(dim):
                    for j in range(dim):
                        mapping[(i,j)]=i*int(math.pow(10,dim//10))+j
                nx.relabel_nodes(G,mapping,False)
            pos_services=conf["services"]
            pos_workloads=conf["wl"]

            experiment_name = exp["name"]+"_"+conf["name"]
            diameter = nx.diameter(G)

            episodes = exp["episodes"]
            render = conf["render"]

            t_start = time.time()
            print("\tLoading environment")

            services = [Service() for i in range(len(pos_services))]
            for i, service in enumerate(services):
                service.name = 'service %d' % i
                service.pos = pos_services[i]

            workloads = [Workload() for i in range(len(pos_workloads))]
            for i, workload in enumerate(workloads):
                workload.name = 'workload %d' % i
                workload.pos = pos_workloads[i]

            env = NetworkEnv(G,services,workloads,diameter)
            env.seed(2019)
            obs = env.reset()
            # print("---Matrix--")
            # print(env.matrix)


            print("Starting Agent")
            trainer = DQNAgent(env.dim)
            max_episode_len = episodes
            batch_size = 5 # Warning: This parameter

            final_ep_mean_rewards,final_ep_std_rewards = [],[]
            mean_score = []
            episode_step = 0
            train_step = 0
            rmax = len(pos_workloads)*diameter

            print("\tPreparing Render")
            disk_dir = Path("../experiment1/results/%s"%experiment_name)
            disk_dir.mkdir(parents=True, exist_ok=True)
            disk_dir_render = Path("../experiment1/results/%s/images"%experiment_name)
            disk_dir_render.mkdir(parents=True, exist_ok=True)
            image_name = experiment_name
            image_id = 0

            print('\tStarting iterations...')
            path = str(disk_dir) + "/network.png"
            env.render(label="",path=path)

            for e in range(max_episode_len):
                time_reward = []
                episode_score = []

                state = env.reset()
                assert len(get_services_position(state))==len(env.services),"Number of services is incorrect in the init."
                print("EPISODE: ",e)
                print("-"*10)
                print("STATE\n",state)
                for step in range(diameter*len(env.workloads)):
                    if render:
                        path = str(disk_dir_render) + "/%s_%05d.png" % (image_name, image_id)
                        if len(time_reward)<1:
                            rshow= 0.0
                        else:
                            rshow = time_reward[step - 1]
                        env.render(label="e:%i-s:%i - r:%0.3f" % (e, step,rshow), path=path)
                        image_id += 1

                    # get action
                    action = trainer.act(obs)
                    print("ACTION\n",action)
                    new_obs, rew, done, info = env.step(action)
                    print("REW\n",rew)
                    trainer.remember(obs, action, rew, new_obs, done)
                    obs = new_obs
                    print("NEWSTATE\n",obs)

                    # Get the maximum rewards for all number of services in play
                    time_reward.append(env.get_acc_reward(rew,get_services_position(new_obs)))

                    if step%10==0:
                        # print("Step: episode: {}/{}, score: {}, e: {:.2}".format(e, max_episode_len, step, trainer.epsilon))
                        pass
                    if done:
                        # print("episode: {}/{}, score: {}, e: {:.2}, re: {}".format(e, max_episode_len, step, trainer.epsilon,np.mean(time_reward)))
                        episode_score.append(step)
                        break

                    if len(trainer.memory) > batch_size:
                        # print("Training model")
                        loss = trainer.replay(batch_size)

                final_ep_mean_rewards.append(np.mean(time_reward))
                final_ep_std_rewards.append(np.std(time_reward))
                mean_score.append(np.mean(np.array(episode_score)))


