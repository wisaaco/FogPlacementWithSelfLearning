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
     "episodes": 200,
     "conf": [{
         "name": "n10",
         "args": {"n": 5, "m": 2, "seed": 4},
         "services": [0, 1],
         "wl": [3, 2],
         "render": True
     }]},

    {"name": "random",
     "generator": nx.erdos_renyi_graph,
     "episodes": 300,
     "conf": [{
         "name": "n10",
         "args": {"n": 10, "p": 0.4, "seed": 10},
         "services": [0],
         "wl": [2, 9, 7],
         "render": True
     },
         {
             "name": "n30",
             "args": {"n": 30, "p": 0.4, "seed": 10},
             "services": [0],
             "wl": [8, 21, 19],
             "render": False
         },
         {
             "name": "n40",
             "args": {"n": 40, "p": 0.4, "seed": 10},
             "services": [0],
             "wl": [8, 19, 13],
             "render": False
         },

     ]},
    {"name": "barabasi",
     "generator": nx.barabasi_albert_graph,
     "episodes": 300,
     "conf": [{
         "name": "n10",
         "args": {"n": 10, "m": 2, "seed": 10},
         "services": [0],
         "wl": [8, 9, 5],
         "render": True
     },
         {
             "name": "n30",
             "args": {"n": 30, "m": 2, "seed": 10},
             "services": [0],
             "wl": [8, 21, 29],
             "render": False
         },
         {
             "name": "n40",
             "args": {"n": 40, "m": 2, "seed": 10},
             "services": [0],
             "wl": [8, 32, 29],
             "render": False
         },

     ]},

    {"name": "lobster",
     "generator": nx.random_lobster,
     "episodes": 300,
     "conf": [{
         "name": "n26",
         "args": {"n": 10, "p1": 0.8, "p2": 0.7, "seed": 10},
         "services": [1],
         "wl": [21, 23, 25],
         "render": True
     },

         {
             "name": "n79",
             "args": {"n": 30, "p1": 0.8, "p2": 0.7, "seed": 10},
             "services": [1],
             "wl": [54, 71, 78],
             "render": False
         },

     ]},
    {"name": "grid",
     "generator": nx.grid_graph,
     "episodes": 300,
     "conf": [
         {
             "name": "10x10",
             "args": {"dim": [10, 10]},
             "dim": 10,
             "services": [4],
             "wl": [74, 90, 99],
             "render": True
         },
         {
             "name": "20x20",
             "dim": 20,
             "args": {"dim": [20, 20]},
             "services": [9],
             "wl": [1409, 1900, 1919],
             "render": False
         },
     ]}
]

def make_experiment(G,render,episodes,experiment_name,diameter):
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
        # print("EPISODE: ",e)
        # print("-"*10)
        for step in range(diameter*len(env.workloads)):
            if render:
                path = str(disk_dir_render) + "/%s_%05d.png" % (image_name, image_id)
                if len(time_reward)<1:
                    rshow= 0.0
                else:
                    rshow = time_reward[step - 1]
                env.render(label="e:%i-s:%i - r:%0.3f/%i" % (e, step,rshow,rmax), path=path)
                image_id += 1

            # get action
            action = trainer.act(obs)
            new_obs, rew, done, info = env.step(action)
            trainer.remember(obs, action, rew, new_obs, done)
            obs = new_obs

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

    rew_file_name = str(disk_dir) +'/%s_mean_rewards.pkl'%experiment_name
    with open(rew_file_name, 'wb') as fp:
        pickle.dump(final_ep_mean_rewards, fp)

    rew_file_name =  str(disk_dir) +'/%s_std_rewards.pkl'%experiment_name
    with open(rew_file_name, 'wb') as fp:
        pickle.dump(final_ep_std_rewards, fp)

    score_file_name = str(disk_dir) +'/%s_score.pkl'%experiment_name
    with open(score_file_name, 'wb') as fp:
        pickle.dump(mean_score, fp)



    mean_score = np.array(mean_score)
    mean_reward = np.array(final_ep_mean_rewards)
    std_reward = np.array(final_ep_std_rewards)

    x = range(max_episode_len)
    fig, ax = plt.subplots(2, 1, constrained_layout=True)
    # fig.suptitle("24 -H-H- relu - DIM: 20")
    fig.suptitle(" %s "%experiment_name)
    ax[0].plot(x,mean_reward, 'k-')
    ax[0].fill_between(x,mean_reward - std_reward, mean_reward + std_reward)
    ax[0].set_xlabel("episode" , fontsize=12)
    ax[0].set_ylabel("mean episode reward", fontsize=12)


    ax[1].plot(x, mean_score, 'k-')
    ax[1].set_ylabel("mean episode score ", fontsize=12)

    plt.savefig(str(disk_dir)+'/graphic%s.pdf'%experiment_name, format='pdf', dpi=600)

    with open(str(disk_dir)+"/modelo_stats_%s.txt"%experiment_name,"w") as f:
        trainer.model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write("\n")
        f.write("Time: %f"%(time.time()-t_start))
    print("\tDone")

    trainer.save("MODEL_%s"%experiment_name)

if __name__ == "__main__":
    for exp in experiments:
        print("Preparing experiment: %s"%exp["name"])
        if exp["name"] != "debug":
            break
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
            # try:
            make_experiment(G, conf["render"], exp["episodes"], experiment_name,diameter)
            # except:
            #     print("Some problems with experiment: %s "%experiment_name)

