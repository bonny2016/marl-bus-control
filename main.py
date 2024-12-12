import argparse
import os
from sim import Sim_Engine as Sim_Engine
from sim import util as U
import numpy as np
import copy
from random import seed
import torch

parser = argparse.ArgumentParser(description='param')
parser.add_argument("--seed", type=int, default=1)  # random seed
parser.add_argument("--model", type=str, default='TD3_Distill')  # caac  ddpg maddpg
parser.add_argument("--data", type=str, default='A_0_1')  # used data prefix
parser.add_argument("--para_flag", type=str, default='A_0_1')  # stored parameter prefix
parser.add_argument("--episode", type=int, default=200)  # training episode
parser.add_argument("--overtake", type=int, default=0)  # overtake=0: not allow overtaking
parser.add_argument("--arr_hold", type=int, default=1)  # arr_hold=1: determine holding once bus arriving bus stop
parser.add_argument("--train", type=int, default=1)  # train=1: training phase
parser.add_argument("--restore", type=int, default=0)  # restore=1: restore the model
parser.add_argument("--share_scale", type=int, default=0)  # restore=1: restore the model
parser.add_argument("--n_students", type=int, default=4)  # restore=1: restore the model
parser.add_argument("--all", type=int,
                    default=1)  # all=0 for considering only forward/backward buses; all=1 for all buses
parser.add_argument("--vis", type=int, default=0)  # vis=1 to visualize bus trajectory in test phase
parser.add_argument("--weight", type=int, default=2)  # weight for action penalty
parser.add_argument("--control", type=int,
                    default=2)  # 0 for no control;  1 for FH; 2 for RL (ddpg, maddpg)

args = parser.parse_args()
print(args)

if args.model == 'CAAC':
    from model.CAAC import Agent
if args.model == 'TD3_Distill':
    from model.TD3_Distill import Agent
if args.model == 'TD3_Distill_2':
    from model.TD3_Distill_2 import Agent
if args.model == 'ddpg':
    from model.DDPG import Agent
if args.model == 'maddpg':
    from model.MADDPG import Agent


def load_student(state_dim):

    student_agent = Agent(state_dim=state_dim, name='', seed=args.seed)
    try:
        student_agent.load(str(args.para_flag) + str('_') + str(args.share_scale) + str('_') + str(args.weight) + str(
            '_') + str(args.model) + str('_'))
    except:
        print("no model saved yet")
    return student_agent


def train(args):
    stop_list, pax_num = U.getStopList(args.data)
    print('Stops prepared, total bus stops: %g' % (len(stop_list)))
    bus_routes = U.getBusRoute(args.data)
    print('Bus routes prepared, total routes :%g' % (len(bus_routes)))
    stop_list_ = copy.deepcopy(stop_list)
    dispatch_times, bus_list, route_list, simulation_step = U.init_bus_list(bus_routes)
    print('init...')
    agents = {}
    agents_pool = []
    student_agent = None
    if args.model != '':
        eng = Sim_Engine.Engine(bus_list=bus_list, busstop_list=stop_list_, control_type=args.control,
                                dispatch_times=dispatch_times,
                                demand=0, simulation_step=simulation_step, route_list=route_list,
                                hold_once_arr=args.arr_hold, is_allow_overtake=args.overtake,
                                share_scale=args.share_scale,
                                all=args.all,
                                weight=args.weight)

        bus_list = eng.bus_list
        bus_stop_list = eng.busstop_list
        state_dim = 7
        agents_pool = [Agent(state_dim=state_dim, name='', seed=args.seed) for i in range(args.n_students)]
        # non share
        if args.share_scale == 0:
            for i, (k, v) in enumerate(eng.bus_list.items()):
                agents[k] = agents_pool[i % args.n_students]
        # share in route
        if args.share_scale == 1:
            agents = {}
            for k, v in eng.route_list.items():
                agent = Agent(state_dim=state_dim, name='', n_stops=len(bus_stop_list), buslist=bus_list,
                              seed=args.seed)
                agents[k] = agent
    for ep in range(args.episode):
        stop_list_ = copy.deepcopy(stop_list)
        bus_list_ = copy.deepcopy(bus_list)
        r = np.random.randint(10, 40) / 10.
        for _, stop in stop_list_.items():
            stop.set_rate(r)

        eng = Sim_Engine.Engine(bus_list=bus_list_, busstop_list=stop_list_, control_type=args.control,
                                dispatch_times=dispatch_times,
                                demand=0, simulation_step=simulation_step, route_list=route_list,
                                hold_once_arr=args.arr_hold, is_allow_overtake=args.overtake,
                                share_scale=args.share_scale, weight=args.weight)

        eng.agents = agents
        if ep > 0:
            if memory_copy != None:
                eng.GM = memory_copy
            for bid, b in eng.bus_list.items():
                if b.is_virtual == 0:
                    eng.GM.temp_memory[bid] = {'s': [], 'a': [], 'fp': [], 'r': []}

        if args.restore == 1 and args.control > 1:
            for k, v in agents.items():
                print(str(args.para_flag) + str('_') + str(args.share_scale) + str('_') + str(args.model) + str('_'))
                v.load(str(args.para_flag) + str('_') + str(args.share_scale) + str('_') + str(args.weight) + str(
                    '_') + str(args.model) + str('_'))

        Flag = True
        while Flag:
            Flag = eng.sim()
        ploss_log = []
        qloss_log = []
        if args.control > 1 and args.restore == 0:
            update_iter = 3
            for _ in range(update_iter):
                if ep >= 0:
                    ploss, qloss, trained = eng.learn()
                    if trained == True:
                        qloss_log.append(qloss)
                        ploss_log.append(ploss)

        if args.control > 1:
            memory_copy = eng.GM
        else:
            memory_copy = None

        log = eng.cal_statistic(
            name=str(args.para_flag) + str('_') + str(args.share_scale) + str('_') + str(args.model) + str('_'),
            train=args.train)
        abspath = os.path.abspath(os.path.dirname(__file__))
        name = abspath + "/log/" + args.data + args.model

        name += str(int(args.weight))
        if args.all == 1:
            name += 'all'

        avg_reward = U.train_result_track(eng=eng, ep=ep, qloss_log=qloss_log, ploss_log=ploss_log, log=log, name=name,
                                          seed=args.seed)
        # if avg_reward > -1.5 and avg_reward > best_reard and ep > 10:
        if avg_reward > -1 and ep > 10 and ep % 2 == 0:
            agent_to_save = agents_pool[np.random.randint(0, 4)]

            agent_to_save.save(
                str(args.para_flag) + str('_') + str(args.share_scale) + str('_') + str(args.weight) + str(
                    '_') + str(args.model) + str('_'))

            student_agent = load_student(state_dim=state_dim)
            student_agent = eng.distill(student_agent)
            student_agent.save(
                str(args.para_flag) + str('_') + str(args.share_scale) + str('_') + str(args.weight) + str(
                    '_') + str(args.model) + str('_'))
            # for k, v in eng.bus_list.items():
            for agent in agents_pool:
                agent.lr_decay()
                agent.load(
                    str(args.para_flag) + str('_') + str(args.share_scale) + str('_') + str(args.weight) + str(
                        '_') + str(args.model) + str('_'))
        eng.close()


def evaluate(args):
    stop_list, pax_num = U.getStopList(args.data)
    print('Stops prepared, total bus stops: %g' % (len(stop_list)))
    bus_routes = U.getBusRoute(args.data)
    print('Bus routes prepared, total routes :%g' % (len(bus_routes)))
    dispatch_times, bus_list, route_list, simulation_step = U.init_bus_list(bus_routes)
    agents = {}
    if args.model != '':
        stop_list_ = copy.deepcopy(stop_list)
        eng = Sim_Engine.Engine(bus_list=bus_list, busstop_list=stop_list_, control_type=args.control,
                                dispatch_times=dispatch_times,
                                demand=0, simulation_step=simulation_step, route_list=route_list,
                                hold_once_arr=args.arr_hold, is_allow_overtake=args.overtake,
                                share_scale=args.share_scale, weight=args.weight)

        bus_list = eng.bus_list
        # non share
        if args.share_scale == 0:
            for k, v in eng.bus_list.items():
                state_dim = 3
                agent = Agent(state_dim=state_dim, name=k, n_stops=len(eng.busstop_list), buslist=eng.bus_list,
                              seed=args.seed)
                agents[k] = agent

        # share in route
        if args.share_scale == 1:
            agents = {}
            for k, v in eng.route_list.items():
                state_dim = 3
                if args.model == 'accf':
                    agent = Agent(state_dim=state_dim, name='', n_stops=len(eng.busstop_list), buslist=eng.bus_list,
                                  seed=args.seed)
                else:
                    agent = Agent(state_dim=state_dim, name='', n_stops=len(eng.busstop_list), buslist=eng.bus_list,
                                  seed=args.seed)
                agents[k] = agent

    rs = [np.random.randint(10, 20) / 10. for _ in range(100)]

    for ep in range(100):
        stop_list_ = copy.deepcopy(stop_list)
        bus_list_ = copy.deepcopy(bus_list)
        r = rs[ep]
        if args.vis == 1:
            r = 1.5

        for _, stop in stop_list_.items():
            stop.set_rate(r)

        eng = Sim_Engine.Engine(bus_list=bus_list_, busstop_list=stop_list_, control_type=args.control,
                                dispatch_times=dispatch_times,
                                demand=0, simulation_step=simulation_step, route_list=route_list,
                                hold_once_arr=args.arr_hold, is_allow_overtake=args.overtake,
                                share_scale=args.share_scale, weight=args.weight)

        eng.agents = agents
        s = str(args.para_flag) + str('_') + str(args.share_scale) + str('_') + str(args.weight) + str('_') + str(
            args.model) + str('_')
        if args.restore == 1 and args.control > 1:
            for k, v in agents.items():
                v.load(s)

        Flag = True
        while Flag:
            Flag = eng.sim()

        log = eng.cal_statistic(
            name=str(args.para_flag) + str('_') + str(args.share_scale) + str('_') + str(args.model) + str('_'),
            train=args.train)

        abspath = os.path.abspath(os.path.dirname(__file__))
        if args.control == 0:
            name = abspath + "/logt/" + args.data + 'nc'

        if args.control == 2:
            name = abspath + "/logt/" + args.data + args.model

            name += str(int(args.weight))
            if args.all == 1:
                name += 'all'

        if args.control == 1:
            name = abspath + "/logt/" + args.data + 'fc'

        U.train_result_track(eng=eng, ep=ep, qloss_log=[0], ploss_log=[0], log=log, name=name,
                             seed=args.seed)

        if args.vis == 1 and args.data == 'SG0':
            if args.control == 0:
                name = abspath + "/vis/visnc/"
            if args.control == 1:
                name = abspath + "/vis/visfc/"
            if args.control == 2:
                name = abspath + "/vis/vis" + args.model + '/'
            try:
                os.makedirs(name)
            except:
                print(name, ' has existed')
            U.visualize_trajectory(engine=eng, name=name + '_' + str(args.data) + str('_'))
            break

        eng.close()


if __name__ == '__main__':
    seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.train == 1:
        train(args)

    else:
        evaluate(args)
