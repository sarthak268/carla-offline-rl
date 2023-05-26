import gym
import logging
# from d4rl.pointmaze import waypoint_controller
# from d4rl.pointmaze_bullet import bullet_maze
# from d4rl.pointmaze import maze_model
# from agents.navigation.controller import VehiclePIDController
from agents.navigation.local_planner import LocalPlanner
import d4rl.carla
import numpy as np
import pickle
import gzip
import h5py
import argparse
import time


def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'timeouts': [],
            'rewards': [],
            # 'infos/goal': [],
            # 'infos/qpos': [],
            # 'infos/qvel': [],
            }

# def append_data(data, s, a, tgt, done, timeout, robot):
#     data['observations'].append(s)
#     data['actions'].append(a)
#     data['rewards'].append(0.0)
#     data['terminals'].append(False)
#     data['timeouts'].append(False)
    # data['infos/goal'].append(tgt)
    # data['infos/goal_reached'].append(done)
    # data['infos/goal_timeout'].append(timeout)
    # data['infos/qpos'].append(robot.qpos.copy())
    # data['infos/qvel'].append(robot.qvel.copy())

def append_data(data, s, act, reward, done, timeout):
    data['observations'].append(s)
    data['actions'].append(act)
    data['rewards'].append(reward)
    data['terminals'].append(done)
    data['timeouts'].append(timeout)
    

def npify(data):
    for k in data:
        if k == 'terminals' or k == 'timeouts':
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render trajectories')
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    parser.add_argument('--env_name', type=str, default='maze2d-umaze-v1', help='Maze type')
    parser.add_argument('--num_samples', type=int, default=int(1e6), help='Num samples to collect')
    args = parser.parse_args()

    # env = gym.make('carla-lane-v0')
    env = gym.make('carla-town-ours-v0')

    vehicle = env.vehicle
    max_episode_steps = env._max_episode_steps

    planner = LocalPlanner(vehicle)
    planner._init_controller()

    if args.render:
        env.render('human')

    planner._compute_next_waypoints()
    s = env.reset()
    act = env.action_space.sample()
    timeout = False

    data = reset_data()
    ts = 0

    for _ in range(args.num_samples):

        control = planner.run_step() 
        # these are control values: VehicleControl(throttle=0.750000, steer=-0.100000, brake=0.000000, hand_brake=False, reverse=False, manual_gear_shift=False, gear=0)
        
        throttle = control.throttle
        steer = control.steer
        brake = control.brake
        act = (throttle, steer)

        # compute new waypoint is agent has reached the previous one
        if planner.done():
            planner._compute_next_waypoints()

        if args.noisy:
            act = act + np.random.randn(*act.shape)*0.5

        # act = np.clip(act, -1.0, 1.0)
        if ts >= max_episode_steps:
            timeout = True

        # ns, _, _, _ = env.step(act)
        ns, reward, done, info = env.step(act)    
        # We might want to add info also    
        
        append_data(data, s, act, reward, done, timeout)
        print (len(data['observations']))

        if len(data['observations']) % 10000 == 0:
            print(len(data['observations']))

        ts += 1
        if done:
            # env.set_target()
            planner._compute_next_waypoints()

            done = False
            ts = 0
        else:
            last_position = s[0:2]
            s = ns

        if args.render:
            env.render('human')

    
    if args.noisy:
        fname = 'noisy-carla.hdf5' 
    else:
        fname = 'carla.hdf5'
    dataset = h5py.File(fname, 'w')
    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')


if __name__ == "__main__":
    main()
