import simpy
import random
import pygame

import numpy as np


class Assembly(object):
    def __init__(self, num_of_processes, len_of_queue, inbound_panel_blocks=None, display_env=False):
        self.env, self.model = self._modeling(num_of_processes)
        self.num_of_processes = num_of_processes
        self.len_of_queue = len_of_queue
        self.action_space = len_of_queue
        self.observation_space = num_of_processes * len_of_queue + num_of_processes
        self.inbound_panel_blocks = inbound_panel_blocks
        self.inbound_panel_blocks_clone = self.inbound_panel_blocks[:]
        self.process = []
        self.queue = []
        self.stage = 0
        self.empty = -1
        if display_env:
            display = AssemblyDisplay()
            display.game_loop_from_space()

    def step(self, action):
        done = False
        block = self.queue.pop(action)
        self.model[0].put(block)
        while True:
            self.env.step()
            if self.model[0].server_num - self.model[0].server.count(None) > 0:
                break
        if len(self.queue) == 0:
            done = True
            self.env.run()
        next_state = self._get_state()
        reward = self._calculate_reward()
        return next_state, reward, done

    def reset(self):
        self.inbound_panel_blocks = self.inbound_panel_blocks_clone[:]
        random.shuffle(self.inbound_panel_blocks)
        self.stage = 0
        return self._get_state()

    def _get_state(self):
        state = np.full(self.observation_space, self.empty)
        if len(self.inbound_panel_blocks) > 0 and len(self.queue) < self.len_of_queue:
            num = min(len(self.inbound_panel_blocks), self.len_of_queue - len(self.queue))
            for i in range(num):
                panel_block = self.inbound_panel_blocks.pop()
                self.queue.append(panel_block)
        state[:self.num_of_processes] = self.process
        
        for i, process in enumerate(self.process):
            pass
        for i, panel_block in enumerate(self.queue):
            pass
        return state

    def _calculate_reward(self):
        pass  # 이다원이 후처리 파일 만들고 나면 만들 것

    def _modeling(self, num_of_processes):
        from environment.SimComponents import Process, Sink
        env = simpy.Environment()
        model = []
        processes = {}
        for i in range(num_of_processes + 1):
            model.append(Process(env, 'Process{0}'.format(i), processes, 1, qlimit=1))
            processes['Process{0}'.format(i)] = model[-1]
            if i == num_of_processes:
                model.append(Sink(env, 'Sink'))
                processes['Sink'] = model[-1]
        return env, model


class AssemblyDisplay(object):
    def __init__(self):
        pass

    def game_loop_from_space(self):
        pass