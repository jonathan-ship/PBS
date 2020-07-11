import simpy
import random
import pygame

import numpy as np


class Assembly(object):
    def __init__(self, num_of_processes, len_of_queue, inbound_panel_blocks=None, display_env=False):
        self.env, self.model = self.DES_modeling(num_of_processes)
        self.num_of_processes = num_of_processes
        self.len_of_queue = len_of_queue
        self.action_space = len_of_queue
        self.observation_space = num_of_processes * len_of_queue + num_of_processes
        self.inbound_panel_blocks = inbound_panel_blocks
        self.inbound_panel_blocks_clone = self.inbound_panel_blocks[:]
        self.processes = []
        self.queue = []
        self.stage = 0
        self.empty = -1
        if display_env:
            display = AssemblyDisplay()
            display.game_loop_from_space()

    def DES_modeling(self, num_of_processes):
        from environment.SimComponents import Source, Sink, Process
        env = simpy.Environment()
        model = []
        Source = Source(env, 'Source', )
        Sink = Sink(env, 'Sink')
        model = [Source, Sink]
        return env, model

    def step(self, action):
        pass

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
        for i, process in enumerate(self.processes):

        for i, panel_block in enumerate(self.queue):

    def _calculate_reward(self):
        pass


class AssemblyDisplay(object):
    def __init__(self):
        pass

    def game_loop_from_space(self):
        pass