import simpy
import pandas as pd
import random
import pygame
import os

import numpy as np

from environment.postprocessing import Utilization


class Assembly(object):
    def __init__(self, num_of_processes, len_of_queue, event_path, inbound_panel_blocks=None, display_env=False):
        self.num_of_processes = num_of_processes
        self.len_of_queue = len_of_queue
        self.event_path = event_path
        self.inbound_panel_blocks = inbound_panel_blocks
        self.inbound_panel_blocks_clone = self.inbound_panel_blocks[:]
        self.a_size = len_of_queue
        self.s_size = num_of_processes * len_of_queue + num_of_processes
        self.env, self.model, self.monitor = self._modeling(self.num_of_processes, self.event_path)
        self.queue = []
        self.time = 0.0
        self.num_of_blocks_put = 0
        self.stage = 0
        if display_env:
            display = AssemblyDisplay()
            display.game_loop_from_space()

    def step(self, action):
        done = False
        if action >= len(self.queue):
            reward = -1
        else:
            block = self.queue.pop(action)
            self.monitor.record(self.env.now, "Source", part_id=block.id, event="part_created")
            self.env.process(self.model['Process0'].put(block, 'Source', 0))
            self.monitor.record(self.env.now, "Source", part_id=block.id, event="part_transferred")
            self.num_of_blocks_put += 1
            while True:
                self.env.step()
                if self.model['Process0'].parts_sent - self.num_of_blocks_put == 0:
                    while self.env.peek() == self.env.now:
                        self.env.run(self.env.timeout(0))
                    break
            if len(self.queue) == 0:
                done = True
            reward = self._calculate_reward_by_lead_time()
            self.time = self.env.now
        next_state = self._get_state()
        if done:
            self.env.run()
        return next_state, reward, done

    def reset(self):
        self.env, self.model, self.monitor = self._modeling(self.num_of_processes, self.event_path)
        self.inbound_panel_blocks = self.inbound_panel_blocks_clone[:]
        for panel_block in self.inbound_panel_blocks:
            panel_block.step = 0
        random.shuffle(self.inbound_panel_blocks)
        self.num_of_blocks_put = 0
        self.stage = 0
        return self._get_state()

    def _get_state(self):
        # 전체 state 변수를 -1로 초기화
        state = np.full(self.s_size, -1.0)

        # queue에 블록을 할당
        if len(self.inbound_panel_blocks) > 0 and len(self.queue) < self.len_of_queue:
            num = min(len(self.inbound_panel_blocks), self.len_of_queue - len(self.queue))
            for i in range(num):
                panel_block = self.inbound_panel_blocks.pop()
                self.queue.append(panel_block)

        # 각 공정별 블록의 남은 작업 시간 정보
        remaining_working_time = []
        now = self.env.now  # 현재 시각
        for i in range(self.num_of_processes):
            process = self.model['Process{0}'.format(i)]  # modeling한 Process
            for server in process.server:
                if not server.part:
                    remaining_working_time.append(-1)
                    continue
                part_id, working_time = server.part.id, server.part.data[(server.part.step, 'process_time')]
                part_start_time = server.working_start
                remaining_working_time.append(working_time - (now - part_start_time))
        state[:self.num_of_processes] = remaining_working_time

        # queue에 있는 블록의 각 공정에서의 작업 시간 정보
        planned_working_time = []
        for panel_block in self.queue:  # queue에 있는 블록 정보
            working_time = panel_block.data[:, 'process_time']
            planned_working_time += list(working_time[:self.num_of_processes])

        state[self.num_of_processes:self.num_of_processes + len(planned_working_time)] = planned_working_time

        return state

    def _calculate_reward_by_lead_time(self):
        reward = 0
        event_tracer = pd.read_csv(self.event_path)
        block_completed = event_tracer[(event_tracer["EVENT"] == "completed")
                                       & (event_tracer["TIME"] > self.time)]
        block_completed = block_completed.reset_index(drop=True)
        block_created = event_tracer[event_tracer["EVENT"] == "part_created"]
        for i, row in block_completed.iterrows():
            start = (block_created["TIME"][block_created["PART"] == row["PART"]]).tolist()[0]
            finish = row["TIME"]
            lead_time = finish - start
            block = event_tracer[(event_tracer['PART'] == row['PART']) &
                                 ((event_tracer["EVENT"] == "work_start") | (event_tracer["EVENT"] == "work_finish"))]
            total_working_time = sum(block["TIME"].groupby(block["PROCESS"]).diff().dropna())
            reward += (total_working_time - lead_time)
        return reward

    def _calculate_reward_by_complete_blocks(self):
        event_tracer = pd.read_csv(self.event_path)
        block_completed = event_tracer[(event_tracer['TIME'] > self.time) & (event_tracer["EVENT"] == "completed")]
        num_of_block_completed = len(block_completed)
        return num_of_block_completed

    def _calculate_reward_by_delay(self):
        event_tracer = pd.read_csv(self.event_path)
        delay_start = event_tracer[(event_tracer['TIME'] > self.time) & (event_tracer["EVENT"] == "delay_start")]
        num_of_delay_start = len(delay_start)
        return - num_of_delay_start

    def _modeling(self, num_of_processes, event_path):
        from environment.SimComponents import Process, Sink, Monitor
        env = simpy.Environment()
        model = {}
        monitor = Monitor(event_path)
        for i in range(num_of_processes + 1):
            model['Process{0}'.format(i)] = Process(env, 'Process{0}'.format(i), 1, model, monitor, qlimit=1)
            if i == num_of_processes:
                model['Sink'] = Sink(env, 'Sink', monitor)
        return env, model, monitor


class AssemblyDisplay(object):
    def __init__(self):
        pass

    def game_loop_from_space(self):
        pass


if __name__ == '__main__':
    from environment.panelblock import *
    panel_blocks = import_panel_block_schedule('../environment/data/PBS_assy_sequence_gen_000.csv')
    num_of_processes = 7
    len_of_queue = 10
    event_path = './simulation_result'
    if not os.path.exists(event_path):
        os.makedirs(event_path)
    assembly = Assembly(num_of_processes, len_of_queue, event_path + '/event_PBS.csv', inbound_panel_blocks=panel_blocks)
    s = assembly.reset()
    t = 0
    r_cum = 0
    print("step 0 ------ reset")
    print(s)
    for i in range(70):
        s_next, r, d = assembly.step(0)
        r_cum += r
        print("step: {0} | parts_sent: {1} | parts_completed: {2} | reward: {3} | cumulative reward: {4} | time: {5}"
              .format(i, assembly.model['Process0'].parts_sent, assembly.model['Sink'].parts_rec, r, r_cum, t))
        s = s_next
        if d:
            break

    print(assembly.env.now)