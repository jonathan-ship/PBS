import simpy
import pandas as pd
import random
import pygame
import os

import numpy as np

random.seed(42)


class Assembly(object):
    def __init__(self, num_of_processes, len_of_queue, inbound_panel_blocks=None, display_env=False):
        self.num_of_processes = num_of_processes
        self.len_of_queue = len_of_queue
        self.a_size = len_of_queue
        self.s_size = num_of_processes * len_of_queue + num_of_processes
        self.inbound_panel_blocks = inbound_panel_blocks
        self.inbound_panel_blocks_clone = self.inbound_panel_blocks[:]
        self.event_tracer = pd.DataFrame(columns=["TIME", "EVENT", "PART", "PROCESS", "SERVER_ID"])
        self.env, self.model = self._modeling(self.num_of_processes, self.event_tracer)
        self.queue = []
        self.time = 0.0
        self.num_of_blocks_put = 0
        self.stage = 0
        self.empty = -1.0
        if display_env:
            display = AssemblyDisplay()
            display.game_loop_from_space()

    def step(self, action):
        done = False
        if action >= len(self.queue):
            reward = -1
        else:
            block = self.queue.pop(action)
            self.env.process(self.model['Process0'].put(block, 'Source', None, 0))
            self.event_tracer.loc[len(self.event_tracer)] = [self.env.now, "part_transferred", block.id, "Source", None]
            self.num_of_blocks_put += 1
            while True:
                self.env.step()
                if self.model['Process0'].parts_sent - self.num_of_blocks_put == 0:
                    while self.env.peek() == self.env.now:
                        self.env.run(self.env.timeout(0))
                    break
            if len(self.queue) == 0:
                done = True
            reward = self._calculate_reward()
            self.time = self.env.now
        next_state = self._get_state()
        return next_state, reward, done

    def reset(self):
        self.event_tracer = pd.DataFrame([], columns=["TIME", "EVENT", "PART", "PROCESS", "SERVER_ID"])
        self.env, self.model = self._modeling(self.num_of_processes, self.event_tracer)
        self.inbound_panel_blocks = self.inbound_panel_blocks_clone[:]
        for panel_block in self.inbound_panel_blocks:
            panel_block.step = 0
        random.shuffle(self.inbound_panel_blocks)
        self.num_of_blocks_put = 0
        self.stage = 0
        return self._get_state()

    def _get_state(self):
        # 전체 state 변수를 -1로 초기화
        state = np.full(self.s_size, self.empty)

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

    def _calculate_reward(self):
        block_completed = self.event_tracer[
            (self.event_tracer['TIME'] > self.time) &
            (self.event_tracer["EVENT"] == "part_transferred") &
            (self.event_tracer["PROCESS"] == 'Process{0}'.format(self.num_of_processes - 1))]
        num_of_block_completed = len(block_completed)
        return num_of_block_completed

    def _calculate_reward_by_throughput(self):
        # throughput
        df_TH = self.event_tracer["TIME"][
            (self.event_tracer["EVENT"] == "part_transferred") &
            (self.event_tracer["PROCESS"] == 'Process{0}'.format(self.num_of_processes - 1))]
        df_TH = df_TH.reset_index(drop=True)

        TH_list = []
        process_throughput = 0
        for i in range(len(df_TH) - 1):
            TH_list.append(df_TH.loc[i + 1] - df_TH.loc[i])
        if not TH_list:
            process_throughput = 1 / np.mean(TH_list)

        return process_throughput

    def _modeling(self, num_of_processes, event_tracer):
        from environment.SimComponents import Process, Sink
        env = simpy.Environment()
        model = {}
        for i in range(num_of_processes + 1):
            model['Process{0}'.format(i)] = Process(env, 'Process{0}'.format(i), 1, model, event_tracer, qlimit=1)
            if i == num_of_processes:
                model['Sink'] = Sink(env, 'Sink')

        return env, model


class AssemblyDisplay(object):
    def __init__(self):
        pass

    def game_loop_from_space(self):
        pass


if __name__ == '__main__':
    from environment.panelblock import *
    panel_blocks, num_of_processes = import_panel_block_schedule('../environment/data/PBS_assy_sequence_gen_000.csv')
    len_of_queue = 10
    assembly = Assembly(num_of_processes, len_of_queue, inbound_panel_blocks=panel_blocks)
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

    # save data
    save_path = './result'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df_event_tracer = pd.DataFrame(assembly.event_tracer)
    df_event_tracer.to_excel(save_path +'/event_PBS_rev.xlsx')


