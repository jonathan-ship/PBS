import simpy
import random
import pygame

import numpy as np


class Assembly(object):
    def __init__(self, num_of_processes, len_of_queue, inbound_panel_blocks=None, display_env=False):
        self.env, self.model, self.event_tracer = self._modeling(num_of_processes)
        self.num_of_processes = num_of_processes
        self.len_of_queue = len_of_queue
        self.a_size = len_of_queue
        self.s_size = num_of_processes * len_of_queue + num_of_processes
        self.inbound_panel_blocks = inbound_panel_blocks
        self.inbound_panel_blocks_clone = self.inbound_panel_blocks[:]
        self.queue = []
        self.stage = 0
        self.empty = -1
        if display_env:
            display = AssemblyDisplay()
            display.game_loop_from_space()

    def step(self, action):
        done = False
        block = self.queue.pop(action)
        self.model['Process0'].put(block)
        while True:
            self.env.step()
            num_of_working_server, _ = self.model['Process0'].get_num_of_part()
            if self.model['Process0'].server_num - num_of_working_server > 0:
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

    def _modeling(self, num_of_processes):
        from environment.SimComponents import Process, Sink, return_event_tracer
        env = simpy.Environment()
        model = {}
        for i in range(num_of_processes + 1):
            model['Process{0}'.format(i)] = Process(env, 'Process{0}'.format(i), 1, model, qlimit=1)
            if i == num_of_processes:
                model['Sink'] = Sink(env, 'Sink')
        event_tracer = return_event_tracer()
        return env, model, event_tracer


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
    for i in range(50):
        print(i)
        s_next, r, d = assembly.step(0)
        print(s_next)
        s = s_next
        if d:
            break
    print(s)



