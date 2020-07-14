import simpy
import random
import pygame

import numpy as np


class Assembly(object):
    def __init__(self, num_of_processes, len_of_queue, inbound_panel_blocks=None, display_env=False):
        self.env, self.model, self.event_tracer = self._modeling(num_of_processes)
        self.num_of_processes = num_of_processes
        self.len_of_queue = len_of_queue
        self.action_space = len_of_queue
        self.observation_space = num_of_processes * len_of_queue + num_of_processes
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

        # 각 공정별 남은 시간
        remaining_working_time = []
        for i in range(self.num_of_processes):
            now = self.env.now  # 현재 시각
            process = self.model['Process{0}'.format(i)]  # modeling한 Process
            part = process.part_in_progress[0]  # [part_id, working_time]
            part_id = part[0]
            working_time = part[1]
            part_start_time = self.event_tracer['time'][
                (self.event_tracer['part'] == part_id) & (self.event_tracer['process'] == 'Process{0}'.format(i)) & (
                            self.event_tracer['event'] == 'work_start')]
            remaining_working_time.append(working_time - (now - part_start_time))

        state[:self.num_of_processes] = remaining_working_time

        for i, panel_block in enumerate(self.queue):  # queue에 있는 블록 정보
            working_time_list = panel_block.data[:, 'process_time']
            working_time_list += list(working_time_list[:self.num_of_processes - 1])

        state[self.num_of_processes:self.num_of_processes+len(working_time_list)] = working_time_list

        return state

    def _calculate_reward(self):
        pass
        # utilization
        # for i in range(self.num_of_processes):  # process 마다 utilization 계산
        #     # 공정에 관한 event 중 work_start인 event의 시간 저장 (총 시간 계산, working time 시간 계산 시 사용)
        #     work_start = self.event_tracer["time"][(self.event_tracer["process"] == 'Process{0}'.format(i)) & (self.event_tracer["event"] == "work_start")]
        #     work_start = work_start.reset_index(drop=True)
        #
        #     # 공정에 관한 event 중 part_transferred인 event의 시간 저장 (총 시간 계산)
        #     part_transferred = self.event_tracer["time"][
        #         (self.event_tracer["process"] == 'Process{0}'.format(i)) & (self.event_tracer["event"] == "part_transferred")]
        #     part_transferred = part_transferred.reset_index(drop=True)
        #
        #     # 공정에 관한 event 중 work_finish인 event의 시간 저장 (working time 시간 계산 시 사용)
        #     work_finish = self.event_tracer["time"][(self.event_tracer["process"] == 'Process{0}'.format(i)) & (self.event_tracer["event"] == "work_finish")]
        #     work_finish = work_finish.reset_index(drop=True)
        #
        #     # 총 가동 시간
        #     total_time = (part_transferred[len(part_transferred) - 1] - work_start[0]) * (self.model['Process{0}'.format(i)].server_num)
        #
        #     # 총 작업 시간
        #     df_working = work_finish - work_start
        #     total_working = np.sum(df_working)
        #
        #     # 가동률
        #     u_dict = {}
        #     u_dict['Process{0}'.format(i)] = total_working / total_time

        # throughput
        # df_TH = self.event_tracer["time"][
        #     (self.self.event_tracer["event"] == "part_transferred") & (self.event_tracer["process"] == 'Process{0}'.format(self.num_of_processes - 1))]
        # df_TH = df_TH.reset_index(drop=True)
        #
        # TH_list = []
        # for i in range(len(df_TH) - 1):
        #     TH_list.append(df_TH.loc[i + 1] - df_TH.loc[i])
        #
        # process_throughput = 1 / np.mean(TH_list)


    def _modeling(self, num_of_processes):
        from environment.SimComponents import Process, Sink, return_event_tracer
        env = simpy.Environment()
        model = {}
        for i in range(num_of_processes + 1):
            model['Process{0}'.format(i)] = Process(env, 'Process{0}'.format(i), model, 1, qlimit=1)
            if i == num_of_processes:
                model['Sink'] = Sink(env, 'Sink', rec_lead_time=True, rec_arrivals=True)
        event_tracer = return_event_tracer()
        return env, model, event_tracer


class AssemblyDisplay(object):
    def __init__(self):
        pass

    def game_loop_from_space(self):
        pass