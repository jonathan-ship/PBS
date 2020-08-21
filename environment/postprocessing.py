import pandas as pd
import numpy as np
import math as m
import gantt
import time
from datetime import date, timedelta

import sys
sys.path.insert(0, 'c:\pyzo2015a\lib\site-packages\plotly')
import plotly as py
import plotly.figure_factory as ff


class Utilization(object):
    def __init__(self, data, process_dict, process, finish_time):
        self.data = data
        self.process_dict = process_dict
        self.process = process  # utilization을 계산할 Process 혹은 Server
        self.type = "Process" if process in process_dict.keys() else "Server"
        self.finish_time = finish_time

    # 공정에 관한 event 중 work_start인 event의 시간 저장 (총 시간 계산, working time 시간 계산 시 사용)
    # 공정에 관한 event 중 part_transferred인 event의 시간 저장 (총 시간 계산)
    # 공정에 관한 event 중 work_finish인 event의 시간 저장 (working time 시간 계산 시 사용)
    def utilization(self):
        idx = self.data["PROCESS"].map(lambda x: self.process in x)
        self.data = self.data.rename(index=idx)
        if True not in self.data.index:
            u, idle, working_time = 0, 0, 0
            return u, idle, working_time
        self.data = self.data.loc[True, :]

        start_time = 0

        work_start = self.data["TIME"][self.data["EVENT"] == "work_start"]
        work_finish = self.data["TIME"][self.data["EVENT"] == "work_finish"]

        work_start = work_start.reset_index(drop=True)
        work_finish = work_finish.reset_index(drop=True)

        if len(work_start) * len(work_finish) == 0:
            u, idle, working_time = 0, 0, 0
            return u, idle, working_time

        # 총 가동 시간
        server_num = self.process_dict[self.process].server_num if self.type == "Process" else 1
        total_time = (self.finish_time - start_time) * server_num

        # 총 작업 시간
        df_working = work_finish - work_start
        total_working = np.sum(df_working)

        # 가동률
        u = total_working / total_time
        idle = total_time - total_working

        return u, idle, total_working


class LeadTime(object):
    def __init__(self, data):
        self.data = data  # Event Tracer
        self.list_LT = []
        self.part_list = list(data["PART"][data["EVENT"] == "part_created"])

    def avg_LT(self):
        for part in self.part_list:
            part_data = self.data[self.data["PART"] == part]
            df_part_start = part_data["TIME"][part_data["EVENT"] == "part_created"]
            df_part_finish = part_data["TIME"][part_data["EVENT"] == "completed"]

            df_part_start = df_part_start.reset_index(drop=True)
            df_part_finish = df_part_finish.reset_index(drop=True)

            if len(df_part_start) * len(df_part_finish) != 0:
                self.list_LT.append(df_part_finish[0] - df_part_start[0])

        return np.mean(self.list_LT)


# class Idle(object):
#     def __init__(self, data, process_dict, process):
#         self.data = data
#         self.process_dict = process_dict
#         self.process = process  # utilization을 계산할 Process 혹은 Server
#         self.type = "Process" if process in process_dict.keys() else "Server"
#
#     # 공정에 관한 event 중 work_start인 event의 시간 저장 (총 시간 계산, working time 시간 계산 시 사용)
#     # 공정에 관한 event 중 part_transferred인 event의 시간 저장 (총 시간 계산)
#     # 공정에 관한 event 중 work_finish인 event의 시간 저장 (working time 시간 계산 시 사용)
#     def idle(self):
#         idx = self.data["PROCESS"].map(lambda x: self.process in x)
#         self.data = self.data.rename(index=idx)
#         if True not in self.data.index:
#             return 0
#         self.data = self.data.loc[True, :]
#
#         work_start = self.data["TIME"][self.data["EVENT"] == "work_start"]
#         part_transferred = self.data["TIME"][self.data["EVENT"] == "part_transferred"]
#         work_finish = self.data["TIME"][self.data["EVENT"] == "work_finish"]
#
#         work_start = work_start.reset_index(drop=True)
#         part_transferred = part_transferred.reset_index(drop=True)
#         work_finish = work_finish.reset_index(drop=True)
#
#         # 총 가동 시간
#         server_num = self.process_dict[self.process].server_num if self.type == "Process" else 1
#         total_time = (part_transferred[len(part_transferred) - 1] - work_start[0]) * server_num
#
#         # 총 작업 시간
#         df_working = work_finish - work_start
#         total_working = np.sum(df_working)
#
#         # 가동률
#         idle = total_time - total_working
#         return idle


class Throughput(object):
    def __init__(self, data, process):
        self.data = data  # Event Tracer
        self.list_time = []
        self.process = process
        self.part_transferred = []
        self.queue_entered = []

    def throughput(self):
       part_transferred = list(self.data["TIME"][(self.data["PROCESS"] == self.process) & (self.data["EVENT"] == "part_transferred")])
       queue_entered = list(self.data["TIME"][(self.data["PROCESS"] == self.process) & (self.data["EVENT"] == "queue_entered")])

       for i in range(len(part_transferred)-1):
           self.list_time.append(part_transferred[i]-queue_entered[i])

       total_time = np.mean(self.list_time)
       return 1/total_time


class WIP(object):
    def __init__(self, event_tracer, WIP_type=None, process=None):
        self.data = event_tracer
        # "WIP_m": model 전체의 WIP, "WIP_p": 특정 process의 WIP, "WIP_q": 특정 process의 WIP_q
        self.WIP_type = WIP_type
        self.process = process
        self.wip_list = None

    def wip_preprocessing(self):
        if self.WIP_type == "WIP_m":
            wip_start = self.data[self.data["EVENT"] == "part_created"]
            wip_finish = self.data[self.data["EVENT"] == "completed"]
        else:  # WIP_type = "WIP_p" / "WIP_q"
            idx = self.data["PROCESS"].map(lambda x: self.process in x)
            self.data = self.data.rename(index=idx)
            self.data = self.data.loc[True, :]
            wip_start = self.data[self.data["EVENT"] == "queue_entered"]
            if self.WIP_type == "WIP_p":
                wip_finish = self.data[self.data["EVENT"] == "part_transferred"]
            else:  # WIP_type = "WIP_q":
                wip_finish = self.data[self.data["EVENT"] == "queue_released"]

        wip_start = wip_start.reset_index(drop=True)
        wip_finish = wip_finish.reset_index(drop=True)

        time_lange = wip_finish["TIME"][len(wip_finish) - 1]
        wip_list = [0 for _ in range(m.ceil(time_lange) + 1)]

        wip_start.sort_values(by=["PART"], inplace=True)
        wip_finish.sort_values(by=["PART"], inplace=True)

        data_len = min(len(wip_start), len(wip_finish))
        wip_start = wip_start[:data_len]
        wip_finish = wip_finish[:data_len]

        return wip_list, wip_start, wip_finish

    def cal_wip(self):
        self.wip_list, wip_start, wip_finish = self.wip_preprocessing()
        wip_start = wip_start["TIME"]
        wip_finish = wip_finish["TIME"]

        for i in range(len(wip_start)):
            for j in range(m.ceil(wip_start[i]), m.ceil(wip_finish[i])):
                self.wip_list[j] += 1

        return self.wip_list

#class Gantt(object):
    # def __init__(self, data, process_list):

      #  self.data = data
       # self.process_list = process_list
        #self.part_transferred = []
       # self.queue_entered = []
       # self.dataframe = []

  #  def gantt(self):
   #     for i in range(len(self.process_list)-1):
    #       part_transferred = list(self.data["TIME"][(self.data["PROCESS"] == self.process_list[i]) & (self.data["EVENT"] == "part_transferred")])
     #      queue_entered = list(self.data["TIME"][(self.data["PROCESS"] == self.process_list[i]) & (self.data["EVENT"] == "queue_entered")])



      #     for j in range(len(part_transferred) - 1):
       #         temp_dic = {'Task' : self.process_list[i], 'Start' : queue_entered[j], 'Finish' : part_transferred[j]}
        #        self.dataframe.append(temp_dic)

      #  fig = ff.create_gantt(self.dataframe)
      #  py.offline.plot(fig, filename='gantt-numeric-variable.html')

# class WIP:
#     def __init__(self,data, process_list, time):
#         self.data = data
#         self.process_list = process_list
#         self.time = time
#         self.delay_start = []
#         self.delay_finish = []
#
#     def wip(self):
#         count_source = 0
#         count_sink = 0
#         count_delay = 0
#
#         a = 0
#         while self.data["TIME"][a] <= self.time:
#             if self.data["EVENT"][a] == "part_created":
#                 count_source += 1
#             a += 1
#
#         b = 0
#         while self.data["TIME"][b] <= self.time:
#             if self.data["EVENT"][b] == "completed":
#                 count_sink += 1
#             b += 1
#
#         for i in range(len(self.process_list)-1):
#             self.delay_start = list(self.data["TIME"][(self.data["EVENT"] == "delay_start") & (self.data["PROCESS"] == self.process_list[i])])
#             self.delay_finish = list(self.data["TIME"][(self.data["EVENT"] == "delay_finish") & (self.data["PROCESS"] == self.process_list[i])])
#
#         for i in range(len(self.delay_start)-1):
#             if (self.delay_start[i] <= self.time) & (self.delay_finish[i] >= self.time):
#                 count_delay += 1
#
#         wip = count_source - count_sink - count_delay
#         return wip


##시간대별로 dict만들어야할지?
class SUBWIP:
    def __init__(self, data, process, time):
        self.data = data
        self.process = process
        self.time = time
        self.work_start = []
        self.work_finish = []

    def subwip(self):
        idx = self.data["PROCESS"].map(lambda x: self.process in x)
        self.data = self.data.rename(index=idx)
        if True not in self.data.index:
            return 0
        self.data = self.data.loc[True, :]

        self.work_start = list(self.data["TIME"][self.data["EVENT"] == "work_start"])
        self.work_finish = list(self.data["TIME"][self.data["EVENT"] == "work_finish"])

        wip = 0
        for i in range(len(self.work_start)-1):
            if (self.work_start[i] <= self.time) & (self.time <= self.work_finish[i]):
                wip += 1

        return wip


class Gantt(object):
    def __init__(self, data, process_list):
        self.data = data
        self.process_list = process_list

    def gantt(self):

   # Change font default
        gantt.define_font_attributes(fill='black', stroke='black', stroke_width=0, font_family="Verdana")
    # Create some tasks
        self.list_part = list(self.data["PART"][self.data["EVENT"] == "part_created"])

        start = date(2020,8,11)
        for part in self.list_part:
            k = 1
            part_data = self.data[self.data["PART"] == part]
            part_start = list(part_data["TIME"][(part_data["EVENT"] == "queue_entered") | (part_data["EVENT"] == "part_created")])
            part_finish = list(part_data["TIME"][part_data["EVENT"] == "part_transferred"])
            for i in range(len(self.process_list)-1):
                while k <= len(part_start)-1:
                    t = gantt.Task(name=self.process_list[i], start = start + timedelta(days=part_start[k]), duration = part_finish[k] - part_start[k])
                    k += 1
                    p = gantt.Project(name=self.process_list[i])
                    p.add_task(t)
                    p.make_svg_for_tasks(filename='Gantt Chart.svg', start=start, end=start + timedelta(days=part_finish[len(part_start)-2]))

