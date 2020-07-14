from collections import deque

EVENT_TRACER = {"event": [], "time": [], "part": [], "process": []}


class Sink(object):

    def __init__(self, env, name, rec_lead_time=False, rec_arrivals=False, absolute_arrivals=False, selector=None):
        self.name = name
        self.env = env
        self.rec_lead_time = rec_lead_time
        self.rec_arrivals = rec_arrivals
        self.absolute_arrivals = absolute_arrivals
        self.selector = selector
        self.lead_time = []
        self.arrivals = []
        self.parts_rec = 0
        self.last_arrival = 0.0

    def put(self, part):
        if not self.selector or self.selector(part):
            now = self.env.now
            if self.rec_lead_time:
                self.lead_time.append(self.env.now - part.time)
            if self.rec_arrivals:
                if self.absolute_arrivals:
                    self.arrivals.append(now)
                else:
                    self.arrivals.append(now - self.last_arrival)
                self.last_arrival = now


class Process(object):

    def __init__(self, env, name, server_num, process_dict, qlimit=None):
        self.name = name
        self.env = env
        self.server_num = server_num
        self.process_dict = process_dict
        self.qlimit = qlimit

        self.server = [None for _ in range(server_num)]
        self.queue = deque([])
        self.waiting = deque([])
        self.parts_rec = 0
        self.parts_sent = 0
        self.flag = False

    def run(self, part, server_id):
        # record: work_start
        self.record(self.env.now, part.id, self.name, event="work_start")
        # work start
        proc_time = part.data[(part.step, 'process_time')]
        yield self.env.timeout(proc_time)

        # record: work_finish
        self.record(self.env.now, part.id, self.name, event="work_finish")

        # next process
        next_process = part.data[(part.step + 1, 'process')]

        if self.process_dict[next_process].__class__.__name__ == 'Process':
            # lag: 후행공정 시작시간 - 선행공정 종료시간
            lag = part.data[(part.step + 1, 'start_time')] - self.env.now
            if lag > 0:
                yield self.env.timeout(lag)

            # 대기 event 발생
            if len(self.process_dict[next_process].queue) + (
                    self.process_dict[next_process].server_num - self.process_dict[next_process].server.count(None)) >= \
                    self.process_dict[next_process].qlimit:
                self.process_dict[next_process].waiting.append(self.env.event())
                # record: delay_start
                self.record(self.env.now, None, self.name, event="delay_start")

                yield self.process_dict[next_process].waiting[-1]

        # record: part_transferred
        self.record(self.env.now, part.id, self.name, event="part_transferred")
        # part_transferred
        self.process_dict[next_process].put(part)
        part.step += 1
        self.parts_sent += 1

        if len(self.queue) > 0:
            part = self.queue.popleft()
            self.server[server_id] = self.env.process(self.run(part, server_id))
            # record: queue_released
            self.record(self.env.now, part.id, self.name, event="queue_released")
        else:
            self.server[server_id] = None

        # 대기 종료
        if len(self.queue) + (self.server_num - self.server.count(None)) < self.qlimit and len(self.waiting) > 0:
            self.waiting.popleft().succeed()
            # record: delay_finish
            pre_process = part.data[(part.step - 1, 'process')] if part.step > 0 else 'Source'
            self.record(self.env.now, None, pre_process, event="delay_finish")

    def put(self, part):
        self.parts_rec += 1
        if self.server.count(None) != 0:  # server에 자리가 있으면
            self.server[self.server.index(None)] = self.env.process(self.run(part, self.server.index(None)))
        else:  # server가 다 차 있으면
            self.queue.append(part)
            # record: queue_entered
            self.record(self.env.now, part.id, self.name, event="queue_entered")

    def record(self, time, part, process, event=None):
        EVENT_TRACER["event"].append(event)
        EVENT_TRACER["time"].append(time)
        EVENT_TRACER["part"].append(part)
        EVENT_TRACER["process"].append(process)


# event tracer을 _get_reward 함수로 return 해 주는 함수
def return_event_tracer():
    return EVENT_TRACER