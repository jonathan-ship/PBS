import pandas as pd


def import_panel_block_schedule(filepath):
    df_schedule = pd.read_csv(filepath, encoding='euc-kr')
    df_schedule = df_schedule.drop(columns=['unit_assy'])
    num_of_processes = len(df_schedule.columns) - 1
    panel_blocks = []
    for i, block in df_schedule.iterrows():
        panel_block = PanelBlock(block['product'], block.drop(['product']))
        panel_blocks.append(panel_block)
    return panel_blocks, num_of_processes


def export_panel_block_schedule(filepath, event_tracer):
    block_list = event_tracer["PART"][
        (event_tracer["EVENT"] == "part_transferred") & (event_tracer["PROCESS"] == "Source")]
    df_block_list = pd.DataFrame(block_list, columns=["RL results"])
    df_block_list.to_excel(filepath + '/results_PBS.xlsx')


class PanelBlock(object):
    def __init__(self, block_id, block_working_time):
        # block_id
        self.id = block_id
        # 작업 시간 저장
        index = pd.MultiIndex.from_product([[i for i in range(len(block_working_time.index)+1)],
                                            ['start_time', 'process_time', 'process']])
        self.data = pd.Series(index=index, dtype=float)
        for i, process in enumerate(block_working_time.index):
            self.data[(i, 'process_time')] = block_working_time[process]
            self.data[(i, 'process')] = 'Process{0}'.format(i)
        self.data[len(block_working_time.index), 'process'] = 'Sink'
        # 지나간 공정 수
        self.step = 0


if __name__ == "__main__":
    panel_blocks, num_of_process = import_panel_block_schedule('../environment/data/PBS_assy_sequence_gen_000.csv')
    print(panel_blocks[0].data)