import pandas as pd
import numpy as np


def import_panel_block_schedule(filepath):
    df_schedule = pd.read_csv(filepath, encoding='euc-kr')
    panel_blocks = []
    for i in range(len(df_schedule)):
        temp = df_schedule.iloc[i]
        panel_block = PanelBlock(temp)
        panel_blocks.append(panel_block)
    return panel_blocks


class PanelBlock(object):
    def __init__(self, panel_block):
        # block_id
        self.id = panel_block['product']
        # 작업 시간 저장
        panel_block.drop(['product', 'unit_assy'], inplace=True)
        index = pd.MultiIndex.from_product([[i for i in range(len(panel_block.index)+1)], ['start_time', 'process_time', 'process']])
        self.data = pd.Series(index=index)
        for i, process in enumerate(panel_block.index):
            self.data[(i, 'process_time')] = panel_block[process]
            self.data[(i, 'process')] = 'Process{0}'.format(i)
        self.data[len(panel_block.index), 'process'] = 'Sink'
        # 지나간 공정 수
        self.step = 0











if __name__ == "__main__":
    panel_blocks = import_panel_block_schedule('../environment/data/PBS_assy_sequence_gen_000.csv')
    print(len(panel_blocks))