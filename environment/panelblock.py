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
    def __init__(self, data):
        # block_id
        self.id = data['product']
        # 작업 시간 저장
        data.drop(['product', 'unit_assy'], inplace=True)
        idx_list = data.index
        index = pd.MultiIndex.from_product([[i for i in range(len(idx_list)+1)], ['start_time', 'process_time', 'process']])
        self.panel_block = pd.Series(index=index)
        for i, process in enumerate(idx_list):
            self.panel_block[(i, 'process_time')] = data[process]
            self.panel_block[(i, 'process')] = process
        self.panel_block[len(idx_list), 'process'] = 'Sink'
        # 지나간 공정 수
        self.step = 0











if __name__ == "__main__":
    panel_blocks = import_panel_block_schedule('../environment/data/PBS_assy_sequence_gen_000.csv')
    print(len(panel_blocks))