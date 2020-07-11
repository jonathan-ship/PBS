import pandas as pd
import numpy as np


def import_panel_block_schedule(filepath):
    df_schedule = pd.read_csv(filepath, encoding='euc-kr')
    panel_blocks = []
    for i, row in df_schedule.iterrows():
        panel_block = PanelBock(row['product'], row['plate_weld'], row['saw_front'], row['turn_over'], row['saw_back'],
                                row['longi_weld'], row['unit_assy'], row['sub_assy'])
        panel_blocks.append(panel_block)
    return panel_blocks


class PanelBock(object):
    def __init__(self, block_id=None, plate_weld=0, saw_front=0, turn_over=0, saw_back=0, longi_weld=0, unit_assy=0, sub_assy=0):
        self.id = str(block_id)
        self.plate_weld = plate_weld
        self.saw_front = saw_front
        self.turn_over = turn_over
        self.saw_back = saw_back
        self.longi_weld = longi_weld
        self.unit_assy = unit_assy
        self.sub_assy = sub_assy


if __name__ == "__main__":
    panel_blocks = import_panel_block_schedule('../environment/data/PBS_assy_sequence_gen_000.csv')
    print(len(panel_blocks))