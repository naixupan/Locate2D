'''
定位数据记录器
用于记录定位数据，方便后续进行统计
'''

import pandas as pd
import numpy as np
from datetime import datetime
import time
import os

COLUMNS = [
    'x_compute',
    'y_compute',
    'z_compute',
    'rz_compute',
    'x_original',
    'y_original',
    'z_original',
    'rz_original',
    'type'
]


def initalize_csv(file_path,headers):
    if not os.path.exists(file_path):
        pd.DataFrame(columns=headers).to_csv(file_path,index=False)
        print(f'已创建CSV文件：{file_path}')
    else:
        print(f'CSV文件已存在：{file_path}')


def append_to_csv(file_path, data):
    existing_df = pd.read_csv(file_path)
    if isinstance(data,dict):
        new_df = pd.DataFrame([data])
    elif isinstance(data,list) and all(isinstance(item, dict) for item in data):
        new_df = pd.DataFrame(data)
    else:
        raise ValueError("数据格式错误，请提供字典或字典列表")
    for col in COLUMNS:
        if col not in new_df.columns:
            new_df[col] = np.nan

        # 重新排序列以匹配表头
    new_df = new_df[COLUMNS]

    # 合并新旧数据
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)

    # 保存回CSV文件
    combined_df.to_csv(file_path, index=False)
    print(f"已追加 {len(new_df)} 条数据到 {file_path}")


def main() :

    csv_path = '../data/test_csv.csv'
    initalize_csv(csv_path,COLUMNS)

    test_data = [6.153095579, -968.0494345, -176.4404891, -94.51955403, 6.0952, -968.7429, -176.9948, -94.5297, 1]
    single_data = {
        'x_compute': test_data[0],
        'y_compute': test_data[1],
        'z_compute': test_data[2],
        'rz_compute': test_data[3],
        'x_original': test_data[4],
        'y_original': test_data[5],
        'z_original': test_data[6],
        'rz_original': test_data[7],
        'type': test_data[8]
    }
    append_to_csv(csv_path, single_data)

if __name__ == '__main__':
    main()