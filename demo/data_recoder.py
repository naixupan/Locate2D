'''
定位数据记录器
用于记录定位数据，方便后续进行统计
'''

import pandas as pd
import numpy as np
from datetime import datetime
import time
import matplotlib.pyplot as plt
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


def calculate_positioning_errors(csv_file, output_file=None):
    """
    计算定位误差统计并生成报告

    :param csv_file: 输入CSV文件路径
    :param output_file: 输出报告文件路径（可选）
    :return: 包含误差统计的DataFrame
    """
    # 1. 读取CSV文件
    df = pd.read_csv(csv_file)

    # 2. 计算误差
    # 位置误差（绝对误差）
    df['△x'] = df['x_compute'] - df['x_original']
    df['△y'] = df['y_compute'] - df['y_original']
    df['△z'] = df['z_compute'] - df['z_original']

    # 旋转角误差（处理角度环绕问题）
    df['△rz'] = df['rz_compute'] - df['rz_original']
    # df['△rz'] = np.minimum(np.abs(angle_diff), 360 - np.abs(angle_diff))



    # 4. 计算基本统计
    stats = pd.DataFrame()

    # 位置误差统计
    for axis in ['x', 'y', 'z']:
        col = f'△{axis}'
        stats[f'{axis}_mean'] = [df[col].mean()]
        stats[f'{axis}_std'] = [df[col].std()]
        stats[f'{axis}_min'] = [df[col].min()]
        stats[f'{axis}_max'] = [df[col].max()]
        stats[f'{axis}_median'] = [df[col].median()]
        stats[f'{axis}_p95'] = [df[col].quantile(0.95)]

    # 旋转角误差统计
    stats['rz_mean'] = [df['△rz'].mean()]
    stats['rz_std'] = [df['△rz'].std()]
    stats['rz_min'] = [df['△rz'].min()]
    stats['rz_max'] = [df['△rz'].max()]
    stats['rz_median'] = [df['△rz'].median()]
    stats['rz_p95'] = [df['△rz'].quantile(0.95)]

    # 3D误差统计
    # stats['3D_mean'] = [df['3D_error'].mean()]
    # stats['3D_std'] = [df['3D_error'].std()]
    # stats['3D_min'] = [df['3D_error'].min()]
    # stats['3D_max'] = [df['3D_error'].max()]
    # stats['3D_median'] = [df['3D_error'].median()]
    # stats['3D_p95'] = [df['3D_error'].quantile(0.95)]

    # 5. 按类型分组统计
    if 'type' in df.columns:
        grouped_stats = df.groupby('type').agg({
            '△x': ['mean', 'std', 'median'],
            '△y': ['mean', 'std', 'median'],
            '△z': ['mean', 'std', 'median'],
            '△rz': ['mean', 'std', 'median']
        })

        # 重命名列名
        grouped_stats.columns = [
            f"{col[0]}_{col[1]}" for col in grouped_stats.columns
        ]

    # 6. 可视化分析
    plt.figure(figsize=(15, 10))

    # 误差分布直方图
    plt.subplot(2, 2, 1)
    for axis in ['x', 'y', 'z']:
        plt.hist(df[f'△{axis}'], bins=50, alpha=0.5, label=f'△{axis}')
    plt.title('位置误差分布')
    plt.xlabel('误差值')
    plt.ylabel('频率')
    plt.legend()

    # 旋转角误差分布
    plt.subplot(2, 2, 2)
    plt.hist(df['△rz'], bins=50, alpha=0.7, color='purple')
    plt.title('旋转角误差分布')
    plt.xlabel('角度误差(度)')
    plt.ylabel('频率')


    # 误差随时间变化（如果有时间戳）
    if 'timestamp' in df.columns:
        plt.subplot(2, 2, 4)
        plt.plot(df['timestamp'], df['3D_error'], 'b-', alpha=0.7)
        plt.title('3D误差随时间变化')
        plt.xlabel('时间戳')
        plt.ylabel('3D误差')

    plt.tight_layout()

    # 7. 输出报告
    print("=" * 50)
    print("定位误差统计分析报告")
    print("=" * 50)
    print(f"数据集大小: {len(df)}个样本")
    print("\n整体误差统计:")
    print(stats.transpose().to_string(float_format="%.4f"))

    if 'type' in df.columns:
        print("\n按类型分组的误差统计:")
        print(grouped_stats.to_string(float_format="%.4f"))

    # 8. 保存结果
    if output_file:
        with pd.ExcelWriter(output_file) as writer:
            # 保存原始数据与误差
            df.to_excel(writer, sheet_name='原始数据与误差', index=False)

            # 保存整体统计
            stats.to_excel(writer, sheet_name='整体统计', index=False)

            # 保存分组统计
            if 'type' in df.columns:
                grouped_stats.to_excel(writer, sheet_name='分组统计')

            # 保存可视化图表
            plt.savefig(writer, bbox_inches='tight', pad_inches=0.5, format='png')

        print(f"\n报告已保存至: {output_file}")

    # 显示图表
    plt.show()

    return df, stats


def main() :

    csv_path = '../data/data_record.csv'
    initalize_csv(csv_path,COLUMNS)

    # test_data = [6.153095579, -968.0494345, -176.4404891, -94.51955403, 6.0952, -968.7429, -176.9948, -94.5297, 1]
    # single_data = {
    #     'x_compute': test_data[0],
    #     'y_compute': test_data[1],
    #     'z_compute': test_data[2],
    #     'rz_compute': test_data[3],
    #     'x_original': test_data[4],
    #     'y_original': test_data[5],
    #     'z_original': test_data[6],
    #     'rz_original': test_data[7],
    #     'type': test_data[8]
    # }
    # append_to_csv(csv_path, single_data)
    static_csv = '../data/static_data'
    calculate_positioning_errors(csv_path, static_csv)


if __name__ == '__main__':
    main()