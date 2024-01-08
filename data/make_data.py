import os

import pandas as pd


def filter_data():
    chunk_size = 5000000  # 每次读取的行数
    file_path = "/run/user/1000/gvfs/smb-share:server=ds918.local,share=数据共享/pems.csv"

    date_time_format = "%m/%d/%Y %H:%M:%S"
    csv_reader = pd.read_csv(
        file_path,
        chunksize=chunk_size,
        parse_dates=["Timestamp"],
        date_format=date_time_format,
    )

    # filtered_data = pd.DataFrame()

    start_date = pd.to_datetime("2019-01-01")  # 开始时间
    end_date = pd.to_datetime("2019-01-31")  # 结束时间

    output_file_prefix = "save/pems_"
    part_number = 1
    for chunk in csv_reader:
        # 筛选字段A和字段B并添加到filtered_data中
        selected_columns = chunk[
            [
                "Timestamp",
                "ID",
                "District",
                "Freeway",
                "Direction",
                "LaneType",
                "StationLength",
                "numSamples",
                "percentObs",
                "TotalFlow",
                "AvgOccupancy",
                "AvgSpeed",
                "LaneNSamples",
                "LaneNFlow",
                "LaneNAvgOcc",
                "LaneNAvgSpeed",
                "LaneNObserved",
            ]
        ]
        selected_columns = selected_columns[
            (chunk["Timestamp"] >= start_date) & (chunk["Timestamp"] <= end_date)
        ]
        # filtered_data = pd.concat([filtered_data, selected_columns])

        if not selected_columns.empty:
            # 保存当前块到一个新的文件
            output_file_path = f"{output_file_prefix}{part_number}.csv"
            selected_columns.to_csv(output_file_path, index=False)

            part_number += 1


def merge_data():
    # 获取所有以'filtered_data_part_'开头的文件
    # file_pattern = "save/pems_*.csv"
    filtered_files = os.listdir("save")

    # 创建一个空的DataFrame来存储合并后的数据
    merged_data = pd.DataFrame()

    # 遍历所有筛选文件并合并它们
    for file_path in filtered_files:
        print(file_path)
        chunk = pd.read_csv(os.path.join("save", file_path))
        merged_data = pd.concat([merged_data, chunk], ignore_index=True)

    # 保存合并后的数据到一个新的文件
    output_merged_file = "merged_filtered_data.csv"
    merged_data.to_csv(output_merged_file, index=False)

    print(f"Merged data saved to {output_merged_file}")


def sort_by_timestamp():
    input_merged_file = "merged_filtered_data.csv"
    output_merged_file = "merged_filtered_data_sorted.csv"

    df = pd.read_csv(
        input_merged_file, parse_dates=["Timestamp"], date_format="%Y-%m-%d %H:%M:%S"
    )
    df.sort_values(by=["Timestamp", "ID"], inplace=True)

    # # 指定起始时间和结束时间, 并使用date_range()生成每5分钟的时间序列
    # start_time = '2019-01-01 00:00:00'
    # end_time = '2019-02-01 00:00:00'
    # time_range = pd.date_range(start=start_time, end=end_time, freq='5T')
    # merged_data = pd.DataFrame()
    # for idx in range(len(time_range)):
    #     start_t = time_range[idx]
    #     end_t = time_range[idx+1]
    #     print(start_t, end_t, flush=True)
    #     selected_columns = df[(df['Timestamp'] >= start_t) & (df['Timestamp'] <= end_t)]
    #     # 按时间戳列进行排序
    #     selected_columns.sort_values(by=["ID"], inplace=True)
    #     merged_data = pd.concat([merged_data, selected_columns], ignore_index=True)
    # # 保存合并后的数据到一个新的文件
    # merged_data.to_csv(output_merged_file, index=False)

    df.to_csv(output_merged_file, index=False)


def data_check():
    # 读取csv文件
    df = pd.read_csv(
        "/root/code/PaddleSTXL/test.csv",
        parse_dates=["Timestamp"],
        date_format="%Y-%m-%d %H:%M:%S",
        index_col="Timestamp",
    )

    start_time = "2019-01-01 00:00:00"
    end_time = "2019-01-31 23:55:00"
    time_range = pd.date_range(start=start_time, end=end_time, freq="5T")

    base_id = df[(df["Timestamp"] == time_range[0])]["ID"]
    for idx in range(len(time_range)):
        current_time = time_range[idx]
        print(current_time, flush=True)
        selected_columns = df[(df["Timestamp"] == current_time)]
        if list(selected_columns["ID"]) == list(base_id):
            print(f"{idx}, {current_time} True")
        else:
            print(
                f"{idx}, {current_time} False, {len(list(selected_columns['ID']))} vs {len(list(base_id))}"
            )


if __name__ == "__main__":
    sort_by_timestamp()
