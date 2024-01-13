from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from dataset.import_data import connect_mysql


# step 1: get id-idx dict
def get_id_idx_dict(cnx):
    id_idx_dict = {}
    sql_str = (
        "SELECT stations_lat_lon.INDEX, stations_lat_lon.ID FROM stations_lat_lon;"
    )
    cursor = cnx.cursor()
    cursor.execute(sql_str)
    res = cursor.fetchall()
    for idx_id in res:
        id_idx_dict[idx_id[1]] = idx_id[0]
    return id_idx_dict


# step 2: get TotalFlow, AvgSpeed, AvgOccupancy
def get_traffic_flow_piece(cnx, id_idx_dict, timestamp):
    n_feat = 1 + 3  # the first feat to flag valid data
    traffic_flow_piece = np.zeros([len(id_idx_dict), n_feat])
    sql_str = f"SELECT ID, TotalFlow, AvgSpeed, AvgOccupancy from pems_5min WHERE Timestamp = '{timestamp}';"
    cursor = cnx.cursor()
    cursor.execute(sql_str)
    res = cursor.fetchall()

    def data_convert(value):
        if value is None:
            return 0
        else:
            return value

    for r in res:
        if r[0] in id_idx_dict:
            idx = id_idx_dict[r[0]] - 1
            traffic_flow_piece[idx, 0] = 1
            traffic_flow_piece[idx, 1] = data_convert(r[1])
            traffic_flow_piece[idx, 2] = data_convert(r[2])
            traffic_flow_piece[idx, 3] = data_convert(r[3])
    return traffic_flow_piece


# step 3: get timestamp_idx_dict
def get_timestamp_idx_dict(cnx):
    timestamp_idx_dict = {}
    sql_str = "SELECT `timestamp`.`index`, `timestamp`.`Timestamp` from `timestamp`;;"
    cursor = cnx.cursor()
    cursor.execute(sql_str)
    res = cursor.fetchall()
    for idx_timestamp in res:
        timestamp_idx_dict[idx_timestamp[1]] = idx_timestamp[0]
    return timestamp_idx_dict


# step 4: construct dataset
def get_traffic_flow_data(cnx, start_date, end_date, freq="5T"):
    id_idx_dict = get_id_idx_dict(cnx)
    timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
    timestamp_idx_dict = get_timestamp_idx_dict(cnx)

    n_feat = 1 + 3  # the first feat to flag valid data
    traffic_flow_data = np.zeros([len(timestamps), len(id_idx_dict), n_feat])
    start_timestamp_idx = timestamp_idx_dict[start_date] - 1
    for timestamp in timestamps:
        print(timestamp)
        traffic_flow_piece = get_traffic_flow_piece(cnx, id_idx_dict, timestamp)
        timestamp_idx = (timestamp_idx_dict[timestamp] - 1) - start_timestamp_idx
        traffic_flow_data[timestamp_idx] = traffic_flow_piece
    return traffic_flow_data, id_idx_dict, timestamp_idx_dict


if __name__ == "__main__":
    cnx = connect_mysql(
        ip="192.168.188.222", port="3306", username="root", password="wonder"
    )

    input_format = "%Y-%m-%d %H:%M:%S"
    output_format = "%Y-%m-%d_%H-%M-%S"

    start_datetime = datetime.strptime("2019-01-24 00:00:00", input_format)
    end_datetime = datetime.strptime("2019-01-31 23:55:00", input_format)
    cur_datetime = start_datetime
    while cur_datetime + timedelta(hours=23) + timedelta(minutes=55) <= end_datetime:
        s_datetime = cur_datetime
        e_datetime = cur_datetime + timedelta(hours=23) + timedelta(minutes=55)
        traffic_flow_data, _, _ = get_traffic_flow_data(
            cnx, s_datetime, e_datetime, "5T"
        )
        filename = f"{s_datetime.strftime(output_format)}-{e_datetime.strftime(output_format)}.npy"
        print("saved", filename)
        np.savez_compressed(filename, data=traffic_flow_data)
        cur_datetime += timedelta(days=1)
