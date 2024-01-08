import gzip
import os
import shutil

import mysql.connector

base_sql_line = (
    "load data infile '{}' "
    "into table pems "
    "fields terminated by ',' "
    "lines terminated by '\\n' "
    "IGNORE 1 LINES "
    "(@Timestamp,@ID,@District,@Freeway,@Direction,@LaneType,@StationLength,@numSamples,"
    "@percentObs,@TotalFlow,@AvgOccupancy,@AvgSpeed,@LaneNSamples,@LaneNFlow,@LaneNAvgOcc,"
    "@LaneNAvgSpeed,@LaneNObserved,@t1,@t2,@t3,@t4,@t5,@t6,@t7,@t8,@t9,@t10,@t11,@t12,@t13,"
    "@t14,@t15,@t16,@t17,@t18,@t19,@t20,@t21,@t22,@t23,@t24,@t25,@t26,@t27,@t28,@t29,@t30,"
    "@t31,@t32,@t33,@t34,@t35) "
    "SET Timestamp=STR_TO_DATE(@Timestamp, '%m/%d/%Y %H:%i:%s'), "
    "ID = NULLIF(@ID, -1), "
    "District = NULLIF(@District, 0), "
    "Freeway = NULLIF(@Freeway, 0), "
    "Direction = NULLIF(@Direction, ''), "
    "LaneType = NULLIF(@LaneType, ''), "
    "StationLength = NULLIF(@StationLength, 0), "
    "numSamples = NULLIF(@numSamples, 0), "
    "percentObs = NULLIF(@percentObs, 0), "
    "TotalFlow = NULLIF(@TotalFlow, 0), "
    "AvgOccupancy = NULLIF(@AvgOccupancy, 0), "
    "AvgSpeed = NULLIF(@AvgSpeed, 0), "
    "LaneNSamples = NULLIF(@LaneNSamples, 0), "
    "LaneNFlow = NULLIF(@LaneNFlow, 0), "
    "LaneNAvgOcc = NULLIF(@LaneNAvgOcc, 0), "
    "LaneNAvgSpeed = NULLIF(@LaneNAvgSpeed, 0), "
    "LaneNObserved = NULLIF(@LaneNObserved, 0), "
    "t1 = NULLIF(@t1, 0), "
    "t2 = NULLIF(@t2, 0), "
    "t3 = NULLIF(@t3, 0), "
    "t4 = NULLIF(@t4, 0), "
    "t5 = NULLIF(@t5, 0), "
    "t6 = NULLIF(@t6, 0), "
    "t7 = NULLIF(@t7, 0), "
    "t8 = NULLIF(@t8, 0), "
    "t9 = NULLIF(@t9, 0), "
    "t10 = NULLIF(@t10, 0), "
    "t11 = NULLIF(@t11, 0), "
    "t12 = NULLIF(@t12, 0), "
    "t13 = NULLIF(@t13, 0), "
    "t14 = NULLIF(@t14, 0), "
    "t15 = NULLIF(@t15, 0), "
    "t16 = NULLIF(@t16, 0), "
    "t17 = NULLIF(@t17, 0), "
    "t18 = NULLIF(@t18, 0), "
    "t19 = NULLIF(@t19, 0), "
    "t20 = NULLIF(@t20, 0), "
    "t21 = NULLIF(@t21, 0), "
    "t22 = NULLIF(@t22, 0), "
    "t23 = NULLIF(@t23, 0), "
    "t24 = NULLIF(@t24, 0), "
    "t25 = NULLIF(@t25, 0), "
    "t26 = NULLIF(@t26, 0), "
    "t27 = NULLIF(@t27, 0), "
    "t28 = NULLIF(@t28, 0), "
    "t29 = NULLIF(@t29, 0), "
    "t30 = NULLIF(@t30, 0), "
    "t31 = NULLIF(@t31, 0), "
    "t32 = NULLIF(@t32, 0), "
    "t33 = NULLIF(@t33, 0), "
    "t34 = NULLIF(@t34, 0), "
    "t35 = NULLIF(@t35, 0) ;"
)


def connect_mysql(ip, port, username, password):
    return mysql.connector.connect(
        host=ip,
        port=port,
        user=username,
        password=password,
        database="traffic_flow",
        auth_plugin="mysql_native_password",
    )


def load_txt_file_into_mysql(mysql_cnx, base_path, temp_path="/var/lib/mysql-files"):
    """list each file in the directory and load it into mysql
        the base sql cmd is base_sql_line.
    Args:
        base_path (_type_): _description_
    """
    cursor = mysql_cnx.cursor()
    for file_name in os.listdir(base_path):
        if not file_name.endswith(".txt.gz"):
            continue

        gz_filepath = os.path.join(base_path, file_name)
        txt_filepath = os.path.join(temp_path, file_name.replace(".gz", ""))
        with gzip.open(gz_filepath, "rb") as f_in:
            with open(txt_filepath, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        cursor.execute(base_sql_line.format(txt_filepath))
        mysql_cnx.commit()

        print(txt_filepath, flush=True)
        os.remove(txt_filepath)


cnx = connect_mysql(ip="127.0.0.1", port="3306", username="root", password="wonder")
load_txt_file_into_mysql(cnx, "/var/lib/mysql-files")
