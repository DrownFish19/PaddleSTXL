import folium
import pandas as pd

# ID,dist,lat,lon
# 308511,3,38.761062,-120.569835
# 308512,3,38.761182,-120.569866
# 311831,3,38.409253,-121.48412
# 311832,3,38.409782,-121.48468
# 311844,3,38.412421,-121.484436
# 311845,3,38.406175,-121.483002

df = pd.read_csv("data/pems_stations.csv")

color_list = [
    "red",
    "blue",
    "green",
    "purple",
    "orange",
    "lightred",
    "beige",
    "darkblue",
    "darkgreen",
    "cadetblue",
    "darkpurple",
    "white",
    "pink",
    "lightblue",
    "lightgreen",
    "gray",
    "black",
    "lightgray",
]


m = [
    folium.Map(location=[df["lon"].mean(), df["lat"].mean()], zoom_start=10)
    for _ in range(12)
]

for i, row in df.iterrows():
    print(i)
    folium.Marker(
        location=[row["lat"], row["lon"]],
        tooltip=int(row["ID"]),
        icon=folium.Icon(color=color_list[int(row["dist"])]),
    ).add_to(m[int(row["dist"]) - 1])

for i in range(12):
    m[i].save(f"stations_map_{i+1}.html")
