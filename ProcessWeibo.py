from Params import Params

DELIM = "\t"

"""
Filter out checkins in a particular region
OBJECTID,0
poiid,1
CTYPE,2
checkin_nu,3
categorys,4
photo_num,5
todo_num,6
herenow_us,7
checkin_us,8
tip_num,9
distance,10
weibo_id,11
pintu,12
enterprise,13
category,14
x,15
y,16
"""
def filter_weibo(param):
    with open("dataset/weibo/checkins.csv") as f:
        with open("dataset/weibo/checkins_filtered.txt", "w") as f_out:
            lines = ""
            next(f)
            for line in f:
                parts = line.split(",")
                lat, lon = float(parts[15]), float(parts[16])
                if param.x_min <= lat <= param.x_max and param.y_min <= lon <= param.y_max:
                    poiid, checkin_us = parts[1], parts[8]
                    lines += DELIM.join(list(map(str, [poiid, lat, lon, checkin_us, "\n"])))

            f_out.write(lines)

    f.close()
    f_out.close()

p = Params(1000)
p.select_dataset()
filter_weibo(p)