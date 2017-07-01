import glob
import csv
from Utils import radixSortPassengers

# compute MBR of dataset
x_min, y_min, x_max, y_max = 90, 180, -90, -180
with open("dataset/tdrive/passengers.txt") as worker_file:
    reader = csv.reader(worker_file, delimiter=',')
    for row in reader:
        x_min = min(float(row[0]), x_min)
        x_max = max(float(row[0]), x_max)
        y_min = min(float(row[1]), y_min)
        y_max = max(float(row[1]), y_max)
print (x_min, y_min, x_max, y_max)


# read data from file to vehicles and passengers
vehicles, passengers = {}, []
fileCount = 0
for file in glob.glob("dataset/20121101/*.txt"):
    with open(file) as f_in:
        for line in f_in:
            if line != "\n":
                parts = line.split(",")
                gpsState = int(parts[8])
                if gpsState == 1: # valid GPS
                    pickup_dropoff = int(parts[1])
                    if pickup_dropoff in [0, 1]: # either pickup or dropoff
                        vehicleId = int(parts[0])
                        time = parts[3]
                        lon, lat = float(parts[4]), float(parts[5])

                        if pickup_dropoff == 1 and vehicleId not in vehicles: # (Become loaded)
                            vehicles[vehicleId] = (lat, lon)
                        elif pickup_dropoff == 0: # (Become Empty)
                            passengers.append((lat, lon, time))

    fileCount += 1
    if fileCount % 10 == 0:
        print ("Processed " + str(fileCount) + " files\t", str(len(vehicles)) + " vehicles\t", str(len(passengers)) + " vehicles")

# write vehicles to file
with open("dataset/tdrive/vehicles.txt", "w") as csv_file:
    writer = csv.writer(csv_file)
    for vid, loc in vehicles.items():
       writer.writerow([loc[0], loc[1], vid])

# sort passengers in decreasing order of time
passengers = radixSortPassengers(passengers, 2)

# write passengers to file
with open("dataset/tdrive/passengers.txt", "w") as csv_file:
    writer = csv.writer(csv_file)
    for tuple in passengers:
        writer.writerow([tuple[0], tuple[1], int(tuple[2])])