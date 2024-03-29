import csv, os
from collections import defaultdict
import statistics
import matplotlib.pyplot as plt
import numpy as np

data = []
def read_csv(filepath):
    trial_num_per_input_type = defaultdict(int)
    if filepath.endswith(".csv"):
        with open(filepath, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            # Check if the required columns are present
            if all(col in reader.fieldnames for col in ['inputType', 'durationPerPixel', 'YdistancePrevTarget']): #TODO: add necessary columns here
                prev_screen = None
                prev_pos = None
                for row in reader:
                    # cast from string to int if numeric value
                    for col in row:
                        val = row[col]
                        try:
                            row[col] = int(val)
                        except ValueError:
                            pass #leave it a string  

                    # add trial number for the current input type, starting from 0
                    input_type = row['inputType']
                    row['trial_num'] = trial_num_per_input_type[input_type]
                    trial_num_per_input_type[input_type] += 1

                    # add diffScreen and diffPos column
                    current_screen = row['targetOnMainScreen']
                    current_pos = row["posNumber"]
                    if prev_screen is not None and current_screen != prev_screen:
                        row["diffScreen"] = True
                    else:
                        row["diffScreen"] = False
                    if prev_pos is not None and current_pos != prev_pos:
                        row["diffPos"] = True
                    else:
                        row["diffPos"] = False
                    prev_screen = current_screen
                    prev_pos = current_pos
                    data.append(row)
            else:
                print(f"Ignoring file {filepath} as it does not have all required columns.")
    else:
        print(f"Ignoring file {filepath} as it is not a CSV-file.")
    return data

def read_folder(folderpath):
    for filename in os.listdir(folderpath):
        filepath = os.path.join(folderpath, filename)
        read_csv(filepath)
    return data

def filter_first_trial(data):
    filtered_data = []
    for row in data:
        if (row['trial_num']) > 0:
            filtered_data.append(row)
        else:
            print("First trial removed.")
    return filtered_data

# filter outliers using interquartile range (IQR) -> threshold: 1.5 * IQR (difference between first and third quartile)
def filter_outliers_iqr(data, column):
    values = [row[column] for row in data]
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    threshold = 1.5
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    print(f"lower bound: {lower_bound}, upper bound: {upper_bound}")
    filtered_data = []
    outliers = []
    for row in data:
        value = row[column]
        if value >= lower_bound and value <= upper_bound:
            filtered_data.append(row)
        else:
            outliers.append(row)
    count_outliers_mouse = 0
    for outlier in outliers:
        print(f"{outlier['inputType']}, trial: {outlier['trial_num']}, value: {outlier['durationPerPixel']}")
        if(outlier['inputType'] == 'Mouse'):
            count_outliers_mouse += 1
    print("number of filtered outliers: ", len(data) - len(filtered_data))
    print("    of which Mouse input:", count_outliers_mouse)
    return filtered_data

def filter_errors_aborted(data):
    filtered_data = []
    for row in data:
        count_outliers_total = 0
        count_outliers_mouse = 0
        if row['errors'] > 0 or row['aborted'] == 'true':
            count_outliers_total += 1
            if row['inputType'] == 'Mouse':
                count_outliers_mouse += 1
        else:
            filtered_data.append(row)
    print("number of filtered trials (errors or aborted): ", count_outliers_total)
    print("    of which Mouse input:", count_outliers_mouse)
    return filtered_data

def calc_stats(data, variable = "durationPerPixel"):
    stats = defaultdict(lambda: {'values': [], 'mean': 0, 'stddev': 0})
    for row in data:
        input_type = row['inputType']
        duration_per_pixel = row[variable]
        stats[input_type]['values'].append(duration_per_pixel)
    for input_type, values_dict in stats.items():
        durations = values_dict['values']
        stats[input_type]['mean'] = statistics.mean(durations)
        stats[input_type]['stddev'] = statistics.stdev(durations)
    return stats

def calc_stats_diffVar(data, variable = "durationPerPixel", diffVar = "diffScreen"):
    stats_diffVar = defaultdict(lambda: {'values': [], 'mean': 0, 'stddev': 0})
    for row in data:
        input_type = row['inputType']
        diffVarTrue = row[diffVar]
        if diffVarTrue:
            stats_diffVar[input_type]['values'].append(row[variable])
    for input_type, values_dict in stats_diffVar.items():
        durations = values_dict['values']
        stats_diffVar[input_type]['mean'] = statistics.mean(durations)
        stats_diffVar[input_type]['stddev'] = statistics.stdev(durations)
    return stats_diffVar

def calc_stats_per_size(data, variable="durationPerPixel"):
    stats_by_sizes = defaultdict(lambda: {'small': {'values': [], 'mean': 0, 'stddev': 0},
                                           'large': {'values': [], 'mean': 0, 'stddev': 0}})
    for row in data:
        input_type = row['inputType']
        size = row['size']
        value = row[variable]
        stats_by_sizes[input_type][size]['values'].append(value)
    for input_type, sizes_dict in stats_by_sizes.items():
        for size, values_dict in sizes_dict.items():
            durations = values_dict['values']
            if durations:
                sizes_dict[size]['mean'] = statistics.mean(durations)
                sizes_dict[size]['stddev'] = statistics.stdev(durations)
    return stats_by_sizes

def calc_stats_per_size_diffVar(data, variable="durationPerPixel", diffVar="diffScreen"):
    stats_perSize_diffVar = defaultdict(lambda: {'small': {'values': [], 'mean': 0, 'stddev': 0},
                                           'large': {'values': [], 'mean': 0, 'stddev': 0}})
    # Calculate statistics for each size
    for row in data:
        input_type = row['inputType']
        diffVarTrue = row[diffVar]
        size = row['size']
        if diffVarTrue:
            stats_perSize_diffVar[input_type][size]['values'].append(row[variable])
    for input_type, sizes_dict in stats_perSize_diffVar.items():      
        for size, values_dict in sizes_dict.items():
            durations = values_dict['values']
            if durations:
                sizes_dict[size]['mean'] = statistics.mean(durations)
                sizes_dict[size]['stddev'] = statistics.stdev(durations)
    return stats_perSize_diffVar

def add_eyePercentage_to_data():
    new_column_header = 'eyePercentage'
    for row in data:
        if row['duration'] != '0':  # Avoid division by zero
            row[new_column_header] = row['eyeIntervalsDuration'] / row['duration']
        else:
            row[new_column_header] = 0  # If 'duration' is zero, set the new value to zero
    return data


marker_map = {
    "Mouse": {
        "large": ['#ff7f0e', "o"],
        "small": ['#ff7f0e', "s"] #'#ffbb78'
    },
    "Eye + Mouse": {
       "large": ['#1f77b4', "o"],
       "small": ['#1f77b4', "s"] #'#aec7e8'
    }
}

def plot_duration_vs_ydistance(data):
    plt.figure()
    for row in data:
        size = row['size']
        input_type = row['inputType']
        color = marker_map[input_type][size][0]
        marker = marker_map[input_type][size][1]
        plt.scatter(row['YdistancePrevTarget'], row['duration'], color=color, marker=marker, label=(input_type +", "+ size), s=20)
    plt.xlabel('Distance to previous target (px)')
    plt.ylabel('Duration (ms)')
    #annotating the x axis with arrow
    plt.annotate('', xy=(-130, -200),xytext=(150,-200),
            arrowprops=dict(arrowstyle='<->',color='#9f9f9f'),   
            annotation_clip=False)
    plt.annotate('', xy=(300, -200),xytext=(2400,-200),                     
            arrowprops=dict(arrowstyle='<->',color='#9f9f9f'),  
            annotation_clip=False)       
    plt.annotate('', xy=(1100, -270),xytext=(2400,-270),                     
            arrowprops=dict(arrowstyle='<->',color='#9f9f9f'),  
            annotation_clip=False)                         
    #text annotation for arrow
    plt.annotate('same pos',xy=(-140,-320),xytext=(-140,-320), color='#9f9f9f',
            annotation_clip=False)
    plt.annotate('different pos',xy=(400,-320),xytext=(400,-320), color='#9f9f9f',
            annotation_clip=False)
    plt.annotate('different screen',xy=(1200,-390),xytext=(1200,-390), color='#9f9f9f',
            annotation_clip=False)
    
    plt.title('Duration vs Distance')
    #remove duplicate labels by putting them in dict
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.show()

# !--------- enter single csv file or or use files from data_to_analyse folder ------------
data = read_csv("analysis\data_to_analyse\experimentResults (42).csv")

#folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_to_analyse")
#data = read_folder(folder_path)
#! --------- how should outliars be filtered? ----------
print("\n------------- Filter ----------------")
#data = filter_outliers_iqr(data, "durationPerPixel") #! simply filtering all outliers is usually not legitimate (unless they occur for a reason that makes filtering them meaningful)
data = filter_errors_aborted(data)
data = filter_first_trial(data)

print("\n-------------- DurationPerPixel --------------")
mean_durationPerPixel = calc_stats(data, "durationPerPixel")
for input_type, stats in mean_durationPerPixel.items():
    print(f"{input_type}: M = {stats['mean']}, SD = {stats['stddev']}")
    
print("\n    ----- only for different positions:")
mean_durationPerPixel_diffPosition = calc_stats_diffVar(data, "durationPerPixel", "diffPos")
for input_type, stats in mean_durationPerPixel_diffPosition.items():
    print(f"    {input_type}: M = {stats['mean']}, SD = {stats['stddev']}")

print("\n    ----- only for different screens:")
mean_durationPerPixel_diffScreen = calc_stats_diffVar(data, "durationPerPixel", "diffScreen")
for input_type, stats in mean_durationPerPixel_diffScreen.items():
    print(f"    {input_type}: M = {stats['mean']}, SD = {stats['stddev']}")

print("\n-------------- DurationPerPixel - per sizes --------------")
mean_durationPerPixel_perSize = calc_stats_per_size(data, variable="durationPerPixel")
for input_type, sizes_dict in mean_durationPerPixel_perSize.items():
    print(f"{input_type}")
    for size, values_dict in sizes_dict.items():
        print(f"{size}")
        print(f"M = {values_dict['mean']}, SD = {values_dict['stddev']}")
    print()

print("\n    ----- only for different positions:")
mean_durationPerPixel_perSize_diffPos = calc_stats_per_size_diffVar(data, "durationPerPixel", "diffPos")
for input_type, sizes_dict in mean_durationPerPixel_perSize_diffPos.items():
    print(f"    {input_type}")
    for size, values_dict in sizes_dict.items():
        print(f"    {size}")
        print(f"    M = {values_dict['mean']}, SD = {values_dict['stddev']}")
    print()

print("\n    ----- only for different screens:")
mean_durationPerPixel_perSize_diffScreens = calc_stats_per_size_diffVar(data, "durationPerPixel", "diffScreen")
for input_type, sizes_dict in mean_durationPerPixel_perSize_diffScreens.items():
    print(f"    {input_type}")
    for size, values_dict in sizes_dict.items():
        print(f"    {size}")
        print(f"    M = {values_dict['mean']}, SD = {values_dict['stddev']}")
    print()

print("\n-------------- eyeIntervalsDuration --------------")

add_eyePercentage_to_data()
print("\nEye Percentage:")
mean_eyePercentage = calc_stats(data, "eyePercentage")
for input_type, stats in mean_eyePercentage.items():
    print(f"{input_type}: M = {stats['mean']}, SD = {stats['stddev']}")


plot_duration_vs_ydistance(data)