import csv, os
from collections import defaultdict
import statistics
import pandas as pd
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
        if not 'numberInBlock' in row or not isinstance(row['numberInBlock'], (int, float)):
            print("Row with missing or invalid 'numberInBlock' detected.")
        else:
            if (row['numberInBlock']) > 0:
                filtered_data.append(row)
            else:
                print("First trial removed.")
    return filtered_data

def filter_outliers_mad(data, column):
    values = [row[column] for row in data]
    median = np.median(values)
    mad = np.median(np.abs(values-median))
    threshold = 2.5 #suggested by Leys et al. (2013) as a reasonable default (adjust: threshold should be justified!)
    lower_bound = median - threshold * mad
    upper_bound = median + threshold * mad
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
        "large": ['#b6b6b6', "o"],
        "small": ['#b6b6b6', "s"]
    },
    "MAGIC": {
       "large": ['#ec7c0c', "o"],
       "small": ['#ec7c0c', "s"] 
    },
    "NINJA": {
       "large": ['#080a9e', "o"],
       "small": ['#080a9e', "s"] 
    }
    ,
    "Touchpad": {
       "large": ['#000000', "o"],
       "small": ['#000000', "s"] 
    }
}

def plot_bar_diagram(data, variable = "durationPerPixel", label='Mean Duration per Pixel (ms)', diffVar = None):
    if(diffVar):
        mean_durationPerPixel = calc_stats_diffVar(data, variable, diffVar)
    else:
        mean_durationPerPixel = calc_stats(data, variable)
        print("no diff var")
    input_types = list(mean_durationPerPixel.keys())
    means = [stats['mean'] for stats in mean_durationPerPixel.values()]
    stddevs = [stats['stddev'] for stats in mean_durationPerPixel.values()]
    plt.figure(figsize=(10, 6))
    plt.grid(axis='y', linestyle='-', alpha=0.7)
    plt.bar(input_types, means, yerr=stddevs, capsize=5, color="grey", zorder=2)
    plt.xlabel('Input Method')
    plt.ylabel(label)
    plt.xticks(rotation=0)
    for spine in plt.gca().spines.values():
        spine.set_visible(False) # Remove the border around the plot
    plt.tight_layout()
    plt.show()

def plot_bar_diagram_per_size(data, variable="durationPerPixel", label='Mean Duration per Pixel (ms)', diffVar=None, diffVarLabel=""):
    if diffVar:
        mean_durationPerPixel = calc_stats_per_size_diffVar(data, variable, diffVar)
    else:
        mean_durationPerPixel = calc_stats_per_size(data, variable)
    input_types = list(mean_durationPerPixel.keys())
    sizes = sorted(set(size for input_type in mean_durationPerPixel.keys() for size in mean_durationPerPixel[input_type].keys()))
    bar_width = 0.15
    plt.figure()
    plt.grid(axis='y', linestyle='-', alpha=0.7)
    ticks = [[], []]
    for index_size, size in enumerate(sizes):
        for index_input_types, input_type in enumerate(input_types):
            if(input_type in marker_map):
                mean = [mean_durationPerPixel[input_type][size]['mean']]
                stddev = [mean_durationPerPixel[input_type][size]['stddev']]
                color = marker_map[input_type][size][0]
                plt.bar(index_size + index_input_types* bar_width, mean, bar_width, yerr=stddev, capsize=5, color=color, zorder=2)
                ticks[0].append(index_size + index_input_types* bar_width)
                ticks[1].append(f"{size}\n{input_type}")
    plt.ylabel(label)
    plt.xticks(ticks[0], ticks[1], rotation=0)
    plt.title(diffVarLabel)
    for spine in plt.gca().spines.values():
        spine.set_visible(False) # Remove the border around the plot
    plt.tight_layout()
    plt.show()

def plot_duration_vs_ydistance(data, bucket_size=1):
    df = pd.DataFrame(data)

    plt.figure()

    for input_type in df['inputType'].unique():
        input_df = df[df['inputType'] == input_type].copy()

        input_df['bucketed_distance'] = (input_df['XdistancePrevTarget'] // bucket_size) * bucket_size

        avg_data = input_df.groupby('bucketed_distance')['duration'].agg(['mean', 'count', 'std']).reset_index()
        avg_data['sem'] = avg_data['std'] / np.sqrt(avg_data['count'])

        for size in input_df['size'].unique():
            size_df = input_df[input_df['size'] == size]
            
            if input_type in marker_map:
                color = marker_map[input_type][size][0]
                marker = marker_map[input_type][size][1]
                # scatterplot
                plt.scatter(size_df['XdistancePrevTarget'], size_df['duration'], 
                            color=color, marker=marker, label=(f"{input_type}, {size}"), s=20)
                # errorbars
                plt.errorbar(avg_data['bucketed_distance'], avg_data['mean'], 
                             yerr=avg_data['sem'], fmt='o', color=color, 
                             label=f'Avg Duration ({input_type}, bucket size={bucket_size})', 
                             markersize=6, linestyle='--', capsize=5)

    plt.xlabel('Distance to previous target (px)')
    plt.ylabel('Duration (ms)')
    plt.title('Duration vs Distance')

    # remove duplicate labels from the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.show()

# !--------- enter single csv file or or use files from data_to_analyse folder ------------
#data = read_csv("analysis\data_to_analyse\experimentResults (52).csv")

folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_to_analyse")
data = read_folder(folder_path)
#! --------- how should outliars be filtered? ----------
print("\n------------- Filter ----------------")
#data = filter_outliers_mad(data, "durationPerPixel") #! simply filtering all outliers is usually not legitimate (unless they occur for a reason that makes filtering them meaningful)
data = filter_errors_aborted(data)
data = filter_first_trial(data)

# print("\n-------------- DurationPerPixel --------------")
# mean_durationPerPixel = calc_stats(data, "durationPerPixel")
# for input_type, stats in mean_durationPerPixel.items():
#     print(f"{input_type}: M = {stats['mean']}, SD = {stats['stddev']}")
    
# print("\n    ----- only for different positions:")
# mean_durationPerPixel_diffPosition = calc_stats_diffVar(data, "durationPerPixel", "diffPos")
# for input_type, stats in mean_durationPerPixel_diffPosition.items():
#     print(f"    {input_type}: M = {stats['mean']}, SD = {stats['stddev']}")

# print("\n    ----- only for different screens:")
# mean_durationPerPixel_diffScreen = calc_stats_diffVar(data, "durationPerPixel", "diffScreen")
# for input_type, stats in mean_durationPerPixel_diffScreen.items():
#     print(f"    {input_type}: M = {stats['mean']}, SD = {stats['stddev']}")

# print("\n-------------- DurationPerPixel - per sizes --------------")
# mean_durationPerPixel_perSize = calc_stats_per_size(data, variable="durationPerPixel")
# for input_type, sizes_dict in mean_durationPerPixel_perSize.items():
#     print(f"{input_type}")
#     for size, values_dict in sizes_dict.items():
#         print(f"{size}")
#         print(f"M = {values_dict['mean']}, SD = {values_dict['stddev']}")
#     print()

# print("\n    ----- only for different positions:")
# mean_durationPerPixel_perSize_diffPos = calc_stats_per_size_diffVar(data, "durationPerPixel", "diffPos")
# for input_type, sizes_dict in mean_durationPerPixel_perSize_diffPos.items():
#     print(f"    {input_type}")
#     for size, values_dict in sizes_dict.items():
#         print(f"    {size}")
#         print(f"    M = {values_dict['mean']}, SD = {values_dict['stddev']}")
#     print()

# print("\n    ----- only for different screens:")
# mean_durationPerPixel_perSize_diffScreens = calc_stats_per_size_diffVar(data, "durationPerPixel", "diffScreen")
# for input_type, sizes_dict in mean_durationPerPixel_perSize_diffScreens.items():
#     print(f"    {input_type}")
#     for size, values_dict in sizes_dict.items():
#         print(f"    {size}")
#         print(f"    M = {values_dict['mean']}, SD = {values_dict['stddev']}")
#     print()

# print("\n-------------- eyeIntervalsDuration --------------")

# add_eyePercentage_to_data()
# print("\nEye Percentage:")
# mean_eyePercentage = calc_stats(data, "eyePercentage")
# for input_type, stats in mean_eyePercentage.items():
#     print(f"{input_type}: M = {stats['mean']}, SD = {stats['stddev']}")


plot_duration_vs_ydistance(data)
# plot_bar_diagram(data, "durationPerPixel", "Duration per Pixel (ms)")
# plot_bar_diagram_per_size(data, "durationPerPixel", 'Mean Duration per Pixel (ms)')
# plot_bar_diagram_per_size(data, "durationPerPixel", 'Mean Duration per Pixel (ms)', "diffScreen", "Results for different Screens")
