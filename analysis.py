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
    values = [int(row[column]) for row in data]
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    threshold = 1.5
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    filtered_data = [row for row in data if int(row[column]) >= lower_bound and int(row[column]) <= upper_bound]
    print("number of filtered trials: ", len(data) - len(filtered_data))
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
    print("of which Mouse input:", count_outliers_mouse)
    return filtered_data

def calc_stats(data, variable = "durationPerPixel"):
    stats = defaultdict(lambda: {'values': [], 'mean': 0, 'stddev': 0})
    for row in data:
        input_type = row['inputType']
        try:
            duration_per_pixel = int(row[variable])
        except ValueError:
            print("The entered variable cannot be cast to int. Please enter a variable with numeric values.")  
        stats[input_type]['values'].append(duration_per_pixel)
    for input_type, values_dict in stats.items():
        durations = values_dict['values']
        stats[input_type]['mean'] = statistics.mean(durations)
        stats[input_type]['stddev'] = statistics.stdev(durations)
    return stats

def calc_stats_diff_position(data, variable = "durationPerPixel"):
    stats_diff_position = defaultdict(lambda: {'values': [], 'mean': 0, 'stddev': 0})
    prev_position = None
    for row in data:
        input_type = row['inputType']
        try: 
            duration_per_pixel = int(row[variable])
        except ValueError:
            print("The entered variable cannot be cast to int. Please enter a variable with numeric values.")    
        position = int(row['posNumber'])
        if prev_position is not None and position != prev_position:
            stats_diff_position[input_type]['values'].append(duration_per_pixel)
        prev_position = position
    
    for input_type, values_dict in stats_diff_position.items():
        durations = values_dict['values']
        stats_diff_position[input_type]['mean'] = statistics.mean(durations)
        stats_diff_position[input_type]['stddev'] = statistics.stdev(durations)
    
    return stats_diff_position

def calc_stats_diff_screens(data, variable = "durationPerPixel"):
    stats_diff_screens = defaultdict(lambda: {'values': [], 'mean': 0, 'stddev': 0})
    prev_target_screen = None
    for row in data:
        input_type = row['inputType']
        try:
            duration_per_pixel = int(row[variable])
        except ValueError:
            print("The entered variable cannot be cast to int. Please enter a variable with numeric values.")  
        target_screen = row['targetOnMainScreen']
        if prev_target_screen is not None and target_screen != prev_target_screen:
            stats_diff_screens[input_type]['values'].append(duration_per_pixel)
        prev_target_screen = target_screen
    
    for input_type, values_dict in stats_diff_screens.items():
        durations = values_dict['values']
        stats_diff_screens[input_type]['mean'] = statistics.mean(durations)
        stats_diff_screens[input_type]['stddev'] = statistics.stdev(durations)
    
    return stats_diff_screens


def plot_duration_vs_ydistance(data):
    input_types = set()
    for row in data:
        input_types.add(row['inputType'])
    
    color_map = {}
    for i, input_type in enumerate(input_types):
        color_map[input_type] = f'C{i}'
    
    for input_type in input_types:
        x_values = []
        y_values = []
        for row in data:
            if row['inputType'] == input_type:
                x_values.append(int(row['YdistancePrevTarget']))
                y_values.append(int(row['duration']))
        plt.scatter(x_values, y_values, color=color_map[input_type], label=input_type)

    plt.xlabel('YdistancePrevTarget')
    plt.ylabel('Duration')
    plt.title('Duration vs YdistancePrevTarget')
    plt.legend()
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

mean_duration_per_pixel = calc_stats(data, "durationPerPixel")
mean_duration_per_pixel_diff_position = calc_stats_diff_position(data, "durationPerPixel")
mean_duration_per_pixel_diff_screens = calc_stats_diff_screens(data, "durationPerPixel")
    
print("Mean Duration Per Pixel:")
for input_type, stats in mean_duration_per_pixel.items():
    print(f"{input_type}: M = {stats['mean']}, SD = {stats['stddev']}")
    
print("\nMean Duration Per Pixel for Different Positions:")
for input_type, stats in mean_duration_per_pixel_diff_position.items():
    print(f"{input_type}: M = {stats['mean']}, SD = {stats['stddev']}")

print("\nMean Duration Per Pixel for Different Screens:")
for input_type, stats in mean_duration_per_pixel_diff_screens.items():
    print(f"{input_type}: M = {stats['mean']}, SD = {stats['stddev']}")

plot_duration_vs_ydistance(data)