import csv, os
from collections import defaultdict
import statistics
import matplotlib.pyplot as plt
import numpy as np

def read_csv(file_path):
    data = []
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            data.append(row)
    return data

def read_folder(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, newline='') as csvfile:
                reader = csv.DictReader(csvfile, delimiter=';')
                # Check if the required columns are present
                if all(col in reader.fieldnames for col in ['inputType', 'durationPerPixel', 'YdistancePrevTarget']): #TODO: add necessary columns here
                    # Read the contents of the CSV file and append to the data list
                    for row in reader:
                        data.append(row)
                else:
                    print(f"Ignoring file {filename} as it does not have all required columns.")
    return data

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
#data = read_csv("experimentResults (42).csv")
folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_to_analyse")
data = read_folder(folder_path)
#! --------- should outliars be filtered? ----------
data = filter_outliers_iqr(data, "durationPerPixel")

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