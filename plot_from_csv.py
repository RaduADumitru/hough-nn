import os
import csv
import matplotlib.pyplot as plt
filename = os.path.join('results', 'nn', 'setup_1.csv')

processor_counts = []
execution_times = []
# columns: cores, time
with open(filename, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if row[0] == '1':
            sequential_time = row[1]
        processor_counts.append(row[0])
        execution_times.append(row[1])

# Remove the first element of the lists (Headers)
processor_counts.pop(0)
execution_times.pop(0)
# Convert the strings to floats
processor_counts = [int(count) for count in processor_counts]
execution_times = [float(time) for time in execution_times]
print(f'Sequential time: {sequential_time}')

print(processor_counts
    , execution_times)
fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
fig.suptitle(f'Parallel performance measurements', fontsize=16)
fig.subplots_adjust(hspace=0.5)

axs[0].plot(processor_counts, execution_times, marker='o')
axs[0].set(xlabel='Number of Processors', ylabel='Execution time (s)',
    title=f'Execution time vs. Number of Processors')

acceleration = [float(sequential_time) / float(execution_time) for execution_time in execution_times]
axs[1].plot(processor_counts, acceleration, marker='o')
axs[1].set(xlabel='Number of Processors', ylabel='Acceleration',
    title=f'Acceleration vs. Number of Processors')

efficiency = [acceleration[i] / int(processor_counts[i]) for i in range(len(processor_counts))]
axs[2].plot(processor_counts, efficiency, marker='o')
axs[2].set(xlabel='Number of Processors', ylabel='Efficiency',
    title=f'Efficiency vs. Number of Processors')

plt.show()