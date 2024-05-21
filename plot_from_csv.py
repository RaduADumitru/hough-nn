import os
import csv
filename = os.path.join('results', 'setup_1.csv')

# columns: cores, time
with open(filename, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        # Process each row of the CSV file here
        # Access the columns using row[index]
        # For example, row[0] will give you the value in the first column
        if row[0] == '1':
            sequential_time = row[1]
            break

print(f'Sequential time: {sequential_time}')

with open(filename, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        pass
        