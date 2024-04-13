import matplotlib.pyplot as plt
import time


def plot_function(f, image, image_name, processor_min, processor_max):

    fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    fig.suptitle(f'{f.__name__} on {image_name}', fontsize=16)
    fig.subplots_adjust(hspace=0.5)
    
    # plot execution time, acceleration, and efficiency of a function f with different number of processors
    execution_times = []
    processors = list(range(processor_min, processor_max + 1))
    for num_processors in processors:
        start_time = time.time()
        f(num_processors, image)
        end_time = time.time()
        execution_time = end_time - start_time
        execution_times.append(execution_time)

    axs[0].plot(processors, execution_times)
    axs[0].set(xlabel='Number of Processors', ylabel='Execution time (s)',
       title=f'Execution time vs. Number of Processors')
    
    # get time of sequential execution
    start_time = time.time()
    f(1, image)
    end_time = time.time()
    sequential_execution_time = end_time - start_time

    # calculate acceleration
    acceleration = [sequential_execution_time / execution_time for execution_time in execution_times]
    axs[1].plot(processors, acceleration)
    axs[1].set(xlabel='Number of Processors', ylabel='Acceleration',
       title=f'Acceleration vs. Number of Processors')

    # calculate efficiency
    efficiency = [acceleration[i] / processors[i] for i in range(len(processors))]
    axs[2].plot(processors, efficiency)
    axs[2].set(xlabel='Number of Processors', ylabel='Efficiency',
       title=f'Efficiency vs. Number of Processors')
    
    plt.show()
