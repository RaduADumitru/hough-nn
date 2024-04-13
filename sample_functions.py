import time

def sample_function(num_processors : int = 1, image = None):
    # Simulate a function that takes time to run, with the time taken inversely proportional to the number of processors
    wait_time = 1 / num_processors
    time.sleep(wait_time)