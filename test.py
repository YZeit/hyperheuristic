import matplotlib.pyplot as plt
import numpy as np
from functions import geneticprogamming
from functions import geneticprogamming_speedup
import time

start_slow = time.time()
geneticprogamming(performance_measure=1, ref_point=True, opt_type="best", project_name="Test1234",
                  reference_point_input=[0, 0, 0], nb_generations=10, population_size=20,
                  crossover_probability=0.9, mutation_probability=0.1,
                  max_depth_crossover=17, max_depth_mutation=7,
                  nb_simulations=5)
end_slow = time.time()

start_fast = time.time()
geneticprogamming_speedup(performance_measure=1, ref_point=True, opt_type="best", project_name="Test1234",
                  reference_point_input=[0, 0, 0], nb_generations=10, population_size=20,
                  crossover_probability=0.9, mutation_probability=0.1,
                  max_depth_crossover=17, max_depth_mutation=7,
                  nb_simulations=5)
end_fast = time.time()

print('time needed slow\t\t', end_slow-start_slow)
print('time needed fast\t\t', end_fast-start_fast)
# Creating dataset

liste = [2, 4, 6, 10]
