import streamlit as st
from functions import geneticprogamming
from functions import start_new_project
from functions import verification
import pandas as pd

st.write("""
# Interactive Decision Support Tool
This is an interactive Tool to help you to optimize your production schedule.
""")

# set parameters for genetic programming
st.sidebar.header('Select the genetic programming parameter')
nb_generations = st.sidebar.slider('number of generations', min_value=1, max_value=100, value=50, step=1)
population_size = st.sidebar.slider('population size', min_value=1, max_value=500, value=200, step=1)
crossover_probability = st.sidebar.slider('crossover probability', min_value=0.01, max_value=0.99, value=0.90, step=0.01)
mutation_probability = st.sidebar.slider('mutation probability', min_value=0.01, max_value=0.99, value=0.10, step=0.01)
max_depth_crossover = st.sidebar.slider('maximum depth for crossover', min_value=1, max_value=20, value=17, step=1)
max_depth_mutation = st.sidebar.slider('maximum depth for mutation', min_value=1, max_value=20, value=7, step=1)

# set parameter for fitness evaluation
st.sidebar.header('Select the fitness evaluation parameter')
nb_simulations = st.sidebar.slider('number of simulation runs', min_value=1, max_value=100, value=10, step=1)

# Performance Names; here fixed values; later as an option to type in or select in app
performance_name = ["makespan", "number tardy jobs", "total tardiness"]

st.header('Start a new project')
project_name = st.text_input("Project name", key="project name")
if st.button("Create a new project"):
    start_new_project(project_name, performance_name)

st.header('Approximate the ideal and nadir point')
if st.button("Initialize"):
    for i in range(3):
        geneticprogamming(performance_measure=i, ref_point=None, opt_type="best", project_name=project_name,
                          reference_point_input=[0,0,0], nb_generations=nb_generations, population_size=population_size,
                          crossover_probability=crossover_probability, mutation_probability=mutation_probability,
                          max_depth_crossover=max_depth_crossover, max_depth_mutation=max_depth_mutation,
                          nb_simulations=1)
        geneticprogamming(performance_measure=i, ref_point=None, opt_type="worst", project_name=project_name, reference_point_input=[0,0,0],
                          nb_generations=nb_generations, population_size=population_size,
                          crossover_probability=crossover_probability, mutation_probability=mutation_probability,
                          max_depth_crossover=max_depth_crossover, max_depth_mutation=max_depth_mutation,
                          nb_simulations=1)
    st.stop()

try:
    initialization_table = pd.read_excel(project_name+'.xlsx', header=0,
                            index_col=0)
    st.write(initialization_table)
except:
    st.error('Oops: the project could not be found! Please check the entered name or create a new project first')
    st.stop()

st.header('Select the reference point')
reference_point = [0,0,0]
reference_point[0] = st.slider('Makespan', min_value=float(initialization_table.at['makespan', 'ideal point']), max_value=float(initialization_table.at['makespan', 'nadir point']))
reference_point[1] = st.slider('Number of tardy jobs', min_value=float(initialization_table.at['number tardy jobs', 'ideal point']), max_value=float(initialization_table.at['number tardy jobs', 'nadir point']))
reference_point[2] = st.slider('Total tardiness', min_value=float(initialization_table.at['total tardiness', 'ideal point']), max_value=float(initialization_table.at['total tardiness', 'nadir point']))

st.header('Evolve dispatching rule')
if st.button("run"):
    geneticprogamming(performance_measure=1, ref_point=True, opt_type="best", project_name=project_name, reference_point_input=reference_point,
                      nb_generations=nb_generations, population_size=population_size,
                      crossover_probability=crossover_probability, mutation_probability=mutation_probability,
                      max_depth_crossover=max_depth_crossover, max_depth_mutation=max_depth_mutation,
                      nb_simulations=nb_simulations)
    function_data=pd.read_excel(project_name + '.xlsx', sheet_name="ref_point", header=0,
                  index_col=0)
    func= function_data.iloc[-1,1]
    verification(performance_measure=1, ref_point=True, reference_point_input=reference_point, best_func=func)




