# Queue management simulation and visualization 

This is the website deployed with streamlit package for Operations Management course.
Its goal is to perform simulation of queueing based on the parameters provided by the user.
The simulation models 3 queueing  techniques: 
- Pooling
- Random assignment
- Allocation to shortest line

## Inputs

User can set the following parameters:
- Simulation time
- number of processing stations
- Customer arrival time distribution with distribution parameters
- Processing time distribution with distribution parameters

## Outputs
After the end of a simulation process the following interactive visualizations and analyses are produced showing comparison between performance of 3 queueing  techniques:
- Instant utilization
- Average utilization
- Customer average waiting time in line
- Average number of customers in line
- Estimations of resulting values according to Little's Law with comparison to valuees produced by the model
