# Comparative Analysis of Levenberg--Marquardt Algorithm and Particle Swarm Optimisation for Time Series and Regression Tasks

## Abstract:
This paper analyses how the Levenberg--Marquardt backpropagation algorithm (LMA) and the Particle Swarm Optimisation (PSO) can be used as a training algorithm of Artificial Neural Networks. In this view, four data sets (two of them for time-series analysis and the other two for fitting applications) were tested with different ANN architectures. The comparison between the two algorithms was made in terms of lower Mean Squared Error (MSE) value and in terms of the coefficient  of  correlation (R). Our results showed that the LMA worked better for time-series analysis and PSO for data fitting tasks.

## Results:
This study highlights two important concepts: time-series analysis and data fitting. It includes using two data sets: milk production and urban traffic behaviours in the São Paulo data set for time-series analysis and QSAR fish toxicity and wine data set for data fitting. 

The experiments were conducted altering the number of neurons in the hidden layer. Two optimisation algorithms (Levenberg-Maquardt algorithm and Particle Swarm Optimisation) were used for both tasks. LMA worked better for time-series analysis and PSO for data fitting tasks. This could be due to the ability of PSO to handle high dimensionality of data than LMA. 

On the one hand, the LMA algorithm was more stable than the PSO algorithm, especially in regression tasks; however, for the same accuracy, the PSO algorithm required a lower number of hidden units in the red and white wine data sets, as can be seen in the following table:

|              | **LMA**        |              |        | **PSO**        |              |        |
| ------------ | -------------- | ------------ | ------ | -------------- | ------------ | ------ |
| **Data set** | **No. Hidden** | **Best MSE** | **R**  | **No. Hidden** | **Best MSE** | **R**  |
| Traffic      | 4              | 0.0584       | 0.9870 | 4              | 101.7226     | 0.2068 |
| Milk         | 4              | 1.2323       | 0.9937 | 12             | 2152.8343    | 0.8150 |
| Fish         | 4              | 0.6901       | 0.7446 | 7              | 0.7354       | 0.7289 |
| Red wine     | 15             | 0.4813       | 0.5534 | 7              | 0.4779       | 0.5430 |
| White wine   | 20             | 0.4788       | 0.4660 | 7              | 0.5020       | 0.4176 |

As future work, we are going to test the PSO algorithm with different parameters, different velocity update equations and different topologies of swarms.

Before using the algorithms to find the optimal ANN's weights, feature selection methods were used for this study and were based on the variance of each feature and on its rank based on its p-values of F-test statistics.

Although for other data sets, combination of the average MSE and R value have suggested neural networks in the way that both were pointing to the same network architecture, this may not always be the case. Finally, it can be concluded that selection of algorithm is dependent on the type of task, as seen in the study.