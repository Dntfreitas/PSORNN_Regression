# Abstract:
This paper analyses how the Levenberg--Marquardt backpropagation algorithm (LMA) and the Particle Swarm Optimisation (PSO) can be used as a training algorithm of Artificial Neural Networks. In this view, four data sets (two of them for time-series analysis and the other two for fitting applications) were tested with different ANN architectures. The comparison between the two algorithms was made in terms of lower Mean Squared Error (MSE) value and in terms of the coefficient  of  correlation (R). Our results showed that the LMA worked better for time-series analysis and PSO for data fitting tasks.

# Results:

The experiments were conducted altering the number of neurons in the hidden layer. Two optimisation algorithms (Levenberg-Maquardt algorithm and Particle Swarm Optimisation) were used for both tasks. LMA worked better for time-series analysis and PSO for data fitting tasks. This could be due to the ability of PSO to handle high dimensionality of data than LMA. 

On the one hand, the LMA algorithm was more stable than the PSO algorithm, especially in regression tasks; however, for the same accuracy, the PSO algorithm required a lower number of hidden units, as can be seen in the following table:

|              | **LMA**        |              |       | **PSO**        |              |        |
| ------------ | -------------- | ------------ | ----- | -------------- | ------------ | ------ |
| **Data set** | **No. Hidden** | **Best MSE** | **R** | **No. Hidden** | **Best MSE** | **R**  |
| Traffic      | 4              | 0,037        | 0,994 | 4              | 139,813      | 0,277  |
| Milk         | 4              | 0,711        | 0,999 | 20             | 7599,916     | -0,412 |
| Fish         | 12             | 0,778        | 0,796 | 7              | 0,735        | 0,729  |
| Red wine     | 15             | 0,411        | 0,611 | 7              | 0,478        | 0,543  |
| White wine   | 20             | 0,577        | 0,519 | 7              | 0,502        | 0,418  |

As future work, we are going to test the PSO algorithm with different parameters, different velocity update equations and different topologies of swarms.

Before using the algorithms to find the optimal ANN's weights, feature selection methods were used for this study and were based on the variance of each feature and on its rank based on its p-values of F-test statistics.

Although for other data sets, combination of the average MSE and R value have suggested neural networks in the way that both were pointing to the same network architecture, this may not always be the case. Finally, it can be concluded that selection of algorithm is dependent on the type of task, as seen in the study.
