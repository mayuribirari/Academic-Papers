# ENGR-E-516-ECC-Final-Project

### Large-scale Artificial Neural Network: MapReduce-based for Machine Learning

### Project goals:

- Understanding how the mapreduce framework works.
- Implementing mapreduce framework using hadoop.
- Using the mrjob library in python on a sample mapreduce use-case.
- Creating datasets for each model (in text form for mrjob).
- Applying mapreduce framework on several machine learning models:
  - Linear regression
  - Logistic regression
  - k-Nearest neighbours
  - Naive bayes
  - Digit classifier neural network
- Implement all the above algorithms using the mapreduce framework (creating mappers and reducers)
- Converting the mapreduce mappers and reducers to mrjob.
- Verifying the results using the scikit-learns implementation of these models.
- Implementing neural network based deep learning using mapreduce framework.
- Comparing the results of mapreduce vs training on single instance.
- Running the algorithms on Jetstream instances parallely.

### Related work and gap analysis:

The paper "Large-scale distributed deep networks" describes the architecture and techniques used by Google to train very large deep neural networks on distributed computing systems. The authors present a system called "DistBelief" that is capable of training neural networks with billions of parameters on thousands of CPU and GPU processors. The system uses a data-parallel approach to distribute the computations and the data across the nodes in the network, and it includes several optimizations to improve efficiency and scalability, such as model and data parallelism, asynchronous SGD, and parameter server architectures.
The paper "Large-scale Artificial Neural Network: MapReduce-based Deep Learning" proposes a system for training large-scale artificial neural networks using the MapReduce paradigm. Although the paper offers a comprehensive analysis of the suggested system, there are certain possible research gaps that might be addressed in subsequent work, such as:
- The proposed system is evaluated on a limited set of datasets. It would be useful to evaluate the system on a wider range of datasets to determine its performance characteristics in different scenarios.
- The paper focuses on the training phase of large-scale neural networks, but does not address issues related to deployment and inference. Future work could explore ways to efficiently deploy and infer large-scale neural networks in a distributed setting.

Overall, the paper "Large-scale Artificial Neural Network: MapReduce-based Deep Learning" makes a valuable contribution to the field of large-scale deep learning, but there is room for further research to address some of the potential gaps in the work.

### Proposed tasks

a) **Understanding and Implementing Map-reduce**: Implement the mapreduce framework on a sample use-case to understand how the framework works using the hadoop and mrjob library.

(b) **Generating dataset for machine learning models**: Generate datasets for classification, regression and digits for the classification, regression and the digit classifier model.

(c) **Implement the classification models in mapreduce architecture**: Convert each of the regression models (Linear regressiom, ridge regression, decision trees) to the mapreduce architecture, implement these algorithms from scratch using mrjob library in python.

(d) **Implement the classification models in mapreduce architecture**: Convert each of the regression models (Logistic regression, kNN, naive bayes) to the mapreduce architecture, implement these algorithms from scratch using mrjob library in python.

(e) **Compare the models**: Compare the results from the mapreduce architecture with the results from scikit-learn implementation of the models.

(f) **Train classifier using the mapreduce framework**: Implement a simple digit classifier and train using the mapreduce framework. Evaluate the model using the weights obtained from the mapreduce.

(g) **Run the algorithms on jetstream instance**: Push the code on the jetstream instance and run the mrjob to obtain results.

(h) **Compare the performance**: Compare the performance and other metrics (CPU usage, time consumption) of the mapreduce implementation and training the algorithms using scikit-learn on a single instance.



### Preliminary Overview:

We have observed that the map-reduce framework is particularly useful for applying machine learning algorithms to large datasets. This is because map-reduce allows us to distribute the processing of data across multiple nodes in a cluster, enabling us to process large amounts of data in parallel. In addition, the map-reduce framework is fault-tolerant, which means that if any of the nodes in the cluster fail during processing, the system will automatically reassign their tasks to other nodes, ensuring that the processing continues without interruption.

Through our analysis, we have found that using map-reduce to implement machine learning algorithms can significantly reduce the time required for processing large datasets compared to traditional single-machine approaches. This is because map-reduce allows us to parallelize the processing of data, which can result in significant speedups when working with large datasets. In addition, we have found that using map-reduce can be particularly effective for algorithms that involve iterative processing, such as training neural networks, as it allows us to distribute the work of each iteration across multiple nodes in the cluster, reducing the overall time required for training.

Overall, our preliminary analysis suggests that using the map-reduce framework for machine learning can offer significant advantages over traditional single-machine approaches, particularly when working with large datasets.


## Experiments:

Lets look the implementations of these algorithms using the MapReduce paradigm:

### Linear Regression:

- The function cholesky_solution_linear_regression() is defined to calculate the Cholesky decomposition of the sample covariance matrix of explanatory variables and the covariance between explanatory variables and the dependent variable. It returns a list of coefficients for the regression model.
- The function extract_variables() is defined to extract relevant features from the input file. It expects the input data to be in the form of comma-separated values, where the first value is the dependent variable and the rest are the explanatory variables.
- The **mapper_lr** function takes each line of input, extracts the relevant features using the extract_variables() function, and updates the x_t_x, x_t_y, and counts variables. If the bias option is set to True, it adds a column of ones to the feature matrix to account for the bias term.
- The **mapper_lr_final** function is called at the end of the map phase. It sends the x_t_x, x_t_y, and counts variables to the reducer by yielding a key-value pair.
- The **reducer_lr** function takes in the key-value pairs from the mappers, aggregates the x_t_x, x_t_y, and counts variables, and then applies the Cholesky decomposition algorithm to obtain the regression coefficients. It returns the coefficients as a comma-separated string.

### Logistic Regression:

- The **mapper** function reads in a line of input data, splits it into a list of values, extracts the target variable (the last value), extracts the features (all values except the last one), and yields a key-value pair where the key is 0 and the value is a tuple containing the features and target.
- The sigmoid function is a mathematical function used in logistic regression that maps any real-valued number to a value between 0 and 1.
- The **reducer** function initializes the model parameters with zeros, sets the learning rate and number of iterations, and creates an empty list to store the data. Then, it loops over the input values for the specified number of iterations and performs gradient descent to update the model parameters. For each input value, the features are converted to a numpy array, the dot product of the model parameters and the features is computed, the sigmoid function is applied to the result, and the model parameters are updated using the gradient descent formula. The updated model parameters are used to predict the target variable, and the accuracy of the predictions is computed.

### Naive Bayes:

- The **mapper** function takes in a key-value pair, where the key is None and the value is a line from the input file. It then splits the line by comma and assigns the terms to terms and the label to label. For each term in terms, it yields a tuple of (term, (label, 1)).
- The **reducer** function takes in a term and a list of tuples, where each tuple contains a label and its count. It then uses Counter to count the occurrences of each label and computes the total count. Finally, it yields a tuple of (term, [(label, count / total_count) for label, count in label_counts.items()]), where label_counts is the dictionary of label-count pairs and total_count is the total count of all labels for the term.
- The mapper_prior function takes in a term and a list of tuples, where each tuple contains a label and its probability. It then yields a tuple of (label, prob) for each label and its probability.

### Ridge Regression:

- Defined a **mapper** function called mapper that takes a key-value pair as input and yields a key-value pair as output. Here, the input key-value pair is ignored and the value is parsed into x and y variables. The x variable is a list of floats representing the features, and y is a float representing the target variable. The mapper yields a key-value pair with None as the key and a tuple containing x and y as the value:
- Defined a **reducer** function called reducer that takes a key-value pair as input and yields a key-value pair as output. Here, the input key is ignored and the value is a list of tuples where each tuple contains x and y values. The reducer concatenates all x and y values into separate lists, converts them to numpy arrays, standardizes X and y, adds a regularization term to the diagonal of X.T @ X, calculates the coefficients using the closed-form solution, and yields a key-value pair with None as the key and a tuple containing the intercept and coefficients as the value.

### KNN:

- We define the first **mapper** function, mapper_compute_distance(), which takes as input a key-value pair where the key is ignored and the value is a line from the input file containing a vector. The function parses the line into a numpy array, loads the query vector from the file, computes the Euclidean distance between the vector and the query vector using numpy, and yields a key-value pair where the key is None (indicating that all the distances should be collected together in the reducer) and the value is a tuple containing the distance and the vector.
- We define the first **reducer** function, reducer_collect_distances(), which takes as input a key-value pair where the key is None and the value is a list of tuples, where each tuple contains a distance and a vector. The function collects all the distances and vectors into separate lists, and yields a key-value pair where the key is None (indicating that all the vectors should be processed together in the next mapper) and the value is a tuple containing the list of distances and the list of vectors.

### Neural Network:

- The feedforward method implements the feedforward algorithm, which calculates the output of the neural network for a given input. It takes the input vector, applies the synaptic weights to it, and passes it through the activation function to get the output of each layer.
- The backpropagation method implements the backpropagation algorithm, which updates the synaptic weights of the neural network using gradient descent. It takes the input vector and the true label as input, and calculates the error gradient for each layer. It then updates the synaptic weights using the error gradient and the learning rate.
- The mapper method reads in a record, applies the neural network to it, and emits a key-value pair with the predicted class and the output of the neural network.
- The reducer_init method initializes the counts and sums for each class.
- The reducer method receives the predicted class and the output of the neural network from the mapper, and updates the counts and sums for the class.
- The reducer_final method calculates the final probabilities and emits a key-value pair with the actual class and the predicted class probabilities.


### References:

[1] Sejnowski TJ. Bell AJ. An information-maximization approach to blind separation and blind deconvolution. In Neural Computation, 1995.

[2] O. Chapelle. Training a support vector machine in the primal. Journal of Machine Learning Research (submitted), 2006.

[3] W. S. Cleveland and S. J. Devlin. Locally weighted regression: An approach to regression analysis by local fitting. In J. Amer. Statist. Assoc. 83, pages 596–610, 1988.

[4] L. Csanky. Fast parallel matrix inversion algorithms. SIAM J. Comput., 5(4):618–623, 1976.

[5] A. Silvescu D. Caragea and V. Honavar. A framework for learning from distributed data using sufficient statistics and its application to learning decision trees.
International Journal of Hybrid Intelligent Systems, 2003.

[6] J.Ferreira,M.B.Vellasco,M.A.C.Pacheco,R.Carlos,andH.Barbosa, “Data mining techniques on the evaluation of wireless churn.” in ESANN, 2004, pp. 483–488.

[7] H. Hwang, T. Jung, and E. Suh, “An ltv model and customer seg- mentation based on customer value: a case study on the wireless telecommunication industry,” Expert systems with applications, vol. 26, no. 2, pp. 181–188, 2004.

[8] L. G. Valiant, “A theory of the learnable,” Communications of the ACM, vol. 27, no. 11, pp. 1134–1142, 1984






