# GeneticSearch
Performs an evolutionary search of the defined ESN parameter space

This C++ project depends on the included Eigen 3.2.7 library (http://eigen.tuxfamily.org/index.php?title=Main_Page) and successfully builds in Visual Studio Express 2013 on a 64bit Windows 10 machine. All included files and source code are freely available for public and private use.

The project was developed from 10/13 to 1/14 and was originally intended as a means for exploring several concepts including Evolutionary Search, Echo State Networks, and Reservoir Computing. This project was also useful for practicing coding in C++, creating a  Multi-threaded application, and utilizing Matrix algebra and related concepts. Due to the exploritory nature of this project it was not intended for publication or distribution to a wider audience, lacks helpful documentation, and is in a pre-release but functional state. 


Brief overview (retrospective):

This application performs a "Genetic Search" of candidate Echo State Networks, selects the most "fit" networks and then generates offspring from pairs of the selected networks. This is a relatively straight-forward interpretation of an evolutionary approach to finding weights of a network. 

The system generates a random starting population and then proceeds to mutate and combine the candidates to produce the candidates for the next generation. Each new candidate is  initialized with an output layer trained to predict a numeric value with Ridge Regression.

The numeric training data provided is a series of 3x5 black and white images with random distortions and inversions applied. The text data files _moderate and _complex have more random noise than the _simple data set.

During the search, a list of defined metrics and other information determined as useful is printed to the log file for later instpection.

In this implementation, fitness is equal to the networks validation_accuracy value.


Metrics

The following metrics are calculated for each candidate:
Reservoir nodes,
Network edges,
Network weights,
Network states,
Training accuracy,
Validation accuracy,
Network mean activation,
Network capacity,
Network memory,
Network sensitivity,
Network connectivity,
Network alpha

For more information regarding these metrics see: 
EchoStateNet::CalculateNetMetrics
EchoStateNet::CalculateNetMemory


config.txt

The system reads the search parameters from config.txt, below is a brief description of each:

population_size - int, the number of candidate networks per epoch

max_generations - int, the number of evolutionary generations or epochs

reservoir_nodes - int, the number of nodes contained in an ESN

new_random_population - float, the percent of new, randomly initialized networks introduced every new generation

mutation_rate - float, the rate of mutation per element

mutation_range - float, the maximum +/- difference between the original attribute and the mutated attribute (for mutating non-binary values)

crossover_rate - float, the rate of genetic (element) crossover when generating a child network from two parent networks

train_sample - float, The percentage of the number of training elements to use when training each network

overwrite_data - bool, if true (1) overwrite the existing binary weights file

log_file_name - string, name of generated search log. A new entry to the log is appended for each search if the file exists. The log file includes run time, progress and metrics.

bin_file_name - string, name of the binary file the application will store weights to and may then later parse to restore a collection of networks. This file is created if it does not yet exist.

train_file_name - string, the file containing the labeled image data where each line contains a training instance in the form: [number label, e1, e2, ..., e15]
where e1-e15 are pixel black-white pixel intensity values in the range of [0,255]




