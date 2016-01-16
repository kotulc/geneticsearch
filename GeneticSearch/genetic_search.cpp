// genetic_search.cpp : Defines methods of the GeneticSearch class
// Copyright 2013, Clayton Kotualk

#include "genetic_search.h"


// Return a subset of the given matrix of size n=matrix_size*subset_percent with
// n random rows from this source matrix.
void GeneticSearch::GetRandomSubset(const MatrixXf& training_set, const MatrixXf& label_set, 
		MatrixXf* train_subset, MatrixXf* label_subset, float subset_percent) {

	int subset_size = int(training_set.rows()*subset_percent);
	int element_index = 0;

	*train_subset = MatrixXf::Zero(subset_size, training_set.cols());
	*label_subset = MatrixXf::Zero(subset_size, label_set.cols());

	std::uniform_int_distribution<int> idx_distribution(0,training_set.rows()-1);

	for (int i=0; i<subset_size; i++) {
		element_index = idx_distribution(random_generator);
		train_subset->row(i) = training_set.row(element_index);
		label_subset->row(i) = label_set.row(element_index);
	}

	return;

}


// With a chance based on mutation_rate, mutate elements of the provided array of length n
void GeneticSearch::MutateFloatArray(float * f_array, int n) {

	// Calculate the total number of mutations with [0.0,2.0]*mutation_rate*n
	int total_mutations = int(float_distribution(random_generator)*2*mutation_rate*n);
	std::uniform_int_distribution<int> idx_distribution(0,n-1);

	for(int i=0;i<total_mutations;i++) {
		int array_idx = idx_distribution(random_generator);
		float f_mutate = mutation_distribution(random_generator);
		f_array[array_idx] += f_mutate;
	}

	total_mutations++;

}


// Swap segments of the float arrays, f_array1 and f_array2 (both of length n) at a random index
// to simulate the crossover of parent chromosomes.
void GeneticSearch::CrossFloatArray(float * f_array1, float * f_array2, int n) {

	std::uniform_int_distribution<int> idx_distribution(0,n-1);

	int crossover_idx = idx_distribution(random_generator);
	float * temp_array = new float[crossover_idx];

	for (int i=0; i<crossover_idx; i++) {
		temp_array[i] = f_array1[i];
	}

	for (int i=0; i<crossover_idx; i++) {
		f_array1[i] = f_array2[i];
	}

	for (int i=0; i<crossover_idx; i++) {
		f_array2[i] = temp_array[i];
	}

	delete [] temp_array;

}

// Add variation to each of the provided matrices with genetic crossover and mutation
void GeneticSearch::AddMatrixGeneticVariation(MatrixXf& child_matrix1, MatrixXf& child_matrix2) { 

	if (child_matrix1.size() != child_matrix2.size() || child_matrix1.size() == 0) return;

	int array_length = child_matrix1.size();

	float crossover = float_distribution(random_generator);
	if (crossover <= crossover_rate) {
		CrossFloatArray(child_matrix1.data(), child_matrix2.data(), array_length);
	}

	MutateFloatArray(child_matrix1.data(), array_length);
	MutateFloatArray(child_matrix2.data(), array_length);

}


// Add variation to each of the provided float values with genetic crossover and mutation
void GeneticSearch::AddFloatGeneticVariation(float * param1, float * param2) {

	float crossover = float_distribution(random_generator);
	if (crossover <= crossover_rate) {
		float temp = *param1;
		*param1 = *param2;
		*param2 = temp;
	}

	float mutate = float_distribution(random_generator);
	if (mutate <= mutation_rate) {
		float mutation_factor = mutation_distribution(random_generator);
		*param1 += mutation_factor;
		mutation_factor = mutation_distribution(random_generator);
		*param2 += mutation_factor;
	}

}


// Select and return a member of the population. The probability of selection is 
// proportional to the member's fitness value
EchoStateNet GeneticSearch::SelectChromosome(std::map<float, EchoStateNet> * population) {

	if (population->size() < 1) return EchoStateNet(MatrixXf(),VectorXf(),0,0,0);

	std::map<float, EchoStateNet>::iterator esn_map_iter;
	esn_map_iter = population->end();
	esn_map_iter--;

	std::uniform_real_distribution<float> fitness_distribution(0.0,(esn_map_iter)->first);
	esn_map_iter = population->lower_bound(fitness_distribution(random_generator));

	return esn_map_iter->second; 

}


void GeneticSearch::AddESNToPopulationMutexed(std::map<float, EchoStateNet> * population, EchoStateNet esn, float * fitness_sum) {

	population_mutex.lock();

	*fitness_sum += esn.GetNetFitness();
	population->insert(std::pair<float, EchoStateNet>(*fitness_sum, esn));

	population_mutex.unlock();

}


void GeneticSearch::GenerateESNThread(const MatrixXf& training_data, const MatrixXf& training_labels, 
		std::map<float, EchoStateNet> * population, float * fitness_sum) {

	EchoStateNet esn = EchoStateNet(training_data, training_labels, reservoir_nodes, 
		float_distribution(random_generator)*1.2, float_distribution(random_generator) ); //add net size param to search

	AddESNToPopulationMutexed(population, esn, fitness_sum);

}


void GeneticSearch::GenerateESNChildrenThread(const MatrixXf& training_data, 
	const MatrixXf& training_labels, std::map<float, EchoStateNet> * current_population, 
	std::map<float, EchoStateNet> * future_population, float * fitness_sum) {

	EchoStateNet parent1 = SelectChromosome(current_population);
	EchoStateNet parent2 = SelectChromosome(current_population);
	std::vector<MatrixXf> w_matrix_vect1 = parent1.GetWeightMatrices();
	std::vector<MatrixXf> w_matrix_vect2 = parent2.GetWeightMatrices();

	for(unsigned int j=0; j<w_matrix_vect1.size(); j++) {
		AddMatrixGeneticVariation(w_matrix_vect1[j], w_matrix_vect2[j]);
	}

	std::vector<float> param_vect1 = parent1.GetESNParameters();
	std::vector<float> param_vect2 = parent2.GetESNParameters();

	for(unsigned int j=0; j<param_vect1.size(); j++) {
		AddFloatGeneticVariation(&param_vect1[j], &param_vect2[j]);
	}

	EchoStateNet child1 = EchoStateNet(training_data, training_labels, param_vect1, w_matrix_vect1);
	EchoStateNet child2 = EchoStateNet(training_data, training_labels, param_vect2, w_matrix_vect1);
	AddESNToPopulationMutexed(future_population, child1, fitness_sum);
	AddESNToPopulationMutexed(future_population, child2, fitness_sum);

}


// Return a new population generated from the provided current_population
std::map<float, EchoStateNet> GeneticSearch::GenerateNextPopulation(const MatrixXf& training_data, 
		const MatrixXf& training_labels, std::map<float, EchoStateNet> current_population) { //Clean up!

	std::map<float, EchoStateNet> future_population;
	std::vector<std::thread> esn_threads;

	float fitness_sum = 0.0;
	float progress_percent = 0.0;
	int half_population = (int)population_size/2;
	int i = 0;

	std::cout << "Progress: " << 0 << " %\r" << std::flush;

	for (; i<(int)half_population*new_random_population; i++) { 
		if ( int( (float)i/half_population * 100.0) > int(progress_percent * 100) ) {
			progress_percent = float(i)/half_population;
			std::cout << "Progress: " << int(progress_percent * 100.0) << " %\r" << std::flush;
		}

		esn_threads.push_back(std::thread(&GeneticSearch::GenerateESNThread, this, training_data, training_labels, &future_population, &fitness_sum));
		//GenerateESNThread(training_data, training_labels, &future_population, &fitness_sum);
	}

	for (; i<half_population; i++) { 
		if ( int( (float)i/half_population * 100.0) > int(progress_percent * 100) ) {
			progress_percent = float(i)/half_population;
			std::cout << "Progress: " << int(progress_percent * 100.0) << " %\r" << std::flush;
		}

		esn_threads.push_back(std::thread(&GeneticSearch::GenerateESNChildrenThread, this, training_data, training_labels, &current_population, &future_population, &fitness_sum));
		//GenerateESNChildrenThread(training_data, training_labels, &current_population, &future_population, &fitness_sum);
	}

	for (int j=0; j<esn_threads.size(); j++) {
		esn_threads[j].join();
	}

	search_out << "Epoch fitness mean: " << (float)fitness_sum/population_size << "\n" << std::endl;
	std::cout << "Epoch fitness mean: " << (float)fitness_sum/population_size << "\n" << std::endl;
	OutputFittestESNMetrics(&future_population);

	return future_population;

}


// Generate and return the starting population with random parameter values 'alpha' and 'connectivity'
std::map<float, EchoStateNet> GeneticSearch::GenerateStartPopulation(const MatrixXf& training_data, const MatrixXf& training_labels) {

	std::map<float, EchoStateNet> population;
	std::vector<std::thread> esn_threads;

	float fitness_sum = 0.0;
	float progress_percent = 0.0;

	std::cout << "Progress: " << 0 << " %\r" << std::flush;

	for (int i=0; i<population_size; i++) {
		if ( int( float(i)/population_size * 100.0) > int(progress_percent * 100) ) {
			progress_percent = float(i)/population_size;
			std::cout << "Progress: " << int(progress_percent * 100.0) << " %\r" << std::flush;
		}

		esn_threads.push_back(std::thread(&GeneticSearch::GenerateESNThread, this, training_data, training_labels, &population, &fitness_sum));
		//GenerateESNThread(training_data, training_labels, &population, &fitness_sum);
	}

	for (int i=0; i<esn_threads.size(); i++) {
		esn_threads[i].join();
	}

	search_out << "Fitness mean: " << (float)fitness_sum/population_size << "\n" << std::endl;
	std::cout << "Initial population generated." << "\nFitness mean: " << (float)fitness_sum/population_size << "\n" << std::endl;
	OutputFittestESNMetrics(&population);

	return population;

}


void GeneticSearch::OutputSearchParameters() {

	search_out << "Search parameters..." << std::endl;
	search_out << "Search population size: " << population_size << std::endl;
	search_out << "Max number of generations: " << max_generations << std::endl;
	search_out << "New random population: " << new_random_population*100 << "%" << std::endl;
	search_out << "Population mutation rate: " << (float)mutation_rate*100 << "%" << std::endl;
	search_out << "Population mutation range: " << mutation_range << std::endl;
	search_out << "Genetic crossover rate: " << crossover_rate*100 << "%" << std::endl;
	search_out << "Training samples per ESN: " << train_sample*100 << "%" << std::endl;

}


void GeneticSearch::OutputFittestESNMetrics(std::map<float, EchoStateNet> * population) {

	float current_fitness = 0;
	float best_fitness = -1;

	EchoStateNet * best_esn = NULL;

	std::map<float, EchoStateNet>::iterator population_iterator;
	for (population_iterator = population->begin(); population_iterator != population->end(); population_iterator++) {
		current_fitness = population_iterator->second.GetNetFitness();
		
		if (current_fitness > best_fitness) {
			best_esn = &(population_iterator->second);
			best_fitness = current_fitness;
		}
	}

	if (best_esn != NULL) {
		search_out << "Most Fit ";
		best_esn->OutputNetMetrics(search_out);
		best_esn->SaveNetToFile(bin_file);
	}

}


void GeneticSearch::RunGeneticSearch(const MatrixXf& training_data, const MatrixXf& training_labels, std::string out_file) {

	search_out.open(out_file, std::ios::out | std::ios::app);

	if (!search_out.is_open()) {
		std::cout << "Error: Could not output log to file " << out_file << "\n" << std::endl;
		return;
	}

	std::time_t begin_time;
    std::tm* time_info;
    char buffer [80];

    std::time(&begin_time);
    time_info = std::localtime(&begin_time);
    std::strftime(buffer,80,"%m-%d-%Y %H:%M:%S",time_info);

	search_out << "\n\n\n" << buffer << " - Processing new search...\n" << std::endl;

	OutputSearchParameters();

	std::cout << "Generating initial population..." << std::endl;
	search_out << "\nGenerating initial population..." <<  std::endl;

	MatrixXf data_subset;
	MatrixXf label_subset;

	GetRandomSubset(training_data, training_labels, &data_subset, &label_subset, train_sample);

	std::map<float, EchoStateNet> population = GenerateStartPopulation(data_subset, label_subset);
	for (int i=1; i<=max_generations; i++) {
		std::cout << "Generating population epoch " << i << "..." << std::endl;
		search_out << "\nGenerating population epoch " << i << "..." << std::endl;
		GetRandomSubset(training_data, training_labels, &data_subset, &label_subset, train_sample);
		population = GenerateNextPopulation(data_subset, label_subset, population);
	}

	std::time_t end_time;
	std::time(&end_time);
	time_info = std::localtime(&end_time);
	std::strftime(buffer,80,"%m-%d-%Y %H:%M:%S",time_info);

	search_out << "\n" << buffer;

	double elapsed_time = ( std::difftime(end_time,begin_time) );

	int hours = elapsed_time/3600;
	int minutes = (elapsed_time - (hours*3600))/60;
	int seconds = elapsed_time - (hours*3600 + minutes*60);

	search_out << " - Search completed.\nTime elapsed " << hours << ":" << minutes << ":" << seconds << "\n\n";
	search_out.close();

}


// Initialize the genetic search with the provided parameters
GeneticSearch::GeneticSearch(const MatrixXf& training_data, const MatrixXf& training_labels, int res_nodes=20, 
	int pop_size=100, int max_gens=10, float new_rand_pop=0.1, float m_rate=0.001, float m_range=1.0, 
	float c_rate=0.7, float t_sample=0.10, std::string out_file="search_log.txt", std::string b_file="search_data.bin") {

	// Perform simple input validation
	reservoir_nodes = (res_nodes <= 0) ? 20 : res_nodes;
	population_size = (pop_size <= 0) ? 100 : pop_size;
	max_generations = (max_gens <= 0) ? 10 : max_gens;
	new_rand_pop = (new_rand_pop > 1.0) ? 0.1 : new_rand_pop;
	new_random_population = (new_rand_pop < 0) ? 0.1 : new_rand_pop;
	m_rate = (m_rate > 1.0) ? 1.0 : m_rate;
	mutation_rate = (m_rate < 0) ? 0.001 : m_rate;
	mutation_range = (m_range < 0) ? 1.0 : m_range;
	crossover_rate = (c_rate < 0) ? 0.7 : c_rate;
	t_sample = (t_sample > 1.0) ? 1.0 : t_sample;
	train_sample = (t_sample < 0) ? 0.10 : t_sample;

	bin_file = b_file;

	std::mt19937 r_gen(random_seed());
	random_generator = r_gen;

	std::uniform_real_distribution<float> f_dist(0.0,1.0); 
	float_distribution = f_dist;

	std::uniform_real_distribution<float> m_dist(-1*mutation_range,mutation_range);
	mutation_distribution = m_dist;

	RunGeneticSearch(training_data, training_labels, out_file);

}
