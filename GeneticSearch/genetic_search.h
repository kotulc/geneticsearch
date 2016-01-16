#ifndef GENETIC_SEARCH_H
#define GENETIC_SEARCH_H

#include "stdafx.h"
#include "echo_state_net.h"


class GeneticSearch {

	private:
	int reservoir_nodes;
	int population_size;
	int max_generations;

	float new_random_population;
	float mutation_rate;
	float mutation_range;
	float crossover_rate;
	float train_sample;

	std::string bin_file;
	std::ofstream search_out;
	std::mutex population_mutex;
	std::random_device random_seed;
	std::mt19937 random_generator;
	std::uniform_real_distribution<float> float_distribution;
	std::uniform_real_distribution<float> mutation_distribution;


	void GetRandomSubset(const MatrixXf& training_set, const MatrixXf& label_set, MatrixXf* train_subset,
		MatrixXf* label_subset, float subset_percent);

	void MutateFloatArray(float * f_array, int n);

	void CrossFloatArray(float * f_array1, float * f_array2, int n);

	void AddMatrixGeneticVariation(MatrixXf& child_matrix1, MatrixXf& child_matrix2);

	void AddFloatGeneticVariation(float * param1, float * param2);

	EchoStateNet SelectChromosome(std::map<float, EchoStateNet> * population);

	void AddESNToPopulationMutexed(std::map<float, EchoStateNet> * population, EchoStateNet esn, float * fitness_sum);

	void GenerateESNThread(const MatrixXf& training_data, const MatrixXf& training_labels, 
		std::map<float, EchoStateNet> * population, float * fitness_sum);

	void GenerateESNChildrenThread(const MatrixXf& training_data, const MatrixXf& training_labels, 
		std::map<float, EchoStateNet> * current_population, std::map<float, EchoStateNet> *future_population, 
		float * fitness_sum);

	std::map<float, EchoStateNet> GenerateNextPopulation(const MatrixXf& training_data, 
		const MatrixXf& training_labels, const std::map<float, EchoStateNet> current_population);

	std::map<float, EchoStateNet> GenerateStartPopulation(const MatrixXf& training_data, const MatrixXf& training_labels);

	void OutputSearchParameters();

	void OutputFittestESNMetrics(std::map<float, EchoStateNet> * population);

	void RunGeneticSearch(const MatrixXf& training_data, const MatrixXf& training_labels, std::string out_file);

	public:
		GeneticSearch(const MatrixXf& training_data, const MatrixXf& training_labels, int reservoir_nodes,  
			int population_size, int max_generations, float new_random_population, float mutation_rate, 
			float mutation_range, float crossover_rate, float train_sample, std::string out_file, std::string bin_file);

};

#endif