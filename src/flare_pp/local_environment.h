#ifndef LOCAL_ENVIRONMENT_H
#define LOCAL_ENVIRONMENT_H

#include "structure.h"
#include "descriptor.h"

// Local environment class.
class LocalEnvironment{
    public:
        std::vector<int> environment_indices, environment_species,
            neighbor_list;
        int central_index, central_species, noa, sweep;
        std::vector<double> rs, xs, ys, zs, xrel, yrel, zrel;
        double cutoff, structure_volume;

        // Store cutoffs for each kernel and indices of atoms inside each cutoff sphere.
        std::vector<double> n_body_cutoffs, many_body_cutoffs;
        std::vector<std::vector<int>> n_body_indices, many_body_indices;

        // Triplet indices and cross bond distances are stored only if the 3-body kernel is used
        std::vector<std::vector<int>> three_body_indices;
        std::vector<double> cross_bond_dists;

        // Store descriptor calculators for each many body cutoff.
        std::vector<DescriptorCalculator *> descriptor_calculators;
        std::vector<Eigen::VectorXd> descriptor_vals;
        std::vector<Eigen::MatrixXd> descriptor_force_dervs,
            descriptor_stress_dervs, force_dot, stress_dot;
        std::vector<double> descriptor_norm;

        LocalEnvironment();

        LocalEnvironment(const Structure & structure, int atom,
                         double cutoff);

        // n-body
        LocalEnvironment(const Structure & structure, int atom,
                         double cutoff, std::vector<double> n_body_cutoffs);

        // many-body
        LocalEnvironment(const Structure & structure, int atom, double cutoff,
                         std::vector<double> many_body_cutoffs,
                         std::vector<DescriptorCalculator *>
                            descriptor_calculators);

        // n-body + many-body
        LocalEnvironment(const Structure & structure, int atom, double cutoff,
                         std::vector<double> n_body_cutoffs,
                         std::vector<double> many_body_cutoffs,
                         std::vector<DescriptorCalculator *>
                            descriptor_calculator);

        static void compute_environment(const Structure & structure,
                                 int noa, int atom,
                                 double cutoff, int sweep_val,
                                 std::vector<int> & environment_indices,
                                 std::vector<int> & environment_species,
                                 std::vector<int> & neighbor_list,
                                 std::vector<double> & rs,
                                 std::vector<double> & xs,
                                 std::vector<double> & ys,
                                 std::vector<double> & zs,
                                 std::vector<double> & xrel,
                                 std::vector<double> & yrel,
                                 std::vector<double> & zrel);

        void compute_indices();
        void compute_descriptors();
};

#endif