
#include <vector>
#include <random>
#include <iostream>

#include <PLS/pls.h>

using namespace PLS;

int main(int argc, char *argv[]) {

    if (argc < 4) { 
        std::cerr << "Usage: ./pls X_data.csv Y_data.csv num_components" << std::endl;
        std::cerr << "NB: X and Y csvs must be comma delimited, square numerical data, with no headers." << std::endl;
        exit(100); 
    }

    std::string x_filename(argv[1]);
    std::string y_filename(argv[2]);

    Mat2D X_orig  = read_matrix_file(x_filename);
    Mat2D Y_orig  = read_matrix_file(y_filename);

    Mat2D X = colwise_z_scores( X_orig );
    Mat2D Y = colwise_z_scores( Y_orig );

    int ncomp = atoi(argv[3]);

    Model plsm(X, Y, ncomp);

    plsm.print_state();

    plsm.print_explained_variance(X, Y);
    
    Residual looerror = plsm.error<LOO>(X, Y);
    print_validation(looerror, LOO, MSE);

    std::mt19937 rng;

    Residual lsoerror = plsm.error<LSO>(X, Y, 0.3, 10*X.rows(), rng);
    print_validation(lsoerror, LSO, MSE);

    return 0;
}
