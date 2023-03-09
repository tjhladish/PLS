
#include <vector>
#include "pls.h"

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

    //Mat2D X = X_orig ;
    //Mat2D Y = Y_orig ;
    Mat2D X = colwise_z_scores( X_orig );
    Mat2D Y = colwise_z_scores( Y_orig );

    int nobj  = X_orig.rows();
    int npred = X_orig.cols();
    int nresp = Y_orig.cols();
    int ncomp = atoi(argv[3]);

    PLS_Model plsm(X, Y, ncomp);

    plsm.explained_variance(X, Y, ncomp);
        
    cout << "Validation (loo):\n";
    //Mat2D looSSE = plsm.loo_validation(X, Y, PRESS);
    //cout << looSSE << endl;
    Mat2D looRMSEP = plsm.loo_validation(X, Y, RMSEP);
    cout << looRMSEP << endl;

    cout << "Validation (lso):\n";
    //Mat2D lsoSSE = plsm.lso_validation(X, Y, PRESS, 0.3, X.rows());
    //cout << lsoSSE << endl;
    Mat2D lsoRMSEP = plsm.lso_validation(X, Y, RMSEP, 0.3, 10*X.rows());
    cout << lsoRMSEP << endl;
    
    cout << "Optimal number of components (loo):\t" << plsm.loo_optimal_num_components(X,Y) << endl;
    cout << "Optimal number of components (lso):\t" << plsm.lso_optimal_num_components(X,Y,0.3,10*X.rows()) << endl;
    
/*
    cout << plsm.scores() << endl << endl; 

    cout << "Validation (RMSEP):\n";
    cout << plsm.loo_validation(X, Y, RMSEP) << endl;

    cout << "W:\n";
    cout << plsm.W.colwise().sum() << endl << endl;

    cout << "P:\n";
    cout << plsm.P.colwise().sum() << endl << endl;

    cout << "Q:\n";
    cout << plsm.Q.colwise().sum() << endl << endl;

    cout << "R:\n";
    cout << plsm.R.colwise().sum() << endl << endl;

    cout << "X:\n";
    cout << X.sum() << endl << endl;

    cout << "Y:\n";
    cout << Y.sum() << endl << endl;
*/

    return 0;
}
