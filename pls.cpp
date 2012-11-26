#include "pls.h"

int main() { 
    //http://eigen.tuxfamily.org/dox/QuickRefPage.html
    //Mat2D X_orig  = read_matrix_file("toyX.csv", ',');
    //Mat2D Y_orig  = read_matrix_file("toyY.csv", ',');
    //Mat2D X_orig  = read_matrix_file("simpleX_orig.csv", ',');
    //Mat2D Y_orig  = read_matrix_file("simpleY_orig.csv", ',');
    Mat2D X_orig  = read_matrix_file("nir.csv", ',');
    Mat2D Y_orig  = read_matrix_file("octane.csv", ',');
    PLS_Model plsm;

    int nobj  = X_orig.rows();
    int npred = X_orig.cols();
    int nresp = Y_orig.cols();
    
    Mat2D X = colwise_z_scores( X_orig );
    Mat2D Y = colwise_z_scores( Y_orig );


    plsm.initialize(npred, nresp, 10);
    plsm.plsr(X,Y, plsm, KERNEL_TYPE1);

    //cout << setprecision(16) << X << endl;
    //cout << setprecision(6) << Y << endl;

    for (int A = 1; A<11; A++) { // number of components to try
        // How well did we do?
        cout << A << " components\t";
        cout << "explained variance: " << plsm.explained_variance(X, Y, A) << endl;;
    }

    return 0;
}

