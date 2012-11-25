#include <sstream>
#include <vector>
#include <iostream>
#include <fstream>
//#include "/home/tjhladish/work/lib/eigen/unsupported/test/mpreal/mpreal.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MPRealSupport>
#include "Utility.h"

using namespace std;

using namespace Eigen;
//using namespace mpfr;

typedef long double float_type;
//typedef mpreal float_type;
typedef Matrix<float_type,Dynamic,Dynamic> Mat2D;
typedef Matrix<float_type, Dynamic, 1>  Col;
typedef Matrix<float_type, 1, Dynamic>  Row;
typedef Matrix<complex<float_type>,Dynamic,Dynamic> Mat2Dc;
typedef Matrix<complex<float_type>, Dynamic, 1>  Colc;

typedef enum { KERNEL_TYPE1, KERNEL_TYPE2 } ALGORITHM;

struct PLS_Model {
    Mat2Dc P, W, R, Q, T, beta;
    int A;
    void initialize(int num_predictors, int num_responses, int num_components) {
        A = num_components; 
        P.setZero(num_predictors, num_components);
        W.setZero(num_predictors, num_components);
        R.setZero(num_predictors, num_components);
        Q.setZero(num_responses, num_components);
        // T will be initialized if needed
        return;
    }
};

//  Mat2D A(3,4); A  << 1,2,0,1, 1,1,1,-1, 3,1,5,-7;
//  VectorXd b(3);   b << 7,3,1;
//  VectorXd x;
/*
double string2double(const std::string& s){ std::istringstream i(s); double x = 0; i >> x; return x; }

vector<double> read_vector_file(string filename) {
    cerr << "Loading " << filename << endl;
    ifstream myfile(filename.c_str());
    std::stringstream ss;

    vector<double> M;
    if (myfile.is_open()) {
        string line;

        while ( getline(myfile,line) ) {
            M.push_back(string2double(line));
        }
    }
    return M;
}
*/

Mat2D read_matrix_file(string filename, char sep) {

    cerr << "Loading " << filename << endl;
    ifstream myfile(filename.c_str());
    std::stringstream ss;

    vector<vector<double> > M;
    if (myfile.is_open()) {
        string line;

        while ( getline(myfile,line) ) {
            //split string based on "," and store results into vector
            vector<string> fields;
            split(line, sep, fields);

            vector<double>row(fields.size());
            for( int i=0; i < fields.size(); i++ ) {
                row[i] = string2double(fields[i]);
            }
            M.push_back(row);
        }
    }

    Mat2D X( (int) M.size(), (int) M[0].size() );
    for(int i=0; i < M.size(); i++ ) {
        for(int j=0; j < M[i].size(); j++ ) {  
            X(i,j)=M[i][j]; 
        }
    }
    return X;
}

Row col_means( Mat2D mat ) {
    
    /*Row means = Row::zero(mat.cols());
    const double N = mat.rows();
    assert (N>0);

    const double N_inv = 1.0/N;
    for (int i=0; i<mat.cols(); i++) {
        means[i] = mat->col(i).array().sum() * N_inv;
    }*/
    return mat.colwise().sum() / mat.rows();
}


Row col_stdev( Mat2D mat, Row means ) {
    Row stdevs = Row::Zero(mat.cols());
    const double N = mat.rows();
    if ( N < 2 ) return stdevs;

    const double N_    = 1.0/(N-1); // N-1 for unbiased sample variance
    for (int i=0; i<mat.cols(); i++) {
        //stdevs[i] = ( (mat.col(i).array()-means[i]).square().sum() * N_ ).sqrt();
        stdevs[i] = sqrt( (mat.col(i).array()-means[i]).square().sum() * N_ );
    }
    //stdevs = (mat.col(i).array()-means[i]).square().sum() * N_ ).sqrt();
    return stdevs;
}


float_type dominant_eigenvalue( EigenSolver<Mat2Dc> es ){
    Colc  ev = es.eigenvalues();
    //Matrix<complex<float_type>, Dynamic, 1>  ev = es.eigenvalues();
    float_type m = 0;
    for (int i = 0; i<ev.size(); i++) {
        if (imag(ev[i]) == 0) {
            if (abs(ev[i]) > m) m = abs(ev[i]);
        }
    }
    return m;
}


Colc dominant_eigenvector( EigenSolver<Mat2D> es ){
    Colc eval = es.eigenvalues();
    //Matrix<complex<float_type>, Dynamic, Dynamic, 1>  evec = es.eigenvalues();
    ///http://eigen.tuxfamily.org/dox/classEigen_1_1EigenSolver.html#aeb6c0eb89cc982629305f6c7e0791caf
    //MatrixXcd D = es.eigenvalues().asDiagonal();
    //MatrixXcd V = es.eigenvectors();

    //typedef Matrix< std::complex<double> , Dynamic , Dynamic > MatrixXcd;

    float_type m = 0;
    int idx = 0;
    for (int i = 0; i<eval.size(); i++) {
        if (imag(eval[i]) == 0) {
            if (abs(eval[i]) > m) {
                m = abs(eval[i]);
                idx = i;
            }
        }
    }
    return es.eigenvectors().col(idx);

}


//Modified kernel algorithm
void pls_algorithm2(const Mat2D X, const Mat2D Y, PLS_Model& plsm, ALGORITHM algorithm) {
    int A = plsm.A; Mat2Dc& W = plsm.W; Mat2Dc& P = plsm.P; Mat2Dc& Q = plsm.Q; Mat2Dc& R = plsm.R; Mat2Dc& T = plsm.T; Mat2Dc& beta = plsm.beta;

    int M = Y.cols(); // Number of response variables == columns in Y

    if (algorithm == KERNEL_TYPE1) T.setZero(X.rows(), A);

    Mat2D XY = X.transpose() * Y;  // XY=X*Y;
    Mat2D XX;
    if (algorithm == KERNEL_TYPE2) XX = X.transpose() * X;  // XX=X'*X;
    //Mat2Dc XY = (X.transpose() * Y).cast<complex<float_type> > ();  // XY=X*Y;
    //Mat2Dc XX = (X.transpose() * X).cast<complex<float_type> > ();  // XX=X'*X;

    for (int i=0; i<A; i++) {
        Colc w, p, q, r, t; 
        Mat2Dc tt;
        if (M==1) {
            //w = XY;
            w = XY.cast<complex<float_type> >();
        } else {
            EigenSolver<Mat2D> es( (XY.transpose() * XY) );
            q = dominant_eigenvector(es);//C(:,find(diag(D)==max(diag(D))));
            w = (XY*q);
        }

        w /= sqrt((w.transpose()*w)(0,0)); // use normalize function from eigen?
        r=w;
        for (int j=0; j<=i-1; j++) {
            r -= (P.col(j).transpose()*w)(0,0)*R.col(j);
        }
        if (algorithm == KERNEL_TYPE1) {
            t = X*r;
            tt = t.transpose()*t;
            p.noalias() = (X.transpose()*t);
        } else if (algorithm == KERNEL_TYPE2) {
            tt = r.transpose()*XX*r;
            p.noalias() = (r.transpose()*XX).transpose();
        }
        p /= tt(0,0);
        q.noalias() = (r.transpose()*XY).transpose(); q /= tt(0,0);
        XY -= ((p*q.transpose())*tt).real(); // is casting this to 'real' always safe?
        //cout << setprecision(6) << i << "	w="	<< w.sum().real()  << "	p="	<< p.sum().real() << "	q="	<< q.sum().real()  << "	r="	<<  r.sum().real()  << "\ttt=" << tt(0,0) <<  "\tXX" << XX.sum() << endl;
        W.col(i)=w;
        P.col(i)=p;
        Q.col(i)=q;
        R.col(i)=r;
        if (algorithm == KERNEL_TYPE1) T.col(i) = t;
    }

    beta.noalias() = R*Q.transpose(); // compute the regression coefficients
    //cout << beta << endl;
    //cout << T * Q.transpose() << endl;
    //beta = beta.cwiseProduct(Ystdev.transpose()).cwiseQuotient(Xstdev.transpose());
    //cout << beta.size() << endl;
    //cout << Xstdev.size() << endl;
    //cout << Ystdev.size() << endl;
    return; 
}


int main() { 
    //http://eigen.tuxfamily.org/dox/QuickRefPage.html
    //Mat2D X_orig  = read_matrix_file("simpleX_orig.csv", ',');
    //Mat2D Y_orig  = read_matrix_file("simpleY_orig.csv", ',');
    Mat2D X_orig  = read_matrix_file("nir.csv", ',');
    Mat2D Y_orig  = read_matrix_file("octane.csv", ',');
    PLS_Model plsm;

    int nobj  = X_orig.rows();
    int npred = X_orig.cols();
    int nresp = Y_orig.cols();

    // Standardize X and Y, i.e. convert to Z-scores
    Row Xmeans = col_means( X_orig );
    Row Xstdev = col_stdev( X_orig, Xmeans );
    Mat2D X = Mat2D::Zero(X_orig.rows(), X_orig.cols());
    for (int r = 0; r<X.rows(); r++) { X.row(r) = (X_orig.row(r) - Xmeans).cwiseQuotient(Xstdev); }

    Row Ymeans = col_means( Y_orig );
    Row Ystdev = col_stdev( Y_orig, Ymeans );
    Mat2D Y = Mat2D::Zero(Y_orig.rows(), Y_orig.cols());
    for (int r = 0; r<Y.rows(); r++) { Y.row(r) = (Y_orig.row(r) - Ymeans).cwiseQuotient(Ystdev); }
    //cout << X << "\n\n";
    //cout << Y << "\n\n";


    for (int A = 1; A<11; A++) { // number of components to try

        plsm.initialize(npred, nresp, A);
        pls_algorithm2(X,Y, plsm, KERNEL_TYPE1);

        //cout << setprecision(16) << X << endl;
        //cout << setprecision(6) << Y << endl;

        // How well did we do?
        cout << A << " components" << endl;
        Mat2D Ypred = X*plsm.beta.real();
        cout << setprecision(6) << (Y-Ypred).array().square().sum() << endl;
        cout << setprecision(6) << (Y.array() - (Y.sum()/Y.rows())).square().sum() << endl;
        cout << setprecision(6) << 1 - (Y-Ypred).array().square().sum() / (Y.array() - (Y.sum()/Y.rows())).square().sum() << endl;
    }

    return 0;
}

