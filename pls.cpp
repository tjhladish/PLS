#include <sstream>
#include <vector>
#include <iostream>
#include <fstream>
//#include <mpreal.h>
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
typedef Matrix<complex<float_type>,Dynamic,Dynamic> Mat2Dc;
typedef Matrix<complex<float_type>, Dynamic, 1>  Colc;
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

   
float_type dominant_eigenvalue( EigenSolver<Mat2D> es ){
    Matrix<complex<float_type>, Dynamic, 1>  ev = es.eigenvalues();
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
void pls_algorithm2(const Mat2D X, const Mat2D Y, int A, Mat2Dc& W, Mat2Dc& P, Mat2Dc& Q, Mat2Dc& R, Mat2Dc& beta) {
    int M = Y.cols(); //columns in Y?

    //Modified kernel algorithm #2
    Mat2D XY = X.transpose() * Y;  // XY=X*Y;
    Mat2D XX = X.transpose() * X;  // XX=X'*X;

    for (int i=0; i<A; i++) {
        Colc w, p, q, r; 
        if (M==1) {
            w = XY.cast<complex<float_type> >();
            //w = XY.cast< complex<float_type> >();
        } else {
            EigenSolver<Mat2D> es( XY.transpose() * XY );
            //Mat2D C = es.eigenvectors();
            //Mat2D D = es.eigenvalues();

            //[C,D]=eig(XY'*XY);
            q = dominant_eigenvector(es);//C(:,find(diag(D)==max(diag(D))));
            w = (XY*q);
        }
        //complex<float_type> w2 = w.transpose()*w;
        w /= sqrt((w.transpose()*w)(0,0));
        //w=w.cwiseQuotient( (w.transpose()*w).cwiseSqrt() );
        //w=w/sqrt(w'*w);
        r=w;
        for (int j=0; j<i-1; j++) {
            r=r- (P.col(j).transpose()*w)(0,0)*R.col(j);
        }
        Mat2Dc tt = (r.transpose()*XX*r);
        p= (r.transpose()*XX).transpose(); p /= tt(0,0);
        //p= (r.transpose()*XX).transpose().cwiseQuotient(tt);
        q= (r.transpose()*XY).transpose(); q /= tt(0,0);
        //q= (r.transpose()*XY).transpose().cwiseQuotient(tt);
        XY=XY-((p*q.transpose())*tt).real(); // is casting this to 'real' safe?
        W.col(i)=w;
        P.col(i)=p;
        Q.col(i)=q;
        R.col(i)=r;
    }

    beta=R*Q.transpose(); // compute the regression coefficients
    return; 
}



int main() { 
    //http://eigen.tuxfamily.org/dox/QuickRefPage.html
    Mat2D X  = read_matrix_file("nir.csv", ',');
    Mat2D Y  = read_matrix_file("octane.csv", ',');
    int A = 1; 

    Mat2Dc P = Mat2Dc::Zero(X.cols(), A); //P = zeros(size(X)(2),A);
    Mat2Dc W = Mat2Dc::Zero(X.cols(), A); //W = zeros(size(X)(2),A);
    Mat2Dc R = Mat2Dc::Zero(X.cols(), A); //R = zeros(size(X)(2),A);
    Mat2Dc Q = Mat2Dc::Zero(Y.cols(), A);  //Q = zeros(size(Y)(2),A);
    Mat2Dc beta;
    
    //cout << setprecision(16) << X << endl;

    pls_algorithm2(X,Y,A, W,P,Q,R,beta);
    
    //cout << setprecision(6) << Y << endl;
    cout << A << " components" << endl;
    cout << setprecision(6) << (Y-X*beta.real()).cwiseProduct(Y-X*beta.real()).sum() << endl;

    A = 3; 

    P = Mat2Dc::Zero(X.cols(), A); //P = zeros(size(X)(2),A);
    W = Mat2Dc::Zero(X.cols(), A); //W = zeros(size(X)(2),A);
    R = Mat2Dc::Zero(X.cols(), A); //R = zeros(size(X)(2),A);
    Q = Mat2Dc::Zero(Y.cols(), A);  //Q = zeros(size(Y)(2),A);

    pls_algorithm2(X,Y,A, W,P,Q,R,beta);
    
    cout << A << " components" << endl;
    cout << setprecision(6) << (Y-X*beta.real()).cwiseProduct(Y-X*beta.real()).sum() << endl;

    A = 10; 
    P = Mat2Dc::Zero(X.cols(), A); //P = zeros(size(X)(2),A);
    W = Mat2Dc::Zero(X.cols(), A); //W = zeros(size(X)(2),A);
    R = Mat2Dc::Zero(X.cols(), A); //R = zeros(size(X)(2),A);
    Q = Mat2Dc::Zero(Y.cols(), A);  //Q = zeros(size(Y)(2),A);

    pls_algorithm2(X,Y,A, W,P,Q,R,beta);
    
    cout << A << " components" << endl;
    cout << setprecision(6) << (Y-X*beta.real()).cwiseProduct(Y-X*beta.real()).sum() << endl;

    return 0;
}

