#include "utility.h"

void split(const string& s, char c, vector<string>& v) {
    string::size_type i = 0;
    string::size_type j = s.find(c);

    while (j != string::npos) {
        v.push_back(s.substr(i, j-i));
        i = ++j;
        j = s.find(c, j);
    }
    if (j == string::npos) v.push_back(s.substr(i, s.length( )));
}


Mat2D read_matrix_file(string filename, char sep) {
    std::cerr << "Loading " << filename << endl;
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


Row col_stdev( Mat2D mat, Row means ) {
    Row stdevs = Row::Zero(mat.cols());
    const float_type N = mat.rows();
    if ( N < 2 ) return stdevs;

    const float_type N_inv = 1.0/(N-1); // N-1 for unbiased sample variance
    for (int i=0; i<mat.cols(); i++) {
        stdevs[i] = sqrt( (mat.col(i).array()-means[i]).square().sum() * N_inv );
    }
    return stdevs;
}


float_type dominant_eigenvalue( EigenSolver<Mat2Dc> es ){
    Colc  ev = es.eigenvalues();
    float_type m = 0;

    for (int i = 0; i<ev.size(); i++) {
        if (imag(ev[i]) == 0) {
            if (abs(ev[i]) > m) m = abs(ev[i]);
        }
    }
    return m;
}


Colc dominant_eigenvector( EigenSolver<Mat2D> es ){
    Colc eig_val = es.eigenvalues();
    float_type m = 0;
    int idx = 0;

    for (int i = 0; i<eig_val.size(); i++) {
        if (imag(eig_val[i]) == 0) {
            if (abs(eig_val[i]) > m) {
                m = abs(eig_val[i]);
                idx = i;
            }
        }
    }
    return es.eigenvectors().col(idx);
}

Mat2D colwise_z_scores( const Mat2D& mat ) {
    // Standardize values by column, i.e. convert to Z-scores
    Row means = col_means( mat );
    Row stdev = col_stdev( mat, means );
    Mat2D zmat = Mat2D::Zero(mat.rows(), mat.cols());
    for (int r = 0; r<mat.rows(); r++) { zmat.row(r) = (mat.row(r) - means).cwiseQuotient(stdev); }
    return zmat;
}
