#ifndef UTILITY_H
#define UTILITY_H

#include <cstdlib>
#include <sstream>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <iterator>
#include <iomanip>
#include <fstream>

using namespace std;

void split(const string& s, char c, vector<string>& v);

inline double string2double(const std::string& s){ std::istringstream i(s); double x = 0; i >> x; return x; }

#endif
