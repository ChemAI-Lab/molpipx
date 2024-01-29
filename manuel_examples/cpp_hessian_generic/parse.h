#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using std::size_t;
using std::string;
using std::vector;

std::pair<vector<int>, vector<int>> parse_mono(std::string path) {
  std::ifstream file(path, std::ios::in);
  size_t P = std::count(std::istreambuf_iterator<char>(file),
                        std::istreambuf_iterator<char>(), '\n');
  // reset to beginning
  file.seekg(0);
  vector<int> MORSE_indices{}; // -1 since first row is zeros
  vector<int> MONO_vec{};
  std::string line;

  getline(file, line); // skip first row since is zeros
  while (getline(file, line)) {
    if (line[0] == '0') {
      std::stringstream ss(&line[2]);
      int x, pos = 0;
      while (ss >> x) {
        if (x == 1) {
          MORSE_indices.push_back(pos);
          break;
        }
        pos++;
      }
    } else {
      std::stringstream ss(&line[2]);
      int a, b;
      ss >> a >> b;
      MONO_vec.push_back(a);
      MONO_vec.push_back(b);
    }
  }
  return std::make_pair(MORSE_indices, MONO_vec);
}

std::vector<int> parse_poly(std::string path) {
  std::ifstream file(path, std::ios::in);
  size_t P = std::count(std::istreambuf_iterator<char>(file),
                        std::istreambuf_iterator<char>(), '\n');
  // reset to beginning
  file.seekg(0);
  std::vector<int> res;
  std::string line;

  while (getline(file, line)) {
    if (line[0] == '2') {
      res.push_back(2);
    } else {
      res.push_back(3);
    }
    std::stringstream ss(&line[2]);
    int x;
    while (ss >> x)
      res.push_back(x);
    // -1 as line end marker
    res.push_back(-1);
  }

  return res;
}

