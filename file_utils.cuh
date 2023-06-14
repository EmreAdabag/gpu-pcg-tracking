#include <vector>
#include <fstream>
#include <sstream>


std::vector<std::vector<float>> readCSVToVecVec(const std::string& filename) {
    std::vector<std::vector<float>> data;
    std::ifstream infile(filename);

    if (!infile.is_open()) {
        std::cerr << "File could not be opened!\n";
    } else {
        std::string line;


        while (std::getline(infile, line)) {
            std::vector<float> row;
            std::stringstream ss(line);
            std::string val;

            while (std::getline(ss, val, ',')) {
                row.push_back(std::stof(val));
            }

            data.push_back(row);
        }
    }

    infile.close();
    return data;
}
