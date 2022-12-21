#include "src/utils.h"

namespace dyne {

/* *********************************************************************** */
/// Helper function to split a string based on delimiter.
std::vector<std::string> split(const std::string& i_str,
                               const std::string& delimiter) {
  std::vector<std::string> result;

  size_t found = i_str.find(delimiter);
  size_t startIndex = 0;

  while (found != std::string::npos) {
    result.emplace_back(
        std::string(i_str.begin() + startIndex, i_str.begin() + found));
    startIndex = found + delimiter.size();
    found = i_str.find(delimiter, startIndex);
  }
  if (startIndex != i_str.size())
    result.emplace_back(std::string(i_str.begin() + startIndex, i_str.end()));
  return result;
}

/* *********************************************************************** */
std::vector<std::vector<std::string>> ReadCsv(const std::string& file_path,
                                              const std::string& delimiter) {
  std::ifstream file;
  file.open(file_path.c_str());

  std::string line;
  std::vector<std::vector<std::string>> data;

  while (!file.eof()) {
    std::getline(file, line, '\n');

    std::vector<std::string> split_string = split(line, delimiter);
    data.push_back(split_string);
  }
  return data;
}

}  // namespace dyne
