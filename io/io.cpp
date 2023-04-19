#include <fstream>
#include <sstream>
#include <stdexcept>

#include "io.h"

const char *io::readKernel(const char *filename) {
    // Open the file for input and create a string stream to read its contents
    std::ifstream file(filename);

    if (!file) {
        throw std::runtime_error("Failed to open file");
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();

    char* kernel = new char[buffer.str().length() + 1];

    // Copy the file contents from the string stream to the char array
    buffer.str().copy(kernel, buffer.str().length());
    kernel[buffer.str().length()] = '\0';

    // Return the char array containing the file contents
    return kernel;
}