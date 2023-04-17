#pragma once

namespace io {
    /**
     * Reads the contents of a file into a char array and returns a pointer to it.
     *
     * @param filename The name of the file to read.
     * @return A pointer to a char array containing the contents of the file.
     *
     * @throws std::runtime_error if the file could not be opened for reading.
     *
     * @note The caller is responsible for deallocating the memory allocated for the
     *       char array by calling `delete[]` on the returned pointer.
     */

    const char *readKernel(const char *fileName);
}