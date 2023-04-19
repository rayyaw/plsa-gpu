// Stage 2 of 5 in the machine learning pipeline.
// Calculate the per-document smoothed counts, and the background model.
// Save the counts to counts/ folder and the background model as models/bg.plsa.

// C headers
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

// C++ headers
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using std::atoi;
using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::map;
using std::set;
using std::string;
using std::stringstream;
using std::ofstream;
using std::vector;

int main(int argc, char *argv[]) {
    // Open the books/ folder and read in the books list.
    // The first section is all i/o so can't be GPU accelerated.
    if (argc > 2) {
        cerr << "Usage: " << argv[0] << " [max_books - optional]" << endl;
        exit(1);
    }

    vector<string> bookNames = vector<string>();

    // This will actually be LONG_MAX since size_t is unsigned
    size_t max_books = -1;
    size_t book_count = 0;

    if (argc > 1) {
        max_books = atoi(argv[1]);
    }

    DIR *books = opendir("./books");
    dirent *entry;

    // Find all applicable books
    if (books != NULL) {
        while ((entry = readdir(books)) != NULL) {
            string long_name = "books/";
            long_name += entry -> d_name;
            bookNames.push_back(long_name);

            book_count++;

            // Make sure we don't exceed book limit
            if (book_count >= max_books) break;
        }
    }

    cout << "Detected " << bookNames.size() << " books. Converting to counts..." << endl;

    // Map from the book name to its unsmoothed counts (smoothing will come later)
    map<string, map<string, size_t>> document_word_counts = map<string, map<string, size_t>>();
    set<string> vocabulary = set<string>();
    size_t document_count = 0;
    size_t total_words = 0;

    // Convert each book to per-document word counts
    for (string i : bookNames) {
        // Check for non-books (like directories . and ..)
        string extension = ".txt";
        if (i.length() <= extension.length() || i.substr(i.length() - extension.length()) != extension) continue;

        ifstream f(i);
        string content((std::istreambuf_iterator<char>(f)), (std::istreambuf_iterator<char>()));
        
        stringstream content_stream = stringstream();
        content_stream << content;

        size_t words_this_document = 0;

        while (!content_stream.eof()) {
            string word;
            content_stream >> word;

            // strip punctuation and set lowercase (could this be GPU accelerated?)
            // reduce the counts as much as possible
            std::transform(word.begin(), word.end(), word.begin(), [](unsigned char c) { return std::tolower(c); });
            word.erase(std::remove_if(word.begin(), word.end(), [](unsigned char c){ return std::ispunct(c); }), word.end());

            total_words++;
            vocabulary.insert(word);

            // If word in dict, add 1. Else, set the value to 1.
            if (document_word_counts[i].find(word) != document_word_counts[i].end()) {
                document_word_counts[i][word]++;
            } else {
                document_word_counts[i][word] = 1;
            }
        }

        f.close();

        document_count++;

        if (document_count % 100 == 0) cout << document_count << " ";

    }

    cout << endl << "Finished collecting counts from all documents. Generating encodings..." << endl;
    
    // Create a unique int to represent each word. Save the output to a map (and file).
    // The line of the file is the index in the embedding.
    map<string, size_t> word_encodings = map<string, size_t>();
    ofstream word_encoding_file("model/word_encodings.txt");

    size_t word_value = 0;
    for (auto i = vocabulary.begin(); i != vocabulary.end(); i++) {
        word_encodings[*i] = word_value;
        word_encoding_file << *i << endl;
        word_value++;
    }

    word_encoding_file.close();

    // Convert the document maps to use the integer indices (as arrays).
    cout << "Found " << vocabulary.size() << " words. Encoding completed, generating written counts and model..." << endl;

    size_t *encoded_word_counts = new size_t[word_encodings.size() * document_count];

    // We also need to save the document names to files, in order
    ofstream file_encoding_file("model/file_encodings.txt");
    size_t file_value = 0;

    for (auto i : document_word_counts) {
        file_encoding_file << i.first << endl;

        for (auto word : i.second) {
            encoded_word_counts[(file_value * word_encodings.size()) + word_encodings[word.first]] = word.second;            
        }

        file_value++;
    }

    file_encoding_file.close();

    cout << "Encoding counts and saving to file..." << endl;

    // Save the per-document counts to a file
    ofstream count_file2("model/counts.bin", std::ios::binary);
    size_t vocab_size = word_encodings.size();
    const char* docBin = reinterpret_cast<const char*>(&document_count);
    const char* vocabBin = reinterpret_cast<const char*>(&vocab_size);
    count_file2.write(docBin, sizeof(size_t));
    count_file2.write(vocabBin, sizeof(size_t));
    count_file2.write(reinterpret_cast<const char*>(encoded_word_counts), sizeof(size_t)* word_encodings.size() * document_count);

    count_file2.close();


    // Launch a GPU kernel to calculate the background model for every word. No smoothing is required.
    // CPU implementation
    // FIXME - More penalties for common words
    cout << "Calculating background model..." << endl;
    double *background_model = new double[word_encodings.size()];
    double normalization_factor = 0;

    for (size_t i = 0; i < word_encodings.size(); i++) {
        size_t total_count = 0;

        for (size_t j = 0; j < document_count; j++) {
            total_count += encoded_word_counts[j * word_encodings.size() + i];
        }

        // Use the square to help discriminate against common words
        background_model[i] = (double) total_count * total_count;
        normalization_factor += (double) total_count * total_count;
    }

    for (size_t i = 0; i < word_encodings.size(); i++) {
        background_model[i] /= normalization_factor;
    }

    ofstream background_model_file("model/bg.bin", std::ios::binary);
    background_model_file.write(reinterpret_cast<const char*>(background_model), sizeof(double) * vocab_size);
    background_model_file.close();

    // Save the total counts within each document to a file
    cout << "Calculating total document counts..." << endl;
    size_t *document_totals = new size_t[document_count];

    for (size_t i = 0; i < document_count; i++) {
        size_t total_count = 0;

        for (size_t j = 0; j < vocab_size; j++) {
            total_count += encoded_word_counts[i * word_encodings.size() + j];
        }

        document_totals[i] = total_count;
    }

    ofstream total_count_file("model/total_document_counts.bin", std::ios::binary);
    total_count_file.write(reinterpret_cast<const char*>(document_totals), sizeof(size_t) * document_count);
    total_count_file.close();

    // Free resources before returning
    delete[] background_model;
    delete[] document_totals;
    delete[] encoded_word_counts;

    cout << "Done!" << endl;
}