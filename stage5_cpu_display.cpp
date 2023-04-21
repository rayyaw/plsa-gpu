/**
 * @file stage5_cpu_display.cpp
 * @author RileyTech
 * @brief 
 * @date 2023-04-15
 * 
 * @copyright Copyright (c) 2023. Licensed under CC-BY SA.
 * 
 */

// C headers
#include <stdio.h>
#include <stdlib.h>

// C++ headers
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

using std::cerr;
using std::cout;
using std::endl;
using std::getline;
using std::ifstream;
using std::pair;
using std::string;
using std::vector;

// We use char* since this comes from argv
size_t getIndexFromFilename(const char *filename) {
    ifstream file("model/file_encodings.txt");

    string line;
    size_t line_num = 0;

    while (getline(file, line)) {
        // do something with the line
        if (line == filename) {
            file.close();
            return line_num;
        }

        line_num++;
    }

    file.close();
    return -1;
}


// FIXME - This is incorrect, document_coverage.bin is in topic-major order
vector<pair<double, size_t>> getMostCoveredTopics(size_t document_number, int num_topics, size_t total_topics, size_t num_documents, size_t words_per_topic) {
    vector<pair<double, size_t>> most_covered_topics = vector<pair<double, size_t>>(num_topics);
    
    FILE *document_coverage_file = fopen("model/document_coverage.bin", "rb");
    double *document_coverage = new double[total_topics * num_documents];

    if (!document_coverage_file) {
        cerr << "Unable to open input file, exiting" << endl;
        exit(1);
    }

    // Read document_coverage from file
    fread(document_coverage, sizeof(double), total_topics * num_documents, document_coverage_file);
    fclose(document_coverage_file);

    for (int i = 0; i < num_topics; i++) {
        double current = 0;
        size_t current_idx = 0;

        for (size_t j = 0; j < total_topics; j++) {
            // Handle the edge case of being the first element
            if (document_coverage[j*num_documents + document_number] > current && 
               (i == 0 || document_coverage[j*num_documents + document_number] < most_covered_topics[i-1].first)) {
                current = document_coverage[j*num_documents + document_number];
                current_idx = j;
            }
        }

        most_covered_topics[i] = pair<double, size_t>(current, current_idx);
    }

    delete[] document_coverage;
    return most_covered_topics;
}

void printTopicInformation(vector<pair<double, size_t>> most_covered_topics, size_t words_per_topic) {
    // C++ is mean so we need to open another ifstream on the same file
    for (size_t i = 0; i < most_covered_topics.size(); i++) {
        ifstream topic_models("model/topic_summary.txt");
        // Throw away unused data   
        size_t num_topics, num_documents, num_words;
        topic_models >> num_topics >> num_documents >> num_words;

        cout << "---- " << i+1 << "th most covered topic was number " << 
        most_covered_topics[i].second << " at probability " << most_covered_topics[i].first << " ----" << endl;

        // Seek through the file to find the topic
        string word_prob; string word;
        string line;
        // Get rid of some extra data
        getline(topic_models, line);

        for (size_t j = 0; j < most_covered_topics[i].second * words_per_topic; j++) {
            getline(topic_models, line);
        }

        for (size_t j = 0; j < words_per_topic; j++) {
            getline(topic_models, line);
            word_prob = line.substr(0, line.find(" "));
            word = line.substr(line.find(" ") + 1, line.size()); 
            cout << "P(" << word << ") = " << word_prob << endl;
        }

        cout << endl;

        topic_models.close();
    }
}

// stage5 bookFile.txt topTopics
int main (int argc, char *argv[]) {
    if (argc != 3) {
        cerr << argv[0] << " [file name to summarize] [number of topics to analyze]" << endl;
    }

    int topics_to_analyze = atoi(argv[2]);

    ifstream topic_models("model/topic_summary.txt");
    size_t num_topics, num_documents, num_words;

    topic_models >> num_topics >> num_documents >> num_words;
    topic_models.close();

    // Find the file as an index
    size_t file_index = getIndexFromFilename(argv[1]);
    vector<pair<double, size_t>> most_covered_topics = getMostCoveredTopics(file_index, topics_to_analyze, num_topics, num_documents, num_words);
    printTopicInformation(most_covered_topics, num_words);

}