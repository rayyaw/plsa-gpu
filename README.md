# APLSA framework

APLSA (Accelerated Probabilistic Latent Semantic Analysis) is a framework designed to perform topic mining on long documents. 

## Prerequisites
- Python3 with urllib (`apt install python3`, `pip3 install urllib`)
- Any C++ compiler (I use g++ for MinGW64, you may need to modify the Makefile if you use a different compiler)
- OpenCL (this part is complicated, you should lookup and download the SDK for your system)
- If running on Windows, you must use [MinGW](https://www.mingw-w64.org/) as WSL does not currently support GPU peripherals.
- If running the GPU implementation, you must run on a computer with a GPU.


## Data Pipeline

The data pipeline is designed as follows:

1. Data download. This part cannot be GPU accelerated, as the limiting factor is your network speed. This stage took me a total of 15 hours (in several sessions) to complete.

`python3 stage1_download_books.py`

This will download most of the [Project Gutenberg collection](https://www.gutenberg.org/) into the `books/` folder. The only preprocessing at this stage is removing the Project Gutenberg headers and footers (these have information such as the title, author, and license). We include the title and author into the title of the book instead. The output format is `.txt`.

2. Data preprocessing. This phase takes the raw book data and converts it into per-document word counts. Here, we also generate the background language model as output (using maximum likelihood estimation). The background model will be saved to `model/` folder.

CPU version (no acceleration): `make stage2_cpu && bin/stage2_cpu [max_books]`

GPU version (with acceleration): Not available due to time constraints. I focused on optimizing stage 3 instead.

Note that the `max_books` parameter is optional. It specifies the maximum number of books to use. I highly recommend setting this below 500 (at 500 documents, the model generated is already 1.9 GB, and this grows with number of documents as we need to store per-document counts of each word).

3. EM algorithm runtime. This phase takes the counts, background LM, probability of background LM, and topic count, and generates LMs for each topic, and document coverage data using the EM algorithm. This is saved to `model/` folder.

CPU version (no acceleration): `make stage3_cpu && bin/stage3_cpu <number of topics to generate> <background model probability>`

GPU version (with acceleration): `make stage3_gpu && bin/stage3_gpu <number of topics to generate> <background model probability>`

Note that both arguments are required.

Note also that due to C++ makefiles, even the CPU version requires you to have OpenCL installed on your system (though a GPU is not required to run analysis).

4. Postprocessing. This phase takes the output models and converts them into a form that allows for easier human analysis.

CPU version (no acceleration): `make stage4_cpu && bin/stage4_cpu <top_words_per_topic>`

GPU version (with acceleration): Not available due to time constraints.

Note that the `top_words_per_topic` argument is required. This specifies how many words to save per topic into display format.

5. Output Analysis. This script allows you to look at the topic models and document coverage given a specific document. It will print the output to the terminal.

`make stage5_cpu && bin/stage5_cpu <book to analyze> <number of topics to display>`

Note that both arguments are required. The `book to analyze argument` should be a path from the folder `README` is in to the book, for example `books/10-The-Book-Title.txt`. The `number of topics to display` argument is used to show only the top N topics, not including the background model. This stage displays both the document coverage and the top words for that topic.

You can also run `make` to make all targets, and call each stage in the pipeline without needing to run `make` for each stage.

**Important Note:** You may need to modify the `Makefile` depending on the exact location of your OpenCL install. You should replace the `-L` and `-I` arguments with the path to your OpenCL library and header files respectively.

## Model Format

The model is saved in multiple files, which are structured as follows:

### Generated by stage 2

- `model/bg.bin` contains the probability of each word in the background model. This is calculated as a simple maximum likelihood estimate from the entire collection, with no smoothing. It uses a unigram LM, so every word is independent.
- `model/counts.bin` contains the per-document counts of each word, and is the largest file. I considered compressing this by using CSR matrix, but the word matrix is too dense for such a representation to save storage space.
- `model/file_encodings.txt` specifies the ordering of files. The model internally converts files and words to numbers (since the GPU doesn't support maps), so we need to store a consistent ordering which we can use to convert back into human-usable data.
- `model/word_encodings.txt` specifies the ordering of words. See `model/file_encodings.txt` for more information.

### Generated by stage 3
- `model/document_coverage.bin` stores the document coverage for each topic/document pair.
- `model/topic_models.bin` stores the generated topic models. Each model is a unigram LM.

### Generated by stage 4
- `model/topic_summary.txt` stores only the top words in each topic, sorted by probability. Computing this in stage 4 skips the requirement of re-sorting the entire model every time stage 5 is called.

## License
All source code is available under CC-BY SA 4.0. For more details, check https://creativecommons.org/licenses/by-sa/4.0/.