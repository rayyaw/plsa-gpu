# APLSA framework

APLSA (Accelerated Probabilistic Latent Semantic Analysis) is a framework designed to perform topic mining on long documents. 

## Prerequisites
- Python3 with urllib (`apt install python3`, `pip3 install urllib`)
- Any C++ compiler (I use clang for MinGW64)
- OpenCL (this part is complicated, you should lookup and download the SDK for your system)
- If running on Windows, you must use [MinGW](https://www.mingw-w64.org/) as WSL does not currently support GPU peripherals.
- If running the GPU implementation, you must run on a computer with a GPU.


## Data Pipeline

The data pipeline is designed as follows:

1. Data download. This part cannot be GPU accelerated, as the limiting factor is your network speed. This stage took me a total of 15 hours (in several sessions) to complete.

`python3 download-books.py`

This will download most of the [Project Gutenberg collection](https://www.gutenberg.org/) into the `books/` folder. The only preprocessing at this stage is removing the Project Gutenberg headers and footers (these have information such as the title, author, and license). We include the title and author into the title of the book instead. The output format is `.txt`.

2. Data preprocessing. This phase takes the raw book data and converts it into per-document word counts. Here, we also generate the background language model as output (using maximum likelihood estimation). The background model will be saved to `models/bg.plsa`.

CPU version (no acceleration):

GPU version (with acceleration):

3. EM algorithm runtime. This phase takes the counts, background LM, probability of background LM, and topic count, and generates LMs for each topic, and document coverage data using the EM algorithm. This is saved to `models/` folder.

CPU version (no acceleration):

GPU version (with acceleration):

4. Postprocessing. This phase takes the output models and converts them into a form that allows for easier human analysis.

CPU version (no acceleration):

GPU version (with acceleration):

5. Output Analysis. This script allows you to look at the topic models and document coverage given a specific document.