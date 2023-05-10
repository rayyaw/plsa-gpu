# Use build/ to store temp files
# Use bin/ to store final executables

OPENCL_H_PATH='C:/Program Files (x86)/OCL_SDK_Light/include/'
OPENCL_LIB_PATH='C:/Program Files (x86)/OCL_SDK_Light/lib/x86_64/'

all: all_cpu all_gpu

clean:
	rm -rf bin/*.exe
	rm -rf build/*.o

stage2_cpu: stage2_cpu_preprocessing.cpp
	g++ stage2_cpu_preprocessing.cpp -O3 -o bin/stage2_cpu

model_data: stage3/modelData.cpp stage3/modelData.h
	g++ -c stage3/modelData.cpp -o build/model_data.o -I ${OPENCL_H_PATH}

em_step: model_data stage3/emStep.h stage3/emStep.cpp
	g++ -c stage3/emStep.cpp -o build/em_step.o -I ${OPENCL_H_PATH}

stage3_cpu_cpp: stage3_cpu_processing.cpp
	g++ -c stage3_cpu_processing.cpp -O3 -o build/stage3_cpu_cpp.o -I ${OPENCL_H_PATH}

stage4_cpu_cpp: stage4_cpu_postprocessing.cpp
	g++ -c build stage4_cpu_postprocessing.cpp -o build/stage4_cpu_cpp.o -I ${OPENCL_H_PATH}

stage4_cpu: stage4_cpu_cpp em_step gpu io linalg
	g++ build/io.o build/linalg.o build/gpu.o build/stage4_cpu_cpp.o build/em_step.o build/model_data.o -o bin/stage4_cpu -I ${OPENCL_H_PATH} -L ${OPENCL_LIB_PATH} -lOpenCL

stage5_cpu: stage5_cpu_display.cpp
	g++ stage5_cpu_display.cpp -o bin/stage5_cpu

all_cpu: stage2_cpu stage3_cpu stage4_cpu stage5_cpu

gpu: gpu/gpu.cpp
	g++ -o build/gpu.o -c gpu/gpu.cpp -I ${OPENCL_H_PATH}

io: io/io.cpp
	g++ -o build/io.o -c io/io.cpp

linalg: linalg/sgemm.cpp
	g++ -c linalg/sgemm.cpp -o build/linalg.o -I ${OPENCL_H_PATH}

reduce: linalg/reduce.cpp
	g++ -c linalg/reduce.cpp -o build/reduce.o -I ${OPENCL_H_PATH}

stage3_cpu: stage3_cpu_cpp model_data em_step gpu io linalg
	g++ build/stage3_cpu_cpp.o build/model_data.o build/em_step.o build/io.o build/linalg.o build/gpu.o -O3 -o bin/stage3_cpu -I ${OPENCL_H_PATH} -L ${OPENCL_LIB_PATH} -lOpenCL

stage3_gpu_cpp: stage3_gpu_processing.cpp
	g++ -c stage3_gpu_processing.cpp -O3 -o build/stage3_gpu_cpp.o -I ${OPENCL_H_PATH}

stage3_gpu: io linalg gpu stage3_gpu_cpp model_data em_step reduce
	g++ build/reduce.o build/em_step.o build/model_data.o build/stage3_gpu_cpp.o build/io.o build/linalg.o build/gpu.o -o bin/stage3_gpu -I ${OPENCL_H_PATH} -L ${OPENCL_LIB_PATH} -lOpenCL

all_gpu: stage3_gpu