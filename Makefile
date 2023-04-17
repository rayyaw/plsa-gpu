# Use build/ to store temp files
# Use bin/ to store final executables

all: all_cpu #all_gpu

clean:
	rm -rf bin/*.exe
	rm -rf build/*

stage2_cpu: stage2_cpu_preprocessing.cpp
	g++ stage2_cpu_preprocessing.cpp -O3 -o bin/stage2_cpu

model_data: stage3/modelData.cpp stage3/modelData.h
	g++ -c stage3/modelData.cpp -o build/model_data.o

em_step: model_data stage3/emStep.h stage3/emStep.cpp
	g++ -c stage3/emStep.cpp -o build/em_step.o

stage3_cpu_cpp: stage3_cpu_processing.cpp
	g++ -c stage3_cpu_processing.cpp -O3 -o build/stage3_cpu_cpp.o

stage3_cpu: stage3_cpu_cpp model_data em_step
	g++ build/stage3_cpu_cpp.o build/model_data.o build/em_step.o -O3 -o bin/stage3_cpu

stage4_cpu_cpp: stage4_cpu_postprocessing.cpp
	g++ -c build stage4_cpu_postprocessing.cpp -o build/stage4_cpu_cpp.o

stage4_cpu: stage4_cpu_cpp em_step
	g++ build/stage4_cpu_cpp.o build/em_step.o build/model_data.o -o bin/stage4_cpu

stage5_cpu: stage5_cpu_display.cpp
	g++ stage5_cpu_display.cpp -o bin/stage5_cpu

all_cpu: stage2_cpu stage3_cpu stage4_cpu stage5_cpu

gpu: gpu/gpu.cpp
	clang++ -o build/gpu.o -c gpu/gpu.cpp -I 'C:/Program Files (x86)/OCL_SDK_Light/include/'

io: io/io.cpp
	clang++ -o build/io.o -c io/io.cpp

stage2_gpu_cpp: stage2_gpu_preprocessing.cpp
	clang++ -c stage2_gpu_preprocessing.cpp -o build/stage2_gpu_cpp.o -I 'C:/Program Files (x86)/OCL_SDK_Light/include/'

stage2_gpu: gpu io stage2_gpu_cpp
	clang++ build/stage2_gpu_cpp.o build/io.o build/gpu.o -o bin/stage2_gpu -I 'C:/Program Files (x86)/OCL_SDK_Light/include/' -L 'C:/Program Files (x86)/OCL_SDK_Light/lib/x86_64/' -lOpenCL

all_gpu: stage2_gpu