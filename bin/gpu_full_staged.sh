#!/usr/bin/bash
# Bash is required, as sh cannot interface with the GPU, and this breaks stage 3

bin/stage2_cpu.exe 500

echo "Stage 2 completed ----"
echo ""

bin/stage3_gpu.exe 3 0.9

echo "Stage 3 completed ----"
echo ""

bin/stage4_cpu.exe 25

echo "Stage 4 completed ----"
echo ""

bin/stage5_cpu.exe "books/100 - The Complete Works of William Shakespeare, William Shakespeare.txt" 3