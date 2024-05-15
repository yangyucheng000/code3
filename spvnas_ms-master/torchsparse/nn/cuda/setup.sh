#! /bin/bash
cd convolution
python3 setup.py convolution_cuda.cu convolution_cuda.so

cd ..
cd devoxelize
python3 setup.py devoxelize_cuda.cu devoxelize_cuda.so

cd ..
cd 'hash'
python3 setup.py hash_cuda.cu hash_cuda.so

cd ..
cd hashmap
python3 setup.py hashmap_cuda.cu hashmap_cuda.so

cd ../others
python3 setup.py count_cuda.cu count_cuda.so
python3 setup.py query_cuda.cu query_cuda.so

cd ../voxelize
python3 setup.py voxelize_cuda.cu voxelize_cuda.so
