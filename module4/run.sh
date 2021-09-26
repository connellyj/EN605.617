#!/bin/bash
nvcc assignment.cu -o assignment -std=c++11
./assignment 10000000
