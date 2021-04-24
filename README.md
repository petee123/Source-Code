# Project Title: Neural Feature Search: A Neural Architecture for Automated Feature Engineering

# testing system environment
- Operating system: Windows 10 version 20H2 build:19042.928
- CPU: Intel(R) Core(TM) i5-8400 CPU @ 2.80GHz   2.81 GHz
- Main Memory: DDR4 16GB
- Display Card: NVIDIA Geforce GTX 1060 5G
- IDE: Visual studio code
- Program Language: Python 3.8.5


# setup python environment

1. install python

# install the necessary library from requirements.txt
2. execute command "pip install -r requirements.txt"

3. install correct version NVIDIA CUDA Toolkit (dependent on the display card model)

4. install correct tensorflow library  (need to install tensorflow 2 or above) 
Reference link:
https://www.tensorflow.org/install/pip?hl=zh-tw

5. install visual studio and install the python extension

# compile and execute the program
6. create the python launch.json 

7. open the file main.py

8. execute the program. (shortcut key: F5) (if there are some error related to the library, need to install the missing library follow the message)

# the description of each source file
Folder: 
1. databset: save all test data set file
2. log: save all log and print information, include the test result file
3. .vscode: include the sample launch.json

File:
1. requirement.txt: include the required librarys for this project which can be generated on the requirement file
2. main.py: main program 

# an example to show how to run the program
1. press shortcut key F5 in visual studio code or execute the cmd "python main.py"