This is the implementation of the proposed method in the paper "Noise Detection with Spectator Qubits and Quantum Feature Engineering" in https://arxiv.org/abs/ The implementation is based on Tensorflow 2.3. The "datsets_and_models" folder constains the datasets created and used for generating the results in the paper as well as the trained models for reference.

The "src" folder contains the following source files:

-Makefile        : This is the GNU MAKEFILE that allows running the code easily from any Unix-like system

-Generating datasets:
	-dataset_gen.py  : This module implements functions for generating the datasets used for training and the testing of the proposed algorithm
	-utilities.py    : This module implements helper functions for the simulations
	-simulator.py    : This module implements a noisy qubit simulator using TF
	
-Training models:

  -train_model.py  : This module is for training the ML model using the generated datasets
  -qubitmlmodel.py : This module implements the machine learning-based model for the qubit
  

-Analysis and results:
	-Detector.py     : This module implements the main class for the quantum noise detector 
	-Example.py      : This module runs an example of training the detector given the trained ML models for the qubits 
	-Outputs.py      : This module generates the plots used in the paper
  
In order to run the provided code, run the Makefile in the src folder (run the following command from the terminal: make all). If you want to use our generated datasets, copy the "*.ds " files from the "datasets_and_models" folder to the "src" folder and run the Makefile. The trained ML models are those files with extension ".mlmodel".
