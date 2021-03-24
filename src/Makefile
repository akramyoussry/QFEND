all: Outputs.out


G_1q_X_Z_N0.ds: simulator.py dataset_gen.py utilities.py
	python dataset_gen.py 0
G_1q_X_Z_N1.ds: simulator.py dataset_gen.py utilities.py
	python dataset_gen.py 1
G_1q_X_Z_N2.ds: simulator.py dataset_gen.py utilities.py
	python dataset_gen.py 2
G_1q_X_Z_N3.ds: simulator.py dataset_gen.py utilities.py
	python dataset_gen.py 3
G_1q_X_Z_N4.ds: simulator.py dataset_gen.py utilities.py
	python dataset_gen.py 4
G_1q_X_Z_N5.ds: simulator.py dataset_gen.py utilities.py
	python dataset_gen.py 5

	
G_1q_X_Z_N0.mlmodel: G_1q_X_Z_N0.ds qubitmlmodel.py train_model.py
	python train_model.py 0
G_1q_X_Z_N1.mlmodel: G_1q_X_Z_N1.ds qubitmlmodel.py train_model.py
	python train_model.py 1
G_1q_X_Z_N2.mlmodel: G_1q_X_Z_N2.ds qubitmlmodel.py train_model.py
	python train_model.py 2
G_1q_X_Z_N3.mlmodel: G_1q_X_Z_N3.ds qubitmlmodel.py train_model.py
	python train_model.py 3
G_1q_X_Z_N4.mlmodel: G_1q_X_Z_N4.ds qubitmlmodel.py train_model.py
	python train_model.py 4
G_1q_X_Z_N5.mlmodel: G_1q_X_Z_N5.ds qubitmlmodel.py train_model.py
	python train_model.py 5


Example_0.out: G_1q_X_Z_N0.ds G_1q_X_Z_N0.ds G_1q_X_Z_N0.ds G_1q_X_Z_N0.ds G_1q_X_Z_N0.ds G_1q_X_Z_N0.ds G_1q_X_Z_N0.mlmodel G_1q_X_Z_N1.mlmodel G_1q_X_Z_N2.mlmodel G_1q_X_Z_N3.mlmodel G_1q_X_Z_N4.mlmodel G_1q_X_Z_N5.mlmodel Detector.py 
	python Example.py 0 > Example_0.out
Example_1.out: G_1q_X_Z_N0.ds G_1q_X_Z_N0.ds G_1q_X_Z_N0.ds G_1q_X_Z_N0.ds G_1q_X_Z_N0.ds G_1q_X_Z_N0.ds G_1q_X_Z_N0.mlmodel G_1q_X_Z_N1.mlmodel G_1q_X_Z_N2.mlmodel G_1q_X_Z_N3.mlmodel G_1q_X_Z_N4.mlmodel G_1q_X_Z_N5.mlmodel Detector.py 
	python Example.py 1 > Example_1.out
Example_2.out: G_1q_X_Z_N0.ds G_1q_X_Z_N0.ds G_1q_X_Z_N0.ds G_1q_X_Z_N0.ds G_1q_X_Z_N0.ds G_1q_X_Z_N0.ds G_1q_X_Z_N0.mlmodel G_1q_X_Z_N1.mlmodel G_1q_X_Z_N2.mlmodel G_1q_X_Z_N3.mlmodel G_1q_X_Z_N4.mlmodel G_1q_X_Z_N5.mlmodel Detector.py 
	python Example.py 2 > Example_2.out


Outputs.out: Example_0.out Example_1.out Example_2.out Outputs.py
	python Outputs.py

clean: 
	rm -f *.ds *.npy *.mlmodel *.out *.pdf
