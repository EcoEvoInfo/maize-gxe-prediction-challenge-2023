We trained a 1D-CNN to reduce the dimensionality of input weather data. Eight different models were trained using a 'leave one year out' (LOYO) approach, using one year from the training data as a hold-out validation set.

* `01_generate_weather_dicts.py`: processes input data; scales training, validation, and test inputs based on training set; and outputs a file of dictionaries containing arrays for each environment.  
* `02_train_weather_genobatch.py`: trains a simple 1D-CNN model for each training and validation set. Requires data generators in `myClasses_genobatch.py`. To control for variation attributable to genotype rather than environment, 1D-CNNs were trained in batches of the same hybrid which gave slightly better performance in 7/8 LOYO validation sets in our testing, though computation was much slower due to the greater number of batches in each epoch.   
* `03_reduced_weather_features.py`: extracts outputs from the penultimate layer of each trained model.

Weights and output logs for the best model for each LOYO set can be found in the `best_models` directory.

Best models for each LOYO used different archictures as follows (see `02_train_weather_genobatch.py` for one example or output of running `model.summary()` after loading model in `best_models`: 

| Architecture | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021|
| :--- | ---:|---:|---:|---:|---:|---:|---:|---:| 
| sets of CNN Layers\*|1|1|1|2|3|1|1|2|
| kernel size |12|21|12|21|3|21|24|21|
| # filters in last dense layer | 30 | 3 | 30 |30 |30 |30 |30 |30 |
| RMSE (Mg/ha) | 2.38 | 2.62 |2.44| 2.81 | 2.59 | 3.21 | 2.42 | 2.62| 

\*refers to a set of two 1D convolutional layers followed by a dropout layer and a maxpool layer with parameters in `02_train_weather_genobatch.py`
