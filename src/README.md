
# Deep Learning Approaches

In this folder, you will find the different pretrains and trains generated with Deep Learning.

For the generation of a better temporal representation of the vectors, we have created an Autoencoder.

The trainings have been generated with a very simple MLP and with an RNN with different state cells (LSTM, RNN, GRU).

These are still in the beta phase of development, and the implementation is very basic.



## Documentation

The code is divided as follows:

├src;

│   ├── __data__;

│   ├── __models__;

│   └── __utils__;

The code is divided into the following folders:

- The "data" folder contains the functions related to data collection.
    
- The "models" folder contains the files with the architectures of the models.
    
- Finally, the "utils" folder contains more generic functions for the development of the project.





## Run Locally

To run the code more easily, a bash script has been generated.

To execute the run, use the following commands:

```bash

    cd HP;
    
    bash exec.sh
```



