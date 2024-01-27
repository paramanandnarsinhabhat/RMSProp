
# Neural Network with RMSProp Optimization

This project showcases a simple neural network using RMSProp, a gradient-based optimization technique. It is implemented in Python with the help of NumPy for numerical operations and Matplotlib for visualizing the training process.

## Structure of the Repository

- `/myenv`: A directory intended to contain a virtual environment for project isolation.
- `/source`: Contains the Python script `rmsprop.py` with the neural network implementation.
- `.gitignore`: Specifies intentionally untracked files to ignore.
- `LICENSE`: The full license description for this project.
- `README.md`: Documentation to guide users on the project setup and usage.
- `requirements.txt`: Lists the dependencies necessary to run the project.

## Installation

To get started with this neural network, you must first install its dependencies. Ensure that you have Python installed on your system, and then execute the following command:

```bash
pip install -r requirements.txt
```

This command will install the exact versions of NumPy and Matplotlib required to run the neural network script.

## Usage

To run the neural network:

1. Navigate to the `/source` directory.
2. Execute the script `rmsprop.py` using Python:

```bash
python rmsprop.py
```

## Neural Network Details

The script `rmsprop.py` performs the following operations:

1. **Imports libraries**: Loads NumPy for numerical operations and Matplotlib for plotting.
2. **Creates input and output data**: Initializes the `X` (features) and `y` (labels) arrays and prints their shapes.
3. **Defines the Sigmoid activation function**: Used for activating neurons during the forward pass.
4. **Sets hyperparameters**: Specifies the learning rate, number of neurons, number of epochs, and initializes weights.
5. **Implements RMSProp optimization**: Runs through epochs, performing forward and backward propagation, and adjusts weights using the RMSProp update rule.
6. **Visualizes the training process**: Plots the error reduction and learning rate adjustments over epochs.

## Visualizing the Training Progress

The script produces plots showing:

- The error reduction over each training epoch.
- The adjustments to the learning rates for the hidden and output layers.

These plots help in understanding the effect of RMSProp optimization on the training process.

## Requirements

The `requirements.txt` includes the following libraries:

```
numpy
matplotlib
```

Please ensure these are installed in your environment as instructed in the Installation section above.

## Contributing

Feel free to fork this repository, make improvements, and submit pull requests. We appreciate your contributions to enhance the neural network's functionality or documentation.

## License

This project is open-sourced under the MIT License. See the `LICENSE` file for more information.
