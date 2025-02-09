# Deep Learning Classification Model

This project implements a deep learning classification model using TensorFlow and Keras. It involves training a model to predict employee attrition based on various features. The project includes data preprocessing, model training, evaluation, and visualization of learning rates and loss curves.

## Project Structure

- `main.py`: The main script for running the deep learning model. It handles data loading, preprocessing, model creation, training, evaluation, and visualization.
- `data_loader.py`: Contains functions to load data from a CSV file.
- `preprocessing.py`: Includes functions for preprocessing the data (e.g., scaling, splitting).
- `modeling.py`: Defines functions for creating, training, and evaluating the deep learning model.
- `visualization.py`: Contains functions for plotting the learning rate vs. loss and training loss curves.
- `requirements.txt`: Lists the required Python packages for the project.

## Installation

1. **Clone the repository:**

    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Create and activate a virtual environment (optional but recommended):**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare your data file:**

   Ensure that your data file `employee_attrition.csv` is in the same directory as `main.py`.

2. **Run the main script:**

    ```sh
    python main.py
    ```

   This script will load the data, preprocess it, train the model, evaluate its performance, and generate the plots for learning rate vs. loss and training loss curves.

## Contributing

Feel free to open issues or submit pull requests for improvements or bug fixes.
#   D e e p - L e a r n i n g  
 