import logging
from data_loader import load_data
from preprocessing import preprocess_data
from modeling import create_model, train_model, evaluate_model
from visualization import plot_loss, plot_loss_learning
from sklearn.metrics import accuracy_score
import tensorflow as tf

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        # Load and preprocess data
        df = load_data('employee_attrition.csv')
        if df is None:
            return

        x_train, x_test, y_train, y_test = preprocess_data(df)
        if x_train is None or x_test is None:
            return

        # Create and train model
        model = create_model(input_shape=(x_train.shape[1],), learning_rate=0.0009, activation='sigmoid')
        if model is None:
            return

        history = train_model(model, x_train, y_train, epochs=50)
        if history is None:
            return

        # Evaluate model
        evaluation = evaluate_model(model, x_test, y_test)
        if evaluation is None:
            return

        # Predict and evaluate accuracy
        y_preds = model.predict(x_test)
        y_preds = tf.round(y_preds)
        accuracy = accuracy_score(y_test, y_preds)
        logging.info(f"Test Accuracy: {accuracy}")

        # Plot loss curves
        plot_loss(history)
        plot_loss_learning(history)

    except Exception as e:
        logging.error(f"An error occurred in the main process: {e}")

if __name__ == "__main__":
    main()
