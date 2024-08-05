import tensorflow as tf
import logging

def create_model(input_shape, learning_rate=0.001, activation='sigmoid'):
    try:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=input_shape),
            tf.keras.layers.Dense(1, activation=activation)  # Output layer
        ])
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                      optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                      metrics=['accuracy'])
        logging.info("Model created successfully.")
        return model
    except Exception as e:
        logging.error(f"Error creating model: {e}")
        return None

def train_model(model, x_train, y_train, epochs=50, verbose=0):
    try:
        history = model.fit(x_train, y_train, epochs=epochs, verbose=verbose)
        logging.info("Model training completed.")
        return history
    except Exception as e:
        logging.error(f"Error in model training: {e}")
        return None

def evaluate_model(model, x_test, y_test):
    try:
        evaluation = model.evaluate(x_test, y_test)
        logging.info("Model evaluation completed.")
        return evaluation
    except Exception as e:
        logging.error(f"Error in model evaluation: {e}")
        return None
