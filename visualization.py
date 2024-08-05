import matplotlib.pyplot as plt
import pandas as pd
import logging

def plot_loss(history):
    try:
        pd.DataFrame(history.history).plot()
        plt.title("Model training curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss/Accuracy")
        plt.show()
        logging.info("Loss plot generated.")
    except Exception as e:
        logging.error(f"Error in plotting loss curves: {e}")
def plot_loss_learning(history):
    try:
         
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Model Training Curves')
        plt.legend()
        plt.show()
    except Exception as e:
        logging.error(f"Error in plotting {e}")