This script is designed for CS464 Homework #2. It includes two main sections: Question 1 and Question 2, each containing sub-questions related to PCA and logistic regression implementation.

Prerequisites:

    Python 3.x
    NumPy library
    Matplotlib library
    Seaborn library
    Gzip library 

Dataset:

The script expects MNIST dataset files in the gzip format.

    train-images-idx3-ubyte.gz
    train-labels-idx1-ubyte.gz
    t10k-images-idx3-ubyte.gz
    t10k-labels-idx1-ubyte.gz

Ensure these files are in the same directory as the script.

Running the Script:

Run the script using a Python interpreter. On launching, the script displays a welcome message and then shows the main menu with options to select either Question 1 or Question 2.

Main Menu Options

    1: Display Results for Question 1
    2: Display Results for Question 2
    0: Exit the program

Sub-Menu for Question 1 and Question 2

Each question has its own submenu where you can choose to proceed to specific sub-questions.

Important Notice for Question 2: For Question 2, it is crucial to run sub-question 2.3 before attempting to run sub-questions 2.4 and 2.5. This ensures the logistic regression model is properly trained and available for subsequent analyses.

Outputs:

The script outputs vary depending on the sub-question selected:

    For PCA questions in Question 1, outputs include eigenvalues, visual representations of principal components, and reconstructed images.
    For logistic Regression Model in Question 2, outputs include model training progress, accuracy metrics, confusion matrices, weight visualizations, and classification scores.
