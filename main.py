import numpy as np
import gzip
import matplotlib.pyplot as plt
import seaborn as sns

# Function to read pixel data from the dataset
def read_pixels(data_path, image_size):
    with gzip.open(data_path) as f:
        pixel_data = np.frombuffer(f.read(), 'B', offset=16).astype('float32')
    normalized_pixels = pixel_data / 255
    # Flatten the normalized pixels
    flattened_pixels = normalized_pixels.reshape(-1, image_size * image_size)
    return flattened_pixels

# Function to read label data from the dataset
def read_labels(data_path):
    with gzip.open(data_path) as f:
        label_data = np.frombuffer(f.read(), 'B', offset=8)
    return label_data

images = read_pixels("train-images-idx3-ubyte.gz",28)
labels = read_labels("train-labels-idx1-ubyte.gz")


# One-hot encoding of the labels
def one_hot_encoding(label_data):
    encoded_labels = np.zeros((label_data.size, 10))
    encoded_labels[np.arange(label_data.size), label_data] = 1
    return encoded_labels

# Function to read pixel data from the dataset
def read_pixels(data_path):
    with gzip.open(data_path) as f:
        pixel_data = np.frombuffer(f.read(), 'B', offset=16).astype('float32')
    normalized_pixels = pixel_data / 255
    flattened_pixels = normalized_pixels.reshape(-1, 28*28)
    return flattened_pixels

# Function to read label data from the dataset
def read_labels(data_path):
    with gzip.open(data_path) as f:
        label_data = np.frombuffer(f.read(), 'B', offset=8)
    one_hot_encoding_labels = one_hot_encoding(label_data)
    return one_hot_encoding_labels

# Function to read the entire dataset
def read_dataset():
    X_train = read_pixels("train-images-idx3-ubyte.gz")
    y_train = read_labels("train-labels-idx1-ubyte.gz")
    X_test = read_pixels("t10k-images-idx3-ubyte.gz")
    y_test = read_labels("t10k-labels-idx1-ubyte.gz")
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = read_dataset()

def question1():
    while True:
        print("\nMenu:")
        print("1 - Proceed to Question 1.1")
        print("2 - Proceed to Question 1.2")
        print("3 - Proceed to Question 1.3")
        print("4 - Proceed to Question 1.4")
        print("5 - Proceed to Question 1.5")
        print("0 - Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            # Code for Question 1.1
            mean_vector = np.mean(images, axis=0)
            centered_data = images - mean_vector
            cov_matrix = np.cov(centered_data, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            sorted_indices = np.argsort(eigenvalues)[::-1]
            sorted_eigenvalues = eigenvalues[sorted_indices]
            sorted_eigenvectors = eigenvectors[:, sorted_indices]
            pca_components = sorted_eigenvectors[:, :10]

            #PVE
            total_variance = np.sum(sorted_eigenvalues)
            pve = sorted_eigenvalues[:10] / total_variance
            print("PVE for the first 10 components:", pve)

        elif choice == '2':
            # Code for Question 1.2
            cumulative_pve = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)
            num_components_70_percent = np.where(cumulative_pve >= 0.7)[0][0] + 1  # Adding 1 since index starts from 0
            print("Number of principal components to explain at least 70% of the variance:", num_components_70_percent)

        elif choice == '3':
            # Code for Question 1.3
            def min_max_scaling(matrix):
                min_val = np.min(matrix)
                max_val = np.max(matrix)
                scaled_matrix = (matrix - min_val) / (max_val - min_val)
                return scaled_matrix

            fig, axes = plt.subplots(2, 5, figsize=(10, 4))

            for i in range(10):
                component = pca_components[:, i].reshape(28, 28)
                scaled_component = min_max_scaling(component)

                ax = axes[i // 5, i % 5]
                im = ax.imshow(scaled_component, cmap='Greys_r')
                ax.set_title(f'Component {i+1}')
                ax.axis('off')

                fig.colorbar(im, ax=ax)

            plt.tight_layout()
            plt.show()

        elif choice == '4':
            # Code for Question 1.4
            projected_data = np.dot(centered_data[:100], pca_components[:, :2])

            # Plotting
            plt.figure(figsize=(8, 6))
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'grey', 'pink']

            for i in range(10):
                points = projected_data[labels[:100] == i]
                plt.scatter(points[:, 0], points[:, 1], c=colors[i], label=str(i))

            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title('Projection onto the First 2 Principal Components')
            plt.legend()
            plt.show()

        elif choice == '5':
            # Code for Question 1.5
            pca_components = sorted_eigenvectors

            def reconstruct_image(image, mean_vector, pca_components, k):
                projected = np.dot(image - mean_vector, pca_components[:, :k])
                reconstructed = np.dot(projected, pca_components[:, :k].T)
                reconstructed += mean_vector
                return reconstructed

            k_values = [1, 50, 100, 250, 500, 784]
            first_image = images[0]
            reconstructions = {}

            for k in k_values:
                reconstructions[k] = reconstruct_image(first_image, mean_vector, pca_components, k)

            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            for i, k in enumerate(k_values):
                ax = axes[i // 3, i % 3]
                ax.imshow(reconstructions[k].reshape(28, 28), cmap='Greys_r')
                ax.set_title(f'Reconstructed with {k} components')
                ax.axis('off')
            plt.tight_layout()
            plt.show()

        elif choice == '0':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")
            
            
def question2():
    ## Code For the Model
    class LogisticRegression:
                def __init__(self, n_features, n_classes, batch_size=200, learning_rate=5e-4, lambda_reg=1e-4, epochs=100, weight_initialization='normal',validation_split=0.2):
                    self.n_features = n_features
                    self.n_classes = n_classes
                    self.batch_size = batch_size
                    self.learning_rate = learning_rate
                    self.lambda_reg = lambda_reg
                    self.epochs = epochs
                    self.validation_split = validation_split

                    # Weight initialization
                    if weight_initialization == 'normal':
                        self.weights = np.random.normal(0, 1, (n_features, n_classes))
                    elif weight_initialization == 'zero':
                        self.weights = np.zeros((n_features, n_classes))
                    elif weight_initialization == 'uniform':
                        self.weights = np.random.uniform(-1, 1, (n_features, n_classes))

                    self.bias = np.zeros((1, n_classes))

                def softmax(self, z):
                    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


                def cross_entropy_loss(self, y_true, y_pred):
                    # Cross-entropy loss calculation with L2 regularization
                    ce_loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
                    l2_reg = self.lambda_reg / 2 * np.sum(np.square(self.weights))
                    total_loss = ce_loss + l2_reg
                    return total_loss

                def calculate_gradient(self, X, y_true, y_pred):
                    # Gradient of the cross-entropy loss with L2 regularization
                    gradient = np.dot(X.T, (y_pred - y_true))
                    l2_reg_gradient = self.lambda_reg * self.weights
                    total_gradient = gradient + l2_reg_gradient
                    bias_gradient = np.sum(y_pred - y_true, axis=0)
                    return total_gradient, bias_gradient


                def split_data(self, X, y):
                    split_idx = int(X.shape[0] * (1 - self.validation_split))
                    X_train, X_val = X[:split_idx], X[split_idx:]
                    y_train, y_val = y[:split_idx], y[split_idx:]
                    return X_train, y_train, X_val, y_val

                def train(self, X, y):

                    # Split the data into training and validation
                    X_train, y_train, X_val, y_val = self.split_data(X, y)
                    val_accuracies = []

                    for epoch in range(self.epochs):
                        for i in range(0, X_train.shape[0], self.batch_size):
                            X_batch = X_train[i:i+self.batch_size]
                            y_batch = y_train[i:i+self.batch_size]

                            logits = np.dot(X_batch, self.weights) + self.bias
                            y_pred = self.softmax(logits)

                            gradient, bias_gradient = self.calculate_gradient(X_batch, y_batch, y_pred)

                            self.weights -= self.learning_rate * gradient
                            self.bias -= self.learning_rate * bias_gradient


                        # Compute training accuracy after each epoch
                        train_logits = np.dot(X_train, self.weights) + self.bias
                        train_pred = self.softmax(train_logits)
                        val_logits = np.dot(X_val, self.weights) + self.bias
                        val_pred = self.softmax(val_logits)
                        val_accuracy = self.accuracy(np.argmax(y_val, axis=1), np.argmax(val_pred, axis=1))
                        val_accuracies.append(val_accuracy)

                        if epoch % 10 == 0:
                            print(f'Epoch {epoch+1}/{self.epochs}: '
                                  f'Validation Accuracy: {100*val_accuracy:.4f}%')

                    return val_accuracies

                def predict(self, X):
                    logits = np.dot(X, self.weights) + self.bias
                    y_pred = self.softmax(logits)
                    return np.argmax(y_pred, axis=1)

                def accuracy(self, y_true, y_pred):
                    return np.mean(y_true == y_pred)

                def test(self, X_test, y_test):
                    y_pred = self.predict(X_test)
                    test_accuracy = self.accuracy(np.argmax(y_test, axis=1), y_pred)
                    return y_pred, test_accuracy

                def compute_confusion_matrix(self, true, pred):
                    conf_matrix = np.zeros((self.n_classes, self.n_classes), dtype=int)
                    for t, p in zip(true, pred):
                        conf_matrix[t, p] += 1
                    return conf_matrix

                def plot_confusion_matrix(self, conf_matrix):
                    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
                    plt.xlabel('Predicted Label')
                    plt.ylabel('True Label')
                    plt.title('Confusion Matrix')
                    plt.show()
    while True:
        print("\nMenu for Question 2:")
        print("1 - Proceed to Question 2.1")
        print("2 - Proceed to Question 2.2")
        print("3 - Proceed to Question 2.3")
        print("4 - Proceed to Question 2.4")
        print("5 - Proceed to Question 2.5")
        print("0 - Return to Main Menu")
        choice = input("Enter your choice: ")

        if choice == '1':
            # Code for Question 2.1
            model = LogisticRegression(n_features=X_train.shape[1], n_classes=y_train.shape[1])

            # Train the model
            print("\nTraining Started ...\n")
            model.train(X_train, y_train)

            y_pred, test_accuracy = model.test(X_test, y_test)
            print(f"\nTest accuracy with default hyperparameters: {100*test_accuracy:.4f}%\n")

            conf_matrix = model.compute_confusion_matrix(np.argmax(y_test, axis=1), y_pred)
            model.plot_confusion_matrix(conf_matrix)
        elif choice == '2':
            
            # Code for Question 2.2
            print("\nBatch Size Experiment\n")
            batch_sizes = [1, 64, 50000]
            for batch_size in batch_sizes:
                print(f"\nTraining with batch size: {batch_size}")
                model = LogisticRegression(n_features=X_train.shape[1], n_classes=y_train.shape[1], batch_size=batch_size)
                print("Training Started ...\n")
                accuracies = model.train(X_train, y_train)
                y_pred, test_accuracy = model.test(X_test, y_test)
                print(f"\nTest accuracy with batch size {batch_size}: {100*test_accuracy:.4f}%\n")
                plt.plot(accuracies, label=f'Batch Size {batch_size}')

            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs. Epochs for Different Batch Sizes')
            plt.legend()
            plt.show()
            
            print("\nWeight Initialization Experiment\n")
            initializations = ['zero', 'uniform', 'normal']
            for init in initializations:
                print(f"\nTraining with initialization: {init}")
                model = LogisticRegression(n_features=X_train.shape[1], n_classes=y_train.shape[1], weight_initialization=init)
                print("Training Started ...\n")
                accuracies = model.train(X_train, y_train)
                y_pred, test_accuracy = model.test(X_test, y_test)
                print(f"\nTest accuracy with initialization {init}: {100*test_accuracy:.4f}%\n")
                plt.plot(accuracies, label=f'Initialization {init}')

            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Model Accuracy vs. Epochs for Different Weight Initializations')
            plt.legend()
            plt.show()
            
            print("\nLearning Rate Experiment\n")
            learning_rates = [0.1, 1e-3, 1e-4, 1e-5]
            for lr in learning_rates:
                print(f"\nTraining with Learning Rate: {lr}")
                model = LogisticRegression(n_features=X_train.shape[1], n_classes=y_train.shape[1], learning_rate=lr,batch_size=200)
                print("Training Started ...\n")
                accuracies = model.train(X_train, y_train)
                y_pred, test_accuracy = model.test(X_test, y_test)
                print(f"\nTest accuracy with Learning Rate {lr}: {100*test_accuracy:.4f}%\n")
                plt.plot(accuracies, label=f'Learning Rate {lr}')

            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Model Accuracy vs. Epochs for Different Learning Rates')
            plt.legend()
            plt.show()
            
            print("\nRegularization Coefficient Experiment\n")
            lambdas = [1e-2, 1e-4, 1e-9]
            for lam in lambdas:
                print(f"\nTraining with Regularization coefficient (λ): {lam}")
                model = LogisticRegression(n_features=X_train.shape[1], n_classes=y_train.shape[1], lambda_reg=lam)
                print("Training Started ...\n")
                accuracies = model.train(X_train, y_train)
                y_pred, test_accuracy = model.test(X_test, y_test)
                print(f"\nTest accuracy with Regularization coefficient (λ) {lam}: {test_accuracy:.4f}\n")
                plt.plot(accuracies, label=f'Regularization λ {lam}')

            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Model Accuracy vs. Epochs for Different Regularization Coefficients')
            plt.legend()
            plt.show()
            
        elif choice == '3':
            # Code for Question 3.3
            best_model = LogisticRegression(n_features=X_train.shape[1], n_classes=y_train.shape[1], batch_size=1, weight_initialization='zero', learning_rate=1e-3, lambda_reg=1e-9)
            print("Training Started ...\n")
            best_model.train(X_train, y_train)
            y_pred, test_accuracy = best_model.test(X_test, y_test)
            print(f"\nTest accuracy with best hyperparameters: {100*test_accuracy:.4f}%\n")

            conf_matrix = best_model.compute_confusion_matrix(np.argmax(y_test, axis=1), y_pred)
            best_model.plot_confusion_matrix(conf_matrix)
            
        elif choice == '4':
            # Code for Question 4.4
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            for i in range(10):
                # Get the weights for the ith class
                weight = best_model.weights[:, i].reshape(28, 28)

                # Display the weight image
                ax = axes[i // 5, i % 5]
                ax.matshow(weight, cmap=plt.cm.gray, vmin=0.5 * weight.min(), vmax=0.5 * weight.max())
                ax.set_title(f"Digit {i}")
                ax.axis('off')

            plt.show()
            
        elif choice == '5':
            # Code for Question 5.5
            def calculate_classification_scores(conf_matrix):
                precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
                recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
                f1_score = 2 * (precision * recall) / (precision + recall)
                f2_score = 5 * (precision * recall) / (4*precision + recall)

                return precision, recall, f1_score, f2_score

            conf_matrix = best_model.compute_confusion_matrix(np.argmax(y_test, axis=1), y_pred)
            precision, recall, f1_score, f2_score = calculate_classification_scores(conf_matrix)

            for i in range(model.n_classes):
                print(f"Class {i}: Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, "
                      f"F1 Score: {f1_score[i]:.4f}, F2 Score: {f2_score[i]:.4f}")
        elif choice == '0':
            break
        else:
            print("Invalid choice. Please try again.")
            
            
def main():
    print('Welcome to the CS464 Homework #2')

    while True:
        print("\nMain Menu:")
        print("1 - Display Results for Question 1")
        print("2 - Display Results for Question 2")
        print("0 - Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            question1()
        elif choice == '2':
            question2()
        elif choice == '0':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()


