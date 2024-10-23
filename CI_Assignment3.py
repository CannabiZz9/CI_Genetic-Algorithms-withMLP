import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Load and preprocess the WDBC dataset manually
def load_data(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            line_data = line.strip().split(',')
            features = list(map(float, line_data[2:]))  # Features from index 3 to 32
            diagnosis = 1 if line_data[1] == 'M' else 0  # Diagnosis: M = 1, B = 0
            data.append((features, diagnosis))
    return data

# Normalize the features
def normalize(data):
    X = np.array([x[0] for x in data])
    y = np.array([x[1] for x in data])
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y

# Split the data into train and validation sets for cross-validation
def k_fold_split(data, k=10):
    random.shuffle(data)
    fold_size = len(data) // k
    return [data[i * fold_size:(i + 1) * fold_size] for i in range(k)]

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Softmax function for output layer
def softmax(x):
    exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

# Multilayer Perceptron implementation
class MLP:
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.layers = [input_size] + hidden_layer_sizes + [output_size]
        self.weights = [np.random.randn(self.layers[i], self.layers[i+1]) * 0.1 for i in range(len(self.layers)-1)]
        self.biases = [np.random.randn(1, self.layers[i+1]) * 0.1 for i in range(len(self.layers)-1)]
    
    def forward(self, X):
        self.a = [X]
        for i in range(len(self.weights)-1):
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            self.a.append(sigmoid(z))
        z = np.dot(self.a[-1], self.weights[-1]) + self.biases[-1]
        self.output = softmax(z)
        return self.output

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

# Genetic Algorithm implementation
class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, generations):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.history = []

    def initialize_population(self, input_size, hidden_layers, output_size):
        return [MLP(input_size, hidden_layers, output_size) for _ in range(self.population_size)]
    
    def crossover(self, parent1, parent2):
        child = MLP(parent1.input_size, parent1.hidden_layer_sizes, parent1.output_size)
        for i in range(len(parent1.weights)):
            mask = np.random.rand(*parent1.weights[i].shape) > 0.5
            child.weights[i] = np.where(mask, parent1.weights[i], parent2.weights[i])
        return child
    
    def mutate(self, mlp):
        for i in range(len(mlp.weights)):
            if np.random.rand() < self.mutation_rate:
                mlp.weights[i] += np.random.randn(*mlp.weights[i].shape) * 0.1
    
    def fitness(self, mlp, X, y):
        predictions = mlp.predict(X)
        return np.mean(predictions == y)
    
    def evolve(self, X_train, y_train, X_val, y_val):
        population = self.initialize_population(X_train.shape[1], [40, 25], 2)
        for generation in range(self.generations):
            population = sorted(population, key=lambda mlp: self.fitness(mlp, X_val, y_val), reverse=True)
            best_fitness = self.fitness(population[0], X_val, y_val)
            self.history.append(best_fitness)
            print(f"Generation {generation+1}/{self.generations}, Best fitness: {best_fitness:.4f}")
            new_population = population[:self.population_size//2]  # Selection
            for _ in range(self.population_size//2):  # Crossover
                parent1, parent2 = random.sample(new_population, 2)
                child = self.crossover(parent1, parent2)
                new_population.append(child)
            population = new_population
            for mlp in population:  # Mutation
                self.mutate(mlp)
        return population[0]
    
    def plot_progress(self):
        plt.plot(self.history)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('GA Optimization Progress')
        plt.show()

# Perform k-fold cross-validation
def k_fold_cross_validation(data, k=10):
    folds = k_fold_split(data, k)
    accuracies = []
    for i in range(k):
        val_data = folds[i]
        train_data = [sample for j in range(k) if j != i for sample in folds[j]]
        X_train, y_train = normalize(train_data)
        X_val, y_val = normalize(val_data)
        
        # Genetic Algorithm with MLP
        ga = GeneticAlgorithm(population_size=20, mutation_rate=0.01, generations=100)
        best_model = ga.evolve(X_train, y_train, X_val, y_val)
        
        # Test on validation set using the GA's fitness function
        accuracy = ga.fitness(best_model, X_val, y_val)
        accuracies.append(accuracy)
    
    avg_accuracy = np.mean(accuracies)
    print("Cross-validation accuracies:", accuracies)
    print("Average accuracy:", avg_accuracy)
    
    return accuracies, avg_accuracy, ga, best_model

# Plot accuracies with average
def plot_accuracies_with_avg(accuracies, avg_accuracy):
    folds = range(1, len(accuracies) + 1)
    plt.bar(folds, accuracies, label='Accuracy per Fold')
    plt.axhline(avg_accuracy, color='r', linestyle='--', label=f'Average Accuracy: {avg_accuracy:.4f}')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('K-Fold Cross-Validation Accuracies with Average')
    plt.legend()
    plt.show()

# Plot train data vs predicted data
def plot_train_vs_predicted(best_model, X_train, y_train):
    predictions = best_model.predict(X_train)
    
    plt.scatter(range(len(y_train)), y_train, color='b', label='Actual')
    plt.scatter(range(len(y_train)), predictions, color='r', alpha=0.5, label='Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Class Label')
    plt.title('Train Data: Actual vs Predicted')
    plt.legend()
    plt.show()


# Function to plot confusion matrix for actual vs predicted data
def plot_confusion_matrix(best_model, X_train, y_train):
    predictions = best_model.predict(X_train)
    
    cm = confusion_matrix(y_train, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])  # Benign (0) and Malignant (1)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix: Actual vs Predicted')
    plt.show()

    
# Main function to execute
if __name__ == '__main__':
    data_path = 'wdbc.data'
    data = load_data(data_path)
    accuracies, avg_accuracy, ga, best_model = k_fold_cross_validation(data)
    
    # Plot GA progress
    ga.plot_progress()

    # Plot accuracies and average accuracy
    plot_accuracies_with_avg(accuracies, avg_accuracy)
    
    # Normalize the training data again to plot actual vs predicted
    X_train, y_train = normalize(data)
    
    # Show confusion matrix
    plot_confusion_matrix(best_model, X_train, y_train)
