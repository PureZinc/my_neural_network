import numpy as np
from dataclasses import dataclass
from typing import Callable
import pickle


class NeuronLayer:
    _activations = {
        'sigmoid': lambda Z: 1 / (1 + np.exp(-Z)),
        'relu': lambda Z: np.maximum(0, Z),
        'tanh': lambda Z: np.tanh(Z)
    }

    _activations_derivatives = {
        'sigmoid': lambda Z: (1 / (1 + np.exp(-Z))) * (1 - (1 / (1 + np.exp(-Z)))),
        'relu': lambda Z: (Z > 0).astype(float),
        'tanh': lambda Z: 1 - np.tanh(Z)**2
    }

    def __init__(self, input: int, output: int, activation='sigmoid'):
        self.input_size = input
        self.output_size = output

        if activation == 'relu':
            self.W = np.random.randn(output, input) * np.sqrt(2 / input)
        elif activation in ['sigmoid', 'tanh']:
            self.W = np.random.randn(output, input) * np.sqrt(1 / input)
        else:
            self.W = np.random.randn(output, input) * 0.01

        self.b = np.zeros((output, 1))
        self.Z = None
        self.A = None
        self.set_activation(activation)
    
    @classmethod
    def get_activations(cls):
        return cls._activations
    
    @classmethod
    def get_activations_derivatives(cls):
        return cls._activations_derivatives
    
    def set_activation(self, activation: str) -> None:
        if activation not in self._activations.keys():
            raise ValueError(f"{activation} is not a valid activation value!")
        self.activation = activation
    
    def get_act_func(self, derivative=False) -> Callable[[np.ndarray], None]:
        act = self._activations_derivatives if derivative else self._activations
        act_func = act.get(self.activation, None)
        if act_func is None:
            raise ValueError(f"{self.activation} is NOT an activation function!")
        return act_func

    def activate(self, Z):
        return self.get_act_func()(Z)
    
    def activate_derivative(self, Z):
        return self.get_act_func(derivative=True)(Z) 


class NeuralNetwork:
    def __init__(self, *layers: list[int], activations: dict[int, str] | None) -> None:
        length = len(layers)
        if length < 1:
            raise ValueError("At least one layer is required.")
        
        if activations is None:
            activations = {}
        
        self.layers: list[NeuronLayer] = []
        for i in range(length - 1):
            activation = activations.get(i, 'sigmoid')
            self.layers.append(NeuronLayer(layers[i], layers[i + 1], activation=activation))

    def compute_cost(self, AL, Y: np.ndarray):
        m = Y.shape[1]
        AL = np.clip(AL, 1e-10, 1 - 1e-10)
        cost = -np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)) / m
        return np.squeeze(cost)
    
    def forward_propagation(self, X):
        A = X
        cache = {'A0': A}
        for i, layer in enumerate(self.layers, start=1):
            layer.Z = np.dot(layer.W, A) + layer.b
            A = layer.activate(layer.Z)
            layer.A = A
            cache[f'Z{i}'] = layer.Z
            cache[f'A{i}'] = A
        return A, cache
    
    def backward_propagation(self, cache: dict, X: np.ndarray, Y: np.ndarray) -> dict:
        gradients = {}
        L = len(self.layers)
        m = X.shape[1]
        AL = cache[f'A{L}']

        # Gradient of loss with respect to AL (Cross-Entropy Loss)
        dA = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        for l in reversed(range(1, L + 1)):
            Z = cache[f'Z{l}']
            A_prev = cache[f'A{l - 1}'] if l > 1 else X

            # Activation derivative
            activation = self.layers[l - 1].activate_derivative(Z)
            dZ = dA * activation

            # Gradients
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            gradients[f'dW{l}'] = dW
            gradients[f'db{l}'] = db

            if l > 1:
                W = self.layers[l - 1].W
                dA = np.dot(W.T, dZ)

        return gradients
    
    def update_parameters(self, gradients, learning_rate):
        for l, layer in enumerate(self.layers, start=1):
            layer.W -= learning_rate * gradients[f'dW{l}']
            layer.b -= learning_rate * gradients[f'db{l}']
    
    def train(self, X, Y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            AL, cache = self.forward_propagation(X)
            cost = self.compute_cost(AL, Y)
            gradients = self.backward_propagation(cache, X, Y)
            self.update_parameters(gradients, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.4f}")

    def save(self, filename: str) -> None:
        model_data = {
            'architecture': [(layer.input_size, layer.output_size, layer.activation) for layer in self.layers],
            'parameters': [{'W': layer.W, 'b': layer.b} for layer in self.layers]
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename: str) -> 'NeuralNetwork':
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        architecture = model_data['architecture']
        parameters = model_data['parameters']
        
        # Create a new neural network with the saved architecture
        activations = {i: layer[2] for i, layer in enumerate(architecture)}
        nn = cls(*[layer[0] for layer in architecture], activations=activations)
        
        # Load saved parameters
        for layer, param in zip(nn.layers, parameters):
            layer.W = param['W']
            layer.b = param['b']
        
        print(f"Model loaded from {filename}")
        return nn


#Testing
if __name__ == "__main__":
    import os
    load_choice = input("Load saved NN? (1: YES | 0: NO): ")
    if load_choice == "1" and os.path.exists("saved.pkl"):
        nn = NeuralNetwork.load("saved.pk1")
    else:
        nn = NeuralNetwork(3, 4, 1, activations={0: 'relu', 1: 'sigmoid'})
        print("Initialized a new neural network.")

        # Create training data
        np.random.seed(0)
        X = np.random.randn(3, 100)  # 3 features, 100 examples
        Y = (np.random.rand(1, 100) > 0.5).astype(int)  # Binary labels

        # Train the network
        nn.train(X, Y, epochs=1000, learning_rate=0.01)

    # Test the network
    X_test = np.random.randn(3, 5)  # 5 new examples
    predictions, _ = nn.forward_propagation(X_test)
    predicted_classes = (predictions > 0.5).astype(int)
    print(f"Predictions: {predicted_classes}")

    # User choice to save the model
    save_choice = input("Save NN parameters? (1: YES | 0: NO): ")
    if save_choice == "1":
        nn.save("saved.pk1")
