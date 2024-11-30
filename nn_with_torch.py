import tkinter as tk, tkinter.simpledialog as simpledialog
from tkinter import messagebox
from PIL import ImageGrab
import numpy as np
import torch, torch.nn as nn
from typing import Callable
import torch.optim as optim


class DigitRecognizerApp:
    def __init__(self, model: 'NeuralNetwork'):
        self.model = model
        self.window = tk.Tk()
        self.window.title("Digit Recognizer")

        self.canvas = tk.Canvas(self.window, width=280, height=280, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)

        self.predict_button = tk.Button(self.window, text="Predict", command=self.predict)
        self.predict_button.pack()

        self.clear_button = tk.Button(self.window, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.training_data = []

    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="black", outline="black")

    def clear_canvas(self):
        self.canvas.delete("all")

    def predict(self):
        img_tensor = self.process_canvas()
        prediction = self.get_prediction(img_tensor)
        messagebox.showinfo("Prediction", f"The predicted digit is: {prediction}")
        self.prompt_for_true_answer(prediction, img_tensor)

    def process_canvas(self):
        x = self.window.winfo_rootx() + self.canvas.winfo_x()
        y = self.window.winfo_rooty() + self.canvas.winfo_y()
        img = ImageGrab.grab((x, y, x+280, y+280)).convert("L")
        img = img.resize((28, 28)) 
        img_array = np.array(img) / 255.0 
        return torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).view(-1, 784)

    def get_prediction(self, img_tensor):
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)
            return output.argmax(dim=1).item()

    def prompt_for_true_answer(self, predicted_digit, img_tensor):
        answer = simpledialog.askstring("True Answer", f"Enter the true answer (predicted: {predicted_digit}):")
        if answer is None or not answer.isdigit() or not (0 <= int(answer) <= 9):
            messagebox.showerror("Invalid Input", "Please enter a valid digit between 0 and 9.")
            return
        
        true_answer = int(answer)
        self.training_data.append((img_tensor, true_answer))
        self.retrain_model()

    def retrain_model(self):
        if not self.training_data:
            return

        images, labels = zip(*self.training_data)
        images = torch.cat(images)
        labels = torch.tensor(labels)
        
        self.model.train_model(images, labels, epochs=5, learning_rate=0.001)
        self.training_data.clear()

        messagebox.showinfo("Model Retrained", "Model has been retrained with new data.")

    def run(self):
        self.window.mainloop()

class NeuralNetwork(nn.Module):
    def __init__(self, *layers, activations: dict[int, Callable] | None = None):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()

        activations = activations if activations else {}

        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                act_func = activations.get(i, nn.ReLU())
                self.layers.append(act_func)
        
        self.network = nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.network(x)
    
    def train_model(self, x_train, y_train, epochs, learning_rate):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()  # Choose appropriate loss function
        
        for epoch in range(epochs):
            self.train()  # Set model to training mode
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self(x_train)
            loss = criterion(outputs, y_train)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
if __name__ == "__main__":
    model = NeuralNetwork(784, 128, 64, 10)  # For digit recognition
    app = DigitRecognizerApp(model)
    app.run()