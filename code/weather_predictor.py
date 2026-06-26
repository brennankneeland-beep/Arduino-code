import numpy as np
import matplotlib.pyplot as plt
import pickle

from neuralnetworkBIG import (
    Activation_Softmax_Loss_CategoricalCrossentropy,
    Layer_Dense,
    Activation_ReLU,
    Optimizer_SGD
)

# load data
with open("weather_inputs.pkl", "rb") as f:
    X = pickle.load(f)

with open("weather_labels.pkl", "rb") as f:
    y = pickle.load(f)

# load scaler (IMPORTANT)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

X = (X - scaler["mean"]) / scaler["std"]

# convert labels once
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)

# model
Dense1 = Layer_Dense(12, 64)
activation1 = Activation_ReLU()

Dense2 = Layer_Dense(64, 64)
activation2 = Activation_ReLU()

Dense3 = Layer_Dense(64, 2)

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Optimizer_SGD(learning_rate= 0.005, decay=1e-4)

epochs = []
accuracies = []
losses = []
split = int(len(X) * 0.8)

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

for epoch in range(10001):

    for start in range(0, len(X_train), 32):
        X_batch = X_train[start:start+32]
        y_batch = y_train[start:start+32]

        Dense1.forward(X_batch)
        activation1.forward(Dense1.output)

        Dense2.forward(activation1.output)
        activation2.forward(Dense2.output)

        Dense3.forward(activation2.output)

        loss = loss_activation.forward(Dense3.output, y_batch)

        loss_activation.backward(loss_activation.output, y_batch)

        Dense3.backward(loss_activation.dinputs)
        activation2.backward(Dense3.dinputs)
        Dense2.backward(activation2.dinputs)
        activation1.backward(Dense2.dinputs)
        Dense1.backward(activation1.dinputs)

        optimizer.pre_update_params()
        optimizer.update_params(Dense1)
        optimizer.update_params(Dense2)
        optimizer.update_params(Dense3)
        optimizer.post_update_params()

    # TEST EVALUATION
    Dense1.forward(X_test)
    activation1.forward(Dense1.output)

    Dense2.forward(activation1.output)
    activation2.forward(Dense2.output)

    Dense3.forward(activation2.output)

    loss = loss_activation.forward(Dense3.output, y_test)

    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y_test)
    rain_percent = np.mean(predictions == 0) * 100
    sunny_percent = np.mean(predictions == 1) * 100

    if epoch % 10 == 0:
        print(f"epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, sun:{sunny_percent:.3f}")
        epochs.append(epoch)
        accuracies.append(accuracy)
        losses.append(loss)
# plots
plt.plot(epochs, accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epoch')
plt.grid(True)
plt.show()

plt.plot(epochs, losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.grid(True)
plt.show()