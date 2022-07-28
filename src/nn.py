import numpy as np

from .base import BaseMLP

class NeuralNetworkClassifier(BaseMLP):

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        # Data properties
        N_OBS = x.shape[0]
        INPUT_DIM = x.shape[1]
        NUM_CLASSES = len(np.unique(y))

        # Initialize random weights
        self.W1 = 0.01 * np.random.randn(INPUT_DIM, self.hidden_layer_sizes)
        self.b1 = np.zeros((1, self.hidden_layer_sizes))
        self.W2 = 0.01 * np.random.randn(self.hidden_layer_sizes, NUM_CLASSES)
        self.b2 = np.zeros((1, NUM_CLASSES))

        # Initialize random number generator
        SEED = None if self.random_state is None else self.random_state
        rng = np.random.default_rng(SEED)

        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be greater than zero.")

        self.momentum = np.array(self.momentum)
        if np.any(self.momentum < 0):
            raise ValueError("Momentum must be between zero and one")

        diff = 0

        # Training loop
        for iter in range(self.max_iter):
            z1 = np.dot(x, self.W1) + self.b1
            a1 = np.maximum(0, z1)

            logits = np.dot(a1, self.W2) + self.b2

            # Normalize to obtain class probabilities
            exp_logits = np.exp(logits)
            y_hat = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Compute loss
            class_probs = -np.log(y_hat[range(len(y_hat)), y])
            loss = np.sum(class_probs) / len(y)

            # Progress
            if iter % 100 == 0:
                y_pred = np.argmax(logits, axis=1)
                accuracy =  np.mean(np.equal(y, y_pred))
                print(f"[INFO] Iteration: {iter}, loss: {loss}, accuracy: {accuracy}")

            # Compute gradients
            dscores = y_hat
            dscores[range(len(y_hat)), y] -= 1
            dscores /= len(y)
            dW2 = np.dot(a1.T, dscores)
            db2 = np.sum(dscores, axis=0, keepdims=True)

            dhidden = np.dot(dscores, self.W2.T)
            dhidden[a1 <= 0] = 0
            dW1 = np.dot(x.T, dhidden)
            db1 = np.sum(dhidden, axis=0, keepdims=True)

            diff_dW2 = self.momentum*diff - self.learning_rate*dW2
            diff_dW1 = self.momentum*diff - self.learning_rate*dW1

            # Update weights
            # self.W1 += -self.learning_rate * dW1
            self.W1 += diff_dW1
            self.b1 += -self.learning_rate * db1
            # self.W2 += -self.learning_rate * dW2
            self.W2 += diff_dW2
            self.b2 += -self.learning_rate * db2
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        z1 = np.dot(x, self.W1) + self.b1
        a1 = np.maximum(0, z1)
        logits = np.dot(a1, self.W2) + self.b2
        exp_logits = np.exp(logits)
        y_hat = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        return y_hat
