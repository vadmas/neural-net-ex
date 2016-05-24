from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# Static configuration variables 
NN_INPUT_DIM = 2
NN_OUTPUT_DIM = 2
EPSILON = 0.01
REG_LAMBDA = 0.01


def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y


def predict(model, x):
    W1, b1, W2, b2 = model["W1"], model["b1"], model["W2"], model["b2"]
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2 
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


def visualize(X, y, model, title):
    plot_decision_boundary(lambda x: predict(model, x), X, y)
    plt.title(title)
    plt.show()


def forward(W1, b1, W2, b2, x):
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    y_hat = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return y_hat, z1, a1, z2


# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    num_examples = len(X)
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)

    W1 = np.random.randn(NN_INPUT_DIM, nn_hdim) / np.sqrt(NN_INPUT_DIM)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, NN_OUTPUT_DIM) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, NN_OUTPUT_DIM))

    # The model we'll return at the end 
    model = {}

    # Gradient descent. For each batch - 
    for i in xrange(0, num_passes):
        # Forward prop
        probs, z1, a1, z2 = forward(W1, b1, W2, b2, X)

        # Backprop
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms 
        dW2 += REG_LAMBDA * W2
        dW1 += REG_LAMBDA * W1

        W1 += -EPSILON * dW1 
        b1 += -EPSILON * db1 
        W2 += -EPSILON * dW2 
        b2 += -EPSILON * db2 

        # Assign new parameters to the model
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
            print "Loss after iteration %i: %f" % (i, calculate_loss(model, X, y))

    return model


def calculate_loss(model, X, y):
    num_examples = len(X)
    W1, b1, W2, b2 = model["W1"], model["b1"], model["W2"], model["b2"]
    # Forward propagation to calculate our predictions
    y_hat, z1, a1, z2 = forward(W1, b1, W2, b2, X)
    # Calculate the loss
    correct_logprobs = -np.log(y_hat[range(num_examples), y])
    data_loss = np.sum(correct_logprobs)
    # Add regularization term to loss 
    data_loss += REG_LAMBDA / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1. / num_examples * data_loss


def main():
    X, y = generate_data()
    model = build_model(X, y, 3, print_loss=True)
    visualize(X, y, model, "NN: Hidden layer size = 3")

if __name__ == '__main__':
    main()
