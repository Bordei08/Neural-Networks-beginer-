import pickle, gzip, random
import numpy as np


def read_data(filename):
        with gzip.open(filename,'rb') as file:
            return pickle.load(file, encoding='latin')    


# Functia de softmax, preluata din curs
def __softmax(z):
    return np.exp(z) / (np.sum(np.exp(z)))
    

# Functie de sigmoid, preluata din curs
def __sigmoid(z):
    return 1 /( 1 + np.exp(-z))

def train():            
    training_set, validation_set,testing_set = read_data('mnist.pkl.gz')
    first_weights = (1/(np.sqrt(784)))*np.random.randn(100,784)
    first_biases = (1/(np.sqrt(784)))*np.random.randn(100,1)
    second_weights = (1/(np.sqrt(100)))*np.random.randn(10,100)
    second_biases = (1/(np.sqrt(100)))*np.random.randn(10,1)
    learning_rate = 0.039
    allClassified = False
    iterations = 5
    while not allClassified and iterations > 0 :
            accuracy = 0
            indices = list(range(0, len(training_set[0])))
            random.shuffle(indices)
            n = len(training_set[0])

            for index in indices:
                xi = training_set[0][index].reshape(784,1)

                first_z = np.dot(first_weights, xi) + first_biases
                first_y = __sigmoid(first_z)

                second_z = np.dot(second_weights, first_y) + second_biases
                second_y = __softmax(second_z)

                target = int(training_set[1][index])
                target_list = np.zeros((10,1))
                target_list[target] = 1

                last_error = (second_y-target_list)
                
                second_delta_bias = last_error
                second_delta_weight = last_error * first_y.reshape(1,100)
                
                first_error = first_y*(1-first_y)*np.dot(last_error.reshape(1,10),second_weights.reshape(10,100)).reshape(100,1)
                first_delta_bias = first_error
                first_delta_weight = first_error*xi.reshape(1,784)

                second_weights -= (learning_rate*second_delta_weight)
                second_biases -= (learning_rate*second_delta_bias)

                first_weights -= (learning_rate*first_delta_weight)
                first_biases -= (learning_rate*first_delta_bias)

                second_y = second_y.reshape(1,10)

                if np.argmax(second_y) == target:
                    accuracy +=1

            iterations -= 1
            print(f"{iterations}. {accuracy*100/n}%")

    return first_weights,first_biases,second_weights,second_biases


def test():
    training_set, validation_set,testing_set = read_data('mnist.pkl.gz')
    first_weights,first_biases,second_weights,second_biases = train()
    accuracy = 0
    indices = list(range(0, len(testing_set[0])))
    random.shuffle(indices)

    n = len(testing_set[0])

    for index in indices:
        xi = testing_set[0][index].reshape(784,1)

        first_z = np.dot(first_weights, xi) + first_biases
        first_y = __sigmoid(first_z)

        second_z = np.dot(second_weights, first_y) + second_biases
        second_y = __softmax(second_z)

        target = int(testing_set[1][index])
        second_y = second_y.reshape(1,10)
        if np.argmax(second_y) == target:
            accuracy +=1
        
    print(f"Accuracy : {accuracy*100/n}")

  
test()



