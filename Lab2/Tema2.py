import pickle, gzip, random
import numpy as np
from numpy.core.fromnumeric import reshape


class Tema2:

    def __init__(self):
        self.read_data()
        self.weights = np.random.uniform(-0.2, 0.2, (10, 784))
        self.bias = np.array([1] * 10, dtype=float).reshape(10, 1)
        self.learn_rate = 0.001

    def read_data(self, filename = 'mnist.pkl.gz'):
        with gzip.open(filename, 'rb') as file:
            self.training_set, self.validation_set, self.testing_set = pickle.load(file, encoding='latin')

    def activation(self, input) -> int:
        if input > 0:
            return 1
        return 0

    def learn(self):
        iterations = 100
        allClassified = False
        print ("Learning: ")
        while not allClassified and iterations > 0:
            allClassified = True
            accurate = 0

            # Facem shuffle la training set
            indices = list(range(0, len(self.training_set[0])))
            random.shuffle(indices)
            for i in range(0, len(self.training_set[0])):
                # Facem un array de 10 elemente dupa label
                target = int(self.training_set[1][indices[i]])
                target_list = [0] * 10
                target_list[target] = 1
                target_list = np.array(target_list).reshape(1, 10)

                # Matricea de pixeli
                xi = self.training_set[0][indices[i]]
                xi = xi.reshape(1, 784)

                # Calculam output-ul, dupa ce facem produsul dintre greutati si matrice + bias la final
                results = np.inner(xi, self.weights) + self.bias
               
                output = list(list(map(self.activation, results[0])))

                # Actualizam variabilele
                self.weights = self.weights + np.array((target_list - np.array(output))).reshape(10, 1) * xi + self.learn_rate
                self.bias = self.bias + np.array((target_list - np.array(output))) * self.learn_rate

                # Comparam target-list-ul cu output-ul
                # Daca sunt egale => am antrenat bine
                if target_list[0].tolist() == output:
                    accurate += 1
                    allClassified = False

            iterations -= 1
            print (str(iterations) + " -> " + str(accurate/len(self.training_set[0])))

    def test(self):
        accurate = 0
        print ("Testing: ")
        for i in range(0, len(self.testing_set[0])):
            target = int(self.testing_set[1][i])
            target_list = [0] * 10
            target_list[target] = 1
            target_list = np.array(target_list).reshape(1, 10)
            xi = self.testing_set[0][i]
            xi = xi.reshape(1, 784)
            results = (np.inner(xi, self.weights) + self.bias).tolist()[0]
            result = results.index(max(results))
            if target == result:
                accurate += 1

            print (str(accurate / len(self.testing_set[0])))


grupPerceptroni = Tema2()
grupPerceptroni.learn()
grupPerceptroni.test()