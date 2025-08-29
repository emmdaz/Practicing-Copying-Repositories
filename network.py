
#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes): # El self se utiliza ya que se está haciendo 
                               # alusión al propio objeto que se está creando
        """
        Tal como se mencionaba en el código original, con la función iniciadora
        __init__ se configura la red neuronal. 
        La lista sizes define el número de capas y número de neuronas en cada
        capa; la primer componente de la capa corresponde al número de entradas
        de la red, en este caso, la "capa" inicial tiene un total de 28*28=784 
        entradas, correspondientes al número total de píxeles de cada imagen; 
        por otro lado, las capas intermedias correspoden a las "capas ocultas" 
        de la red que en este caso se decidió fuera una sola con un total de 30
        neuronas; finalmente, la capa de salida, como su nombre indica, 
        corresponde al número de outputs de la red, en este caso, al tratarse 
        de una red cuyo fin es clasificar números entre el 0 y 9 hace que se 
        tengan 10 neuronas de salida. 
        Ahora, en esta función también se inicializan los pesos y los bias de
        forma aleatoria, haciendo uso de una distribución normal estándar de 
        media 0 y varianza 1.""" 
        self.num_layers = len(sizes) # El número de capas corresponde a la
                                     # dimensión (tamaño) de la lista
        self.sizes = sizes 
        # Aquí se inicializan los pesos y los bias
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        """
        Se crea un lista de arreglos n- dimensionales de numpy que tienen los 
        bias inicializados.
        Esto es, a partir del tamaño de las capas ocultas y la capa de salida, 
        acorde el tamaño de cada una de estas, se les genera un valor de una 
        distribución normal estándar. n corresponde al tamaño de cada capa, 
        inlcuyendo a la de salida.
        Por decir, se tiene una capa oculta de 15 neuronas y una capa de salida
        de 7, entonces se genera una lista con dos arreglos de numpy con 15 y 7 
        valores de bias respectivamente.
        """
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])] 
        """ 
        Se genera una lista con arreglos de numpy: Esta capa contendrá tantos 
        arreglos de numpy como capas ocultas tenga + 1 (correspondiente a las
        entradas), cada uno n dimensional con k datos aleatorios de una 
        distribución normal estándar; con n siendo el tamaño de cada capa
        oculta y la capa de salida y k siendo el tamaño de la capa de entrada
        y cada capa oculta.
        Por ej: Si se tiene una red con 2 entradas, 2 capas ocultas, una con
        3 neuronas y otra con 4; y una capa de salida de 5 neuronas, se tiene
        entonces una lista con 3 arreglos de numpy, el primero con 15 valores
        generados, el segundo con 20 y el tercero con 4.     
        """                      

    def feedforward(self, a):
        """
        Esta es una función que será la función de activación del perceptrón
        sigmoidal. Se realiza un produto punto entre la entrada "a" y los pesos
        "w". Regresa el resultado de esta operación.
        """
        for b, w in zip(self.biases, self.weights): # Se hace un ciclo que une 
                                                    # las listas generadas en 
                                                    # la función anterior para 
                                                    # los pesos y los bias.
            a = sigmoid(np.dot(w, a)+b) 
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """
        Se crea una función que corresponderá al algoritmo de Stochastic 
        Gradient Descent (SDG) con el que se buscará modificar los pesos y los
        bias, con el objeto de que las salidas producidas por la red se 
        aproximen a los valores esperados (los correctos). 
        Los datos de entrada para esta función corresponden al mismo objeto, 
        a los datos de entrada, se definen las épocas, el tamaño de los mini
        batches y el valor de eta, el learning rate. 
        En esta parte se genera primeramente una lista que contendrá al
        conjunto de datos de entrenamiento y se define a "n" como el tamaño de
        esta. 
        Además, en el caso de que en la base de datos también existan datos de 
        prueba, se generará una lista con ellos  y se definirá el tamaño de 
        esta como "n_test". Con esto, en el caso de que se tenga este conjunto 
        al terminar una época se realizará una evalucación de la red con la que
        se pretende mostrar el progreso partial que se tiene después de cada 
        una. 
        """

        training_data = list(training_data) # Definición de la lista con datos
                                            # de entrenamiento
        n = len(training_data) # Tamaño del conjunto de datos de entrenamiento

        if test_data: # En el caso de que se tengan datos de prueba:
            test_data = list(test_data) # Se genera la lsita con estos datos
            n_test = len(test_data)     # Se define el tamaño de esta lista

        for j in range(epochs):
            """
        Acorde al número de épocas se repetirá este proceso:
        Se van a "sortear" los datos de entrenamiento de modo que al generar
        los mini batches, se tengan en cada uno de ellos un conjunto distinto.
        El número de batches depende del tamaño del mini batch, de cuántos
        datos tenga, por ello mismo es importante que se tenga en cuenta esto o 
        de lo contrario pasaría un error pues en si el tamaño del conjunto de
        entrenamiento no es divisible con residuo cero del tamaño del mini 
        batch, podría haber uno que tenga menos datos. 
        En esta  función se actualizan los pesos y bias de la red, de modo que 
        se consiga minimizar la función de costo con el SGD usando la función 
        "update_mini_batch". 
        También, en el caso de que se tengan datos de prueba, al final de cada 
        época, se evaluará el desempeño de la red.
            """
            random.shuffle(training_data) # Se sortean aleatoriamente los datos
                                          # de entrenamiento.
            mini_batches = [ # Se generan los mini batches.
                training_data[k:k+mini_batch_size] # Se toman tantos datos del
                                                   # conjunto de entrenamiento
                                                   # como sean necesarios para
                                                   # crear el mini batch en
                                                   # cada época.
                for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches: # Para cada mini batch
                self.update_mini_batch(mini_batch, eta) # Se actualizan los
                                                        # valores de los pesos
                                                        # y los bias acorde al 
                                                        # learning rate, usando
                                                        # SDG
            
            if test_data: # Si hay datos de prueba:
                # Muestra el desempeño de la red al evaluarla
                print("Epoch {} : {} / {}".format(j+1,self.evaluate(test_data),n_test))
            else:
                # Se muestra que terminó el entrenamiento
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """
        Se actualizan los pesos y bias de los mini batches después de aplicar
        SGD. Se genera una lista con valores para el gradiente de los bias y 
        los pesos. Ahora, para los valores correspondientes de la lista 
        "mini_batch" se generan los valores para el cambio de los valores antes
        mencionados, con ayuda del algoritmo de back para luego modificar los
        valores tanto del gradiente de los bias y de los pesos. Luego acorde a 
        esto, se actualizan los valores de los pesos y de los bias, de modo que 
        estos hagan que el valor de la función de costo vaya disminuyendo. 
        Aclarar también que "eta" corresponde al "learning rate". 
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Se genera una tupla de datos con los  gradientes de los bias y de los
        pesos que representan el valor del gradiente de cada función de costo
        C_x. 
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # feedforward
        activation = x # Corresponde a un valor de activación de una neurona
        activations = [x] # Aquí se hace una lista para almacenar todas las 
                          # activaciones, de cada capa
        zs = [] # Se crea una lista para almacenar todos los valores de los 
                # vectores, de cada capa 
                
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Esta es una función para evaluar a la red neuronal, dando la cantidad 
        de entradas que corresponden de forma correctlaa a las salidas que dio 
        la red neuronal; siendo las salidas el argumento más de activación de 
        lista entre los valores posibles de la capa final de salidas. 
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Función que define elvector de derivadas parciales de la función de
        costo C_x con respecto a "a" para las activaciones de salida. 
        """
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    # Función sigmoide
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    # Derivada de la función sigmoide
    return sigmoid(z)*(1-sigmoid(z))
