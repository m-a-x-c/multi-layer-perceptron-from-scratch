import numpy as np
import copy
import time
import MLMath

class MultiLayerPerceptron:
    def __init__(self, data, metadata):
        self.X = data["inputs"]
        self.y = data["targets"]
        self.layer_config = metadata["layer configuration"]
        self.num_of_layers = len(metadata["layer configuration"])
        self.learning_rate = metadata["learning rate"]
        self.num_of_epochs = metadata["number of epochs"]
        self.act_func = metadata["activation function"]
        self.error_func = metadata["error function"]

        self.weights = []
        self.w_from_in = []
        self.w_from_out = []
        self.w_to_in = []
        self.w_to_out = []
        self.w_target = []

        self.total_num_of_updates = 0

        for i in range(self.num_of_layers - 1):
            n = self.layer_config[i]
            m = self.layer_config[i+1]
            layer = np.random.rand(n, m)
            self.weights.append(layer)

        self.nodes = []
        self.nodes_no_act = []
        for num_of_nodes in self.layer_config:
            self.nodes.append(np.ones((num_of_nodes)))
            self.nodes_no_act.append(np.ones((num_of_nodes)))
        



    def train(self):
        start = time.time()

        print("...TRAINING...\n")
        for n in range(self.num_of_epochs):
            for i in range(len(self.X)):
                self.feedforward(self.X[i])
                self.calc_derivative(self.y[i])
                self.update_weights()
                self.total_num_of_updates += 1

        elapsed = time.time() - start

        for i in range(len(self.X)):
            self.feedforward(self.X[i], with_print=True)
        
        print("Total number of updates:", self.total_num_of_updates)
        print("Total number of epochs:", self.num_of_epochs)
        print("Total time taken:", round(elapsed, 2), "seconds")
        print("------------------------------------\n")

        # print(self.nodes)
        # print("\n")
        # print(self.weights)

    
    def activation_func(self, x):
        if self.act_func == "sigmoid":
            return MLMath.sigmoid(x)
        else:
            return MLMath.sigmoid(x)
    
    def activation_func_derivative(self, x):
        return MLMath.sigmoid_derivative(x)
    
    def error(self, y, y_pred):
        if self.error_func == "mse":
            return MLMath.mse(y, y_pred)
        else:
            return MLMath.mse(y, y_pred)

    def error_derivative(self, y, y_pred):
        return MLMath.mse_derivative(y, y_pred)
    




    def feedforward(self, instance, with_print=False):
        self.nodes_no_act[0] = np.array(instance)
        self.nodes[0] = np.array(instance)
        for i in range(self.num_of_layers - 1):
            val = np.matmul(self.nodes[i], self.weights[i])
            self.nodes_no_act[i+1] = val
            self.nodes[i+1] = self.activation_func(val)
        
        if with_print:
            print("INPUT NODES:", instance)
            print("OUTPUT NODES:", self.nodes[-1])
            print("")





    def create_weights_from_list(self, nodes, layer_config):
        # this function creates a weights array with the value of the
        # node the weight comes from, instead of the value of the weight

        w_from = []

        for i in range(len(layer_config) - 1):
            m = layer_config[i] # number of nodes in current layer
            n = layer_config[i+1] # number of nodes in next layer

            new_layer = np.repeat(nodes[i], n).reshape(m, n)
            # new_layer = new_layer.astype('float16')
            w_from.append(new_layer)
        
        return w_from

    def create_weights_to_list(self, nodes, layer_config):
        # this function creates a weights array with the value of the
        # node the weight goes to, instead of the value of the weight

        w_to = []

        for i in range(1, len(layer_config)):
            n = layer_config[i-1] # number of nodes in previous layer

            new_layer = np.tile( nodes[i], (n, 1) )
            # new_layer = new_layer.astype('float16')
            w_to.append(new_layer)
        
        return w_to

    def create_weights_target_list(self, target, layer_config):
        # target = [y_output_node_1, y_output_node_2, ..., y_output_node_n]
        # e.g. target = [0,1,0,1]
        # note target is the output values for a single instance only

        w_target = np.tile( np.array(target), (layer_config[-2], 1) )
        # w_target = w_target.astype('float16')

        return w_target
    

    def update_weights_lists(self, instance_y):
        self.w_from_in = self.create_weights_from_list(self.nodes_no_act, self.layer_config)
        self.w_from_out = self.create_weights_from_list(self.nodes, self.layer_config)
        self.w_to_in = self.create_weights_to_list(self.nodes_no_act, self.layer_config)
        self.w_to_out = self.create_weights_to_list(self.nodes, self.layer_config)
        self.w_target = self.create_weights_target_list(instance_y, self.layer_config)


        self.w_to_in_act_deriv = []
        for layer in self.w_to_out:
            self.w_to_in_act_deriv.append(np.multiply(np.subtract(1, layer), layer))


    def terminal_layer_derivative(self):
        dE_dOuty = self.error_derivative(self.w_target, self.w_to_out[-1])
        dOuty_dIny = self.w_to_in_act_deriv[-1]
        dIny_dW = self.w_from_out[-1]

        derivative = MLMath.hadamard_product([dE_dOuty, dOuty_dIny, dIny_dW])
        
        for_next_layer = MLMath.hadamard_product([dE_dOuty, dOuty_dIny, self.weights[-1]])
        for_next_layer = np.sum(for_next_layer, axis=1)

        return derivative, for_next_layer
    
    def non_terminal_layer_derivative(self, layer_index, prev):
        m = self.layer_config[layer_index]
        n = self.layer_config[layer_index+1]
        dE_dOuty = np.tile( prev, (m, 1) )

        
        dOuty_dIny = self.w_to_in_act_deriv[layer_index]
        dIny_dW = self.w_from_out[layer_index]
        derivative = MLMath.hadamard_product([dE_dOuty, dOuty_dIny, dIny_dW])


        for_next_layer = MLMath.hadamard_product([dE_dOuty, dOuty_dIny, self.weights[layer_index]])
        for_next_layer = np.sum(for_next_layer, axis=1)

        return derivative, for_next_layer

    def calc_derivative(self, instance_y):
        self.update_weights_lists(instance_y)

        self.derivatives = []
        prev = None
        for i in reversed(range(len(self.weights))):
            if i == (len(self.weights) - 1):
                d, prev = self.terminal_layer_derivative()
            else:
                d, prev = self.non_terminal_layer_derivative(i, prev)
            
            self.derivatives.insert(0,d)
            
        # print(self.nodes)
        # print("\n")
        # print(self.weights)
        # print("\n")
        # print(self.derivatives)
    
    def update_weights(self):
        for i in range(len(self.weights)):
            self.weights[i] = np.subtract(self.weights[i], (self.learning_rate * self.derivatives[i]))

        



        

        
if __name__ == "__main__":
    X = [
        [1,1],
        [0,1],
        [1,0],
        [0,0],
    ]

    y = [
        [0],
        [1],
        [1],
        [0]
    ]

    data = {
        "inputs" : X,
        "targets" : y
    }

    metadata = {
        "layer configuration" : [2, 2, 1],
        "learning rate" : 0.1,
        "number of epochs" : 100000,
        "batch size" : 1,
        "activation function" : "sigmoid",
        "optimizer" : "adam",
        "error function" : "mse",
    }

    mlp = MultiLayerPerceptron(data, metadata)
    mlp.train()