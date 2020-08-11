from keras.layers import Layer, Dense
import keras.backend as K
import copy

class ReuseLayer(Layer):
    """定义新的Layer，增加reuse方法，允许在定义Layer时调用现成的层"""

    def reuse(self, layer, *args, **kwargs):
        if not layer.built:
            if len(args) > 0:
                inputs = args[0]
            else:
                inputs = kwargs['inputs']
            if isinstance(inputs, list):
                input_shape = [K.int_shape(x) for x in inputs]
            else:
                input_shape = K.int_shape(inputs)
            layer.build(input_shape)
        outputs = layer.call(*args, **kwargs)
        for w in layer.trainable_weights:
            if w not in self._trainable_weights:
                self._trainable_weights.append(w)
        for w in layer.non_trainable_weights:
            if w not in self._non_trainable_weights:
                self._non_trainable_weights.append(w)
        return outputs


class CoustomDense(ReuseLayer):
    """原来是继承Layer类，现在继承OurLayer类"""
    def __init__(self, hidden_dim, output_dim, hidden_act='linear', output_act='linear', **kwargs):
        super(CoustomDense, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_activation = hidden_act
        self.output_activation = output_act

    def build(self, input_shape):
        """在build方法里边添加需要重用的层，当然也可以像标准写法一样条件可训练的权重。"""
        super(CoustomDense, self).build(input_shape)
        self.h_dense = Dense(self.hidden_dim, activation=self.hidden_activation)
        self.o_dense = Dense(self.output_dim, activation=self.output_activation)

    def call(self, inputs):
        h = self.reuse(self.h_dense, inputs)
        o = self.reuse(self.o_dense, h)
        return o
    
    def compute_output_shape(self, input_shape):
        return input_shape[:-1]+(self.output_dim)


class OurBidirectional(ReuseLayer):
    """自己封装双向RNN，允许传入mask，保证对齐
    """
    def __init__(self, layer, **args):
        super(OurBidirectional, self).__init__(**args)
        self.forward_layer = copy.deepcopy(layer)
        self.backward_layer = copy.deepcopy(layer)
        self.forward_layer.name = 'forward_' + self.forward_layer.name
        self.backward_layer.name = 'backward_' + self.backward_layer.name
    
    def reverse_sequence(self, x, mask):
        """这里的mask.shape是[batch_size, seq_len, 1]
        """
        seq_len = K.round(K.sum(mask, 1)[:, 0])
        seq_len = K.cast(seq_len, 'int32')
        return K.tf.reverse_sequence(x, seq_len, seq_dim=1)
    
    def call(self, inputs):
        x, mask = inputs
        x_forward = self.reuse(self.forward_layer, x)
        x_backward = self.reverse_sequence(x, mask)
        x_backward = self.reuse(self.backward_layer, x_backward)
        x_backward = self.reverse_sequence(x_backward, mask)
        x = K.concatenate([x_forward, x_backward], 2)
        return x * mask

    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1], self.forward_layer.units * 2)