import numpy as np
import tensorflow as tf
from tqdm import tqdm


def load_yolov3_weights(global_vars, file_path):
    """Load YOLOv3 weights.

    :param global_vars: The global variables of the currrent TensorFlow session.
    :param file_path: The path to the weights file.
    """

    pbar = tqdm(unit='layers', total=366)
    with open(file_path, 'rb') as f:
        # First 5 values are not needed
        _ = np.fromfile(f, dtype=np.int32, count=5)
        weights = np.fromfile(f, dtype=np.float32)

        var_iter = 0
        weight_iter = 0
        assign_ops = []

        while var_iter < len(global_vars) - 1:
            var1 = global_vars[var_iter]
            var2 = global_vars[var_iter + 1]

            # Convolution layer
            if 'Conv' in var1.name.split('/')[-2]:
                # Check next layer's type
                if 'BatchNorm' in var2.name.split('/')[-2]:
                    gamma, beta, mean, variance = global_vars[var_iter + 1:var_iter + 5]
                    batch_norm_vars = [beta, gamma, mean, variance]
                    for var in batch_norm_vars:
                        shape = var.shape.as_list()
                        num_params = np.prod(shape)
                        var_weights = weights[weight_iter:weight_iter + num_params].reshape(shape)
                        weight_iter += num_params
                        assign_ops.append(tf.assign(var, var_weights, validate_shape=True))

                    # Move up 4 variables
                    var_iter += 4
                    pbar.update(4)
                elif 'Conv' in var2.name.split('/')[-2]:
                    bias = var2
                    shape = bias.shape.as_list()
                    num_params = np.prod(shape)
                    bias_weights = weights[weight_iter:weight_iter + num_params].reshape(shape)
                    weight_iter += num_params
                    assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))

                    # Move up 1 variable
                    var_iter += 1
                    pbar.update(1)

                shape = var1.shape.as_list()
                num_params = np.prod(shape)

                var_weights = weights[weight_iter:weight_iter + num_params].reshape((shape[3], shape[2],
                                                                                     shape[0], shape[1]))
                var_weights = np.transpose(var_weights, (2, 3, 1, 0))
                weight_iter += num_params
                assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
                var_iter += 1
                pbar.update(1)

        return assign_ops
