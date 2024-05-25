import numpy as np

class Conv1D:
    """
    Implémentation de la couche de convolution 1D.
    """
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size) / filter_size

    def iterate_regions(self, input):
        """
        Génère toutes les régions 1D pour la convolution.
        """
        for i in range(input.shape[0] - self.filter_size + 1):
            region = input[i:i + self.filter_size]
            yield region, i

    def forward(self, input):
        """
        Effectue la passe avant de la convolution.
        """
        self.last_input = input
        output = np.zeros((input.shape[0] - self.filter_size + 1, self.num_filters))
        for region, i in self.iterate_regions(input):
            output[i] = np.sum(region * self.filters, axis=1)
        return output

    def backward(self, d_L_d_out, learn_rate):
        """
        Effectue la rétropropagation de la convolution.
        """
        d_L_d_filters = np.zeros(self.filters.shape)
        for region, i in self.iterate_regions(self.last_input):
            for j in range(self.num_filters):
                d_L_d_filters[j] += d_L_d_out[i, j] * region

        self.filters -= learn_rate * d_L_d_filters
        return None


class Conv2D:
    """
    Implémentation de la couche de convolution 2D.
    """
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / (filter_size * filter_size)

    def iterate_regions(self, image):
        """
        Génère toutes les régions 2D pour la convolution.
        """
        h, w = image.shape
        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                region = image[i:i + self.filter_size, j:j + self.filter_size]
                yield region, i, j

    def forward(self, input):
        """
        Effectue la passe avant de la convolution.
        """
        self.last_input = input
        h, w = input.shape
        output = np.zeros((h - self.filter_size + 1, w - self.filter_size + 1, self.num_filters))
        for region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(region * self.filters, axis=(1, 2))
        return output

    def backward(self, d_L_d_out, learn_rate):
        """
        Effectue la rétropropagation de la convolution.
        """
        d_L_d_filters = np.zeros(self.filters.shape)
        for region, i, j in self.iterate_regions(self.last_input):
            for k in range(self.num_filters):
                d_L_d_filters[k] += d_L_d_out[i, j, k] * region

        self.filters -= learn_rate * d_L_d_filters
        return None


class MaxPool1D:
    """
    Implémentation de la couche de max-pooling 1D.
    """
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def iterate_regions(self, input):
        """
        Génère toutes les régions 1D pour le max-pooling.
        """
        for i in range(0, input.shape[0], self.pool_size):
            region = input[i:i + self.pool_size]
            yield region, i

    def forward(self, input):
        """
        Effectue la passe avant du max-pooling.
        """
        self.last_input = input
        output = np.zeros((input.shape[0] // self.pool_size, input.shape[1]))
        for region, i in self.iterate_regions(input):
            output[i // self.pool_size] = np.amax(region, axis=0)
        return output

    def backward(self, d_L_d_out):
        """
        Effectue la rétropropagation du max-pooling.
        """
        d_L_d_input = np.zeros(self.last_input.shape)
        for region, i in self.iterate_regions(self.last_input):
            h, w = region.shape
            amax = np.amax(region, axis=0)

            for i2 in range(h):
                for j in range(w):
                    if region[i2, j] == amax[j]:
                        d_L_d_input[i + i2, j] = d_L_d_out[i // self.pool_size, j]

        return d_L_d_input


class AvgPool1D:
    """
    Implémentation de la couche d'average-pooling 1D.
    """
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def iterate_regions(self, input):
        """
        Génère toutes les régions 1D pour l'average-pooling.
        """
        for i in range(0, input.shape[0], self.pool_size):
            region = input[i:i + self.pool_size]
            yield region, i

    def forward(self, input):
        """
        Effectue la passe avant de l'average-pooling.
        """
        self.last_input = input
        output = np.zeros((input.shape[0] // self.pool_size, input.shape[1]))
        for region, i in self.iterate_regions(input):
            output[i // self.pool_size] = np.mean(region, axis=0)
        return output

    def backward(self, d_L_d_out):
        """
        Effectue la rétropropagation de l'average-pooling.
        """
        d_L_d_input = np.zeros(self.last_input.shape)
        for region, i in self.iterate_regions(self.last_input):
            h, w = region.shape
            avg = np.mean(region, axis=0)

            for i2 in range(h):
                for j in range(w):
                    d_L_d_input[i + i2, j] = d_L_d_out[i // self.pool_size, j] / self.pool_size

        return d_L_d_input


class Flatten:
    """
    Implémentation de la couche de flattening.
    """
    def forward(self, input):
        """
        Effectue la passe avant du flattening.
        """
        self.last_shape = input.shape
        return input.flatten().reshape(input.shape[0], -1)

    def backward(self, d_L_d_out):
        """
        Effectue la rétropropagation du flattening.
        """
        return d_L_d_out.reshape(self.last_shape)
