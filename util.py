import tensorflow as tf
from tensorflow_probability import edward2 as ed
import numpy as np
import itertools
import math
import matplotlib
import matplotlib.pyplot as plt




def set_values(**model_kwargs):
    """Creates a value-setting interceptor."""

    def interceptor(f, *args, **kwargs):
        """Sets random variable values to its aligned value."""
        name = kwargs.get("name")
        if name in model_kwargs:


            kwargs["value"] = model_kwargs[name]
        else:
            print(f"set_values not interested in {name}.")
        return ed.interceptable(f)(*args, **kwargs)

    return interceptor



def optimizing_loop(fn, optimizer, sess, num_epochs=1000, verbose=True):
    t = []
    train = optimizer.minimize(-fn)
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(num_epochs):
        sess.run(train)
        t.append(sess.run([fn]))
        if verbose:
            if i % 200 == 0: print("\n" + str(i) + " epochs\t" + str(t[-1]), end="", flush=True)
            if i % 10 == 0: print(".", end="", flush=True)

    return t



def get_pred_loglk(x_test, fn_p, fn_q, qweights_dict, qbias_dict):

    qweights_dict_ = {k:tf.stop_gradient(v) for k,v in qweights_dict.items()}
    qbias_dict_ = {k:tf.stop_gradient(v) for k,v in qbias_dict.items()}


    qz, _, _ = fn_q()

    with ed.interception(set_values(z=qz, x=x_test, **qweights_dict_, **qbias_dict_)):
        pz, _, _, px = fn_p()

    energy = tf.reduce_sum(pz.distribution.log_prob(pz.value)) + \
             tf.reduce_sum(px.distribution.log_prob(px.value))

    entropy = tf.reduce_sum(qz.distribution.log_prob(qz.value))

    elbo = energy - entropy

    return elbo # = loglk


def plot_imgs(data , nx=3, ny=3):
    fig, ax = plt.subplots(nx, ny, figsize=(12, 12))
    fig.tight_layout(pad=0.3, rect=[0, 0, 0.9, 0.9])
    for x, y in [(i, j) for i in list(range(nx)) for j in list(range(ny))]:
        img_i = data[x + y * nx].reshape((28, 28))
        i = (x, y) if nx > 1 else y
        ax[i].imshow(img_i, cmap='gray')
    plt.show()



def preprocess_data(x_data, y_data, N, C, num_pixels):
    x_data = x_data[np.isin(y_data, C)]
    y_data = y_data[np.isin(y_data, C)]
    y_data = y_data[:N]
    x_data = np.reshape(x_data[:N], (N,num_pixels))  #serialize the data
    x_data = np.float32(x_data)
    return x_data, y_data



def plot_hidden(name, post, y_train, C, step, save=True, log=None):
    # plot_imgs(w_loc_inf)
    import matplotlib
    matplotlib.rcParams.update({'font.size': 14})

    markers = ["x", "+", "o"]
    colors = [plt.get_cmap("gist_rainbow")(0.05),
              plt.get_cmap("gnuplot2")(0.08),
              plt.get_cmap("gist_rainbow")(0.33)]
    transp = [0.9, 0.9, 0.5]

    fig = plt.figure()

    for c in range(0, len(C)):
        col = colors[c]
        plt.scatter(post["z"][y_train == C[c], 0], post["z"][y_train == C[c], 1], color=col,
                    label=C[c], marker=markers[c], alpha=transp[c], s=60)
        plt.legend(fontsize=16)


    #plt.ylim(post["z"][0])
    #
    # max_z = np.max(post["z"], axis=0)
    # min_z = np.min(post["z"], axis=0)
    #
    #
    # plt.ylim(min_z[1]*0.99, max_z[1]*1.1)
    # plt.xlim(min_z[0] * 0.99, max_z[0] * 1.1)

    if save:
        fig.savefig(name)
    if log != None:
        log.log_fig(name, fig, step)


    plt.close(fig)




def parameter_iterator(*args, **kwargs):

    def parameters(max_iter=math.inf):
        n = 1
        for i in itertools.product(*(list(*args) + list(kwargs.values()))):
            yield i
            if n>=max_iter: break
            n=n+1

    return parameters



def get_tfvar(name):
    for v in tf.trainable_variables():
        if str.startswith(v.name, name):
            return v


