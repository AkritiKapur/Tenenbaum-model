import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
H = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

h = {}


def get_length(left, right):
    lx, ly = left
    rx, ry = right

    return np.sqrt(np.square(lx-rx) + np.square(ly-ry))


# Task 0
def get_priors(Hypothesis_space, sig):

    prior = {}
    for i in Hypothesis_space:
        left = (i, i)
        right = (-i, i)
        side_len = get_length(left, right)
        prior[i] = np.exp(- 2 * side_len / sig)

    denom = sum(prior.values())
    prior = {k: prior[k] / denom for k in prior}
    return prior


def get_uniformative_prior(Hypothesis_space):
    prior = {}
    for i in Hypothesis_space:
        left = (i, i)
        right = (-i, i)
        side_len = get_length(left, right)
        prior[i] = 1 / (side_len * side_len)

    denom = sum(prior.values())
    prior = {k: prior[k] / denom for k in prior}
    plt.bar(H, prior.values())
    plt.title('uninformative prior distribution')
    plt.xlabel('side len / 2')
    plt.ylabel('priors')
    plt.show()
    return prior


def task_1():
    prior_6 = get_priors(H, 6)
    prior_12 = get_priors(H, 12)
    plt.bar(H, prior_6.values())
    plt.title('prior distribution with sigma_1 = sigma_2 = 6')
    plt.xlabel('side len / 2')
    plt.ylabel('priors')
    plt.show()

    plt.bar(H, prior_12.values())
    plt.title('prior distribution with sigma_1 = sigma_2 = 12')
    plt.xlabel('side len / 2')
    plt.ylabel('priors')
    plt.show()


def task_2(Hypothesis_space, X, sig):
    priors = get_priors(H, sig)
    likelihood = {}
    posterior = {}
    for i in Hypothesis_space:
        left = (i, i)
        right = (-i, i)
        all_present = True
        for sample in X:
            x, y = sample
            if -i <= x <= i and -i <= y <= i:
                pass
            else:
                all_present = False

        likelihood[i] = 0
        if all_present:
            area = np.square(get_length(left, right))
            likelihood[i] = 1 / pow(area, len(X))

        posterior[i] = likelihood[i] * priors[i]

        # print(likelihood[i], priors[i], posterior[i])

    denom = sum(posterior.values())
    posterior = {i: posterior[i] / denom for i in posterior}
    plt.bar(Hypothesis_space, [posterior[i] for i in Hypothesis_space])
    plt.title('posterior with {} point(s)'.format(len(X)))
    plt.xlabel('side len / 2')
    plt.ylabel('posterior')
    plt.show()

    return posterior


def weak_posterior(Hypothesis_space, X, sig):
    priors = get_priors(H, sig)
    likelihood = {}
    posterior = {}
    for i in Hypothesis_space:
        all_present = True
        for sample in X:
            x, y = sample
            if -i <= x <= i and -i <= y <= i:
                pass
            else:
                all_present = False

        likelihood[i] = 0
        if all_present:
            likelihood[i] = 1

        posterior[i] = likelihood[i] * priors[i]

        # print(likelihood[i], priors[i], posterior[i])

    denom = sum(posterior.values())
    posterior = {i: posterior[i] / denom for i in posterior}
    # plt.bar(Hypothesis_space, [posterior[i] for i in Hypothesis_space])
    # plt.title('posterior with {}')
    # plt.xlabel('side len / 2')
    # plt.ylabel('posterior')
    # plt.show()

    return posterior


def get_posteriors_uninformative(Hypothesis_space, X):
    priors = get_uniformative_prior(H)
    likelihood = {}
    posterior = {}
    for i in Hypothesis_space:
        left = (i, i)
        right = (-i, i)
        all_present = True
        for sample in X:
            x, y = sample
            if -i <= x <= i and -i <= y <= i:
                pass
            else:
                all_present = False

        likelihood[i] = 0
        if all_present:
            area = np.square(get_length(left, right))
            likelihood[i] = 1 / pow(area, len(X))

        posterior[i] = likelihood[i] * priors[i]

    denom = sum(posterior.values())
    posterior = {i: posterior[i] / denom for i in posterior}

    # plt.plot(Hypothesis_space, [posterior[i] for i in Hypothesis_space])
    # plt.show()

    return posterior


def task_3(x_range, y_range, X, sig):
    posteriors = task_2(H, X, sig)
    prediction_probabilities = defaultdict(list)
    for i in range(x_range[0], x_range[1]):
        for j in range(y_range[0], y_range[1]):
            val = 0
            for hh in H:
                if -hh <= i <= hh and -hh <= j <= hh:
                    val += posteriors[hh]

            prediction_probabilities[i].append(val)
    x = np.arange(x_range[0], x_range[1])
    y = np.arange(y_range[0], y_range[1])

    z = plt.contourf(x, y, [prediction_probabilities[i] for i in range(x_range[0], x_range[1])])
    plt.colorbar(z)
    plt.show()


def task_5():
    task_3([-10, 11], [-10, 11], [(2.2, -0.2)], 30)
    task_3([-10, 11], [-10, 11], [(2.2, -0.2), (0.5, 0.5)], 30)
    task_3([-10, 11], [-10, 11], [(2.2, -0.2), (0.5, 0.5), (1.5, 1)], 30)


def uniformative_prob(x_range, y_range, X):
    # using uninformative prior
    posteriors = get_posteriors_uninformative(H, X)
    prediction_probabilities = defaultdict(list)
    for i in range(x_range[0], x_range[1]):
        for j in range(y_range[0], y_range[1]):
            val = 0
            for hh in H:
                if -hh <= i <= hh and -hh <= j <= hh:
                    val += posteriors[hh]

            prediction_probabilities[i].append(val)
    x = np.arange(x_range[0], x_range[1])
    y = np.arange(y_range[0], y_range[1])

    z = plt.contourf(x, y, [prediction_probabilities[i] for i in range(x_range[0], x_range[1])])
    plt.colorbar(z)
    plt.show()


def weak_bayes(x_range, y_range, X, sig):
    posteriors = weak_posterior(H, X, sig)
    print(posteriors)
    prediction_probabilities = defaultdict(list)
    for i in range(x_range[0], x_range[1]):
        for j in range(y_range[0], y_range[1]):
            val = 0
            for hh in H:
                if -hh <= i <= hh and -hh <= j <= hh:
                    val += posteriors[hh]

            prediction_probabilities[i].append(val)
    x = np.arange(x_range[0], x_range[1])
    y = np.arange(y_range[0], y_range[1])

    z = plt.contourf(x, y, [prediction_probabilities[i] for i in range(x_range[0], x_range[1])])
    plt.colorbar(z)
    plt.show()


def task_6():
    uniformative_prob([-10, 11], [-10, 11], [(2.2, -0.2)])
    # uniformative_prob([-10, 11], [-10, 11], [(2.2, -0.2), (0.5, 0.5)])
    # uniformative_prob([-10, 11], [-10, 11], [(2.2, -0.2), (0.5, 0.5), (1.5, 1)])

    # weak bayes
    # weak_bayes([-10, 11], [-10, 11], [(2.2, -0.2)], 30)
    # weak_bayes([-10, 11], [-10, 11], [(2.2, -0.2), (0.5, 0.5)], 30)
    # weak_bayes([-10, 11], [-10, 11], [(2.2, -0.2), (0.5, 0.5), (1.5, 1)], 30)


if __name__ == "__main__":
    # task_1()
    # task_2(H, [(1.5, 0.5)], 12)
    # task_3([-10, 11], [-10, 11], [(1.5, 0.5)], 10)
    # task_3([-10, 11], [-10, 11], [(4.5, 2.5)], 10)
    # task_5()
    task_6()
