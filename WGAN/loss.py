import torch


def get_gradient(crit, real, fake, epsilon):
    """
    Returns the gradient of the critic's scores with respect to mixes of real and fake images
    :param crit: a critic model
    :param real: a batch of real_images
    :param fake: a batch of fake_images
    :param epsilon: a vector of the uniformly random proportions of real and fake per mixed image
    :return: gradient: the gradient of the critic's scores wrt the mixed image
    """
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)
    mixed_scores = crit(mixed_images)
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]
    return gradient


def gradient_penalty(gradient):
    """
    Return the gradient penalty of a gradient
    Given a batch of image gradients , you calculate magnitude of each image gradients
    and penalize the mean quadratic distance of each magnitude to 1
    :param gradient: the gradient of critic's scores wrt mixed_image
    :return: penalty: the gradient penalty
    """
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)

    # Penalize the mean squared distance of the gradient norms from 1
    penalty = torch.nn.MSELoss()(gradient_norm, torch.ones_like(gradient_norm))
    return penalty


def get_gen_loss(crit_fake_pred):
    """
    Return the loss of a generator given the critic's scores of generator's fake images.
    :param crit_fake_pred: the critic's scores of the fake images
    :return: gen_loss: a scalar loss value for current batch of generator
    """
    gen_loss = -1 * crit_fake_pred.mean()
    return gen_loss


def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    """
    Return the loss of critic given the critic's scores for fake and real images
    :param crit_fake_pred: the critic's scores of the fake images
    :param crit_real_pred: the critic's scores of the real images
    :param gp: the unweighted gradient penalty
    :param c_lambda: the current weight of the gradient penalty
    :return: crit_loss: a scalar for the critic's loss, accounting for the relevant factors
    """
    # crit_loss =  (crit_fake_pred - crit_real_pred + gp * c_lambda).mean()
    crit_loss = crit_fake_pred.mean() - crit_real_pred.mean() + gp * c_lambda
    return crit_loss
