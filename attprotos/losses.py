from keras import backend as K


def prototype_loss(y_true, y_pred):
    if len(y_pred.shape) == 3:
        y_pred = K.mean(y_pred, axis=-1)
    if len(y_pred.shape) == 4:
        y_pred = K.mean(y_pred, axis=(2, 3))

    error_1 = K.mean(K.min(y_pred, axis=0))
    error_2 = K.mean(K.min(y_pred, axis=1))

    return 1.0*error_1 + 1.0*error_2


def dummy_loss(y_true, y_pred):
    return K.sum(y_pred)
