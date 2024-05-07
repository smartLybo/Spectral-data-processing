from sklearn.metrics import mean_absolute_error

def get_MAE(y_true, y_pred): # MAE
    C = mean_absolute_error(y_true, y_pred)
    return C