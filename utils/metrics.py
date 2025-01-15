from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, data_gen):
    y_true = data_gen.classes
    y_pred = model.predict(data_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(classification_report(y_true, y_pred_classes, target_names=data_gen.class_indices.keys()))
    print(confusion_matrix(y_true, y_pred_classes))