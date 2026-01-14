import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def train(rows, label_key, epochs=400, lr=0.1):
    if not rows:
        return {"weights": {}, "bias": 0.0, "accuracy": 0.0}

    features = [key for key in rows[0].keys() if key != label_key]
    weights = {feature: 0.0 for feature in features}
    bias = 0.0

    for _ in range(epochs):
        grad_w = {feature: 0.0 for feature in features}
        grad_b = 0.0
        for row in rows:
            x = [float(row[feature]) for feature in features]
            y = float(row[label_key])
            z = sum(weights[feature] * value for feature, value in zip(features, x)) + bias
            pred = sigmoid(z)
            error = pred - y
            for feature, value in zip(features, x):
                grad_w[feature] += error * value
            grad_b += error
        for feature in features:
            weights[feature] -= lr * grad_w[feature] / len(rows)
        bias -= lr * grad_b / len(rows)

    accuracy = evaluate(rows, weights, bias, label_key, features)
    return {
        "weights": {key: round(value, 4) for key, value in weights.items()},
        "bias": round(bias, 4),
        "accuracy": round(accuracy, 3),
        "features": features,
    }


def predict(row, weights, bias, features):
    z = sum(weights.get(feature, 0.0) * float(row.get(feature, 0.0)) for feature in features) + bias
    prob = sigmoid(z)
    return prob


def evaluate(rows, weights, bias, label_key, features):
    correct = 0
    for row in rows:
        prob = predict(row, weights, bias, features)
        pred = 1 if prob >= 0.5 else 0
        if pred == int(row[label_key]):
            correct += 1
    return correct / len(rows) if rows else 0.0
