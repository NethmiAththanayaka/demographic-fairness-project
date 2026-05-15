from implicit.als import AlternatingLeastSquares

def train_als(
    user_items,
    factors=64,
    regularization=0.01,
    iterations=20
):
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations
    )

    model.fit(user_items)

    return model
