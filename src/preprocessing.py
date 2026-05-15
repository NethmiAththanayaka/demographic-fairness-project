from sklearn.model_selection import train_test_split

def create_train_test(ratings, test_size=0.2, random_state=42):
    train, test = train_test_split(
        ratings,
        test_size=test_size,
        random_state=random_state
    )

    return train, test
