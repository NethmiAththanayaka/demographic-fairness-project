import pandas as pd

def load_movielens(data_dir="data/ml-1m"):
    ratings = pd.read_csv(
        f"{data_dir}/ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    users = pd.read_csv(
        f"{data_dir}/users.dat",
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"]
    )

    movies = pd.read_csv(
        f"{data_dir}/movies.dat",
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["movie_id", "title", "genres"]
    )

    return ratings, users, movies
