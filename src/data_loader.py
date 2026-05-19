import os
import zipfile
import urllib.request
import pandas as pd


def download_movielens(data_root="data"):
    os.makedirs(data_root, exist_ok=True)

    zip_path = os.path.join(data_root, "ml-1m.zip")
    extract_dir = os.path.join(data_root, "ml-1m")

    if os.path.exists(extract_dir):
        print("MovieLens already exists.")
        return extract_dir

    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

    print("Downloading MovieLens-1M...")
    urllib.request.urlretrieve(url, zip_path)

    print("Extracting MovieLens-1M...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_root)

    return extract_dir


def load_movielens(data_dir="data/ml-1m"):
    if not os.path.exists(data_dir):
        download_movielens(data_root="data")

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


def load_lastfm(data_dir="data/lastfm"):
    """
    Placeholder for Last.fm loading.

    You should update file names here based on the exact Last.fm dataset files
    used in your notebook.
    """

    users_path = os.path.join(data_dir, "users.dat")
    artists_path = os.path.join(data_dir, "artists.dat")
    user_artists_path = os.path.join(data_dir, "user_artists.dat")

    users = pd.read_csv(users_path, sep="\t")
    artists = pd.read_csv(artists_path, sep="\t")
    interactions = pd.read_csv(user_artists_path, sep="\t")

    return interactions, users, artists