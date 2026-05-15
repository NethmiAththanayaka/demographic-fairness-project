from scipy.sparse import coo_matrix

def build_sparse_matrix(train):
    user_ids = train["user_id"].unique()
    item_ids = train["movie_id"].unique()

    u2i = {u: i for i, u in enumerate(user_ids)}
    i2u = {i: u for u, i in u2i.items()}

    m2i = {m: i for i, m in enumerate(item_ids)}
    i2m = {i: m for m, i in m2i.items()}

    rows = train["user_id"].map(u2i)
    cols = train["movie_id"].map(m2i)

    matrix = coo_matrix(
        (train["rating"], (rows, cols))
    ).tocsr()

    return matrix, u2i, i2u, m2i, i2m
