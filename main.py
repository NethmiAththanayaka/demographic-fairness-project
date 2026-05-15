from src.data_loader import load_movielens
from src.preprocessing import create_train_test
from src.matrix_builder import build_sparse_matrix
from src.model import train_als
from src.evaluation import recall_at_k_for_users_model
from src.fairness_metrics import demo_table
from src.evaluation import max_gap

ratings, users, movies = load_movielens()

train, test = create_train_test(ratings)

user_items, u2i, i2u, m2i, i2m = build_sparse_matrix(train)

model = train_als(user_items)

recall_df = recall_at_k_for_users_model(
    model,
    user_items,
    test,
    users,
    u2i,
    m2i
)

result = demo_table(recall_df, users, demographic="gender")

print(result)
print("Max Gap:", max_gap(result))
