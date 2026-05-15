def recommend_with_fair_rerank(
    model,
    user_items,
    user_id,
    u2i,
    longtail_items,
    K=10,
    C=200,
    alpha=0.1
):

    if user_id not in u2i:
        return []

    uidx = u2i[user_id]

    recs = model.recommend(
        uidx,
        user_items[uidx],
        N=C
    )

    reranked = []

    for item_idx, score in recs:

        bonus = alpha if item_idx in longtail_items else 0

        reranked.append((item_idx, score + bonus))

    reranked.sort(key=lambda x: x[1], reverse=True)

    return reranked[:K]
