from services.embeddings import generate_embeddings, cosine_similarity

texts = [
    "Agua Bonafont botella 1L",
    "Coca-Cola lata 355ml",
    "Papas Sabritas Fuego 170g",
    "Red Bull Energy Drink 250ml",
    "Galletas Oreo chocolate 117g",
    "Leche entera Lala 1L",
    "Atún Dolores en agua 140g",
    "Cereal Zucaritas 300g"
]

embs = generate_embeddings(texts)

query = "bebida energética con cafeína"
q_emb = generate_embeddings(query)

scores = [cosine_similarity(q_emb[0], e) for e in embs]

best_index = scores.index(max(scores))
print("Best: ", texts[best_index])
print("Score: ", scores[best_index])

print("Scores: ", scores)