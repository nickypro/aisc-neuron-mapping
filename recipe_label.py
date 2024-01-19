import numpy as np
from sklearn import linear_model
from taker import Model

def train_predictor_layer(m: Model):
    recipe_train = """Ingredients: 1 lb chicken breasts, 2 cups broccoli florets, 1 cup sliced carrots, 1/4 cup soy sauce, 2 tbsp sesame oil, 2 tbsp honey, 2 cloves garlic (minced), 1 tsp ginger (grated), 2 cups cooked rice. Instructions: Cook chicken in sesame oil until brown. Add broccoli, carrots, garlic, and ginger. Mix soy sauce and honey, add to pan. Cook until veggies are tender. Serve over rice."""
    stream_train = m.get_residual_stream(recipe_train).to('cpu')

    recipe_test = """Ingredients: 1 lb shrimp, 2 cups bell peppers (sliced), 1 cup pineapple chunks, 1/4 cup teriyaki sauce, 2 tbsp olive oil, 2 cloves garlic (minced), 1 tsp chili flakes, 2 cups cooked quinoa. Instructions: Sauté shrimp, peppers, and garlic in olive oil. Add pineapple, teriyaki sauce, chili flakes. Cook until shrimp are pink. Serve over quinoa. Enjoy!"""
    stream_test = m.get_residual_stream(recipe_test).to('cpu')

    for layer in range(m.cfg.n_layers*2 + 1):
        #reg = linear_model.LinearRegression()
        reg = linear_model.LogisticRegression()
        X_train = np.array(stream_train[layer])
        labels = np.zeros(144)
        labels[89:] = 1
        X_test = np.array(stream_test[layer])
        labels_test = np.zeros(139)
        labels_test[82:] = 1
        reg = reg.fit(X_train, labels)
        score = reg.score(X_test, labels_test)
        print('Layer', layer, score)

m = Model("nickypro/tinyllama-15M")
train_predictor_layer(m)
