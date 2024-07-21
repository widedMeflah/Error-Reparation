from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import mutual_info_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow_probability as tfp
from sklearn.preprocessing import LabelEncoder
import scipy
from sklearn.metrics import mutual_info_score
import itertools
from sklearn.metrics import normalized_mutual_info_score
import time
import psutil
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import tensorflow.keras as keras
from sklearn.preprocessing import StandardScaler



app = Flask(__name__)

# Charger le jeu de données en utilisant pandas
data = pd.DataFrame({
 "Num": ["1", "2", "3", "4", "5", "6", "7"],
     "NSS": ["123-45-678-901-23", "123-45-678-901-23", "987-65-432-109-87", "456-78-901-234-56", "789-01-234-567-89", "234-56-789-012-34", "567-89-012-345-67"],
    "Nom": ["Jean Dupont", "Marie Martin", "Jean Dupont", "Maria Garcia", "Hans Müller", "Chihiro Tanaka", "Antoine Yann"],
    "Telephone": ["011234567", "011234568", "042125551", "042125551", "041234567", "031234565", "031235676"],
    "indicatifPays": ["+33", "+33", "+33", "+33", "+33", "+81", "+81"],
    "Commune": ["Paris", "Paris", "Montpellier", "Paris", "Montpellier", "Tokyo", "Tokyo"],
    "Pays": ["France", "France", "France", "France", "France", "Japon", "Japon"],
    "Zip": ["75000", "75000", "34000", "75000", "34000", "100-0001", "100-0001"]
})

@app.route('/')
def tableau():
    # Convertir le DataFrame en HTML
    table_html = data.to_html(classes='table table-striped table-bordered table-hover', index=False)

    return render_template('tableau.html', table_html=table_html)


dependances = ['indicatifPays → Zip', 'Telephone → Nom', 'NSS → Telephone', 'Pays → Commune']
dependancesValides = ['indicatifPays → Pays', 'NSS → Nom', 'NSS → Telephone', 'Zip → Commune']

def verifie_dependances(dataframe, dependances):
    # Initialise un dictionnaire pour stocker les résultats pour chaque dépendance fonctionnelle
    resultats = {}

    # Parcourt chaque dépendance fonctionnelle
    for FD in dependances:
        # Sépare la partie gauche et la partie droite de la FD
        partie_gauche, partie_droite = FD.split(' → ')

        # Groupe les données par la partie gauche de la FD et récupère les indices des groupes
        groupes = dataframe.groupby(partie_gauche).groups

        # Initialise une liste pour stocker les indices des lignes enfreignant la FD
        lignes_non_respectees = []

        # Parcourt chaque groupe
        for groupe in groupes.values():
            # Si le groupe a plus d'une ligne, vérifie si la partie droite est la même pour toutes les lignes du groupe
            if len(groupe) > 1:
                valeurs_partie_droite = dataframe.loc[groupe, partie_droite].values
                if len(set(valeurs_partie_droite)) > 1:
                    # La dépendance fonctionnelle n'est pas respectée pour ce groupe de lignes
                    lignes_non_respectees.extend(groupe)

        # Récupère les indices des lignes uniques
        indices_lignes_non_respectees = list(set(lignes_non_respectees))

        # Stocke les résultats pour cette dépendance fonctionnelle dans le dictionnaire
        resultats[FD] = indices_lignes_non_respectees

    # Crée un DataFrame pour les résultats
    resultat_final = pd.DataFrame(resultats.items(), columns=['DF', 'LignesViolations'])

    # Retourne le DataFrame final
    return resultat_final
def statistique1(dataframe, resultat_dependances):
    resultat = verifie_dependances(data, dependances)
    resultat_dependances['TailleViolations'] = resultat_dependances['LignesViolations'].apply(lambda x: len(x) / len(dataframe))
    return resultat_dependances

def statistique2(dataframe, dependances,violations_data):

  resultats = []

  for fd in dependances:
    left_column, right_column = fd.split(' → ')
    resultat_fd = []

    # Sélectionner les lignes correspondant à cette violation de dépendance fonctionnelle
    violation_indices = violations_data['LignesViolations'][dependances.index(fd)]
    violation_data = data.iloc[violation_indices]

    # Utiliser groupby pour regrouper par chaque valeur unique dans la colonne de gauche
    grouped = violation_data.groupby(left_column)[right_column].nunique()

    # Utiliser value_counts pour compter le nombre d'occurrences de chaque valeur de la colonne de gauche
    count_values = violation_data[left_column].value_counts()

    # Parcourir les valeurs uniques de la colonne de gauche
    for value in grouped.index:
        # Récupérer le résultat de groupby pour cette valeur
        groupby_result = grouped.loc[value]

        # Récupérer le résultat de value_counts pour cette valeur
        count_result = count_values.get(value, 0)

        # Éviter une division par zéro en vérifiant que count_result est différent de zéro
        if count_result != 0:
            # Effectuer la division et ajouter le résultat à la liste resultat_fd
            resultat_fd.append(groupby_result / count_result)

        else:
            resultat_fd.append(0)  # En cas de division par zéro, ajouter 0 comme résultat

    # Ajouter la liste resultat_fd à la liste de résultats finaux
    resultats.append(resultat_fd)
  violations_data['Moyenne'] = [np.mean(resultat) for resultat in resultats]

  return violations_data
def build_generator():
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation='relu', input_shape=(2,)))
    model.add(keras.layers.Dense(2, activation='linear'))  # Générateur génère des données de même dimension que X
    return model

# Créez le discriminateur
def build_discriminator():
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation='relu', input_shape=(2,)))
    model.add(keras.layers.Dense(1, activation='sigmoid'))  # Discriminateur génère une seule sortie (réel ou faux)
    return model

# Fonction pour créer et entraîner le GAN
def train_gan(X_train):
    X_train_scaled = StandardScaler().fit_transform(X_train)  # Normalisation des données

    generator = build_generator()
    discriminator = build_discriminator()

    discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    discriminator.trainable = False  # Gèle les poids du discriminateur lors de l'entraînement du générateur

    gan_input = keras.Input(shape=(2,))
    x = generator(gan_input)
    gan_output = discriminator(x)

    gan = keras.Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')

    batch_size = 32
    epochs = 1000  # Nombre d'itérations d'entraînement du GAN

    for epoch in range(epochs):
        for _ in range(X_train_scaled.shape[0] // batch_size):
            noise = np.random.normal(0, 1, (batch_size, 2))  # Génère du bruit aléatoire
            generated_data = generator.predict(noise)  # Génère des données synthétiques
            real_data = X_train_scaled[np.random.randint(0, X_train_scaled.shape[0], batch_size)]

            X = np.concatenate([real_data, generated_data])
            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = 1  # Étiquettes pour les données réelles

            discriminator_loss = discriminator.train_on_batch(X, y_dis)

            noise = np.random.normal(0, 1, (batch_size, 2))
            y_gen = np.ones(batch_size)  # Étiquettes pour les données générées

            generator_loss = gan.train_on_batch(noise, y_gen)

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, D Loss: {discriminator_loss[0]}, G Loss: {generator_loss}")

    return generator.predict(np.random.normal(0, 1, (X_train_scaled.shape[0], 2)))
def coder(df):
    label_encoder = LabelEncoder()

    # Supprimer la colonne 'row_num' si nécessaire
    if 'Num' in df.columns:
        df = df.drop(columns=['Num'])

    df_encoded = df.copy()

    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            df_encoded[col] = label_encoder.fit_transform(df_encoded[col])

    return df_encoded

def classes(dataframe, dependances, algorithm):
    resultatt = verifie_dependances(dataframe, dependances)
    resultat = statistique1(dataframe, resultatt)
    statitstique2 = statistique2(dataframe, dependances, resultat)

    X = statitstique2[["TailleViolations", "Moyenne"]]
    y = statitstique2["class"]

    if algorithm == "LogisticRegression":
        classifier = LogisticRegression()
    elif algorithm == "SVM":
        classifier = SVC()
    elif algorithm == "RandomForest":
        classifier = RandomForestClassifier()
    elif algorithm == "DecisionTree":
        classifier = DecisionTreeClassifier()
    else:
        raise ValueError("Algorithm not supported")

    num_repeats = 1

    for _ in range(num_repeats):
        # Entraînez le modèle sur l'ensemble de données complet
        classifier.fit(X, y)

        # Faites des prédictions sur l'ensemble de données complet
        y_pred = classifier.predict(X)
        accuracy = accuracy_score(y, y_pred)

    # Sélectionnez les colonnes "fd" et "class"
    statitstique2_subset = statitstique2[["DF", "source"]]

    return statitstique2_subset
def generer_combinaisons(attributs_sup_jeme_moy, j):
    # Utiliser itertools.combinations pour générer les combinaisons de taille j
    combinaisons = list(itertools.combinations(attributs_sup_jeme_moy, j))

    return combinaisons

def ajouter_attributs_gauche(df, contrainte_erronee, n):
    # Extraire les attributs du côté gauche et du côté droit de la flèche "→"
    attributs_gauche, attributs_droite = [part.strip() for part in contrainte_erronee.split(' → ')]

    # Convertir les listes en ensembles pour faciliter les opérations ensemblistes
    attributs_droite = set(attributs_droite.split(', '))
    attributs_gauche = set(attributs_gauche.split(', '))

    # Calculer les attributs restants (tous les attributs - attributs du côté droit - attributs du côté gauche)
    attributs_tous = set(df.columns)
    attributs_restants = attributs_tous - attributs_droite - attributs_gauche

    if n is None:
        n = len(attributs_restants)

    # Étape 1 : Initialiser les listes pour stocker les informations mutuelles et les moyennes
    info_mutuelles = []
    moyennes_info_mutuelles = []

    # Étape 2 : Calculer l'information mutuelle pour chaque paire d'éléments
    for attr_r in attributs_restants:
        info_mutuelle_attr_r = []
        for attr_d in attributs_droite:
            # Récupérer les données correspondant aux attributs d'intérêt
            data_d = df[attr_d]
            data_r = df[attr_r]

            # Calculer l'information mutuelle entre les deux attributs
            info_mutuelle = normalized_mutual_info_score(data_d, data_r)

            info_mutuelle_attr_r.append(info_mutuelle)

        info_mutuelles.append(info_mutuelle_attr_r)

    # Étape 3 : Calculer les moyennes d'information mutuelle pour chaque attribut restant
    for info_mutuelle_attr_r in info_mutuelles:
        moyenne_info_mutuelle = np.mean(info_mutuelle_attr_r)

        moyennes_info_mutuelles.append(moyenne_info_mutuelle)

    # Étape 4 : Trier les attributs restants en fonction des moyennes d'information mutuelle calculées
    resultats_tries = [x for _, x in sorted(zip(moyennes_info_mutuelles, attributs_restants), reverse=True)]

    moyennes_info_mutuelles.sort(reverse=True)

    # Trouver l'attribut avec la moyenne la plus élevée parmi tous les attributs triés
    attribut_max_moyenne = [attr for attr in resultats_tries if moyennes_info_mutuelles[resultats_tries.index(attr)] == moyennes_info_mutuelles[0]]
    # Convertir l'ensemble attributs_gauche en liste
    attributs_gauche_liste = list(attributs_gauche)

    # Étape 5 : Construire les nouvelles contraintes avec l'attribut ayant la meilleure moyenne d'information mutuelle
    nouvelles_contraintes = []
    for attr in attribut_max_moyenne:
        nouvelles_contrainte = f"{', '.join(attributs_gauche_liste + [attr])} → {', '.join(attributs_droite)}"
        nouvelles_contraintes.append(nouvelles_contrainte)
     # Étape 6 : Trouver les  attributs avec les n meilleurs moyennes
    if n > 1:
        for j in range(2, n + 1):
            attributs_sup_jeme_moy = [attr for attr in resultats_tries if moyennes_info_mutuelles[resultats_tries.index(attr)] >= moyennes_info_mutuelles[j - 1]]
            combinaisons_attributs = generer_combinaisons(attributs_sup_jeme_moy, j)

            # Étape 7 : Ajouter les combinaisons au côté gauche de la contrainte pour créer de nouvelles contraintes
            for combinaison in combinaisons_attributs:
                nouvelle_contrainte = f"{', '.join(attributs_gauche_liste + list(combinaison))} → {', '.join(attributs_droite)}"
                nouvelles_contraintes.append(nouvelle_contrainte)

    return nouvelles_contraintes


def generer_combinaisons(attributs_sup_jeme_moy, j):
    # Utiliser itertools.combinations pour générer les combinaisons de taille j
    combinaisons = list(itertools.combinations(attributs_sup_jeme_moy, j))

    return combinaisons

def ajouter_attributs_droite(df, contrainte_erronee, n):
    # Extraire les attributs du côté gauche et du côté droit de la flèche "→"
    attributs_gauche, attributs_droite = [part.strip() for part in contrainte_erronee.split(' → ')]

    # Convertir les listes en ensembles pour faciliter les opérations ensemblistes
    attributs_droite = set(attributs_droite.split(', '))
    attributs_gauche = set(attributs_gauche.split(', '))

    # Calculer les attributs restants (tous les attributs - attributs du côté droit - attributs du côté gauche)
    attributs_tous = set(df.columns)
    attributs_restants = attributs_tous - attributs_droite - attributs_gauche

    if n is None:
        n = len(attributs_restants)

    # Étape 1 : Initialiser les listes pour stocker les informations mutuelles et les moyennes
    info_mutuelles = []
    moyennes_info_mutuelles = []

    # Étape 2 : Calculer l'information mutuelle pour chaque paire d'éléments
    for attr_r in attributs_restants:
        info_mutuelle_attr_r = []
        for attr_d in attributs_gauche:
            # Récupérer les données correspondant aux attributs d'intérêt
            data_d = df[attr_d]
            data_r = df[attr_r]

            # Calculer l'information mutuelle entre les deux attributs
            info_mutuelle = mutual_info_score(data_d, data_r)

            info_mutuelle_attr_r.append(info_mutuelle)

        info_mutuelles.append(info_mutuelle_attr_r)

    # Étape 3 : Calculer les moyennes d'information mutuelle pour chaque attribut restant
    for info_mutuelle_attr_r in info_mutuelles:
        moyenne_info_mutuelle = np.mean(info_mutuelle_attr_r)

        moyennes_info_mutuelles.append(moyenne_info_mutuelle)

    # Étape 4 : Trier les attributs restants en fonction des moyennes d'information mutuelle calculées
    resultats_tries = [x for _, x in sorted(zip(moyennes_info_mutuelles, attributs_restants), reverse=True)]

    moyennes_info_mutuelles.sort(reverse=True)

    # Trouver l'attribut avec la moyenne la plus élevée parmi tous les attributs triés
    attribut_max_moyenne = [attr for attr in resultats_tries if moyennes_info_mutuelles[resultats_tries.index(attr)] == moyennes_info_mutuelles[0]]
    # Convertir l'ensemble attributs_droite en liste
    attributs_droite_liste = list(attributs_droite)

    # Étape 5 : Construire les nouvelles contraintes avec l'attribut ayant la meilleure moyenne d'information mutuelle
    nouvelles_contraintes = []
    for attr in attribut_max_moyenne:

        nouvelles_contrainte = f"{', '.join(attributs_gauche)} → {', '.join(attributs_droite_liste + [attr])}"
        nouvelles_contraintes.append(nouvelles_contrainte)
    # Étape 6 : Trouver les  attributs avec les n meilleurs moyennes
    if n > 1:
        for j in range(2, n + 1):
            attributs_sup_jeme_moy = [attr for attr in resultats_tries if moyennes_info_mutuelles[resultats_tries.index(attr)] >= moyennes_info_mutuelles[j - 1]]

            combinaisons_attributs = generer_combinaisons(attributs_sup_jeme_moy, j)

            # Étape 7 : Ajouter les combinaisons au côté droite de la contrainte pour créer de nouvelles contraintes
            for combinaison in combinaisons_attributs:

                nouvelle_contrainte = f"{', '.join(attributs_gauche)} → {', '.join(attributs_droite_liste + list(combinaison))}"
                nouvelles_contraintes.append(nouvelle_contrainte)

    return nouvelles_contraintes



def generer_combinaisons(attributs_sup_jeme_moy, j):
    # Utiliser itertools.combinations pour générer les combinaisons de taille j
    combinaisons = list(itertools.combinations(attributs_sup_jeme_moy, j))

    return combinaisons

def supprimer_attributs_droite(df, contrainte_erronee, n):
    # Extraire les attributs du côté gauche et du côté droit de la flèche "→"
    attributs_gauche, attributs_droite = [part.strip() for part in contrainte_erronee.split(' → ')]

    # Convertir les listes en ensembles pour faciliter les opérations ensemblistes
    attributs_droite = set(attributs_droite.split(', '))
    attributs_gauche = set(attributs_gauche.split(', '))

    if n is None:
        n = len(attributs_droite)
    if n == 0:
            return None
    else:
            # Étape 1 : Initialiser les listes pour stocker les informations mutuelles et les moyennes
            info_mutuelles = []
            moyennes_info_mutuelles = []

            # Étape 2 : Calculer l'information mutuelle pour chaque paire d'éléments
            for attr_r in attributs_droite:
                info_mutuelle_attr_r = []
                for attr_d in attributs_gauche:
                    # Récupérer les données correspondant aux attributs d'intérêt
                    data_d = df[attr_d]
                    data_r = df[attr_r]

                    # Calculer l'information mutuelle entre les deux attributs
                    info_mutuelle = mutual_info_score(data_d, data_r)

                    info_mutuelle_attr_r.append(info_mutuelle)

                info_mutuelles.append(info_mutuelle_attr_r)

            # Étape 3 : Calculer les moyennes d'information mutuelle pour chaque attribut droite
            for info_mutuelle_attr_r in info_mutuelles:
                moyenne_info_mutuelle = np.mean(info_mutuelle_attr_r)

                moyennes_info_mutuelles.append(moyenne_info_mutuelle)

            # Étape 4 : Trier les attributs droites en fonction des moyennes d'information mutuelle calculées
            resultats_tries = [x for _, x in sorted(zip(moyennes_info_mutuelles, attributs_droite), reverse=False)]

            moyennes_info_mutuelles.sort(reverse=False)

            # Trouver l'attribut avec la moyenne la plus basse parmi tous les attributs triés
            attribut_min_moyenne = [attr for attr in resultats_tries if moyennes_info_mutuelles[resultats_tries.index(attr)] == moyennes_info_mutuelles[0]]

            # Convertir l'ensemble attributs_droite en liste
            attributs_droite_liste = list(attributs_droite)

            # Étape 5 : Construire les nouvelles contraintes en enlevant l'attribut ayant la  moyenne la plus basse  d'information mutuelle
            nouvelles_contraintes = []
            for attr in attribut_min_moyenne:
                # Créez une copie de la liste attributs_droite_liste
                attributs_droite_liste_copy = attributs_droite_liste.copy()

                # Supprimez l'attribut de la copie
                attributs_droite_liste_copy.remove(attr)

                # Créez la nouvelle contrainte avec la copie modifiée
                nouvelles_contrainte = f"{', '.join(attributs_gauche)} → {', '.join(attributs_droite_liste_copy)}"

                # Ajoutez la nouvelle contrainte à la liste nouvelles_contraintes
                nouvelles_contraintes.append(nouvelles_contrainte)
            # Étape 6 : Trouver les  attributs avec les n mauvaises moyennes
            if n > 1:
                for j in range(2, n + 1):
                    attributs_sup_jeme_moy = [attr for attr in resultats_tries if moyennes_info_mutuelles[resultats_tries.index(attr)] <= moyennes_info_mutuelles[j - 1]]

                    combinaisons_attributs = generer_combinaisons(attributs_sup_jeme_moy, j)

                    # Étape 7 : Enlever les combinaisons du côté droite de la contrainte pour créer de nouvelles contraintes
                    for combinaison in combinaisons_attributs:
                        nouvelle_contrainte = f"{', '.join(attributs_gauche)} → {', '.join([attr for attr in attributs_droite_liste if attr not in combinaison])}"

                        nouvelles_contraintes.append(nouvelle_contrainte)

            return nouvelles_contraintes




def generer_combinaisons(attributs_sup_jeme_moy, j):
    # Utiliser itertools.combinations pour générer les combinaisons de taille j
    combinaisons = list(itertools.combinations(attributs_sup_jeme_moy, j))

    return combinaisons

def supprimer_attributs_gauche(df, contrainte_erronee, n):
    # Extraire les attributs du côté gauche et du côté droit de la flèche "→"
    attributs_gauche, attributs_droite = [part.strip() for part in contrainte_erronee.split(' → ')]

    # Convertir les listes en ensembles pour faciliter les opérations ensemblistes
    attributs_droite = set(attributs_droite.split(', '))
    attributs_gauche = set(attributs_gauche.split(', '))



    if n is None:
        n = len(attributs_gauche)
    if n == 0:
            return None
    else:

            # Étape 1 : Initialiser les listes pour stocker les informations mutuelles et les moyennes
            info_mutuelles = []
            moyennes_info_mutuelles = []

            # Étape 2 : Calculer l'information mutuelle pour chaque paire d'éléments
            for attr_r in attributs_gauche:
                info_mutuelle_attr_r = []
                for attr_d in attributs_droite:
                    # Récupérer les données correspondant aux attributs d'intérêt
                    data_d = df[attr_d]
                    data_r = df[attr_r]

                    # Calculer l'information mutuelle entre les deux attributs
                    info_mutuelle = mutual_info_score(data_d, data_r)

                    info_mutuelle_attr_r.append(info_mutuelle)

                info_mutuelles.append(info_mutuelle_attr_r)

            # Étape 3 : Calculer les moyennes d'information mutuelle pour chaque attribut gauche
            for info_mutuelle_attr_r in info_mutuelles:
                moyenne_info_mutuelle = np.mean(info_mutuelle_attr_r)

                moyennes_info_mutuelles.append(moyenne_info_mutuelle)

            # Étape 4 : Trier les attributs gauches en fonction des moyennes d'information mutuelle calculées
            resultats_tries = [x for _, x in sorted(zip(moyennes_info_mutuelles, attributs_gauche), reverse=False)]

            moyennes_info_mutuelles.sort(reverse=False)

            # Trouver l'attribut avec la moyenne la plus basses parmi tous les attributs triés
            attribut_min_moyenne = [attr for attr in resultats_tries if moyennes_info_mutuelles[resultats_tries.index(attr)] == moyennes_info_mutuelles[0]]

            # Convertir l'ensemble attributs_gauche en liste
            attributs_gauche_liste = list(attributs_gauche)

            # Étape 5 : Construire les nouvelles contraintes avec l'attribut ayant la mauvaise moyenne d'information mutuelle
            nouvelles_contraintes = []
            for attr in attribut_min_moyenne:
              # Créez une copie de la liste attributs_gauche_liste
              attributs_gauche_liste_copy = attributs_gauche_liste.copy()

              # Supprimez l'attribut de la copie
              attributs_gauche_liste_copy.remove(attr)

              # Créez la nouvelle contrainte avec la copie modifiée
              nouvelles_contrainte = f"{', '.join(attributs_gauche_liste_copy)} → {', '.join(attributs_droite)}"

              # Ajoutez la nouvelle contrainte à la liste nouvelles_contraintes
              nouvelles_contraintes.append(nouvelles_contrainte)

            # Étape 6 : Trouver les  attributs avec les n mauvaises moyennes
            if n > 1:
                for j in range(2, n + 1):
                    attributs_sup_jeme_moy = [attr for attr in resultats_tries if moyennes_info_mutuelles[resultats_tries.index(attr)] <= moyennes_info_mutuelles[j-1]]

                    combinaisons_attributs = generer_combinaisons(attributs_sup_jeme_moy, j)

                    # Étape 7 : supprimer les combinaisons du côté gauche de la contrainte pour créer de nouvelles contraintes
                    for combinaison in combinaisons_attributs:

                        nouvelle_contrainte = f"{', '.join  ([attr for attr in attributs_gauche_liste if attr not in combinaison])} → {', '.join(attributs_droite)}"


                        nouvelles_contraintes.append(nouvelle_contrainte)

            return nouvelles_contraintes


def remplacer_attribut_gauche(df, contrainte_errone):


    nouvelles_contraintes_supprimees = supprimer_attributs_gauche(df, contrainte_errone, 1)
    nouvelles_contraintes = []

    for nouvelle_contrainte_supprimee in nouvelles_contraintes_supprimees:
        nouvelles_contraintes_ajoutees = ajouter_attributs_gauche(df, nouvelle_contrainte_supprimee, 1)
        nouvelles_contraintes.extend(nouvelles_contraintes_ajoutees)

    return nouvelles_contraintes
def remplacer_attribut_droite(df, contrainte_errone):


    nouvelles_contraintes_supprimees = supprimer_attributs_droite(df, contrainte_errone, 1)
    nouvelles_contraintes = []

    for nouvelle_contrainte_supprimee in nouvelles_contraintes_supprimees:
        nouvelles_contraintes_ajoutees = ajouter_attributs_droite(df, nouvelle_contrainte_supprimee, 1)
        nouvelles_contraintes.extend(nouvelles_contraintes_ajoutees)

    return nouvelles_contraintes

def GenererModificationsContraintes(data, contrainte_erronee):
    contraintes_generees = []  # liste vide pour stocker les contraintes générées

    ajouter_attributs_gauche_resultat = ajouter_attributs_gauche(data, contrainte_erronee, None)
    contraintes_generees.extend(ajouter_attributs_gauche_resultat)

    ajouter_attributs_droite_resultat = ajouter_attributs_droite(data, contrainte_erronee,None)
    contraintes_generees.extend(ajouter_attributs_droite_resultat)

    supprimer_attributs_droite_resultat = supprimer_attributs_droite(data, contrainte_erronee, None)
    contraintes_generees.extend(supprimer_attributs_droite_resultat)

    supprimer_attributs_gauche_resultat = supprimer_attributs_gauche(data, contrainte_erronee, None)
    contraintes_generees.extend(supprimer_attributs_gauche_resultat)

    remplacer_attribut_gauche_resultat = remplacer_attribut_gauche(data, contrainte_erronee)
    contraintes_generees.extend(remplacer_attribut_gauche_resultat)

    remplacer_attribut_droite_resultat = remplacer_attribut_droite(data, contrainte_erronee)
    contraintes_generees.extend(remplacer_attribut_droite_resultat)

    return contraintes_generees
def filtrer_contraintes(contraintes_generees):    # fonction pour supprime les contraintes qu ont pas l'un des 2 cotées
    contraintes_filtrees = []
    for nouvelle_contrainte in contraintes_generees:
        # Séparer les deux côtés de la contrainte
        cote_gauche, cote_droit = nouvelle_contrainte.split(" → ")

        # Vérifier si les deux côtés existent
        if cote_gauche and cote_droit:
            contraintes_filtrees.append(nouvelle_contrainte)

    return contraintes_filtrees


def nettoyer_contrainte(contrainte):     # fonction pou Supprimer les virgules inutiles
    cote_gauche, cote_droit = contrainte.split(" → ")

    # Supprimer la virgule du côté gauche s'il commence par une virgule
    if cote_gauche.startswith(","):
        cote_gauche = cote_gauche[1:].strip()

    # Supprimer la virgule du côté droit s'il commence par une virgule
    if cote_droit.startswith(","):
        cote_droit = cote_droit[1:].strip()

    return f"{cote_gauche} → {cote_droit}"

def nettoyer_contraintes(contraintes_generees):
    contraintes_nettoyees = []
    for nouvelle_contrainte in contraintes_generees:
        contrainte_modifiee = nettoyer_contrainte(nouvelle_contrainte)
        contraintes_nettoyees.append(contrainte_modifiee)

    return contraintes_nettoyees

def parse_fd(fd_str):
    # Diviser la chaîne de contrainte en parties gauche et droite
    left_side, right_side = fd_str.split(" → ")
    left_attributes = [attr.strip() for attr in left_side.split(",")]
    right_attributes = [attr.strip() for attr in right_side.split(",")]
    return (left_attributes, right_attributes)



def are_similar_constraint(fd_str1, fd_str2):
    fd1 = parse_fd(fd_str1)
    fd2 = parse_fd(fd_str2)

    # Convertir les contraintes en ensembles pour ignorer l'ordre des attributs
    left_side_fd1 = set(fd1[0])
    right_side_fd1 = set(fd1[1])

    left_side_fd2 = set(fd2[0])
    right_side_fd2 = set(fd2[1])

    # Vérifier si les côtés gauches et droits sont similaires
    left_similar = left_side_fd1 == left_side_fd2
    right_similar = right_side_fd1 == right_side_fd2


    return left_similar and right_similar
contraintes_systemes = ["state → city","city → state"]
def generer_contrainte(df, contrainte_erronee, contraintes_systemes):
    df = coder(df)
    contraintes_testees = []

    contraintes_generees = GenererModificationsContraintes(df, contrainte_erronee)
    contraintes_nettoyees = nettoyer_contraintes(contraintes_generees)
    contraintes_filtrees = filtrer_contraintes(contraintes_nettoyees)
    for nouvelle_contrainte in contraintes_filtrees:
        similar_present = any(are_similar_constraint(nouvelle_contrainte, contrainte_systeme) for contrainte_systeme in contraintes_systemes)
        if not similar_present:
            contraintes_testees.append(nouvelle_contrainte)
    return contraintes_testees



def check_functional_dependencies(df, constraints):
    valid_constraints = []

    for constraint in constraints:
        lhs, rhs = constraint.split(" → ")
        lhs = lhs.strip().split(', ')
        rhs = rhs.strip().split(', ')

        # Vérification si toutes les colonnes existent dans le DataFrame
        for col in lhs + rhs:
            if col not in df.columns:
                raise ValueError(f"La colonne '{col}' n'existe pas dans le DataFrame.")

        # Agréger les données en fonction de lhs et vérifier la dépendance fonctionnelle pour chaque combinaison unique
        grouped_data = df.groupby(lhs)[rhs].nunique()

        # Si le groupe unique a une seule valeur distincte pour chaque colonne rhs,
        # la contrainte est satisfaite pour toutes les lignes du DataFrame.
        if (grouped_data == 1).all().all():  # Ajouter un second .all() ici pour vérifier tous les groupes uniques.
            valid_constraints.append(constraint)

    return valid_constraints
def levenshtein_distance(str1, str2):
    m = len(str1)
    n = len(str2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[m][n]

def most_similar_constraint(erroneous_constraint, constraint_set):
    min_distance = float('inf')
    most_similar = None

    for proposal in constraint_set:
        distance = levenshtein_distance(erroneous_constraint, proposal)
        if distance < min_distance:
            min_distance = distance
            most_similar = proposal

    return most_similar


def Reparer(df, liste_contraintes_erronees, contraintes_systemes):
    # Liste pour stocker les résultats finaux des contraintes réparées
    contraintes_reparees = []

    for contrainte_erronee in liste_contraintes_erronees:
        # Générer les contraintes potentielles à partir de la contrainte erronée et des contraintes système
        contraintes_generees = generer_contrainte(df, contrainte_erronee, contraintes_systemes)

        # Valider les contraintes générées pour déterminer lesquelles sont correctes et applicables
        contraintes_valides = check_functional_dependencies(df, contraintes_generees)

        # Trouver la contrainte valide la plus similaire à la contrainte erronée
     

        if len(contraintes_valides) > 1:
            contrainte_finale = most_similar_constraint(contrainte_erronee, contraintes_valides)
        else:
            contrainte_finale = contraintes_valides[0]  # Prendre la première contrainte valide directement

        # Ajouter la contrainte réparée à la liste des contraintes réparées
        contraintes_reparees.append((contrainte_erronee, contrainte_finale))


      
    return contraintes_reparees












@app.route('/dependances')
def get_correspondances():
    # Convertir le dictionnaire en une réponse JSON valide
    return jsonify(dependances)
@app.route('/dependancesValides')
def get_correspondancess():
    # Convertir le dictionnaire en une réponse JSON valide
    return jsonify(dependancesValides)
@app.route('/verifie_dependances')
def verifier_dependances():
    # Appelez la fonction pour vérifier les dépendances
    resultats = verifie_dependances(data, dependances)

    # Créez un dictionnaire pour stocker les résultats structurés
    resultats_structures = {}

    for FD, lignes_violations in resultats.values:
        # Convertissez les indices des lignes en listes de noms de lignes
        lignes_violations_nom = [data.loc[idx]['Num'] for idx in lignes_violations]

        # Stockez les résultats dans le dictionnaire structuré
        resultats_structures[str(FD)] = lignes_violations_nom

    # Renvoyez le dictionnaire JSON des résultats structurés
    return jsonify(resultats_structures)


@app.route('/classes')
def identifier_source():
    # Appelez la fonction pour vérifier les dépendances
    resultats = classes(data, dependances)
    resultats_structures = {}
    for FD, mesclasses in resultats.values:
      resultats_structures[str(FD)] = mesclasses
   
    return jsonify(resultats_structures)

@app.route('/generer_contrainte')
def generer_contrainte_endpoint():
    # Récupérez la contrainte erronée depuis les paramètres de la requête
    contrainte_erronee = request.args.get('contrainte_erronee')

    # Vous pouvez maintenant appeler votre fonction de traitement
    contraintes_testees = generer_contrainte(data, contrainte_erronee, contraintes_systemes)

    # Convertissez les résultats en JSON
    result_json = {'contraintes_testees': contraintes_testees}

    return jsonify(result_json)




@app.route('/valider')
def valider():
  

  

    
    # Récupérez la contrainte erronée depuis les paramètres de la requête
    contrainte_erronee = request.args.get('contrainte_erronee')
    contraintes_testees = generer_contrainte(data, contrainte_erronee, contraintes_systemes)
    contraintes_valides = check_functional_dependencies(data, contraintes_testees)

    # Convertissez les résultats en JSON
    result_json = {'contraintes_testees': contraintes_valides}

    return jsonify(result_json)

@app.route('/select')
def select():
  
 # Récupérez la contrainte erronée depuis les paramètres de la requête
    contrainte_erronee = request.args.get('contrainte_erronee')
    contraintes_testees = generer_contrainte(data, contrainte_erronee, contraintes_systemes)
    contraintes_valides = check_functional_dependencies(data, contraintes_testees)
    contrainte_finale = most_similar_constraint(contrainte_erronee, contraintes_valides)
    # Convertissez les résultats en JSON
    result_json = {'Réparation suggérée': contrainte_finale}

    return jsonify(result_json)

@app.route('/reparer')
def reparer():
    contrainte_erronee = ["indicatifPays → Zip","Pays → Commune"]
    contraintes_reparees = Reparer(data, contrainte_erronee, contraintes_systemes)
    result_json = []

    for contrainte_erronee, contrainte_reparee in contraintes_reparees:
        result_json.append({
            "Contrainte erronee": contrainte_erronee,
            "Reparation": contrainte_reparee
        })

    return jsonify(result_json)
@app.route('/reparerhyb')
def reparerhyb():


    

    contraintes_reparees = Reparer(data, contrainte_erronee, contraintes_systemes)
    result_json = []

    for contrainte_erronee, contrainte_reparee in contraintes_reparees:
        result_json.append({
            "Contrainte erronee": contrainte_erronee,
            "Reparation": contrainte_reparee
        })

    return jsonify(result_json)
if __name__ == '__main__':
    app.run(debug=True)
