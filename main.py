import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
import os

# Creiamo un esempio di DataFrame
data = pd.read_csv("./data/merged_dataset_300.csv")

watchlist_dataset = pd.DataFrame(data)

# Trasformiamo la colonna 'Generi' in variabili dummy
mlb = MultiLabelBinarizer()
generi_dummy = pd.DataFrame(mlb.fit_transform(watchlist_dataset['Genere'].str.split(',')), columns=mlb.classes_)

# Uniamo le variabili dummy con la colonna 'AnnoMedio'
X = pd.concat([generi_dummy, watchlist_dataset['MediaAnno']], axis=1)

# Standardizziamo i dati
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applichiamo l'algoritmo K-Means
kmeans = KMeans(n_clusters=20, random_state=42) #0.11028786040103278

watchlist_dataset['Cluster'] = kmeans.fit_predict(X_scaled)

def format_json(input_json):
    # Manipola il valore associato alla chiave 'Genere' per ottenere il formato desiderato
    input_json['Genere'] = input_json['Genere'].strip('[]').split(',')
    # Rimuove eventuali spazi bianchi attorno ai generi
    input_json['Genere'] = [genre.strip() for genre in input_json['Genere']]
    return input_json

def rimuovi_ridondanze_generi(input_utente):
    # Rimuovi le ridondanze dai generi
    generi_unici = list(set(input_utente['Genere']))
    input_utente_no_ridondanze = {'Genere': generi_unici, 'MediaAnno': input_utente['MediaAnno']}
    return input_utente_no_ridondanze

def raccomanda_film(input_utente):
    input_utente = format_json(input_utente)

    # Rimuovi ridondanze dai generi
    input_utente = rimuovi_ridondanze_generi(input_utente)

    # Trasformiamo i generi dell'utente in variabili dummy
    generi_utente = pd.DataFrame(mlb.transform([input_utente['Genere']]), columns=mlb.classes_)

    # Creiamo un DataFrame per l'utente con la stessa struttura di X (features originali)
    input_utente_df = pd.DataFrame(0, columns=X.columns, index=[0])

    # Aggiungiamo le variabili dummy
    input_utente_df[generi_utente.columns] = generi_utente.values

    # Aggiungiamo l'anno medio
    input_utente_df['MediaAnno'] = input_utente['MediaAnno']

    # Standardizziamo i dati
    input_utente_scaled = scaler.transform(input_utente_df)

    # Otteniamo il cluster a cui appartiene l'utente
    cluster_utente = kmeans.predict(input_utente_scaled.reshape(1, -1))

    # Restituisci tutti gli id film nel cluster (senza ridondanze)
    id_film_senza_ridondanze = watchlist_dataset[watchlist_dataset['Cluster'] == cluster_utente[0]]['ID_Film'].str.split(',').explode().unique().tolist()

    return id_film_senza_ridondanze, cluster_utente[0]


def scrivi_su_file(json_input):
    file_path = './data/Valutazioni.csv'

    # Converti le stringhe di generi e film consigliati in liste
    generi = json_input['Genere']
    consigliati = json_input['Consigliati']

    # Creiamo un DataFrame con i dati dell'utente
    df = pd.DataFrame({
        'Genere': [generi],
        'MediaAnno': [json_input['MediaAnno']],
        'Valutazione': [json_input['Valutazione']],
        'Consigliati': [consigliati]
    })

    # Scrivi il DataFrame su un file CSV
    try:
        if not os.path.exists(file_path):
            df.to_csv(file_path, index=False)
        else:
            df.to_csv(file_path, mode='a', header=False, index=False)
        return "Grazie del feedback!"
    except Exception as e:
        print(f"Si è verificato un errore durante la scrittura del file: {e}")
        return "Non siamo riusciti a registrare il tuo feedback! Riprova più tardi."
