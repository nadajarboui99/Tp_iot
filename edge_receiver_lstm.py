
#PROGRAMME 2 : EDGE RECEIVER (IoT Edge)


import numpy as np
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt
import json
import time
from matplotlib.animation import FuncAnimation
from tensorflow.keras.models import load_model


# CONFIGURATION


# Configuration MQTT
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "ecg/data"
MQTT_CLIENT_ID = "EdgeReceiver"

# Stockage des données reçues
donnees_recues = {
    'timestamps': [],
    'values': [],
    'indices': []
}

# Flags
reception_terminee = False
total_attendu = 0


# CHARGEMENT DU SIGNAL ORIGINAL


print("=" * 60)
print("EDGE RECEIVER - Réception et reconstruction du signal ECG")
print("=" * 60)

try:
    signal_original = np.load('signal_original.npy')
    temps_original = np.load('temps_original.npy')
    print(f" Signal original chargé : {len(signal_original)} points")
except FileNotFoundError:
    print(" Fichiers originaux non trouvés.")
    print("   Lancez d'abord capteur_ecg.py pour générer les données.")
    signal_original = None
    temps_original = None

# CHARGEMENT DU MODÈLE LSTM
print("\n" + "=" * 60)
print("CHARGEMENT DU MODÈLE LSTM")
print("=" * 60)

try:
    model_lstm = load_model('ecg_lstm_model.keras')
    print("✓ Modèle LSTM chargé avec succès")
    UTILISER_LSTM = True
except Exception as e:
    print(f"✗ Erreur chargement modèle : {e}")
    print("  → Lancer d'abord lstm_training.py")
    model_lstm = None
    UTILISER_LSTM = False
# CALLBACKS MQTT


def on_connect(client, userdata, flags, rc, properties=None):
    """Appelé lors de la connexion au broker"""
    if rc == 0:
        print(f"Connecté au broker MQTT")
        print(f"Abonnement au topic : '{MQTT_TOPIC}'")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f" Échec connexion, code : {rc}")

def on_message(client, userdata, msg, properties=None):
    """Appelé à chaque réception de message"""
    global total_attendu, reception_terminee
    
    try:
        # Décoder le message JSON
        data = json.loads(msg.payload.decode())
        
        # Extraire les données
        timestamp = data['timestamp']
        value = data['value']
        index = data['index']
        total = data.get('total', 0)
        
        # Stocker
        donnees_recues['timestamps'].append(timestamp)
        donnees_recues['values'].append(value)
        donnees_recues['indices'].append(index)
        
        # Mise à jour du total attendu
        if total > 0:
            total_attendu = total
        
        # Afficher progression
        nb_recus = len(donnees_recues['values'])
        if nb_recus % 10 == 0 or nb_recus == total_attendu:
            print(f"  [{nb_recus}/{total_attendu}] échantillons reçus")
        
        # Vérifier si réception terminée
        if total_attendu > 0 and nb_recus >= total_attendu:
            reception_terminee = True
            print(f"\n Réception terminée : {nb_recus} échantillons")
    
    except json.JSONDecodeError:
        print(f"  Erreur décodage JSON : {msg.payload}")
    except Exception as e:
        print(f"  Erreur traitement message : {e}")

def on_subscribe(client, userdata, mid, granted_qos, properties=None):
    """Appelé après abonnement réussi"""
    print(" Abonnement confirmé, en attente de données...\n")

# ==========================================
# CONNEXION MQTT
# ==========================================

print("\n" + "=" * 60)
print("CONNEXION AU BROKER MQTT")
print("=" * 60)

# Créer le client MQTT (compatible v2.0+)
try:
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=MQTT_CLIENT_ID
    )
except:
    # Fallback pour anciennes versions
    client = mqtt.Client(MQTT_CLIENT_ID)

# Configurer les callbacks
client.on_connect = on_connect
client.on_message = on_message
client.on_subscribe = on_subscribe

# Se connecter
try:
    print(f"Connexion à {MQTT_BROKER}:{MQTT_PORT}...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
except Exception as e:
    print(f" Erreur de connexion : {e}")
    print("  Vérifiez que Mosquitto est démarré :")
    print("    brew services start mosquitto")
    exit(1)

# ==========================================
# RÉCEPTION DES DONNÉES
# ==========================================

print("\n" + "=" * 60)
print("RÉCEPTION DES ÉCHANTILLONS")
print("=" * 60)
print("(Lancez maintenant capteur_ecg.py dans un autre terminal)")
print("(Ctrl+C pour arrêter)\n")

# Démarrer la boucle MQTT en arrière-plan
client.loop_start()

# Attendre la réception des données
try:
    while not reception_terminee:
        time.sleep(0.1)
    
    # Petit délai pour s'assurer que tout est reçu
    time.sleep(1)

except KeyboardInterrupt:
    print("\n  Interruption par l'utilisateur")

# Arrêter la boucle MQTT
client.loop_stop()
client.disconnect()

# ==========================================
# VÉRIFICATION DES DONNÉES REÇUES
# ==========================================

print("\n" + "=" * 60)
print("ANALYSE DES DONNÉES REÇUES")
print("=" * 60)

if len(donnees_recues['values']) == 0:
    print(" Aucune donnée reçue !")
    print("   Assurez-vous que capteur_ecg.py est lancé et envoie des données.")
    exit(1)

# Convertir en arrays numpy
temps_recu = np.array(donnees_recues['timestamps'])
signal_recu = np.array(donnees_recues['values'])

print(f" Données reçues : {len(signal_recu)} échantillons")
print(f"   - Durée : {temps_recu[-1]:.2f} secondes")
print(f"   - Amplitude min : {signal_recu.min():.3f} mV")
print(f"   - Amplitude max : {signal_recu.max():.3f} mV")

# ==========================================
# RECONSTRUCTION DU SIGNAL
# ==========================================

print("\n" + "=" * 60)
print("RECONSTRUCTION DU SIGNAL")
print("=" * 60)

if signal_original is not None:
    # 1. RECONSTRUCTION PAR INTERPOLATION LINÉAIRE (méthode simple)
    signal_reconstruit_lineaire = np.interp(temps_original, temps_recu, signal_recu)
    print("✓ Reconstruction par interpolation linéaire")
    
    # Métriques pour interpolation linéaire
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse_lineaire = mean_squared_error(signal_original, signal_reconstruit_lineaire)
    mae_lineaire = mean_absolute_error(signal_original, signal_reconstruit_lineaire)
    rmse_lineaire = np.sqrt(mse_lineaire)
    erreur_relative_lineaire = (mae_lineaire / np.abs(signal_original).mean()) * 100
    
    print(f"\n📊 Métriques SANS LSTM (interpolation linéaire) :")
    print(f"   - MSE  : {mse_lineaire:.6f}")
    print(f"   - MAE  : {mae_lineaire:.6f}")
    print(f"   - RMSE : {rmse_lineaire:.6f}")
    print(f"   - Erreur relative : {erreur_relative_lineaire:.2f}%")
    
    # 2. RECONSTRUCTION AVEC LSTM (méthode intelligente)
    if UTILISER_LSTM and model_lstm is not None:
        print("\n" + "-" * 60)
        print("🧠 RECONSTRUCTION AVEC LSTM")
        print("-" * 60)
        
        sequence_length = 50  # Même valeur que dans l'entraînement
        signal_reconstruit_lstm = []
        
        # Initialiser avec les premiers points reçus interpolés
        signal_interpole = np.interp(temps_original, temps_recu, signal_recu)
        
        # Utiliser LSTM pour prédire point par point
        for i in range(len(temps_original)):
            if i < sequence_length:
                # Pas assez de points historiques, utiliser interpolation
                signal_reconstruit_lstm.append(signal_interpole[i])
            else:
                # Prendre les 50 derniers points
                sequence = np.array(signal_reconstruit_lstm[-sequence_length:])
                sequence = sequence.reshape(1, sequence_length, 1)
                
                # Prédire le point suivant avec LSTM
                prediction = model_lstm.predict(sequence, verbose=0)[0][0]
                signal_reconstruit_lstm.append(prediction)
        
        signal_reconstruit_lstm = np.array(signal_reconstruit_lstm)
        print("✓ Reconstruction LSTM terminée")
        
        # Métriques pour LSTM
        mse_lstm = mean_squared_error(signal_original, signal_reconstruit_lstm)
        mae_lstm = mean_absolute_error(signal_original, signal_reconstruit_lstm)
        rmse_lstm = np.sqrt(mse_lstm)
        erreur_relative_lstm = (mae_lstm / np.abs(signal_original).mean()) * 100
        
        print(f"\n📊 Métriques AVEC LSTM :")
        print(f"   - MSE  : {mse_lstm:.6f}")
        print(f"   - MAE  : {mae_lstm:.6f}")
        print(f"   - RMSE : {rmse_lstm:.6f}")
        print(f"   - Erreur relative : {erreur_relative_lstm:.2f}%")
        
        # Comparaison
        print(f"\n📈 AMÉLIORATION APPORTÉE PAR LSTM :")
        amelioration_mse = ((mse_lineaire - mse_lstm) / mse_lineaire) * 100
        amelioration_mae = ((mae_lineaire - mae_lstm) / mae_lineaire) * 100
        print(f"   - Réduction MSE  : {amelioration_mse:.1f}%")
        print(f"   - Réduction MAE  : {amelioration_mae:.1f}%")
        
        if amelioration_mae > 0:
            print(f"   ✓ LSTM améliore la reconstruction de {amelioration_mae:.1f}% !")
        else:
            print(f"   ⚠ LSTM n'améliore pas significativement (données trop peu échantillonnées)")
    else:
        signal_reconstruit_lstm = None
        print("\n⚠ Reconstruction LSTM non disponible (modèle non chargé)")
else:
    signal_reconstruit_lineaire = None
    signal_reconstruit_lstm = None
    print("✗ Pas de signal original pour comparaison")

 # ==========================================
# VISUALISATION
# ==========================================

print("\n" + "=" * 60)
print("GÉNÉRATION DES GRAPHIQUES")
print("=" * 60)

pts_par_cycle = int(len(signal_recu) / temps_recu[-1]) if len(signal_recu) > 0 else "inconnu"

if UTILISER_LSTM and signal_reconstruit_lstm is not None:
    # VERSION AVEC LSTM : 4 graphiques
    fig = plt.figure(figsize=(16, 12))
    
    # Graphique 1 : Signal reçu
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(temps_recu, signal_recu, 'ro-', markersize=5, linewidth=1.5)
    ax1.set_title(f'1. Signal ECG reçu au Edge ({len(signal_recu)} points échantillonnés)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Temps (s)')
    ax1.set_ylabel('Amplitude (mV)')
    ax1.grid(True, alpha=0.3)
    
    # Graphique 2 : Original vs Échantillons
    ax2 = plt.subplot(4, 1, 2)
    ax2.plot(temps_original, signal_original, 'b-', linewidth=0.8, 
             label='Signal original', alpha=0.6)
    ax2.plot(temps_recu, signal_recu, 'ro', markersize=5, 
             label=f'Échantillons reçus (~{pts_par_cycle} pts/cycle)')
    ax2.set_title('2. Comparaison : Signal Original vs Échantillons Reçus', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Temps (s)')
    ax2.set_ylabel('Amplitude (mV)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2)  # Zoom sur 2 cycles
    
    # Graphique 3 : Reconstruction linéaire
    ax3 = plt.subplot(4, 1, 3)
    ax3.plot(temps_original, signal_original, 'b-', linewidth=1.5, 
             label='Signal original', alpha=0.7)
    ax3.plot(temps_original, signal_reconstruit_lineaire, 'orange', linewidth=1.5, 
             linestyle='--', label='Interpolation linéaire', alpha=0.8)
    ax3.set_title(f'3. Reconstruction SANS LSTM (Erreur: {erreur_relative_lineaire:.2f}%)', 
                  fontsize=14, fontweight='bold', color='orange')
    ax3.set_xlabel('Temps (s)')
    ax3.set_ylabel('Amplitude (mV)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 2)
    
    info_text = f"MSE : {mse_lineaire:.6f}\nMAE : {mae_lineaire:.6f}"
    ax3.text(0.02, 0.98, info_text, transform=ax3.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
    
    # Graphique 4 : Reconstruction LSTM
    ax4 = plt.subplot(4, 1, 4)
    ax4.plot(temps_original, signal_original, 'b-', linewidth=1.5, 
             label='Signal original', alpha=0.7)
    ax4.plot(temps_original, signal_reconstruit_lstm, 'g-', linewidth=1.5, 
             label='Reconstruction LSTM', alpha=0.8)
    ax4.set_title(f'4. Reconstruction AVEC LSTM 🧠 (Erreur: {erreur_relative_lstm:.2f}%)', 
                  fontsize=14, fontweight='bold', color='green')
    ax4.set_xlabel('Temps (s)')
    ax4.set_ylabel('Amplitude (mV)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 2)
    
    info_text = f"MSE : {mse_lstm:.6f}\nMAE : {mae_lstm:.6f}\nAmélioration : {amelioration_mae:.1f}%"
    ax4.text(0.02, 0.98, info_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    filename = f'edge_reconstruction_LSTM_{pts_par_cycle}pts.png'
    
else:
    # VERSION SANS LSTM : 3 graphiques
    fig = plt.figure(figsize=(16, 10))

    # Graphique 1 : Signal reçu (points échantillonnés uniquement)
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(temps_recu, signal_recu, 'ro-', markersize=5, linewidth=1.5, label='Échantillons reçus')
    ax1.set_title(f'Signal ECG reçu au Edge ({len(signal_recu)} points, ~{pts_par_cycle} pts/cycle)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Temps (s)', fontsize=12)
    ax1.set_ylabel('Amplitude (mV)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Graphique 2 : Comparaison Original vs Reçu
    if signal_original is not None:
        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(temps_original, signal_original, 'b-', linewidth=0.8, 
                 label='Signal original (500 pts/s)', alpha=0.6)
        ax2.plot(temps_recu, signal_recu, 'ro', markersize=5, 
                 label=f'Échantillons reçus (~{pts_par_cycle} pts/cycle)')
        ax2.set_title('Comparaison : Signal Original vs Échantillons Reçus', 
                      fontsize=14, fontweight='bold')
        ax2.set_xlabel('Temps (s)', fontsize=12)
        ax2.set_ylabel('Amplitude (mV)', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1.5)

    # Graphique 3 : Signal reconstruit vs Original
    if signal_original is not None and signal_reconstruit_lineaire is not None:
        ax3 = plt.subplot(3, 1, 3)
        ax3.plot(temps_original, signal_original, 'b-', linewidth=1.5, 
                 label='Signal original', alpha=0.7)
        ax3.plot(temps_original, signal_reconstruit_lineaire, 'r--', linewidth=1.5, 
                 label='Signal reconstruit (interpolation linéaire)', alpha=0.8)
        ax3.set_title('Reconstruction par Interpolation Linéaire', 
                      fontsize=14, fontweight='bold')
        ax3.set_xlabel('Temps (s)', fontsize=12)
        ax3.set_ylabel('Amplitude (mV)', fontsize=12)
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # Ajouter les métriques sur le graphique
        info_text = f"MSE : {mse_lineaire:.6f}\nMAE : {mae_lineaire:.6f}\nErreur : {erreur_relative_lineaire:.2f}%"
        ax3.text(0.02, 0.98, info_text, transform=ax3.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    filename = f'edge_reconstruction_{pts_par_cycle}pts.png'

# ==========================================
# ANALYSE FINALE
# ==========================================

print("\n" + "=" * 60)
print("ANALYSE FINALE - PARTIE 2")
print("=" * 60)

if signal_original is not None:
    print("\n📊 Résumé de la reconstruction :")
    print(f"   - Points originaux : {len(signal_original)}")
    print(f"   - Points reçus : {len(signal_recu)}")
    print(f"   - Réduction du flux : {(1 - len(signal_recu)/len(signal_original))*100:.1f}%")
    
    print(f"\n   SANS LSTM (interpolation) : Erreur = {erreur_relative_lineaire:.2f}%")
    
    if UTILISER_LSTM and signal_reconstruit_lstm is not None:
        print(f"   AVEC LSTM (prédiction)    : Erreur = {erreur_relative_lstm:.2f}%")
        print(f"\n   🎯 GAIN LSTM : {amelioration_mae:.1f}% d'amélioration")
        
        if amelioration_mae > 20:
            print("\n   ✅ LSTM apporte une AMÉLIORATION SIGNIFICATIVE !")
        elif amelioration_mae > 5:
            print("\n   ✓ LSTM améliore modérément la reconstruction")
        else:
            print("\n   ⚠ LSTM n'améliore pas beaucoup (trop peu de points échantillonnés)")

print("\n💡 CONCLUSION PARTIE 2 :")
print("   Le modèle LSTM permet de reconstruire intelligemment")
print("   les points manquants en apprenant la forme typique ECG.")
print("   → Réduction du flux de 80-95% avec maintien de la qualité !")
print("   → Applicable pour Edge Computing IoT en temps réel.")

print("\n" + "=" * 60)