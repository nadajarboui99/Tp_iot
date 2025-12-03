
#PROGRAMME 2 : EDGE RECEIVER (IoT Edge)


import numpy as np
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt
import json
import time
from matplotlib.animation import FuncAnimation


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

# Interpolation linéaire pour reconstruire le signal complet
if signal_original is not None:
    signal_reconstruit = np.interp(temps_original, temps_recu, signal_recu)
    print(f" Signal reconstruit par interpolation linéaire")
    
    # Calculer l'erreur de reconstruction
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    mse = mean_squared_error(signal_original, signal_reconstruit)
    mae = mean_absolute_error(signal_original, signal_reconstruit)
    rmse = np.sqrt(mse)
    
    # Calculer le pourcentage d'erreur relative
    erreur_relative = (mae / np.abs(signal_original).mean()) * 100
    
    print(f"\n Métriques de qualité :")
    print(f"   - MSE  : {mse:.6f}")
    print(f"   - MAE  : {mae:.6f}")
    print(f"   - RMSE : {rmse:.6f}")
    print(f"   - Erreur relative : {erreur_relative:.2f}%")
else:
    signal_reconstruit = None
    print("  Pas de signal original pour comparaison")

# ==========================================
# VISUALISATION
# ==========================================

print("\n" + "=" * 60)
print("GÉNÉRATION DES GRAPHIQUES")
print("=" * 60)

# Déterminer le nombre de points par cycle
if len(signal_recu) > 0 and temps_recu[-1] > 0:
    pts_par_cycle = int(len(signal_recu) / temps_recu[-1])
else:
    pts_par_cycle = "inconnu"

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
    
    # Zoom sur un cycle pour mieux voir
    ax2.set_xlim(0, 1.5)

# Graphique 3 : Signal reconstruit vs Original
if signal_original is not None and signal_reconstruit is not None:
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(temps_original, signal_original, 'b-', linewidth=1.5, 
             label='Signal original', alpha=0.7)
    ax3.plot(temps_original, signal_reconstruit, 'r--', linewidth=1.5, 
             label='Signal reconstruit (interpolation linéaire)', alpha=0.8)
    ax3.set_title('Reconstruction par Interpolation Linéaire', 
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('Temps (s)', fontsize=12)
    ax3.set_ylabel('Amplitude (mV)', fontsize=12)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Ajouter les métriques sur le graphique
    info_text = f"MSE : {mse:.6f}\nMAE : {mae:.6f}\nErreur : {erreur_relative:.2f}%"
    ax3.text(0.02, 0.98, info_text, transform=ax3.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.tight_layout()

# Sauvegarder
filename = f'edge_reconstruction_{pts_par_cycle}pts.png'
plt.savefig(filename, dpi=150)
print(f" Graphique sauvegardé : {filename}")

plt.show()


# ANALYSE FINALE


print("\n" + "=" * 60)
print("ANALYSE FINALE")
print("=" * 60)

if signal_original is not None:
    print("\n Résumé de la reconstruction :")
    print(f"   - Points originaux : {len(signal_original)}")
    print(f"   - Points reçus : {len(signal_recu)}")
    print(f"   - Réduction : {(1 - len(signal_recu)/len(signal_original))*100:.1f}%")
    print(f"   - Qualité (erreur) : {erreur_relative:.2f}%")
    
    if erreur_relative > 20:
        print("\n  QUALITÉ INSUFFISANTE pour usage médical !")
        print("   → Trop peu de points échantillonnés")
        print("   → Détails importants perdus (ondes P, T)")
    elif erreur_relative > 10:
        print("\n QUALITÉ MOYENNE")
        print("   → Forme générale préservée")
        print("   → Certains détails perdus")
    else:
        print("\n QUALITÉ ACCEPTABLE")
        print("   → Signal exploitable pour monitoring basique")

print("\n CONCLUSION PARTIE 1 :")
print("   L'échantillonnage simple (sans IA) entraîne une perte")
print("   d'information importante, même avec 20 points/cycle.")
print("   → La Partie 2 (LSTM) permettra d'améliorer cela !")

print("\n" + "=" * 60)