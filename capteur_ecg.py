"""
PROGRAMME 1 : CAPTEUR ECG (IoT Sensor)
Génère un signal ECG synthétique, l'échantillonne et l'envoie via MQTT
Tester avec ECHANTILLONS_PAR_CYCLE = 5 puis 20
"""

import numpy as np
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt
import json
import time

# Configuration
DUREE_SECONDES = 5
POINTS_PAR_SECONDE = 500
NOMBRE_CYCLES = 5
ECHANTILLONS_PAR_CYCLE = 20  # Modifier pour les tests (5 ou 20)
NIVEAU_BRUIT = 0.05

# MQTT
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "ecg/data"
MQTT_CLIENT_ID = "CapteurECG_Sensor"

def ecg_synthetique(t):
    """Génère une onde ECG synthétique avec ondes P, Q, R, S, T"""
    return (
        0.1 * np.sin(2 * np.pi * t * 1) +
        -0.15 * np.exp(-((t - 0.25) ** 2) / 0.001) +
        1.0 * np.exp(-((t - 0.3) ** 2) / 0.0005) +
        -0.2 * np.exp(-((t - 0.35) ** 2) / 0.001) +
        0.3 * np.exp(-((t - 0.5) ** 2) / 0.01)
    )

# Génération du signal
print("Génération du signal ECG...")
t = np.linspace(0, 1, POINTS_PAR_SECONDE)
cycle = ecg_synthetique(t)
bruit = NIVEAU_BRUIT * np.random.normal(size=cycle.shape)
cycle_bruite = cycle + bruit
ecg_complet = np.tile(cycle_bruite, NOMBRE_CYCLES)
t_total = np.linspace(0, NOMBRE_CYCLES, len(ecg_complet))

print(f"Signal généré: {len(ecg_complet)} points sur {NOMBRE_CYCLES}s à {POINTS_PAR_SECONDE}Hz")

# Échantillonnage
pas = POINTS_PAR_SECONDE // ECHANTILLONS_PAR_CYCLE
indices_echantillons = np.arange(0, len(ecg_complet), pas)
signal_echantillonne = ecg_complet[indices_echantillons]
temps_echantillonne = t_total[indices_echantillons]

reduction_pourcent = (1 - len(signal_echantillonne) / len(ecg_complet)) * 100
print(f"Échantillonnage: {len(signal_echantillonne)} points ({ECHANTILLONS_PAR_CYCLE}/cycle), réduction: {reduction_pourcent:.1f}%")

# Sauvegarde des signaux
np.save('signal_original.npy', ecg_complet)
np.save('temps_original.npy', t_total)
np.save('signal_echantillonne.npy', signal_echantillonne)
np.save('temps_echantillonne.npy', temps_echantillonne)

# Callbacks MQTT
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connecté au broker MQTT")
    else:
        print(f"Échec connexion MQTT, code: {rc}")

def on_publish(client, userdata, mid):
    pass

# Connexion MQTT
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, MQTT_CLIENT_ID)
client.on_connect = on_connect
client.on_publish = on_publish

try:
    print(f"Connexion à {MQTT_BROKER}:{MQTT_PORT}...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
    time.sleep(1)
except Exception as e:
    print(f"Erreur de connexion: {e}")
    print("Vérifier que Mosquitto est démarré: brew services start mosquitto")
    exit(1)

# Envoi des données
print(f"\nEnvoi de {len(signal_echantillonne)} échantillons vers '{MQTT_TOPIC}'")
print("(Ctrl+C pour arrêter)\n")

compteur = 0
try:
    for i in range(len(signal_echantillonne)):
        message = {
            "timestamp": float(temps_echantillonne[i]),
            "value": float(signal_echantillonne[i]),
            "index": int(i),
            "total": len(signal_echantillonne)
        }
        
        client.publish(MQTT_TOPIC, json.dumps(message))
        
        compteur += 1
        if compteur % 10 == 0 or compteur == len(signal_echantillonne):
            print(f"[{compteur}/{len(signal_echantillonne)}] échantillons envoyés")
        
        time.sleep(0.02 + np.random.uniform(-0.005, 0.005))
    
    print(f"\nTransmission terminée: {compteur} échantillons envoyés")

except KeyboardInterrupt:
    print("\nInterruption par l'utilisateur")

# Visualisation
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(t_total, ecg_complet, 'b-', linewidth=0.8, 
             label='Signal original (avec bruit)', alpha=0.7)
axes[0].plot(temps_echantillonne, signal_echantillonne, 'ro', 
             markersize=5, label=f'Points échantillonnés ({ECHANTILLONS_PAR_CYCLE}/cycle)')
axes[0].set_title(f'Signal ECG - Échantillonnage à {ECHANTILLONS_PAR_CYCLE} points par cycle', 
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel('Temps (s)', fontsize=12)
axes[0].set_ylabel('Amplitude (mV)', fontsize=12)
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

axes[1].plot(temps_echantillonne, signal_echantillonne, 'ro-', 
             markersize=5, linewidth=1.5)
axes[1].set_title('Signal transmis via MQTT (après échantillonnage)', 
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel('Temps (s)', fontsize=12)
axes[1].set_ylabel('Amplitude (mV)', fontsize=12)
axes[1].grid(True, alpha=0.3)

info_text = f"Réduction : {reduction_pourcent:.1f}%\nPoints envoyés : {len(signal_echantillonne)}/{len(ecg_complet)}"
axes[1].text(0.02, 0.98, info_text, transform=axes[1].transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
filename = f'capteur_ecg_{ECHANTILLONS_PAR_CYCLE}pts.png'
plt.savefig(filename, dpi=150)
print(f"Graphique sauvegardé: {filename}")
plt.show()

# Fermeture
time.sleep(0.5)
client.loop_stop()
client.disconnect()
print("Capteur ECG arrêté")