# coding=utf-8
"""Eigene Änderung (Schritt 1, Task 'Label-Normalisierung').

Warum: Die 6 Zieldimensionen (l_shoulder_z, l_shoulder_y, l_arm_x, l_elbow_y,
l_wrist_z, l_wrist_x) haben unterschiedliche Wertebereiche. Ohne Normalisierung
dominiert beim MSE-Loss die Dimension mit dem größten Wertebereich das
Training, die anderen werden praktisch ignoriert - das ist vermutlich der
Hauptgrund für die schlechten MAE/MSE-Werte aus dem ersten (unnormalisierten)
Eval-Lauf.

Vorgehen:
  - Mean/Std werden NUR aus der Trainings-CSV (FLAGS.csv_path) berechnet, NIE
    aus dem Test-Set - sonst würden Informationen aus dem Testset ins
    Training/die Normalisierung einfließen (Leakage).
  - Die Stats werden zusammen mit dem Checkpoint in
    <model_dir>/label_stats.json gespeichert. So kann --mode=eval (das
    FLAGS.csv_path absichtlich nicht braucht, siehe run.py) dieselben Stats
    wiederverwenden, ohne selbst Zugriff auf die Trainings-CSV zu benötigen.
  - Normalisiert wird NUR der Loss-Term (das Modell lernt also, Werte in
    normalisierter Skala vorherzusagen). Für MAE/MSE-Metriken werden die
    Modell-Outputs in run.py wieder zurückskaliert (denormalisiert), damit
    die Metriken weiterhin in den ursprünglichen Einheiten interpretierbar
    und mit dem ersten, unnormalisierten Lauf vergleichbar bleiben.

Dieses Modul ist bewusst reine numpy/json-Logik (kein TensorFlow), damit es
unabhängig von der tf.data-Pipeline (custom_data.py/dataset_loader.py)
bleibt - die beiden Dateien werden für die Normalisierung NICHT verändert.
"""

import json

from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf

# Eigene Änderung: dieselbe Spaltenreihenfolge wie in dataset_loader.py
# (create_dataset) - muss synchron bleiben, falls sich die Label-Spalten
# dort mal ändern.
LABEL_COLUMNS = [
    'l_shoulder_z', 'l_shoulder_y', 'l_arm_x', 'l_elbow_y', 'l_wrist_z',
    'l_wrist_x'
]


def compute_label_stats(csv_path):
  """Berechnet Mean/Std pro Zieldimension aus einer Label-CSV.

  Args:
    csv_path: Pfad zur (Trainings-)CSV mit den LABEL_COLUMNS.

  Returns:
    (mean, std) als zwei float32 numpy-Arrays der Form [6].
  """
  # Lazy import, damit dieses Modul auch ohne pandas importierbar bleibt,
  # falls es mal isoliert genutzt wird. dataset_loader.py importiert pandas
  # global auf die gleiche Weise.
  import pandas as pd

  df = pd.read_csv(csv_path)
  values = df[LABEL_COLUMNS].to_numpy(dtype='float32')
  mean = values.mean(axis=0)
  std = values.std(axis=0)

  # Eigene Änderung: Sicherheitsnetz falls eine Dimension (nahezu) konstant
  # ist (std=0) - sonst Division durch 0 bei der Normalisierung.
  std = np.where(std < 1e-6, 1.0, std).astype('float32')
  mean = mean.astype('float32')

  return mean, std


def save_label_stats(mean, std, path):
  """Speichert Mean/Std als JSON, z.B. unter <model_dir>/label_stats.json."""
  data = {
      'columns': LABEL_COLUMNS,
      'mean': np.asarray(mean, dtype='float64').tolist(),
      'std': np.asarray(std, dtype='float64').tolist(),
  }
  with tf.io.gfile.GFile(path, 'w') as f:
    json.dump(data, f)
  logging.info('Label-Stats gespeichert unter %s: %s', path, data)


def load_label_stats(path):
  """Lädt Mean/Std, die zuvor mit save_label_stats() gespeichert wurden."""
  with tf.io.gfile.GFile(path, 'r') as f:
    data = json.load(f)
  mean = np.asarray(data['mean'], dtype='float32')
  std = np.asarray(data['std'], dtype='float32')
  return mean, std


def load_label_stats_or_default(path):
  """Wie load_label_stats(), aber mit Fallback statt Crash.

  Eigene Änderung: Für --mode=eval auf einem ÄLTEREN Checkpoint, der noch
  ohne Normalisierung trainiert wurde (also kein label_stats.json im
  model_dir hat), soll die Auswertung trotzdem funktionieren - mean=0/std=1
  ist dann ein No-Op (Output bleibt unverändert), genau das Verhalten von
  vor dieser Änderung.

  Args:
    path: Pfad zu label_stats.json.

  Returns:
    (mean, std), ggf. Default-Werte [0]*6 / [1]*6 falls die Datei fehlt.
  """
  if not tf.io.gfile.exists(path):
    logging.info(
        'Kein label_stats.json unter %s gefunden - vermutlich ein '
        'Checkpoint aus der Zeit vor der Label-Normalisierung. Verwende '
        'mean=0/std=1 (= keine Rückskalierung).', path)
    num_dims = len(LABEL_COLUMNS)
    return (np.zeros(num_dims, dtype='float32'),
            np.ones(num_dims, dtype='float32'))
  return load_label_stats(path)
