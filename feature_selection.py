# Wir haben 81 Features, von denen viele Features berechnet werden durch
# Features, die bereits da sind.
# Das ist zwar zum Verständnis sinnvoll, das Netzwerk braucht aber eigentlich
# nur die minimale Anzahl an Features und soll daraus eigene Features
# ermitteln.
# Daher müssen wir eine FEATURE SELECTION betreiben.
# Das machen wir mit SCIKIT-LEARN.

# Wir testen auf RELEVANZ und REDUNDANZ, d.h.:
#
# * Check für Relevanz: Alle Dimensionen extrahieren mit einer Varianz nahe an
#   Null. Das bedeutet nämlich, dass diese Dimensionen keine Informationen
#   enthalten. Sie sind immer gleich.
# * Check für Redundanz: Berechne Korrelation aller Dimensionen und aller
#   Beispiele. Feature Paare mit einer hohen (nahe bei 1) oder niedrigen (nahe
#   bei -1) Korrelation, enthalten keine weiteren Informationen und daher kann
#   eines der beiden Features exkludiert werden.
#
# Andere Feature Selection Probleme sind nur schwer anzuwenden, da wir Features
# von Regionen betrachten und nicht globale Features auf einem Bild. Wir können
# mit Hilfe einer modellbasierten Feature Selection daher nicht sagen, welche
# Features für das entsprechende Modelproblem relevant sind.

# KOVARIANZMATRIX:
#
# Die Kovarianzmatrix als Matrix aller paarweisen Kovarianzen der Elemente
# enthält Informationen über seine Streuung (Varianz) und über Korrelationen
# zwischen seinen Komponenten.
# Diese kann für einen Featurevector berechnet werden bzw. für alle und daraus
# der Mean Kovarianzmatrix berechnet werden. Zero Feature vectors sollten nicht
# berücksichtigt werden.
#
# Es gibt anscheinend verschiedene Arten der Kovarianzberechnung:
#
# * Sparse inverse covariance:
#   Die Inverse der Kovarianzmatrix heißt Precision Matrix und gibt an, ob 2
#   Features konditionell unabhängig voneinander sind. Dann ist deren Wert in
#   der Matrix gleich 0.

# Wir testen das ganze mal:

import tensorflow as tf
import numpy as np

a = np.array([
    [10, 0.8, 0.2],
    [20, 1, -0.2],
    [30, 1.2, 0],
    [40, 1.3, 0],
    ])

cov = np.corrcoef(a, rowvar=False)
print(cov)
