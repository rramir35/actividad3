# Actividad 3 - Métodos de Aprendizaje Supervisado

## Inteligencia Artificial – Sistema Inteligente de Transporte

### Descripción del proyecto

Este proyecto implementa un modelo de aprendizaje supervisado utilizando un Árbol de Decisión para predecir tiempos de viaje en un sistema de transporte masivo.

El sistema utiliza variables como:

* Origen
* Destino
* Distancia en kilómetros
* Hora pico

El objetivo es estimar el tiempo real de desplazamiento entre estaciones.

---

## Dataset utilizado

El dataset fue creado manualmente con información simulada basada en rutas de transporte masivo.

Archivo:

* dataset_transporte.csv

Variables:

* origen
* destino
* distancia_km
* hora_pico
* tiempo_real_min

---

## Algoritmo utilizado

Se implementó un modelo de:

* Árbol de Decisión (DecisionTreeRegressor)

Librerías utilizadas:

* pandas
* numpy
* scikit-learn
* matplotlib

---

## Resultados obtenidos

El modelo permitió realizar predicciones de tiempo de viaje utilizando variables relacionadas con las rutas y condiciones del tráfico.

También se generó una visualización del árbol de decisión:

* arbol_decision.png

---

## Archivos del proyecto

* actividad3.py
* dataset_transporte.csv
* arbol_decision.png

---

## Autor

Rosa Cecilia Ramírez González
