# actividad3_transporte_supervisado.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from datetime import datetime

# ============================================
# 1. CREAR DATASET (basado en el sistema de transporte)
# ============================================
print("="*50)
print("🚍 ACTIVIDAD 3 - APRENDIZAJE SUPERVISADO")
print(f"📅 Ejecutado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*50)

# Datos basados en las conexiones reales de la Actividad 2
datos = {
    "origen": ["Portal Norte", "Calle 100", "Calle 72", "Calle 26", "Portal Norte", "Calle 100", "Calle 72", "Calle 26", "Avenida Jiménez"],
    "destino": ["Calle 100", "Calle 72", "Calle 26", "Portal Sur", "Calle 72", "Calle 26", "Calle 100", "Avenida Jiménez", "Calle 26"],
    "distancia_km": [4.0, 3.5, 2.8, 5.0, 6.5, 4.2, 3.5, 1.5, 1.8],
    "hora_pico": [1, 0, 0, 1, 1, 0, 0, 1, 0],  # 1 = sí, 0 = no
    "tiempo_real_min": [8, 6, 5, 12, 14, 7, 6, 11, 7]
}

df = pd.DataFrame(datos)
print("\n📊 DATASET DE ENTRENAMIENTO:")
print(df)

# Guardar dataset (requisito: archivo de fuentes de datos)
df.to_csv("dataset_transporte.csv", index=False)
print("\n✅ Dataset guardado como 'dataset_transporte.csv'")

# ============================================
# 2. DESCRIPCIÓN DE LOS DATOS
# ============================================
print("\n" + "="*50)
print("📋 DESCRIPCIÓN DE LOS DATOS:")
print("="*50)
print(f"- Número de registros: {len(df)}")
print(f"- Características: origen, destino, distancia_km, hora_pico")
print(f"- Variable objetivo: tiempo_real_min (minutos)")
print("- Tipo de problema: Regresión (predecir tiempo de viaje)")
print("\nEstadísticas descriptivas:")
print(df.describe())

# ============================================
# 3. MODELO DE APRENDIZAJE SUPERVISADO
# ============================================
# Codificar variables categóricas (origen y destino)
origenes = {est: i for i, est in enumerate(df["origen"].unique())}
destinos = {est: i for i, est in enumerate(df["destino"].unique())}

df["origen_cod"] = df["origen"].map(origenes)
df["destino_cod"] = df["destino"].map(destinos)

# Características (X) y variable objetivo (y)
X = df[["origen_cod", "destino_cod", "distancia_km", "hora_pico"]]
y = df["tiempo_real_min"]

# Dividir en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar modelo (Árbol de Decisión - fácil de explicar)
modelo = DecisionTreeRegressor(max_depth=3, random_state=42)
modelo.fit(X_train, y_train)

print("\n" + "="*50)
print("🤖 MODELO ENTREÑADO: Árbol de Decisión")
print("="*50)
print(f"- Precisión en entrenamiento: {modelo.score(X_train, y_train):.2f}")
print(f"- Precisión en prueba: {modelo.score(X_test, y_test):.2f}")

# Predicciones
y_pred = modelo.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📈 MÉTRICAS DE RENDIMIENTO:")
print(f"- Error absoluto medio (MAE): {mae:.2f} minutos")
print(f"- Coeficiente R²: {r2:.2f}")

# ============================================
# 4. PRUEBAS REALIZADAS
# ============================================
print("\n" + "="*50)
print("🔍 PRUEBAS REALIZADAS:")
print("="*50)

# Prueba 1: Predicción con datos existentes
print("\n✅ Prueba 1 - Predicción con datos de entrenamiento:")
for i in range(min(3, len(X_test))):
    pred = modelo.predict([X_test.iloc[i]])[0]
    real = y_test.iloc[i]
    print(f"   Predicción: {pred:.1f} min | Real: {real} min | Diferencia: {abs(pred-real):.1f} min")

# Prueba 2: Nueva ruta (Calle 100 → Portal Sur)
print("\n✅ Prueba 2 - Nueva ruta (Calle 100 → Portal Sur):")
nueva_ruta = pd.DataFrame({
    "origen_cod": [origenes.get("Calle 100", 0)],
    "destino_cod": [destinos.get("Portal Sur", 0)],
    "distancia_km": [12.0],
    "hora_pico": [1]
})
tiempo_predicho = modelo.predict(nueva_ruta)[0]
print(f"   Tiempo predicho: {tiempo_predicho:.1f} minutos")

# Prueba 3: Hora valle vs hora pico
print("\n✅ Prueba 3 - Comparación hora valle vs hora pico:")
ruta_valle = pd.DataFrame({
    "origen_cod": [origenes.get("Portal Norte", 0)],
    "destino_cod": [destinos.get("Calle 100", 0)],
    "distancia_km": [4.0],
    "hora_pico": [0]  # hora valle
})
ruta_pico = pd.DataFrame({
    "origen_cod": [origenes.get("Portal Norte", 0)],
    "destino_cod": [destinos.get("Calle 100", 0)],
    "distancia_km": [4.0],
    "hora_pico": [1]  # hora pico
})
tiempo_valle = modelo.predict(ruta_valle)[0]
tiempo_pico = modelo.predict(ruta_pico)[0]
print(f"   Hora valle: {tiempo_valle:.1f} minutos")
print(f"   Hora pico: {tiempo_pico:.1f} minutos")
print(f"   Diferencia: {tiempo_pico - tiempo_valle:.1f} minutos adicionales")

# ============================================
# 5. GUARDAR VISUALIZACIÓN DEL ÁRBOL
# ============================================
plt.figure(figsize=(12, 8))
plot_tree(modelo, feature_names=["origen_cod", "destino_cod", "distancia_km", "hora_pico"], 
          filled=True, rounded=True)
plt.title("Árbol de Decisión para Predicción de Tiempo de Viaje")
plt.savefig("arbol_decision.png", dpi=150, bbox_inches="tight")
print("\n📊 Visualización del árbol guardada como 'arbol_decision.png'")

# ============================================
# RESUMEN FINAL
# ============================================
print("\n" + "="*50)
print("📋 RESUMEN DEL SISTEMA SUPERVISADO")
print("="*50)
print("✅ Algoritmo: Árbol de Decisión (Regresión)")
print("✅ Dataset creado con 9 registros")
print("✅ Variables: distancia_km, hora_pico")
print("✅ Objetivo: predecir tiempo_real_min")
print("✅ Métricas: MAE y R²")
print("✅ Archivos generados:")
print("   - dataset_transporte.csv")
print("   - arbol_decision.png")
print("="*50)
print("🏁 FIN DE LA EJECUCIÓN")