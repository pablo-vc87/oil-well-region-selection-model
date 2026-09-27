import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def linear_regresion_model_trainer(
        dataset, target_col, 
        feature_cols, verbose=False, 
        test_size=0.25, random_state=54321
):
    """
    Función que entrena un modelo de regresión lineal 
    y entrega un diccionario con predicciones, target,
    el volumen medio de reservas y RCME del modelo; 
        - dataset: dataframe con los datos a usar
        - target_col: nombre de la columna objetivo
        - feature_cols: lista con los nombres de las columnas de características
        - verbose: booleano para imprimir información adicional, 
        volúmen medio y RCME del modelo
        - test_size: proporción del conjunto de datos para validación
        - random_state: semilla para generación aleatoria
    """
    features = dataset[feature_cols]
    target = dataset[target_col]
    train_features, valid_features, train_target, valid_target = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )
    model = LinearRegression()
    model.fit(train_features, train_target) 
    predictions_valid = model.predict(valid_features) 
    rmse = mean_squared_error(valid_target, predictions_valid)**0.5 
    if verbose:
        print(f"Volumen medio de reservas: {dataset[target_col].mean():.2f}")
        print(f"RECM del modelo de regresión lineal en el conjunto de validación: {rmse:.2f}")
        print(f"El promedio de la desviación estandar es {rmse/(max(dataset[target_col]) - min(dataset[target_col]))*100:.2f}%")
    return {
        'predictions': predictions_valid,
        'target': valid_target,
        'average_volume': dataset[target_col].mean(),
        'rmse': rmse
    }
#=======================================
def calculate_profit(df_response, total_budget, unit_income, total_wells):
    """Función que calcula la ganancia total basada en las predicciones del modelo, el presupuesto total, la ganancia por unidad de producto y el número total de pozos a perforar.
    Args:
    df_response (dict): Diccionario con las predicciones y los valores reales del target.
    total_budget (float): Presupuesto total disponible.
    unit_income (float): Ganancia por unidad de producto.
    total_wells (int): Número total de pozos a perforar.
    Returns:
    float: Ganancia total calculada.
    """
    min_well_income = total_budget / (unit_income * total_wells)
    df_best = pd.DataFrame(df_response)[['predictions', 'target']].sort_values('predictions', ascending=False).head(total_wells)
    profit = (df_best['target'].sum() * unit_income) - total_budget
    return profit