import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Título da página
st.title('Robo-Advisor para Investimentos')

# Entrada de dados do usuário
user_input = st.text_input("Insira seus dados de investimento", "100,200,150,1.5")

# Processamento dos dados de entrada
data = np.array([float(x) for x in user_input.split(',')]).reshape(1, -1)

# Carregar dados históricos (exemplo simplificado)
# Em um caso real, você carregaria de uma fonte confiável ou banco de dados
df = pd.DataFrame({
    'open': np.random.rand(100) * 100,
    'close': np.random.rand(100) * 100
})

# Mostrar dados históricos
st.write("Dados Históricos de Investimento", df)

# Botão para realizar previsões
if st.button('Prever Investimentos Futuros'):
    X_train, X_test, y_train, y_test = train_test_split(df[['open']], df['close'], test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(data)
    st.write(f'Previsão de fechamento: {prediction[0]}')

# Rodar o Streamlit
# Para executar este script, salve-o como `robo_advisor.py` e execute `streamlit run robo_advisor.py` no terminal.