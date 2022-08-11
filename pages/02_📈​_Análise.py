import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from quantile_forest import RandomForestQuantileRegressor
import bridge

df_precos = pd.DataFrame(columns=['PrecoMin', 'PrecoMed', 'PrecoMax'])

st.image('data/CARculadora_logo.png', width=304)

if bridge.button_click:
    
    st.title('Análise do preço')
    st.markdown('Aqui está a previsão do modelo para os próximos doze meses! **Mas fique atento**:')
    st.caption('- O processo inflacionário e diversas outras variáveis não são consideradas na análise;')
    st.caption('- Não funciona corretamente em veículos muito antigos.')

    df = bridge.df

    with st.spinner('Aquecendo os motores...'):
        lista_drop = ['Marca', 'Tipo', 'Ano', 'Potencia', 'Portas', 'Modelo', 'Dias', 'KmLimite']
        for var in lista_drop:
            df.pop(var, None)

        X = pd.DataFrame(columns=df.keys())
        X = X.append(df, ignore_index=True)
            
        # One Hot Encoder
        var_ohe = ['UF', 'Combustivel', 'Cambio', 'Direcao', 'Cor']
        ohe = pickle.load(open('data/ohe.pkl', 'rb'))
        df_ohe = pd.DataFrame(ohe.transform(X[var_ohe]).toarray(), columns=ohe.get_feature_names_out(), index=X.index)
        X = X.join(df_ohe)
        X = X.drop(var_ohe, axis=1)

        # Random Forest Quantile Regression
        rfqr = bridge.model
        y_pred = rfqr.predict(X, quantiles=[0.05, 0.5, 0.95])

        df_precos = df_precos.append({'PrecoMin': y_pred[0][0], 'PrecoMed': y_pred[0][1], 'PrecoMax': y_pred[0][2]}, ignore_index=True)
        
        KmDiasDefault = 35 # O brasileiro roda em média 35 km por dia
        KmDiasMin = 10 # Será considerada uma quilometragem mínima diária para os cálculos
        if df['KmPorDias'] < KmDiasMin: df['KmPorDias'] = KmDiasMin
        
        for i in range(1,13): # Ao longo de um ano
            # Um carro perde cerca de 1,67% ao mês
            df['PrecoFipe'] = df['PrecoFipe']*0.9833
            df['PrecoFipeLog'] = np.log(df['PrecoFipe'])
            df['PrecoFipeAjustado'] = df['PrecoFipe']*(1+df['DifMarca']+df['DifTipo']+df['DifAno']+df['DifPotencia']+df['DifNumOpcionais']+df['DifCombustivel']+df['DifDirecao']+df['DifUF'])
            df['Km'] = df['Km'] + 30*df['KmPorDias']
            df['KmSqrt'] = np.sqrt(df['Km'])
            df['KmSqrtPorDias'] = df['KmSqrt']/(df['Km']/df['KmPorDias'])

            X = pd.DataFrame(columns=df.keys())
            X = X.append(df, ignore_index=True)
            
            # One Hot Encoder
            var_ohe = ['UF', 'Combustivel', 'Cambio', 'Direcao', 'Cor']
            ohe = pickle.load(open('data/ohe.pkl', 'rb'))
            df_ohe = pd.DataFrame(ohe.transform(X[var_ohe]).toarray(), columns=ohe.get_feature_names_out(), index=X.index)
            X = X.join(df_ohe)
            X = X.drop(var_ohe, axis=1)

            # Random Forest Quantile Regression
            rfqr = bridge.model
            y_pred = rfqr.predict(X, quantiles=[0.05, 0.5, 0.95])

            df_precos = df_precos.append({'PrecoMin': y_pred[0][0], 'PrecoMed': y_pred[0][1], 'PrecoMax': y_pred[0][2]}, ignore_index=True)

    st.header('Evolução do preço')
    st.line_chart(data=df_precos, width=0, height=0, use_container_width=True)

    df_precos['PrecoMes-1'] = df_precos['PrecoMed'].shift()
    df_precos['Delta'] = df_precos['PrecoMed']/df_precos['PrecoMes-1']-1

    st.header('Traduzindo os dados')
    delta_max = round(df_precos['PrecoMed'].loc[12] - df_precos['PrecoMed'].loc[0],2)
    delta_max_percent = round((int(delta_max)/df_precos['PrecoMed'].loc[0])*100,2)
    mes_desv = df_precos['Delta'].idxmin()
    st.markdown('- Seu veículo pode desvalorizar R$ ' + str(abs(delta_max)) + ' ao longo de 12 meses;')
    st.markdown('- Essa diferença representa ' + str(abs(delta_max_percent)) +'\% do valor de seu carro;')
    st.markdown('- A maior desvalorização do automóvel ocorrerá do mês ' + str(mes_desv-1) + ' para o mês ' + str(mes_desv) + ';')
    st.markdown('- A desvalorização nesse mês será de ' + str(abs(round(100*df_precos['Delta'].loc[mes_desv],2))) +'\%.')

else:
    st.header('Ops! Ainda não recebemos seus dados')
    st.markdown('Preencha os dados de seu veículo na nossa CARculadora!')
