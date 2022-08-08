import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import random
import requests
import pickle
from sklearn.preprocessing import OneHotEncoder
from quantile_forest import RandomForestQuantileRegressor
import bridge
import json

with st.spinner('Aquecendo os motores...'):
    user_agent_list = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
    ]

    df = dict.fromkeys(['Marca', 'Tipo', 'Ano', 'UF', 'Km', 'Potencia',
        'Combustivel', 'Cambio', 'Direcao', 'Cor', 'FinalDePlaca', 'Portas',
        'UnicoDono', 'AceitaTrocas', 'ArCondicionado', 'AirBag', 'Alarme',
        'TravaEletrica', 'VidroEletrico', 'Som', 'SensorDeRe', 'CameraDeRe',
        'Blindado', 'Modelo', 'Dias', 'KmLimite', 'KmPorDias',
        'NumOpcionais', 'KmSqrt', 'PotenciaNum', 'PrecoFipe', 'DifMarca', 'DifTipo',
        'DifAno', 'DifPotencia', 'DifNumOpcionais', 'DifCombustivel', 'DifDirecao', 'DifUF',
        'PrecoFipeLog', 'PrecoFipeAjustado', 'KmSqrtPorDias'])

    df_modelos = pd.read_csv('data/modelos_fipe.csv')
    df_anos = pd.read_csv('data/anos_fipe.csv')
    df_precos = pd.read_csv('data/preco_fipe.csv')

    # Caminhão Leve e Buggy foram retirados da lista de tipos
    lista_tipos = ['Passeio', 'Hatch', 'Sedã', 'SUV', 'Pick-up', 'Van/Utilitário', 'Conversível']
    lista_estados = ['RJ', 'AM', 'DF', 'RS', 'PA', 'PR', 'CE', 'SP', 'RO', 'MG', 'PB', 'ES', 'GO', 'SC', 'BA', 'PE', 'MA', 'RN', 'SE', 'AL', 'MS', 'MT', 'TO', 'RR', 'AC', 'PI', 'AP']
    lista_potencia = ['1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9', '2.0 - 2.9', '3.0 - 3.9', '4.0 ou mais']
    lista_cores = ['Branco', 'Prata', 'Preto', 'Cinza', 'Vermelho', 'Azul', 'Verde', 'Amarelo', 'Laranja', 'Outra']
    lista_opcionais = ['Ar condicionado', 'Air bag', 'Alarme', 'Trava elétrica', 'Vidro elétrico', 'Som', 'Sensor de ré', 'Câmera de ré', 'Blindado']

    marca = open('data/marca_v_fipe.json')
    json_marca = json.load(marca)
    tipo = open('data/tipo_v_fipe.json')
    json_tipo = json.load(tipo)
    ano = open('data/ano_v_fipe.json')
    json_ano = json.load(ano)
    potencia = open('data/potencia_v_fipe.json')
    json_potencia = json.load(potencia)
    numopcionais = open('data/numopcionais_v_fipe.json')
    json_numopcionais = json.load(numopcionais)
    combustivel = open('data/combustivel_v_fipe.json')
    json_combustivel = json.load(combustivel)
    direcao = open('data/direcao_v_fipe.json')
    json_direcao = json.load(direcao)
    uf = open('data/uf_v_fipe.json')
    json_uf = json.load(uf)

st.image('data/CARculadora_logo.png', width=304)
st.markdown('Tenha acesso aos preços reais de mercado! Entre com os dados de seu veículo abaixo')

## Input de dados

lista_marcas = df_modelos['Marca'].unique()
marca = st.selectbox("Marca", lista_marcas)
df['Marca'] = marca
cod_marca = df_modelos.loc[df_modelos['Marca']==marca]['CodMarca'].values[0]

lista_modelos = df_modelos.loc[df_modelos['Marca']==marca]['Modelo'].unique()
modelo = st.selectbox("Modelo", lista_modelos)
df['Modelo'] = modelo.split(' ')[0]
cod_modelo = df_modelos.loc[(df_modelos['Marca']==marca)&(df_modelos['Modelo']==modelo)]['CodModelo'].values[0]

lista_anos = df_anos.loc[(df_anos['Marca']==marca)&(df_anos['Modelo']==modelo.upper())]

# Flag caso os anos do modelo estejam disponíveis em "anos_fipe.csv"
na_fipe = False

if len(lista_anos)==0: # Os anos do modelo não foram consultados
    # Realiza a consulta do modelo
    url_ano = 'https://parallelum.com.br/fipe/api/v1/carros/marcas/'+str(cod_marca)+'/modelos/'+str(cod_modelo)+'/anos'
    user_agent = random.choice(user_agent_list)
    headers = { 
        'User-Agent'      : user_agent, 
        'Referer'         : 'https://www.google.com.br/',
        'Accept'          : 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8', 
        'Accept-Language' : 'en-US,en;q=0.5',
        'DNT'             : '1', 
        'Connection'      : 'close'
    }
    try:
        source_ano = requests.get(url_ano, headers=headers, timeout=random.randint(5,10))
        df_ano = pd.DataFrame(source_ano.json()).rename({'nome':'Ano', 'codigo':'CodAno'}, axis=1)
        df_ano['Marca'] = marca
        df_ano['CodMarca'] = cod_marca
        df_ano['Modelo'] = modelo.upper()
        df_ano['CodModelo'] = cod_modelo
        df_anos = df_anos.append(df_ano, ignore_index=True)
        #df_anos.to_csv('data/anos_fipe.csv', index=False) # Retirar caso não seja possível salvar arquivos
        
        lista_anos = df_anos.loc[(df_anos['Marca']==marca)&(df_anos['Modelo']==modelo.upper())]
        na_fipe = True
    except:
        st.markdown('O modelo não foi encontrado na Tabela Fipe')
else: # Os anos do modelo já foram consultados 
    na_fipe = True

if na_fipe:
    col1, col2 = st.columns(2)
    with col1:
        ano_gas = st.selectbox("Ano/Combustível", lista_anos)
        ano = int(ano_gas.split(' ')[0])
        df['Ano'] = ano
        cod_ano = df_anos.loc[(df_anos['Marca']==marca)&(df_anos['Modelo']==modelo.upper())&(df_anos['Ano']==ano_gas)]['CodAno'].values[0]
    with col2:
        st.caption("Flex")
        flex = st.checkbox("")
        combustivel = ano_gas.split(' ')[1]
        combustivel = 'Flex' if flex else combustivel
        df['Combustivel'] = combustivel

    st.markdown('A Tabela Fipe termina aqui. Mas a gente pode te oferecer uma previsão ainda melhor com as informações abaixo!')
    
    tipo = st.radio('Tipo de carroceria', lista_tipos, horizontal=True)
    df['Tipo'] = tipo

    col1, col2 = st.columns(2)
    with col1:
        uf = st.selectbox("Estado", lista_estados)
        df['UF'] = uf

        km = st.number_input("Quilometragem", min_value=0, max_value=500000, step=1)
        df['Km'] = km
        
        potencia = st.select_slider("Potência", lista_potencia)
        df['Potencia'] = potencia

        cor = st.selectbox('Cor', lista_cores)
        df['Cor'] = cor
    with col2:
        final_de_placa = st.number_input('Final de placa', min_value=0, max_value=9, step=1)
        df["FinalDePlaca"] = final_de_placa

        direcao = st.selectbox("Direção", ['Mecânica', 'Hidráulica', 'Elétrica', 'Assistida'])
        df['Direcao'] = direcao

        cambio = st.select_slider("Câmbio", ['Manual', 'Semi-Automático', 'Automático'])
        df['Cambio'] = cambio

        portas = st.selectbox("Portas", ['2 portas', '4 portas'])
        df['Portas'] = int(portas.split(' ')[0])

    unico_dono = st.checkbox("Único Dono") 
    df['UnicoDono'] = 1 if unico_dono else 0

    aceita_trocas = st.checkbox("Aceita Trocas")
    df['AceitaTrocas'] = 1 if aceita_trocas else 0

    opcionais = st.multiselect("Opcionais", lista_opcionais)
    df['ArCondicionado'] = 1 if 'Ar condicionado' in opcionais else 0
    df['AirBag'] = 1 if 'Air bag' in opcionais else 0
    df['Alarme'] = 1 if 'Alarme' in opcionais else 0
    df['TravaEletrica'] = 1 if 'Trava elétrica' in opcionais else 0
    df['VidroEletrico'] = 1 if 'Vidro elétrico' in opcionais else 0
    df['Som'] = 1 if 'Som' in opcionais else 0
    df['SensorDeRe'] = 1 if 'Sensor de ré' in opcionais else 0
    df['CameraDeRe'] = 1 if 'Câmera de ré' in opcionais else 0
    df['Blindado'] = 1 if 'Blindado' in opcionais else 0

    ## Feature Engineering
    data_coleta = '04/07/2022'
    data_coleta = datetime.strptime(data_coleta, '%d/%m/%Y')
    dias = (data_coleta - datetime.strptime(f'01/07/{ano-1}', '%d/%m/%Y')).days
    df['Dias'] = dias

    max_km_dia = 350
    df['KmLimite'] = dias*max_km_dia
    df['KmPorDias'] = df['Km']/df['Dias']
    df['NumOpcionais'] = len(opcionais)
    df['KmSqrt'] = np.sqrt(df['Km'])
    df['PotenciaNum'] = float(df['Potencia'].replace('2.0 - 2.9', '2.5').replace('3.0 - 3.9', '3.5').replace('4.0 ou mais', '4.0'))
    df['KmSqrtPorDias'] = df['KmSqrt']/df['Dias']

    try:
        df['DifMarca'] = json_marca['DifPercentual'][df['Marca']]
    except:
        df['DifMarca'] = 0
    try:
        df['DifTipo'] = json_tipo['DifPercentual'][df['Tipo']]
    except:
        df['DifTipo'] = 0
    try:
        df['DifAno'] = json_ano['DifPercentual'][df['Ano']]
    except:
        df['DifAno'] = 0
    try:
        df['DifPotencia'] = json_potencia['DifPercentual'][df['Potencia']]
    except:
        df['DifPotencia'] = 0
    try:
        df['DifNumOpcionais'] = json_numopcionais['DifPercentual'][df['NumOpcionais']]
    except:
        df['DifNumOpcionais'] = 0
    try:
        df['DifCombustivel'] = json_combustivel['DifPercentual'][df['Combustivel']]
    except:
        df['DifCombustivel'] = 0
    try:
        df['DifDirecao'] = json_direcao['DifPercentual'][df['Direcao']]
    except:
        df['DifDirecao'] = 0
    try:
        df['DifUF'] = json_marca['DifPercentual'][df['UF']]
    except:
        df['DifUF'] = 0

    ## Processamento
    col1, col2, col3, col4, col5 = st.columns(5)
    with col5:
        calcular = st.button("Clique para calcular!")

    if calcular == True:
        chave_fipe = str(cod_marca) + ' ' + str(cod_modelo) + ' ' + cod_ano
        if chave_fipe in df_precos['ChaveFipe'].unique():
            preco_fipe = df_precos.loc[df_precos['ChaveFipe']==chave_fipe]['PrecoFipe'].values[0]
            df['PrecoFipe'] = float(preco_fipe.split(' ')[1].replace('.', '').replace(',', '.'))
        else: # Não está na base
            url_preco = 'https://parallelum.com.br/fipe/api/v1/carros/marcas/'+str(cod_marca)+'/modelos/'+str(cod_modelo)+'/anos/'+cod_ano
            user_agent = random.choice(user_agent_list)
            headers = { 
                'User-Agent'      : user_agent, 
                'Referer'         : 'https://www.google.com.br/',
                'Accept'          : 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8', 
                'Accept-Language' : 'en-US,en;q=0.5',
                'DNT'             : '1',
                'Connection'      : 'close'
            }
            try:
                source_preco = requests.get(url_preco, headers=headers, timeout=random.randint(5,10))
                preco = source_preco.json()['Valor']
                df_precos = df_precos.append({'CodMarca': cod_marca, 'CodModelo': cod_modelo, 'CodAno': cod_ano, 'PrecoFipe': preco, 'ChaveFipe': str(cod_marca) + ' ' + str(cod_modelo) + ' ' + cod_ano}, ignore_index=True)
                #df_precos.to_csv('data/preco_fipe.csv', index=False)  # Retirar caso não seja possível salvar arquivos
                preco_fipe = df_precos.loc[df_precos['ChaveFipe']==chave_fipe]['PrecoFipe'].values[0]
                df['PrecoFipe'] = float(preco_fipe.split(' ')[1].replace('.', '').replace(',', '.'))
            except:
                st.markdown('O modelo não foi encontrado na Tabela Fipe')

        df['PrecoFipeLog'] = np.log(df['PrecoFipe'])
        df['PrecoFipeAjustado'] = df['PrecoFipe']*(1+df['DifMarca']+df['DifTipo']+df['DifAno']+df['DifPotencia']+df['DifNumOpcionais']+df['DifCombustivel']+df['DifDirecao']+df['DifUF'])

        bridge.df = df
        bridge.button_click = True

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
        rfqr = pickle.load(open('data/rfqr.pkl', 'rb'))
        y_pred = rfqr.predict(X, quantiles=[0.05, 0.5, 0.95])
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<span style="color:green">**Mínimo**</span>', True)
            preco_min = 'R$ ' + str(round(y_pred[0][0],2)).replace('.',',')
            st.subheader(preco_min)
        with col2:
            st.markdown('**Preço calculado**')
            preco_final = 'R$ ' + str(round(y_pred[0][1],2)).replace('.',',')
            st.header(preco_final)

            st.markdown('<span style="color:orange">**Tabela Fipe**</span>', True)
            st.subheader(preco_fipe.replace('.',''))
        with col3:
            st.markdown('<span style="color:red">**Máximo**</span>', True)
            preco_max = 'R$ ' + str(round(y_pred[0][2],2)).replace('.',',')
            st.subheader(preco_max)