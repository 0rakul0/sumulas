# coding: utf-8
import re
import pandas as pd
import datetime

def is_text_valid(text):
    return isinstance(text, str) and any(c.isalnum() for c in text.strip())

with open('./data/sumulasSTF.txt', encoding='utf-8') as f:
    sumula = f.read()

sumulas_sem_quebras = sumula.replace('\n', '')
sumulas = sumulas_sem_quebras.split('SÚMULA')

dados = []

for match in sumulas:
    match = re.search(r'(\d+) (.*) Data de Aprovação Sessão Plenária de (\d{2}/\d{2}/\d{4})', match)
    try:
        id_sumula = match.group(1)
        texto_sumula = match.group(2)
        data_aprovacao = match.group(3)
        data_aprovacao = datetime.datetime.strptime(data_aprovacao, '%d/%m/%Y')
        texto_sumula = re.split('(\d{2}/\d{2}/\d{4})', texto_sumula)[0]
        id_sumula = f'sumula {id_sumula}'
        if is_text_valid(texto_sumula):
            dados.append([id_sumula, texto_sumula, data_aprovacao])
        else:
            dados.append([id_sumula, "Texto da Súmula não encontrado ou não é válido.", data_aprovacao])
    except Exception as e:
        print(f"Erro ao processar a súmula: {e}")
        pass

df = pd.DataFrame(dados, columns=['ID', 'Texto da Súmula', 'Data de Aprovação'])

df.to_csv('./data/sumulas.csv', index=False, encoding='utf-8')

print("Arquivo CSV 'sumulas.csv' criado com sucesso!")
