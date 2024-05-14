from turtle import title
import numpy as np
import pandas as pd
import google.generativeai as genai

GOOGLE_API_KEY = "YOUR_API_KEY"
genai.configure(api_key=GOOGLE_API_KEY)

for m in genai.list_models():
    if 'embedContent' in m.supported_generation_methods:
        print(m.name)

#Exemplo de embedding
title = "A próxima geração de IA para desenvolvedores e Google workspace"
texto_simples = ("Título: A próxima geração de IA para desenvolvedores e Google workspace"
                 "\n"
                 "Artigo completo: \n"
                 "\n"
                 "Gemini API & Google AI Studio: Uma maneira acessivel de explorar e criar protótipos com aplicação de IA generativa")

embeddings = genai.embed_content(model="models/embedding-001",
                                 content=texto_simples,
                                 title=title,
                                 task_type="RETRIEVAL_DOCUMENT") #tipo de cenário que ta trabalhando https://ai.google.dev/gemini-api/tutorials/document_search?hl=pt-br#api_changes_to_embeddings_with_model_embedding-001

#print (embeddings)

#Listagem de documentos que serão buscados
DOCUMENT1 = {
    "Título": "Operando o sistema de controle climático",
    "Conteúdo": "Seu Googlecar possui um sistema de controle de temperatura que permite ajustar a temperatura e o fluxo de ar no carro. Para operar o sistema de controle climático, use os botões e botões localizados no console central. Temperatura: O botão de temperatura controla a temperatura dentro do carro. Gire o botão no sentido horário para aumentar a temperatura ou no sentido anti-horário para diminuir a temperatura. Fluxo de ar: O botão de fluxo de ar controla a quantidade de fluxo de ar dentro do carro. Gire o botão no sentido horário para aumentar o fluxo de ar ou no sentido anti-horário para diminuí-lo. Velocidade do ventilador: O botão de velocidade do ventilador controla a velocidade do ventilador. Gire o botão no sentido horário para aumentar a velocidade do ventilador ou no sentido anti-horário para diminuir a velocidade do ventilador. Modo: O botão de modo permite selecionar o modo desejado. Os modos disponíveis são: Auto: O carro ajustará automaticamente a temperatura e o fluxo de ar para manter um nível confortável. Legal: O carro soprará ar frio para dentro do carro. Calor: O carro soprará ar quente para dentro do carro. Descongelamento: O carro soprará ar quente no pára-brisa para descongelá-lo."}
DOCUMENT2 = {
    "Título": "Touchscreen",
    "Conteúdo": "Seu Googlecar possui uma grande tela sensível ao toque que fornece acesso a uma variedade de recursos, incluindo navegação, entretenimento e controle de temperatura. Para utilizar o display touchscreen, basta tocar no ícone desejado. Por exemplo, você pode tocar no ícone \"Navegação\" para obter rotas até seu destino ou tocar no ícone \"Música\" para reproduzir suas músicas favoritas."}
DOCUMENT3 = {
    "Título": "Mudança de marcha",
    "Conteúdo": "Seu Googlecar possui transmissão automática. Para mudar de marcha, basta mover a alavanca de câmbio para a posição desejada. Estacionar: Esta posição é usada quando você está estacionado. As rodas estão travadas e o carro não pode se mover. Reverso: Esta posição é usada para fazer backup. Neutro: Esta posição é usada quando você está parado em um semáforo ou no trânsito. O carro não está engatado e não se moverá a menos que você pressione o pedal do acelerador. Drive: Esta posição é usada para avançar. Baixo: Esta posição é usada para dirigir na neve ou em outras condições escorregadias."}

documents = [DOCUMENT1, DOCUMENT2, DOCUMENT3]

#Ver a estrutura do data frame
df = pd.DataFrame(documents)
df.columns = ["Título", "Conteúdo"]
print(df.columns)

model = "models/embedding-001"

#Função para retornar os valores de embedding da linha 21
def embed_function(title, text):
    return genai.embed_content(model=model,
                                 content=text,
                                 title=title,
                                 task_type="RETRIEVAL_DOCUMENT")["embedding"]

df["Embeddings"] = df.apply(lambda row: embed_function(row["Título"], row["Conteúdo"]), axis=1)

#Função para realizar a consulta de embedding
def gerar_e_buscar_consulta(consulta, base, model):
    embedding_da_consulta = genai.embed_content(model=model,
                                 content=consulta,
                                 task_type="RETRIEVAL_QUERY")
    
    produtos_escalares = np.dot(np.stack(df["Embeddings"]), embedding_da_consulta["embedding"])

    indice = np.argmax(produtos_escalares)
    return df.iloc[indice]["Conteúdo"]

consulta = "Como faço para trocar marcha em um carro do Google?"

resultado = gerar_e_buscar_consulta(consulta, df, model)
#print(resultado)

generation_config = {
  "temperature": 0,
  "candidate_count": 1
}

prompt = f"Reescreva esse texto de uma forma mais descontraída, sem adicionar informações que não façam parte do texto: {resultado}"

model_2 = genai.GenerativeModel("gemini-1.0-pro", generation_config=generation_config)
response = model_2.generate_content(prompt)
print(response.text)