# Imports

# Manipulação de dados e imagens
import os
import cv2
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Pytorch
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Pacotes para o relatório de hardware
import gc
import types
import pkg_resources
import pytorch_lightning as pl

# Seed para reproduzir os mesmos resultados
np.random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed(10)


class OrganizaDados(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y


if __name__ == '__main__':

    ''' Verificando o Ambiente de Desenvolvimento '''

    # Verifica se uma GPU está disponível e define o dispositivo apropriado

    processing_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define o device (GPU ou CPU)
    device = torch.device(processing_device)
    print(device)

    # Mapeamento das Imagens
    # Obtemos todos os caminhos das imagens e fazemos o match com as informações em HAM10000_metadata.csv.

    # Pasta com as imagens (Imgens coletadas do site: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
    pasta_imagens = 'input'

    # Obtém o caminho de cada imagem
    caminho_imagens = glob(os.path.join(pasta_imagens, '*', '*.jpg'))

    # Cria um dicionário
    dict_map_imagem_caminho = {os.path.splitext(os.path.basename(x))[0]: x for x in caminho_imagens}

    # Leitura do arquivo de metadados
    df_original = pd.read_csv(os.path.join(pasta_imagens, 'HAM10000_metadata.csv'))

    # Adiciona o path
    df_original['path'] = df_original['image_id'].map(dict_map_imagem_caminho.get)

    # Tipos de lesões que serão analisadas
    dict_tipo_lesao = {'nv': 'Melanocytic nevi',
                       'mel': 'dermatofibroma',
                       'bkl': 'Benign keratosis-like lesions ',
                       'bcc': 'Basal cell carcinoma',
                       'akiec': 'Actinic keratoses',
                       'vasc': 'Vascular lesions',
                       'df': 'Dermatofibroma'}

    df_original['cell_type'] = df_original['dx'].map(dict_tipo_lesao.get)

    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes

    # Pré-Processamento

    # Extraindo Média e Desvio Padrão das Imagens

    # Função para cálculo de média e desvio
    def calcula_img_mean_std(image_paths):

        # Define altura e largura que usaremos nas imagens
        img_h, img_w = 224, 224

        # Listas de controle
        imgs = []
        means, stdevs = [], []

        # Loop de leitura e resize das imagens
        for i in tqdm(range(len(image_paths))):
            img = cv2.imread(image_paths[i])
            img = cv2.resize(img, (img_h, img_w))
            imgs.append(img)

        # Stack de imagens
        imgs = np.stack(imgs, axis=3)
        print(imgs.shape)

        # Normalização
        imgs = imgs.astype(np.float32) / 255.

        # Loop de cálculo da média e desvio
        for i in range(3):
            pixels = imgs[:, :, i, :].ravel()
            means.append(np.mean(pixels))
            stdevs.append(np.std(pixels))

        # BGR --> RGB
        means.reverse()
        stdevs.reverse()

        print("normMean = {}".format(means))
        print("normStd = {}".format(stdevs))

        return means, stdevs


    # Retorna a média e o padrão de cada canal RGB.
    norm_mean, norm_std = calcula_img_mean_std(caminho_imagens)

    # Preparação do Dataset de Validação

    # Vamos verificar quantas imagens estão associadas a cada lesion_id
    df_undup = df_original.groupby('lesion_id').count()

    # Agora filtramos lesion_ids que possuem apenas uma imagem associada
    df_undup = df_undup[df_undup['image_id'] == 1]

    # Reset do índice
    df_undup.reset_index(inplace=True)


    # Função para identificar lesion_ids que possuem imagens duplicadas e aqueles que possuem apenas uma imagem
    def get_duplicates(x):
        unique_list = list(df_undup['lesion_id'])
        if x in unique_list:
            return 'unduplicated'
        else:
            return 'duplicated'


    # Cria uma nova coluna que seja uma cópia da coluna lesion_id
    df_original['duplicates'] = df_original['lesion_id']

    # Aplica a função a esta nova coluna
    df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)

    # Vamos contar as duplicatas
    df_original['duplicates'].value_counts()

    # Agora filtramos as imagens que não têm duplicatas
    df_undup = df_original[df_original['duplicates'] == 'unduplicated']

    # Agora criamos um val set usando df_undup porque temos certeza de que nenhuma dessas imagens tem duplicatas
    y = df_undup['cell_type_idx']
    _, df_val = train_test_split(df_undup, test_size=0.2, random_state=101, stratify=y)

    # Separação das Amostras de Treino e Validação

    # Esta função identifica se uma imagem faz parte do conjunto train ou val
    def get_val_rows(x):
        val_list = list(df_val['image_id'])
        if str(x) in val_list:
            return 'val'
        else:
            return 'train'


    # Identifica treino ou validação
    df_original['train_or_val'] = df_original['image_id']

    # Aplica a função a esta nova coluna
    df_original['train_or_val'] = df_original['train_or_val'].apply(get_val_rows)

    # Filtra as linhas de treino
    df_treino = df_original[df_original['train_or_val'] == 'train']

    print(len(df_treino))
    print(len(df_val))

    df_treino['cell_type_idx'].value_counts()

    # Dataset Augmentation

    # Taxa de dataset augmentation a ser usada em cada classe
    data_aug_rate = [15, 10, 5, 50, 0, 40, 5]

    # Loop para o dataset augmentation
    for i in range(7):

        if data_aug_rate[i]:
            # Equaliza a proporção de imagens por classe nos dados de treino
            # Geramos novas imagens multiplicando as imagens existentes pela taxa definida na lista de taxas
            df_treino = df_treino._append([df_treino.loc[df_treino['cell_type_idx'] == i, :]] * (data_aug_rate[i] - 1),
                                         ignore_index=True)

    # Reset do índice
    df_treino = df_treino.reset_index()

    # Preparação das Amostras de Treino, Validação e Teste

    # Podemos dividir o conjunto de validação em um conjunto de validação e um conjunto de teste
    df_val, df_teste = train_test_split(df_val, test_size=0.5)

    # Reset do índice
    df_val = df_val.reset_index()
    df_teste = df_teste.reset_index()

    # Modelagem

    # Função de Inicialização do Modelo e Definição de Arquitetura com Transfer Learning

    # feature_extracting é um booleano que define se estamos fazendo um ajuste fino ou extração de recursos.
    # Se feature_extracting = False, o modelo é ajustado e todos os parâmetros do modelo são atualizados.
    # Se feature_extracting = True, apenas os parâmetros da última camada são atualizados, os outros permanecem fixos.
    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False


    # Função para inicializar diferentes arquiteturas de Deep Learning
    def inicializa_modelo(model_name, num_classes, feature_extract, use_pretrained=True):

        model_ft = None
        input_size = 0

        # Usaremos o modelo resnet50
        if model_name == "resnet":
            # Tamanho (pixels) das imagens de entrada
            input_size = 224

            # Carregamos o modelo pré-treinado com todos os pesos
            model_ft = models.resnet50(pretrained=use_pretrained)

            # Treinamos o modelo e atualizamos os pesos durante o treinamento
            set_parameter_requires_grad(model_ft, feature_extract)

            # Define o número de atributos de entrada
            num_ftrs = model_ft.fc.in_features

            # Camada linear final para prever a probabilidade das 7 classes com as quais estamos trabalhando
            model_ft.fc = nn.Linear(num_ftrs, num_classes)

        else:
            print("Modelo inválido...")
            exit()

        return model_ft, input_size

    # Inicializando o Modelo Escolhido e Definindo Transformações

    # Modelo que será treinado
    nome_modelo = 'resnet'

    num_classes = 7

    # Vamos treinar o modelo e sempre atualizar os pesos
    feature_extract = False

    # Inicializa o modelo
    model_ft, input_size = inicializa_modelo(nome_modelo, num_classes, feature_extract, use_pretrained=True)

    # Define o device
    device = processing_device

    # Coloca o modelo no device
    model = model_ft.to(device)

    # Transformações das imagens de treino
    transform_treino = transforms.Compose([transforms.Resize((input_size, input_size)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(), transforms.RandomRotation(20),
                                           transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                           transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])

    # Transformações das imagens de validação
    transform_val = transforms.Compose([transforms.Resize((input_size, input_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])

    # Criando os DataLoaders

    # Defina um organizador de dados para modelo PyTorch



    # Organiza e transforma os dados de treino
    set_treino = OrganizaDados(df_treino, transform=transform_treino)

    # Cria o dataloader de treino
    loader_treino = DataLoader(set_treino, batch_size=32, shuffle=True, num_workers=4)

    # O mesmo em validação
    set_val = OrganizaDados(df_val, transform=transform_val)
    loader_val = DataLoader(set_val, batch_size=32, shuffle=False, num_workers=4)

    # O mesmo em teste
    set_teste = OrganizaDados(df_teste, transform=transform_val)
    loader_teste = DataLoader(set_teste, batch_size=32, shuffle=False, num_workers=4)

    # Usaremos o otimizador Adam
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Usaremos cross entropy loss como função de perda
    criterion = nn.CrossEntropyLoss().to(device)

    # Treinamento

    # Funções Para o Loop de Treino e Validação

    # Função para calcular erro em treino e validação durante o treinamento
    class CalculaMetricas(object):

        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


    # Listas para erro e acurácia em treino
    total_loss_train, total_acc_train = [], []


    # Função de treino do modelo
    def treina_modelo(treino_loader, model, criterion, optimizer, epoch):

        # Coloca o modelo em modo de treino
        model.train()

        # Inicializa objetos de cálculo de métricas
        train_loss = CalculaMetricas()
        train_acc = CalculaMetricas()

        # Iteração
        curr_iter = (epoch - 1) * len(treino_loader)

        # Loop de treino
        for i, data in enumerate(treino_loader):

            # Extra os dados
            images, labels = data

            # Tamanho da imagem
            N = images.size(0)

            # Coloca imagens e labels no device
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            # Zera os gradientes
            optimizer.zero_grad()

            # Previsão do modelo
            outputs = model(images)

            # Erro do modelo
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Obtem a previsão de maior probabilidade
            prediction = outputs.max(1, keepdim=True)[1]

            # Atualiza as métricas
            train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item() / N)
            train_loss.update(loss.item())

            # Iteração
            curr_iter += 1

            # Print e update das métricas
            # A condição *** and curr_iter < 1000 *** pode ser removida se você quiser treinar com o dataset completo
            if (i + 1) % 100 == 0 and curr_iter < 1000:
                print('[epoch %d], [iter %d / %d], [train loss %.5f], [train acc %.5f]' % (epoch,
                                                                                           i + 1,
                                                                                           len(treino_loader),
                                                                                           train_loss.avg,
                                                                                           train_acc.avg))
                total_loss_train.append(train_loss.avg)
                total_acc_train.append(train_acc.avg)

        return train_loss.avg, train_acc.avg


    # Listas para erro e acurácia em validação
    total_loss_val, total_acc_val = [], []


    # Função para validação
    def valida_modelo(val_loader, model, criterion, optimizer, epoch):

        # Coloca o modelo em modo de validação
        model.eval()

        # Inicializa objetos de cálculo de métricas
        val_loss = CalculaMetricas()
        val_acc = CalculaMetricas()

        # Validação
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                images, labels = data

                N = images.size(0)

                images = Variable(images).to(device)

                labels = Variable(labels).to(device)

                outputs = model(images)

                prediction = outputs.max(1, keepdim=True)[1]

                val_acc.update(prediction.eq(labels.view_as(prediction)).sum().item() / N)

                val_loss.update(criterion(outputs, labels).item())

        print('------------------------------------------------------------')
        print('[epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, val_loss.avg, val_acc.avg))
        print('------------------------------------------------------------')

        return val_loss.avg, val_acc.avg


    # Treinamento do Modelo

    # Hiperparâmetros
    epoch_num = 3
    best_val_acc = 0

    for epoch in range(1, epoch_num + 1):

        # Execute a função de treino
        loss_train, acc_train = treina_modelo(loader_treino, model, criterion, optimizer, epoch)

        # Executa a função de validação
        loss_val, acc_val = valida_modelo(loader_val, model, criterion, optimizer, epoch)

        # Calcula as métricas
        total_loss_val.append(loss_val)
        total_acc_val.append(acc_val)

        # Verifica a acurácia em validação
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            print('*****************************************************')
            print('Melhor Resultado: [epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, loss_val, acc_val))
            print('*****************************************************')

        # Salvando o modelo
        torch.save(model, 'modelo.pt')