import numpy as np

# Definição da função de ativação (Sigmoide)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada da sigmoide (para o cálculo do gradiente)
def sigmoid_derivative(x):
    return x * (1 - x)

# Definição dos padrões de entrada (dígitos 0 e 1)
X = np.array([
    [0, 1, 1, 0,  
     1, 0, 0, 1,
     1, 0, 0, 1,
     0, 1, 1, 0],  # Dígito 0

    [0, 0, 1, 0,  
     0, 1, 1, 0,
     0, 0, 1, 0,
     0, 0, 1, 0],  # Dígito 1

    [0, 1, 1, 0,  
     1, 0, 0, 1,
     1, 0, 0, 1,
     0, 1, 1, 0],  # Outro exemplo de Dígito 0

    [0, 0, 1, 0,  
     0, 1, 1, 0,
     0, 0, 1, 0,
     0, 0, 1, 0]   # Outro exemplo de Dígito 1
])

# Saídas esperadas (0 para o dígito '0' e 1 para o dígito '1')
y = np.array([
    [1, 0],  # Dígito 0
    [0, 1],  # Dígito 1
    [1, 0],  # Outro dígito 0
    [0, 1]   # Outro dígito 1
])

# Inicialização dos pesos
np.random.seed(42)  # Para reprodutibilidade
pesos_entrada_oculta = np.random.rand(16, 8) - 0.5  # Pesos da camada de entrada para a oculta (16 -> 8)
pesos_oculta_saida = np.random.rand(8, 2) - 0.5  # Pesos da camada oculta para a saída (8 -> 2)
bias_oculta = np.random.rand(1, 8) - 0.5  # Bias da camada oculta
bias_saida = np.random.rand(1, 2) - 0.5  # Bias da camada de saída

# Hiperparâmetros do treinamento
taxa_aprendizado = 0.1
epocas = 10000

# Treinamento com Backpropagation
for epoca in range(epocas):
    # Forward Pass
    camada_oculta = sigmoid(np.dot(X, pesos_entrada_oculta) + bias_oculta)
    saida = sigmoid(np.dot(camada_oculta, pesos_oculta_saida) + bias_saida)

    # Cálculo do erro
    erro = y - saida

    # Backpropagation (Ajuste dos pesos)
    gradiente_saida = erro * sigmoid_derivative(saida)
    erro_oculta = np.dot(gradiente_saida, pesos_oculta_saida.T)
    gradiente_oculta = erro_oculta * sigmoid_derivative(camada_oculta)

    # Atualização dos pesos e bias
    pesos_oculta_saida += np.dot(camada_oculta.T, gradiente_saida) * taxa_aprendizado
    pesos_entrada_oculta += np.dot(X.T, gradiente_oculta) * taxa_aprendizado
    bias_saida += np.sum(gradiente_saida, axis=0, keepdims=True) * taxa_aprendizado
    bias_oculta += np.sum(gradiente_oculta, axis=0, keepdims=True) * taxa_aprendizado

    # Exibir erro a cada 1000 épocas
    if epoca % 1000 == 0:
        print(f"Época {epoca}, Erro: {np.mean(np.abs(erro))}")

# Função de teste
def testar(matriz):
    vetor = np.array(matriz).flatten()
    camada_oculta = sigmoid(np.dot(vetor, pesos_entrada_oculta) + bias_oculta)
    resultado = sigmoid(np.dot(camada_oculta, pesos_oculta_saida) + bias_saida)
    return "Dígito 0" if resultado[0, 0] > resultado[0, 1] else "Dígito 1"

# Teste de novos exemplos
teste_0 = [
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
]

teste_1 = [
    [0, 0, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0]
]

# Exibição dos resultados
print("\nResultados:")
print("Resultado para o dígito 0:", testar(teste_0))  # Esperado: "Dígito 0"
print("Resultado para o dígito 1:", testar(teste_1))  # Esperado: "Dígito 1"
