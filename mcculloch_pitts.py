import random

entradas = [
    [1, 0, 1, 0],
    [1, 1, 0, 1],
    [0, 1, 1, 1],
    [0, 0, 1, 1]
]
saidas_esperadas = [1, -1, 1, -1]


pesos = [random.uniform(-1, 1) for _ in range(4)]


taxa_aprendizado = 0.1


def func_ativacao(u):
    return 1 if u >= 0 else -1

def treinar(epocas=20):
    global pesos
    for epoca in range(epocas):
        erro_total = 0
        for i in range(len(entradas)):
            x = entradas[i]
            y_real = saidas_esperadas[i]
            u = sum([x[j] * pesos[j] for j in range(4)])
            y_pred = func_ativacao(u)
            erro = y_real - y_pred
            erro_total += abs(erro)

            for j in range(4):
                pesos[j] += taxa_aprendizado * erro * x[j]
        
        if erro_total == 0:
            break

def avaliar():
    acertos = 0
    print("Resultados:")
    for i in range(len(entradas)):
        x = entradas[i]
        y_real = saidas_esperadas[i]
        u = sum([x[j] * pesos[j] for j in range(4)])
        y_pred = func_ativacao(u)
        resultado = "✔️" if y_pred == y_real else "❌"
        if y_pred == y_real:
            acertos += 1
        print(f"Entrada: {x} | Esperado: {y_real} | Previsto: {y_pred} {resultado}")
    
    acuracia = acertos / len(entradas)
    print(f"\nAcurácia final: {acuracia * 100:.2f}%")

print("Pesos iniciais:", pesos)
treinar()
print("Pesos finais:", pesos)
avaliar()
