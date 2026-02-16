
import os
import sys

# Define o modelo para esta execução
os.environ["MODEL_NAME"] = "deepseek-r1:8b"

print(f"🚀 Iniciando Corretora com modelo: {os.environ['MODEL_NAME']}")
print("⚠️  Certifique-se de ter rodado: ollama pull deepseek-r1:8b")
print("ℹ️  Nota: Este modelo pode exibir tags <think> no terminal.")

# Importa e roda o script principal
import corretora_refinada
if __name__ == "__main__":
    corretora_refinada.main()
