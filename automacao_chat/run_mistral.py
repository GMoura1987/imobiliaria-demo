
import os
import sys

# Define o modelo para esta execução
os.environ["MODEL_NAME"] = "mistral-nemo:12b"

print(f"🚀 Iniciando Corretora com modelo: {os.environ['MODEL_NAME']}")
print("⚠️  Certifique-se de ter rodado: ollama pull mistral-nemo:12b")

# Importa e roda o script principal
import corretora_refinada
if __name__ == "__main__":
    corretora_refinada.main()
