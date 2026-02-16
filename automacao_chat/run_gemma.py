
import os
import sys

# Define o modelo para esta execução
os.environ["MODEL_NAME"] = "gemma2:9b"

print(f"🚀 Iniciando Corretora com modelo: {os.environ['MODEL_NAME']}")
print("⚠️  Certifique-se de ter rodado: ollama pull gemma2:9b")

# Importa e roda o script principal
import corretora_refinada
if __name__ == "__main__":
    corretora_refinada.main()
