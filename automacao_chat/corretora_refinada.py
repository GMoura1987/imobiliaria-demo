import json
import sqlite3
import re
import difflib
from langchain_ollama import ChatOllama
from langchain_classic.memory import ConversationBufferMemory

# --- CONFIGURAÇÕES E CONSTANTES ---
DB_PATH = "db.sqlite3"

DOCS_LOCACAO = """
- RG e CPF (ou CNH)
- Comprovante de residência atual
- Comprovante de renda: 3 últimos holerites (CLT) ou 6 meses de extrato bancário (PJ)
- Certidão de estado civil (Casamento ou Nascimento)
- Se casado: Identidade do cônjuge
- Se PJ: Comprovante de Pessoa Jurídica (Contrato Social/Cartão CNPJ)
"""

llm_analyst = ChatOllama(model="gemma2:9b", temperature=0.0) 
llm_chat = ChatOllama(model="gemma2:9b", temperature=0.3)

# Memória (Silenciando o warning internamente ou ignorando para foco na lógica)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

session_state = {
    "filtros": {
        "bairro": None,
        "tipo": None, # 'casa', 'apartamento', 'cobertura'
        "preco_max": None,
        "imovel_atual_id": None
    }
}

# === CAMADA DE DADOS ===

def executar_busca_db(filtros):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        query = "SELECT * FROM core_imovel WHERE 1=1"
        params = []
        
        if filtros.get("imovel_especifico_id"):
            cursor.execute("SELECT * FROM core_imovel WHERE id = ?", (filtros["imovel_especifico_id"],))
            res = cursor.fetchone()
            conn.close()
            return [dict(res)] if res else []

        if filtros.get("bairro"):
            # Função de correção simplificada integrada
            bairros_db = [r[0] for r in conn.execute("SELECT DISTINCT bairro FROM core_imovel").fetchall()]
            matches = difflib.get_close_matches(filtros["bairro"], bairros_db, n=1, cutoff=0.6)
            bairro_final = matches[0] if matches else filtros["bairro"]
            query += " AND LOWER(bairro) LIKE ?"
            params.append(f"%{bairro_final.lower()}%")
            
        if filtros.get("tipo"):
            # Busca no título ou descrição se não houver coluna 'tipo'
            query += " AND (LOWER(titulo) LIKE ? OR LOWER(descricao) LIKE ?)"
            params.append(f"%{filtros['tipo'].lower()}%")
            params.append(f"%{filtros['tipo'].lower()}%")

        if filtros.get("preco_max"):
            query += " AND preco_aluguel <= ?"
            params.append(filtros["preco_max"])

        cursor.execute(query, params)
        res = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return res[:5]
    except Exception as e:
        print(f"Erro SQL: {e}")
        return []

# === PASSO 1: O ANALISTA DE INTENÇÃO ===

def extrair_intencao_json(historico, mensagem_usuario, estado_atual):
    default_res = {
        "intencao": "CONVERSA",
        "imovel_especifico_id": None,
        "filtros": estado_atual,
        "solicitou_visita": False
    }

    prompt_analista = f"""
    Analise a mensagem e extraia filtros de busca.
    MENSAGEM: "{mensagem_usuario}"
    ESTADO ANTERIOR: {json.dumps(estado_atual)}

    REGRAS:
    - tipo: Identifique se o usuário quer 'casa', 'apartamento', 'kitnet' ou 'cobertura'.
    - bairro: Identifique o local.
    - Se o usuário mudar de ideia (ex: "quero em outro bairro"), limpe o bairro anterior.

    SAÍDA APENAS JSON:
    {{
        "intencao": "BUSCA" | "CONVERSA",
        "filtros": {{ "bairro": str, "tipo": str, "preco_max": float }},
        "solicitou_visita": bool
    }}
    """
    try:
        response = llm_analyst.invoke(prompt_analista).content
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            extracted = json.loads(match.group(0))
            final = default_res.copy()
            final.update(extracted)
            return final
        return default_res
    except: return default_res

# === PASSO 2: A CORRETORA (A VOZ) ===

def gerar_resposta_ana_paula(contexto, mensagem_usuario):
    prompt_sistema = f"""
    Você é a Ana Paula, corretora em Juiz de Fora. 
    
    REGRAS DE OURO:
    1. Se não houver o imóvel EXATO solicitado (ex: não tem casa em tal bairro), diga: "No momento não tenho [tipo] no [bairro]".
    2. Logo em seguida, apresente as ALTERNATIVAS que o sistema encontrou (outros tipos no mesmo bairro ou o mesmo tipo em outros bairros).
    3. Nunca ignore o pedido do usuário. Se ele pediu casa e você só tem apartamento, explique isso.
    4. Max 1 emoji. Foco em Locação.

    DOCUMENTOS: {DOCS_LOCACAO}
    
    CONTEXTO DO BANCO DE DADOS:
    {contexto}
    """
    historico = memory.load_memory_variables({})["chat_history"]
    prompt_final = f"{prompt_sistema}\n\nHistórico: {historico}\nCliente: {mensagem_usuario}\nAna Paula:"
    return llm_chat.invoke(prompt_final).content.replace('Ana Paula:', '').strip()

# === FLUXO PRINCIPAL COM LÓGICA DE BUSCA SECUNDÁRIA ===

def chat_pipeline(mensagem_usuario):
    global session_state
    
    analise = extrair_intencao_json("", mensagem_usuario, session_state["filtros"])
    filtros_novos = analise.get("filtros", {})
    session_state["filtros"].update(filtros_novos)
    
    # 1. Busca Principal (Exata)
    imoveis = executar_busca_db(session_state["filtros"])
    
    tipo_pedido = session_state["filtros"].get("tipo")
    bairro_pedido = session_state["filtros"].get("bairro")
    
    contexto_extra = ""
    
    # 2. Lógica de Alternativas (Se a busca principal for frustrante)
    if not imoveis or (tipo_pedido and len(imoveis) < 2):
        # Busca Alternativa A: Mesmo tipo em QUALQUER bairro
        if tipo_pedido:
            alternativos_tipo = executar_busca_db({"tipo": tipo_pedido})
            if alternativos_tipo:
                contexto_extra += "\nALTERNATIVAS (Mesmo tipo em outros bairros):\n"
                for i in alternativos_tipo:
                    contexto_extra += f"- ID {i['id']}: {i['titulo']} no bairro {i['bairro']} (R$ {i['preco_aluguel']})\n"
        
        # Busca Alternativa B: Qualquer tipo no MESMO bairro
        if bairro_pedido:
            alternativos_bairro = executar_busca_db({"bairro": bairro_pedido})
            # Filtra para não repetir o que já pode ter vindo na busca principal
            alternativos_bairro = [i for i in alternativos_bairro if i['id'] not in [x['id'] for x in imoveis]]
            if alternativos_bairro:
                contexto_extra += "\nALTERNATIVAS (Outros imóveis neste mesmo bairro):\n"
                for i in alternativos_bairro:
                    contexto_extra += f"- ID {i['id']}: {i['titulo']} ({i['preco_aluguel']})\n"

    # Formatação do Contexto para o LLM
    texto_contexto = "IMÓVEIS ENCONTRADOS (BUSCA EXATA):\n"
    if imoveis:
        for i in imoveis:
            texto_contexto += f"- ID {i['id']}: {i['titulo']} no {i['bairro']} (R$ {i['preco_aluguel']}). {i['descricao']}\n"
    else:
        texto_contexto += "Nenhum imóvel encontrado com os critérios exatos.\n"
    
    texto_contexto += contexto_extra

    resposta = gerar_resposta_ana_paula(texto_contexto, mensagem_usuario)
    memory.save_context({"input": mensagem_usuario}, {"output": resposta})
    return resposta

if __name__ == "__main__":
    print("--- 🏠 Ana Paula AI (Versão 4.2 - Consultiva) ---")
    while True:
        try:
            user_input = input("\n👤 Você: ")
            if user_input.lower() in ['sair', 'parar']: break
            print(f"\n🏠 Ana Paula: {chat_pipeline(user_input)}")
        except Exception as e:
            print(f"Erro: {e}")