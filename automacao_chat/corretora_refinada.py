import json
import sqlite3
import re
from langchain_ollama import ChatOllama
from langchain_classic.memory import ConversationBufferMemory

# --- CONFIGURAÇÕES ---
DB_PATH = "db.sqlite3"
# Temperatura 0 para o Extrator (precisão lógica) e 0.6 para a Ana Paula (criatividade controlada)
llm_analyst = ChatOllama(model="deepseek-r1:8b", temperature=0.0) 
llm_chat = ChatOllama(model="deepseek-r1:8b", temperature=0.3)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Estado Global da Sessão (Persistente)
session_state = {
    "filtros": {
        "bairro": None,
        "preco_max": None,
        "quartos": None,
        "imovel_atual_id": None
    }
}

# === CAMADA DE DADOS ===

def executar_busca_db(filtros):
    """Executa a busca baseada puramente no JSON gerado pelo LLM."""
    print(f"DEBUG SQL: Buscando com filtros {filtros}") # Para você ver o que está acontecendo
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM core_imovel WHERE 1=1"
        params = []
        
        # 1. Filtro de ID (Prioridade Total)
        if filtros.get("imovel_especifico_id"):
            cursor.execute("SELECT * FROM core_imovel WHERE id = ?", (filtros["imovel_especifico_id"],))
            res = cursor.fetchone()
            conn.close()
            return [dict(res)] if res else []

        # 2. Filtros Dinâmicos
        if filtros.get("bairro"):
            query += " AND LOWER(bairro) LIKE ?"
            params.append(f"%{filtros['bairro'].lower()}%")
            
        if filtros.get("preco_max") and filtros["preco_max"] > 0:
            query += " AND preco_aluguel <= ?"
            params.append(filtros["preco_max"])
            
        if filtros.get("quartos"):
            query += " AND quartos >= ?"
            params.append(filtros["quartos"])

        # Excluir o imóvel que já está sendo visto para não repetir recomendação
        if filtros.get("excluir_id"):
             query += " AND id != ?"
             params.append(filtros["excluir_id"])

        cursor.execute(query, params)
        res = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return res[:4] # Limita a 4 resultados
    except Exception as e:
        print(f"Erro SQL: {e}")
        return []

# === PASSO 1: O ANALISTA DE INTENÇÃO (O CÉREBRO) ===

def extrair_intencao_json(historico, mensagem_usuario, estado_atual):
    """
    Usa o LLM para converter linguagem natural em parâmetros de busca SQL (JSON).
    Ele raciocina sobre o contexto para atualizar, manter ou limpar filtros.
    """
    
    schema_json = """
    {
        "intencao": "BUSCA" ou "LINK_DIRETO" ou "CONVERSA",
        "imovel_especifico_id": int ou null,
        "filtros": {
            "bairro": "string ou null (null se o usuario pediu 'outros bairros')",
            "preco_max": float ou null,
            "quartos": int ou null
        },
        "explicacao": "breve motivo da mudança de filtros"
    }
    """

    prompt_analista = f"""
    Você é um motor de busca imobiliária inteligente (API). 
    Sua tarefa é analisar a conversa e gerar um JSON de filtros para consulta SQL.

    ESTADO ATUAL DOS FILTROS: {json.dumps(estado_atual)}
    
    HISTÓRICO RECENTE:
    {historico}
    
    MENSAGEM DO USUÁRIO: "{mensagem_usuario}"

    REGRAS DE LÓGICA (RACIOCINE):
    1. Se o usuário mandar um link (/imovel/123), a intenção é "LINK_DIRETO" e o ID vai para imovel_especifico_id.
    2. Se o usuário disser "tá caro" ou "queria algo até 1500", ATUALIZE o 'preco_max'.
    3. Se o usuário disser "tem em outros bairros?" ou "outra localização", DEFINA "bairro": null (para limpar o filtro).
    4. Se o usuário disser "neste bairro", MANTENHA o bairro atual.
    5. Se o usuário perguntar sobre o imóvel atual (ex: "tem garagem?"), MANTENHA os filtros e a intenção "CONVERSA".

    SAÍDA APENAS JSON VÁLIDO. NADA DE TEXTO ANTES OU DEPOIS.
    SCHEMA: {schema_json}
    """

    try:
        response = llm_analyst.invoke(prompt_analista).content
        # Limpeza para garantir que pegamos só o JSON (DeepSeek as vezes põe markdown)
        json_str = re.search(r'\{.*\}', response, re.DOTALL).group(0)
        return json.loads(json_str)
    except Exception as e:
        print(f"Erro no Analista JSON: {e}")
        # Fallback seguro
        return {"intencao": "CONVERSA", "filtros": estado_atual, "imovel_especifico_id": None}

# === PASSO 2: A CORRETORA (A VOZ) ===

def gerar_resposta_ana_paula(contexto_imoveis, mensagem_usuario, intencao_detectada):
    prompt_sistema = f"""
    Você é a Ana Paula, corretora sênior em Juiz de Fora.
    
    DADOS REAIS DO BANCO DE DADOS (Imutável):
    {contexto_imoveis}

    INSTRUÇÕES:
    1. Se a lista de imóveis estiver vazia, diga a verdade: "Não encontrei opções com esse perfil exato no momento". Sugira alterar os filtros.
    2. Se houver imóveis, apresente-os de forma sedutora mas resumida.
    3. Objetivo final: Agendar Visita.
    4. Tom de voz: Profissional, prestativa e ágil (WhatsApp).
    
    INTENÇÃO DETECTADA: {intencao_detectada}
    """
    
    historico = memory.load_memory_variables({})["chat_history"]
    prompt_final = f"{prompt_sistema}\n\nHistórico: {historico}\nCliente: {mensagem_usuario}\nAna Paula:"
    
    return llm_chat.invoke(prompt_final).content.replace('Ana Paula:', '').strip()

# === FLUXO PRINCIPAL ===

def chat_pipeline(mensagem_usuario):
    global session_state
    
    # 1. Recuperar histórico para contexto
    historico = memory.load_memory_variables({})["chat_history"][-2:] # Pega apenas as 2 ultimas trocas para economizar tokens
    
    # 2. O Analista define o que buscar
    analise = extrair_intencao_json(historico, mensagem_usuario, session_state["filtros"])
    
    # Atualiza o estado da sessão com a nova inteligência
    novos_filtros = analise["filtros"]
    
    # Lógica de "Excluir o atual" se o usuário estiver pedindo "outros"
    excluir_id = session_state["filtros"].get("imovel_atual_id")
    if analise["intencao"] == "BUSCA":
        novos_filtros["excluir_id"] = excluir_id
    
    # 3. Executa a busca no SQL
    imoveis_encontrados = []
    
    if analise["intencao"] == "LINK_DIRETO":
        # Se for link direto, o ID específico prevalece
        session_state["filtros"]["imovel_atual_id"] = analise["imovel_especifico_id"]
        # Reseta filtros para focar neste imovel, mas guarda o bairro dele
        imoveis_encontrados = executar_busca_db({"imovel_especifico_id": analise["imovel_especifico_id"]})
        if imoveis_encontrados:
             novos_filtros["bairro"] = imoveis_encontrados[0]['bairro'] # Atualiza contexto de bairro

    elif analise["intencao"] == "BUSCA" or analise["intencao"] == "CONVERSA":
        imoveis_encontrados = executar_busca_db(novos_filtros)

    # Atualiza o estado global para a próxima rodada
    session_state["filtros"].update(novos_filtros)

    # 4. Formata o contexto para a Ana Paula
    texto_contexto = ""
    if imoveis_encontrados:
        texto_contexto = "IMÓVEIS ENCONTRADOS:\n"
        for i in imoveis_encontrados:
            texto_contexto += f"- {i['titulo']} | Bairro: {i['bairro']} | R$ {i['preco_aluguel']} | Desc: {i['descricao'][:100]}...\n"
    else:
        texto_contexto = "STATUS DO SISTEMA: Nenhum imóvel encontrado com os filtros atuais."

    # 5. Gera a resposta final
    resposta = gerar_resposta_ana_paula(texto_contexto, mensagem_usuario, analise["intencao"])
    
    memory.save_context({"input": mensagem_usuario}, {"output": resposta})
    return resposta

# === LOOP ===
if __name__ == "__main__":
    print("--- 🏠 Ana Paula AI (Agentic Arch) ---")
    while True:
        voce = input("\n👤 Você: ")
        if voce.lower() in ['sair', 'parar']: break
        print(f"\n🏠 Ana Paula: {chat_pipeline(voce)}")