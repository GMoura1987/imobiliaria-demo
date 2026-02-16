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

# --- MIX DE MODELOS PARA VRAM 8GB ---
# Qwen 2.5 3B: Especialista em JSON/Lógica, muito rápido e leve (~2GB VRAM)
llm_analyst = ChatOllama(model="qwen2.5:3b", temperature=0.0) 

# Llama 3.1 8B: Bom em instruções e persona, cabe na VRAM (~5GB)
llm_chat = ChatOllama(model="llama3.1:8b", temperature=0.3)

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
            # Busca pela coluna especificacao (casa, apartamento, kitnet)
            query += " AND LOWER(especificacao) = ?"
            params.append(filtros['tipo'].lower())

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
    - tipo: SOMENTE preencha se o usuário EXPLICITAMENTE pediu 'casa', 'apartamento' ou 'kitnet'. Se o usuário disse apenas "imóveis", "opções" ou algo genérico, tipo DEVE ser null.
    - bairro: Identifique o local mencionado.
    - Se o usuário mudar de ideia (ex: "quero em outro bairro"), limpe o bairro anterior.
    - Se a mensagem for cumprimento, saudação ou conversa casual, intencao deve ser "CONVERSA" e filtros vazios.

    SAÍDA APENAS JSON:
    {{
        "intencao": "BUSCA" | "CONVERSA",
        "filtros": {{ "bairro": str | null, "tipo": str | null, "preco_max": float | null }},
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
    Você é a Ana Paula, corretora de imóveis em Juiz de Fora, MG.
    Seu tom é simpático, profissional e direto.
    
    REGRAS OBRIGATÓRIAS:
    1. Apresente TODOS os imóveis listados no CONTEXTO abaixo. Não omita nenhum.
    2. Se o contexto diz "Nenhum imóvel encontrado", aí sim diga que não encontrou e pergunte se quer buscar em outro bairro ou tipo.
    3. Se há imóveis no contexto, NUNCA diga que não tem. Apenas apresente-os de forma organizada.
    4. Se o cliente fez uma pergunta genérica ("imóveis no bairro X"), mostre todos os tipos disponíveis.
    5. Se o cliente pediu um tipo específico (ex: casa) e você só tem outros tipos, explique: "Não encontrei casas, mas tenho estas opções no mesmo bairro:".
    6. Use no máximo 1 emoji por resposta. Foco em Locação.
    7. Para cada imóvel, apresente o PREÇO TOTAL (Aluguel + Taxas) se o cliente perguntar sobre valores. Use os dados detalhados do contexto.
    9. SE O CLIENTE QUISER VISITAR:
       a) Diga "Que ótimo!" e PERGUNTE IMEDIATAMENTE qual a disponibilidade de dia e horário.
       b) NÃO FALE DA FICHA DE CADASTRO AINDA. Aguarde o cliente responder o horário.
       c) APÓS o cliente definir o horário: Confirme o agendamento e SÓ ENTÃO explique que a **Ficha de Pré-Cadastro** agiliza a análise jurídica.
       d) Liste os documentos necessários ({DOCS_LOCACAO}) logo após explicar a ficha.
       e) Se o cliente NÃO quiser fazer a ficha, diga "Tudo bem, nos encontramos no imóvel". RESSALTE que para realizar a visita, o ÚNICO documento obrigatório em mãos é o **RG**.
    10. NÃO mencione a ficha de cadastro em todas as mensagens. Apenas APÓS agendar a visita.
    11. SE O CLIENTE ACHAR CARO: Tente argumentar sobre o custo-benefício (localização, acabamento) OU ofereça opções mais baratas se houver no contexto.
    
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
    
    # 2. Lógica de Alternativas (SOMENTE se busca principal retornou ZERO resultados)
    if not imoveis:
        # Busca Alternativa A: Mesmo tipo em QUALQUER bairro
        if tipo_pedido:
            alternativos_tipo = executar_busca_db({"tipo": tipo_pedido})
            if alternativos_tipo:
                contexto_extra += "\nALTERNATIVAS (Mesmo tipo em outros bairros):\n"
                for i in alternativos_tipo:
                    contexto_extra += f"- ID {i['id']}: {i['titulo']} no bairro {i['bairro']} (R$ {i['preco_aluguel']}). {i['descricao']}\n"
        
        # Busca Alternativa B: Qualquer tipo no MESMO bairro
        if bairro_pedido:
            alternativos_bairro = executar_busca_db({"bairro": bairro_pedido})
            if alternativos_bairro:
                contexto_extra += "\nALTERNATIVAS (Outros imóveis neste mesmo bairro):\n"
                for i in alternativos_bairro:
                    contexto_extra += f"- ID {i['id']}: {i['titulo']} no {i['bairro']} (R$ {i['preco_aluguel']}). {i['descricao']}\n"

    # Formatação do Contexto para o LLM
    texto_contexto = "IMÓVEIS ENCONTRADOS (BUSCA EXATA):\n"
    if imoveis:
        for i in imoveis:
            custo_total = i['preco_aluguel'] + i['preco_iptu'] + i['preco_condominio']
            pets_txt = "Aceita Pets" if i['aceita_pets'] else "Não aceita pets"
            texto_contexto += (
                f"- ID {i['id']}: {i['titulo']} no {i['bairro']}\n"
                f"  * Aluguel: R$ {i['preco_aluguel']} | IPTU: R$ {i['preco_iptu']} | Condomínio: R$ {i['preco_condominio']}\n"
                f"  * TOTAL MENSAL: R$ {custo_total}\n"
                f"  * Detalhes: {i['quartos']} quartos, {i['banheiros']} banheiros, {i['garagem']} vaga(s), {i['area']}m²\n"
                f"  * Extra: {pets_txt}\n"
                f"  * Descrição: {i['descricao']}\n"
                f"  * Código Bairro: {i['codigo_bairro']}\n\n"
            )
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