import re
import unicodedata
from difflib import SequenceMatcher
import sqlite3
from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_classic.memory import ConversationBufferMemory

# --- CONFIGURAÇÕES ---
DB_PATH = "db.sqlite3"
MODEL_NAME = "llama3.1:8b"

llm = ChatOllama(model=MODEL_NAME, temperature=0.4, top_p=0.9)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
ultimos_imoveis_mostrados = []  # Rastreia últimos imóveis apresentados ao cliente
imovel_em_foco = {}  # Rastreia o imóvel específico que o cliente está interessado

db_langchain = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
sql_agent = create_sql_agent(
    llm, db=db_langchain, verbose=False, handle_parsing_errors=True
)

# === FUNÇÕES DE BANCO DE DADOS ===

def inicializar_fts():
    """Cria/atualiza a tabela FTS5 para busca rápida em texto"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # Cria tabela FTS5 se não existir (indexa titulo, descricao, bairro)
        # DROP + CREATE evita erro de DELETE em tabela contentless
        cursor.execute("DROP TABLE IF EXISTS imovel_fts")
        cursor.execute("""
            CREATE VIRTUAL TABLE imovel_fts 
            USING fts5(imovel_id, titulo, descricao, bairro, content='')
        """)
        cursor.execute("""
            INSERT INTO imovel_fts(imovel_id, titulo, descricao, bairro)
            SELECT id, COALESCE(titulo,''), COALESCE(descricao,''), COALESCE(bairro,'')
            FROM core_imovel
        """)
        conn.commit()
        conn.close()
        print("✅ Índice de busca FTS5 criado com sucesso!")
    except Exception as e:
        print(f"⚠️ Erro ao criar FTS5: {e}")

def buscar_bairros_disponiveis():
    """Busca todos os bairros no banco"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT LOWER(bairro) FROM core_imovel WHERE bairro IS NOT NULL")
        resultado = [row[0] for row in cursor.fetchall()]
        conn.close()
        return resultado
    except:
        return []

def buscar_todos_imoveis():
    """Busca todos os imóveis"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM core_imovel")
        resultado = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return resultado
    except:
        return []

def buscar_imovel_por_id(imovel_id):
    """Busca um imóvel específico pelo ID"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM core_imovel WHERE id = ?", (imovel_id,))
        res = cursor.fetchone()
        conn.close()
        return dict(res) if res else None
    except:
        return None

def buscar_imoveis_filtrados(bairro=None, quartos_min=None, preco_max=None, aceita_pets=None, texto_busca=None):
    """Busca imóveis com filtros. Usa FTS5 para busca em texto (rápido mesmo em bancos grandes)"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Se tem texto_busca, usa FTS5 pra pegar os IDs primeiro (busca rápida)
        if texto_busca:
            # FTS5 MATCH: busca instantânea com índice invertido
            # Suporta queries como "academia OR supermercado OR central"
            cursor.execute(
                "SELECT imovel_id FROM imovel_fts WHERE imovel_fts MATCH ?",
                (texto_busca,)
            )
            ids_encontrados = [row[0] for row in cursor.fetchall()]
            if not ids_encontrados:
                conn.close()
                return []
            placeholders = ",".join("?" * len(ids_encontrados))
            query = f"SELECT * FROM core_imovel WHERE id IN ({placeholders})"
            params = list(ids_encontrados)
        else:
            query = "SELECT * FROM core_imovel WHERE 1=1"
            params = []

        if bairro:
            query += " AND LOWER(bairro) LIKE ?"
            params.append(f"%{bairro.lower()}%")
        if quartos_min:
            query += " AND quartos >= ?"
            params.append(quartos_min)
        if preco_max:
            query += " AND preco_aluguel <= ?"
            params.append(preco_max)
        if aceita_pets is not None:
            query += " AND aceita_pets = ?"
            params.append(1 if aceita_pets else 0)

        cursor.execute(query, params)
        resultado = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return resultado
    except:
        return []

# === EXTRAÇÃO E FORMATAÇÃO ===

def extrair_id_da_url(texto):
    """Extrai ID do imóvel de uma URL"""
    match = re.search(r"/imovel/(\d+)", texto)
    return int(match.group(1)) if match else None

def normalizar_texto(texto):
    """Remove acentos e normaliza para comparação fuzzy"""
    nfkd = unicodedata.normalize('NFKD', texto.lower())
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

def extrair_criterios(mensagem):
    """Extrai critérios da mensagem do usuário"""
    msg = mensagem.lower()
    msg_norm = normalizar_texto(msg)
    criterios = {}

    # Bairros do banco de dados - busca exata primeiro, depois fuzzy
    bairros = buscar_bairros_disponiveis()
    for bairro in bairros:
        if bairro in msg:
            criterios['bairro'] = bairro
            break

    # Fuzzy matching: tolera erros de digitação (ex: "matheus" vs "mateus")
    if 'bairro' not in criterios:
        bairro_norm = normalizar_texto(msg)
        melhor_score = 0
        melhor_bairro = None
        for bairro in bairros:
            b_norm = normalizar_texto(bairro)
            # Procura o nome do bairro na mensagem comparando substrings
            for i in range(len(bairro_norm) - len(b_norm) + 1):
                trecho = bairro_norm[i:i + len(b_norm) + 2]  # margem de +2 chars
                score = SequenceMatcher(None, b_norm, trecho).ratio()
                if score > melhor_score:
                    melhor_score = score
                    melhor_bairro = bairro
        if melhor_score >= 0.75 and melhor_bairro:
            criterios['bairro'] = melhor_bairro

    # Se não achou bairro exato, extrai palavras-chave e busca via FTS5
    if 'bairro' not in criterios:
        # Remove stop words e palavras comuns de conversa, sobram as palavras-chave úteis
        stop_words = {
            'um', 'uma', 'uns', 'umas', 'o', 'a', 'os', 'as', 'de', 'do', 'da', 'dos', 'das',
            'em', 'no', 'na', 'nos', 'nas', 'por', 'para', 'com', 'sem', 'que', 'se', 'mas',
            'ou', 'e', 'é', 'ao', 'à', 'ter', 'ser', 'eu', 'me', 'meu', 'minha', 'você',
            'gostaria', 'quero', 'preciso', 'procuro', 'busco', 'queria', 'tem', 'algum',
            'alguma', 'alguns', 'algumas', 'bom', 'dia', 'boa', 'tarde', 'noite', 'olá', 'oi',
            'apartamento', 'casa', 'imóvel', 'imovel', 'imoveis', 'imóveis', 'alugar', 'aluguel',
            'ver', 'mostra', 'mostre', 'mostrar', 'favor', 'pode', 'poderia', 'região', 'regiao',
            'bairro', 'local', 'lugar', 'área', 'area', 'reais', 'real', 'até', 'entre',
            'quartos', 'quarto', 'muito', 'mais', 'menos', 'bem', 'tudo', 'todo', 'toda',
        }
        # Extrai palavras significativas (3+ caracteres, não numéricas, não stop words)
        palavras = [p for p in re.findall(r'[a-záàâãéêíóôõúç]+', msg)
                     if len(p) >= 3 and p not in stop_words]
        if palavras:
            # Junta com OR para FTS5 encontrar qualquer uma
            criterios['texto_busca'] = " OR ".join(palavras)

    # Quartos
    m = re.search(r'(\d+)\s*quarto', msg)
    if m:
        criterios['quartos_min'] = int(m.group(1))

    # Preço - "até X" ou "máximo X"
    m = re.search(r'(?:até|max|máximo|no máximo)\s*r?\$?\s*(\d+)', msg)
    if m:
        criterios['preco_max'] = float(m.group(1))
    else:
        # "entre X e Y"
        m = re.search(r'entre\s*r?\$?\s*(\d+)\s*e\s*r?\$?\s*(\d+)', msg)
        if m:
            criterios['preco_max'] = float(m.group(2))

    # Pets
    if any(p in msg for p in ['pet', 'cachorro', 'gato', 'animal']):
        criterios['aceita_pets'] = True

    return criterios

def formatar_imovel(imovel, numero=None):
    """Formata imóvel de forma atrativa"""
    total = imovel['preco_aluguel'] + imovel['preco_iptu'] + imovel['preco_condominio']
    pets = "✅ Aceita pets" if imovel['aceita_pets'] else "❌ Não aceita pets"
    header = f"OPÇÃO {numero}:" if numero else "📋 DETALHES DO IMÓVEL:"

    descricao = imovel.get('descricao', '') or ''

    return f"""{header}
📍 {imovel['titulo']}
   Bairro: {imovel['bairro']} | {imovel['rua']}, {imovel['numero']}
   🛏️  {imovel['quartos']} quartos | 🚿 {imovel['banheiros']} banheiros | 🚗 {imovel['garagem']} vagas
   📐 {imovel['area']}m² | {pets}
   💰 Aluguel: R$ {imovel['preco_aluguel']:.2f}
      Condomínio: R$ {imovel['preco_condominio']:.2f} | IPTU: R$ {imovel['preco_iptu']:.2f}
   ✨ TOTAL: R$ {total:.2f}/mês
   📝 Descrição: {descricao}"""

def formatar_imovel_detalhado(imovel):
    """Formata imóvel com todos os campos explicitamente rotulados para o LLM"""
    total = imovel['preco_aluguel'] + imovel['preco_iptu'] + imovel['preco_condominio']
    pets = "Sim" if imovel['aceita_pets'] else "Não"
    descricao = imovel.get('descricao', '') or 'Sem descrição adicional'
    return f"""NOME: {imovel['titulo']}
BAIRRO: {imovel['bairro']}
ENDEREÇO: {imovel['rua']}, {imovel['numero']}
QUARTOS: {imovel['quartos']}
BANHEIROS: {imovel['banheiros']}
GARAGEM: {imovel['garagem']} vagas
ÁREA: {imovel['area']}m²
ACEITA PETS: {pets}
PREÇO DO ALUGUEL: R$ {imovel['preco_aluguel']:.2f}
PREÇO DO CONDOMÍNIO: R$ {imovel['preco_condominio']:.2f}
PREÇO DO IPTU: R$ {imovel['preco_iptu']:.2f}
TOTAL MENSAL (aluguel + condomínio + IPTU): R$ {total:.2f}
DESCRIÇÃO: {descricao}"""

def identificar_imovel_mencionado(mensagem, imoveis):
    """Tenta identificar qual imóvel o usuário está perguntando"""
    msg = mensagem.lower()
    # Palavras-chave para cada tipo de imóvel
    for imovel in imoveis:
        titulo = imovel['titulo'].lower()
        # Verifica se alguma palavra do título aparece na mensagem
        palavras_titulo = ['cobertura', 'duplex', 'apartamento', 'casa', 'kitnet', 'loft',
                          'geminada', 'vila', 'studio', 'flat']
        for palavra in palavras_titulo:
            if palavra in msg and palavra in titulo:
                return imovel
    # Se mencionou opção 1, 2, 3...
    m = re.search(r'op[çc][aã]o\s*(\d+)', msg)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(imoveis):
            return imoveis[idx]
    # Se mencionou primeiro/primeira, segundo/segunda etc
    ordinal_map = {'primeir': 0, 'segund': 1, 'terceir': 2}
    for ordinal, idx in ordinal_map.items():
        if ordinal in msg and idx < len(imoveis):
            return imoveis[idx]
    return None

def top3(imoveis):
    """Retorna os 3 mais baratos (melhor custo-benefício)"""
    for im in imoveis:
        im['preco_total'] = im['preco_aluguel'] + im['preco_iptu'] + im['preco_condominio']
    return sorted(imoveis, key=lambda x: x['preco_total'])[:3]

# === RESPOSTAS COM LLM ===

def resposta_llm_corretora(contexto, dados_imoveis=""):
    """Gera resposta usando LLM com persona de corretora"""
    historico = memory.load_memory_variables({})["chat_history"]
    prompt = f"""Você é a Ana Paula, 32 anos, corretora de imóveis há 8 anos. Você trabalha na região de Juiz de Fora.

COMO VOCÊ SE COMPORTA:
- Você escreve como uma pessoa real no WhatsApp, com frases curtas e diretas
- Você NÃO usa bullet points, listas formatadas, asteriscos ou markdown
- Você NÃO se apresenta como IA, chatbot ou assistente virtual. Você é uma pessoa.
- Você usa "rs", "kkk", "haha" quando apropriado (mas sem exagero)
- Você às vezes quebra a frase em mensagens curtas ao invés de um textão
- Você NÃO repete saudações se já cumprimentou o cliente
- Você é simpática mas profissional, como uma vendedora que quer fechar negócio
- Suas respostas são CURTAS, no máximo 4-5 frases por vez
- Você tenta sempre avançar pro próximo passo: agendar visita, pegar dados, fechar contrato

SEU OBJETIVO: fazer o cliente alugar um imóvel. Você quer fechar negócio.

DOCUMENTOS NECESSÁRIOS PARA LOCAÇÃO (use quando perguntarem):
- Comprovante de renda (últimos 3 holerites OU extrato bancário dos últimos 6 meses)
- RG (identidade)
- CPF
- Certidão de casamento (se for casado)
- RG do cônjuge (se for casado)

REGRAS IMPORTANTÍSSIMAS:
- NÃO invente informações sobre imóveis. Use SOMENTE os dados fornecidos abaixo.
- NUNCA misture informações de bairros diferentes. Se o cliente perguntou sobre São Mateus, fale APENAS dos imóveis em São Mateus.
- Não confunda tipos de imóvel: se é apartamento, diga apartamento. Se é casa, diga casa. Não troque.
- SE DADOS DE IMÓVEIS foram fornecidos abaixo, TODAS as informações estão lá (preço, condomínio, IPTU, quartos, banheiros, etc). USE esses dados para responder.
- Só diga que vai verificar se realmente NÃO existem dados de imóveis abaixo.
- SE A LISTA DE IMÓVEIS ESTIVER VAZIA (ou se o aviso disser que não encontrou), SEJA HONESTA. Diga "Infelizmente não tenho opções nesse bairro/perfil no momento".
- JAMAIS INVENTE IMOVEIS. Se a lista abaixo tem imóveis em Benfica, NÃO DIGA que eles ficam no São Mateus.
- Se a busca retornou "Não encontrei com esses critérios", DEIXE CLARO que os imóveis mostrados são de OUTROS bairros ou perfis.

Histórico da conversa: {historico}

{dados_imoveis}

Cliente: {contexto}

Ana Paula:"""
    return llm.invoke(prompt).content

def quer_busca_explicita(msg):
    """Detecta se o usuário está explicitamente pedindo uma NOVA busca"""
    termos_busca = [
        "procuro", "busco", "gostaria de ver", "tem algum", "você tem", "queria ver", 
        "mostra outro", "outras opções", "mudar de bairro", "ver casas", "ver aptos"
    ]
    return any(termo in msg for termo in termos_busca)


# === CHAT PRINCIPAL ===

def ana_paula_chat(mensagem_usuario):
    historico = memory.load_memory_variables({})["chat_history"]
    ja_saudou = len(historico) > 0
    msg = mensagem_usuario.lower()
    
    # --- FLUXO 1: LINK DIRETO DO IMÓVEL ---
    imovel_id = extrair_id_da_url(mensagem_usuario)
    if imovel_id:
        dados = buscar_imovel_por_id(imovel_id)
        if dados:
            total = dados['preco_aluguel'] + dados['preco_iptu'] + dados['preco_condominio']
            ficha = formatar_imovel(dados)
            # Salva imóvel mostrado para follow-ups
            ultimos_imoveis_mostrados.clear()
            ultimos_imoveis_mostrados.append(dados)
            resposta = resposta_llm_corretora(
                mensagem_usuario,
                f"O cliente quer saber sobre este imóvel específico:\n{ficha}\nPreço total com encargos: R$ {total:.2f}/mês.\nApresente este imóvel de forma atrativa, destaque os pontos fortes e convide para uma visita."
            )
            memory.save_context({"input": mensagem_usuario}, {"output": resposta})
            return resposta
        else:
            resposta = "Não encontrei esse imóvel no nosso catálogo. Quer que eu te mostre outras opções disponíveis?"
            memory.save_context({"input": mensagem_usuario}, {"output": resposta})
            return resposta

    # EXTRAÇÃO DE INTENÇÃO E CRITÉRIOS
    criterios = extrair_criterios(mensagem_usuario)
    quer_ver_todas = any(p in msg for p in ['todas', 'todos', 'tudo', 'qualquer', 'outras', 'opções', 'opcoes', 'disponíveis', 'disponiveis'])
    palavras_busca = ["apartamento", "casa", "imóvel", "imovel", "imóveis", "imoveis", "procuro", "quero", "preciso", "mostra", "mostre", "tem algo", "tem outro", "tem mais", "teria outro"]
    quer_buscar = any(p in msg for p in palavras_busca)

    # Contexto de Bairro: "neste bairro", "nesse bairro", "mesmo bairro", "por aqui"
    if 'bairro' not in criterios and (imovel_em_foco or ultimos_imoveis_mostrados):
        termos_bairro_contexto = ['neste bairro', 'nesse bairro', 'mesmo bairro', 'naquele bairro', 'nessa região', 'nessa regiao', 'por aqui']
        if any(termo in msg for termo in termos_bairro_contexto):
            # Tenta pegar do foco atual ou do último mostrado
            ref = imovel_em_foco if imovel_em_foco else ultimos_imoveis_mostrados[0]
            if ref and 'bairro' in ref:
                criterios['bairro'] = ref['bairro']
                # Se inferiu bairro pelo contexto, reforça que é uma busca
                quer_buscar = True

    # --- FLUXO 2: PRIMEIRA INTERAÇÃO ---
    if not ja_saudou:
        # Se o usuário já chegou pedindo algo específico (tem critérios ou busca), PULA a saudação puramente social
        # e já processa a busca (o LLM pode saudar na resposta da busca)
        tem_intencao_clara = criterios or quer_buscar or quer_ver_todas
        if not tem_intencao_clara:
            bairros = buscar_bairros_disponiveis()
            bairros_txt = ", ".join(b.title() for b in bairros) if bairros else "diversos bairros"
            resposta = f"""Oi! Tudo bem? Sou a Ana Paula, corretora aqui da região de Juiz de Fora 😊

Temos imóveis disponíveis em {bairros_txt}.

Como posso te ajudar hoje?"""
            memory.save_context({"input": mensagem_usuario}, {"output": resposta})
            return resposta


    # Evita que perguntas sobre documentos, visitas, etc. disparem busca de imóveis
    palavras_nao_busca = ['documento', 'fiador', 'fiança', 'contrato', 'agendar', 'visita',
                          'visitação', 'horário', 'horario', 'quando', 'onde', 'fica', 'endereço',
                          'telefone', 'whatsapp', 'obrigad', 'valeu', 'brigad', 'qual', 'quais',
                          'detalhe', 'mais', 'sobre', 'esse', 'essa', 'esses', 'essas', 'aquele',
                          'aquela', 'aqueles', 'aquelas', 'deste', 'desta', 'disso', 'daquilo']
    e_conversa_geral = any(p in msg for p in palavras_nao_busca)

    # Identifica se é apenas um refinamento menor (ex: "aceita pets?", "tem garagem?")
    # Se só tem critérios secundários (sem bairro/texto) e já temos contexto, é pergunta, não busca nova
    criterios_secundarios = ['aceita_pets', 'quartos_min', 'preco_max']
    so_tem_secundarios = all(k in criterios_secundarios for k in criterios.keys()) and criterios
    imovel_focado = bool(imovel_em_foco) or bool(ultimos_imoveis_mostrados)
    eh_pergunta_contexto = (so_tem_secundarios or e_conversa_geral) and imovel_focado and not quer_busca_explicita(msg)

    # Só busca se tiver critérios/intenção E não for uma pergunta de contexto/geral
    # Mas se for busca explicita ("quero ver..."), ignora e_conversa_geral (ex: "quero ver onde fica")
    deve_buscar = (criterios or quer_buscar or quer_ver_todas)
    bloqueio_conversa = e_conversa_geral and not quer_busca_explicita(msg)
    
    # --- FLUXO 2.5: QUALIFICAÇÃO (NEEDS ASSESSMENT) ---
    # Se o usuário quer buscar mas foi muito vago, vamos qualificar melhor antes de consultar o banco.
    # Exceção: se ele disse "mostra tudo", "qualquer um", ou se já temos bastante contexto.
    if deve_buscar and not eh_pergunta_contexto and not bloqueio_conversa and not quer_ver_todas:
        # Critérios mínimos para uma busca eficiente:
        # Bairro + (Preço OU Quartos OU Tipo)
        # OU Texto livre (que indica busca especifica)
        # Se só tiver bairro, é muito amplo.
        
        tem_bairro = 'bairro' in criterios
        tem_preco = 'preco_max' in criterios
        tem_quartos = 'quartos_min' in criterios
        tem_texto = 'texto_busca' in criterios
        
        criterios_insuficientes = False
        perguntas_faltantes = []
        
        if tem_bairro and not (tem_preco or tem_quartos or tem_texto):
            criterios_insuficientes = True
            perguntas_faltantes.append("faixa de preço")
            perguntas_faltantes.append("número de quartos")
            
        elif not criterios and not quer_busca_explicita(msg): 
            # Se não tem criterio NENHUM e não foi explicito ("quero ver imoveis"), 
            # talvez seja só papo furado, mas se passou pelo filtro de busca...
            # Se não tem nada, pergunta tudo.
            criterios_insuficientes = True
            perguntas_faltantes.append("bairro de preferência")
            perguntas_faltantes.append("tipo de imóvel")
            
        if criterios_insuficientes:
            # Gera resposta pedindo detalhes
            prompt_qualificacao = f"""Você é a Ana Paula, corretora. O cliente quer buscar imóveis mas foi muito vago.
Não faça a busca ainda. Em vez disso, faça perguntas para entender melhor o que ele precisa.
O cliente disse: "{mensagem_usuario}"
Critérios que ele JÁ DEU: {criterios}
Informações que FALTAM e você deve pedir (escolha 1 ou 2 principais para não ser chata): {perguntas_faltantes}

Pergunte de forma natural, simpática e curta. Ex: "Legal que você gosta do bairro X! Mas me diz, até qual valor você pretende investir?" """
            
            resposta = llm.invoke(prompt_qualificacao).content
            memory.save_context({"input": mensagem_usuario}, {"output": resposta})
            return resposta

    if deve_buscar and not eh_pergunta_contexto and not bloqueio_conversa:
        # Busca com critérios ou tudo
        if quer_ver_todas and not criterios:
            imoveis = buscar_todos_imoveis()
            aviso = ""
        else:
            imoveis = buscar_imoveis_filtrados(**criterios) if criterios else buscar_todos_imoveis()
            aviso = ""

        # Relaxamento progressivo se não encontrar
        if not imoveis and criterios:
            # Sem filtro de preço
            sem_preco = {k: v for k, v in criterios.items() if k != 'preco_max'}
            if sem_preco:
                imoveis = buscar_imoveis_filtrados(**sem_preco)
                if imoveis:
                    aviso = "\n⚠️ Não encontrei nesse valor, mas separei as opções mais acessíveis pra você:\n"

            # Só bairro
            if not imoveis and 'bairro' in criterios:
                imoveis = buscar_imoveis_filtrados(bairro=criterios['bairro'])
                if imoveis:
                    aviso = "\n⚠️ Flexibilizei os critérios. Veja o que temos no bairro:\n"

            # Tudo (Relaxamento final)
            # SÓ relaxa para "todos" se o usuário NÃO especificou bairro.
            # Se ele pediu um bairro específico e não tem nada lá, é melhor dizer que não tem
            # do que mostrar imóveis de outro lado (o que causa alucinação de local).
            if not imoveis and 'bairro' not in criterios:
                imoveis = buscar_todos_imoveis()
                if imoveis:
                    aviso = "\n⚠️ Não encontrei com esses critérios, mas olha o que temos disponível:\n"

        if not imoveis:
            # Resposta honesta quando não encontra nada no bairro pedido
            if 'bairro' in criterios:
                 resposta = f"Infelizmente não tenho opções disponíveis em {criterios['bairro'].title()} no momento. 😕\n\nQuer dar uma olhada em outros bairros?"
            else:
                 resposta = "No momento não temos imóveis cadastrados com essas características, mas me passa seu contato que assim que surgir algo eu te aviso! 😉"
            
            memory.save_context({"input": mensagem_usuario}, {"output": resposta})
            return resposta

        # TOP 3
        melhores = top3(imoveis)
        qtd = len(imoveis)

        # Salva imóveis mostrados para follow-ups
        ultimos_imoveis_mostrados.clear()
        ultimos_imoveis_mostrados.extend(melhores)
        
        # Limpa foco anterior pois é uma nova busca
        imovel_em_foco.clear()

        fichas = "\n\n".join(formatar_imovel(im, i) for i, im in enumerate(melhores, 1))

        # Usa LLM para apresentar de forma natural
        dados_contexto = f"""Foram encontrados {qtd} imóveis no total.
{aviso}
Apresente EXATAMENTE estes {len(melhores)} imóveis abaixo (não invente outros). 
Copie os dados como estão, depois faça um breve comentário sobre cada um destacando os pontos fortes.
Ao final, pergunte qual agradou mais e ofereça agendar uma visita.

IMPORTANTE: Se o aviso acima diz "Não encontrei", DEIXE CLARO que estas opções são de OUTROS bairros/valores. NÃO minta sobre a localização.

IMÓVEIS ENCONTRADOS:
{fichas}"""
        
        resposta = resposta_llm_corretora(mensagem_usuario, dados_contexto)
        memory.save_context({"input": mensagem_usuario}, {"output": resposta})
        return resposta

    # --- FLUXO 3.5: FOLLOW-UP sobre imóveis já mostrados ---
    # Só entra aqui se NÃO for uma nova busca explícita (para não confundir "tem outros?" com "fale mais desse")
    if ultimos_imoveis_mostrados and not quer_buscar and not quer_busca_explicita(mensagem_usuario):
        # Tenta identificar qual imóvel específico o cliente quer saber
        imovel_especifico = identificar_imovel_mencionado(mensagem_usuario, ultimos_imoveis_mostrados)

        # Se não identificou na mensagem atual, usa o último em foco (contexto implícito)
        if not imovel_especifico and imovel_em_foco:
            # Verifica se o imóvel em foco ainda está na lista de mostrados
            # (para evitar misturar listas antigas se houve nova busca)
            ids_mostrados = [im['id'] for im in ultimos_imoveis_mostrados]
            if imovel_em_foco['id'] in ids_mostrados:
                # IMPORTANTE: .copy() para evitar alias com a global imovel_em_foco, 
                # pois ela será limpa (clear) logo abaixo
                imovel_especifico = imovel_em_foco.copy()
        
        # Se ainda não identificou mas é uma pergunta direta sobre "ele", "esse", "o imóvel"
        # e só temos UM imóvel mostrado, assume que é ele
        if not imovel_especifico and len(ultimos_imoveis_mostrados) == 1:
             imovel_especifico = ultimos_imoveis_mostrados[0].copy()

        if imovel_especifico:
            # Salva o imóvel em foco para próximas interações
            imovel_em_foco.clear()
            imovel_em_foco.update(imovel_especifico)
            ficha_detalhada = formatar_imovel_detalhado(imovel_especifico)
            dados_contexto = f"""O cliente quer saber mais detalhes sobre este imóvel específico (que ele já demonstrou interesse).

TODOS OS DADOS DESTE IMÓVEL (incluindo preços, condomínio, IPTU, etc):
{ficha_detalhada}

Você TEM todas as informações acima. Use-as para responder a pergunta do cliente.
Se a pergunta for sobre algo que está nos dados (ex: "aceita pets?"), responda diretamente com SIM ou NÃO e dê detalhes.
IMPORTANTE: Se a pergunta for sobre endereço (rua, número), forneça exatamente o que está nos dados."""
        else:
            # Contexto geral dos mostrados
            fichas = "\n\n---\n\n".join(formatar_imovel_detalhado(im) for im in ultimos_imoveis_mostrados)
            dados_contexto = f"""O cliente está fazendo uma pergunta sobre os imóveis que você já mostrou.

TODOS OS DADOS DOS IMÓVEIS (incluindo preços, condomínio, IPTU, área, quartos, etc):

{fichas}

Você TEM todas as informações acima. Use-as para responder a pergunta do cliente.
Identifique sobre qual imóvel ele está falando pelo contexto da conversa anterior ou pela pergunta.
Se não souber qual imóvel ele quer, pergunte "De qual imóvel você está falando?".
Se o cliente perguntar sobre um tipo de imóvel que não existe na lista (ex: uma "casa" quando só tem apartamentos), diga que não tem esse tipo e sugira as opções que você tem."""

        resposta = resposta_llm_corretora(mensagem_usuario, dados_contexto)
        memory.save_context({"input": mensagem_usuario}, {"output": resposta})
        return resposta



    # --- FLUXO 4: CONVERSA LIVRE (sempre com contexto de imóveis se houver) ---
    if imovel_em_foco:
        ficha = formatar_imovel_detalhado(imovel_em_foco)
        dados_contexto = f"""Contexto: o cliente estava conversando sobre este imóvel:
{ficha}

Use essas informações se a pergunta do cliente for relacionada a este imóvel."""
        resposta = resposta_llm_corretora(mensagem_usuario, dados_contexto)
    else:
        resposta = resposta_llm_corretora(mensagem_usuario)
    memory.save_context({"input": mensagem_usuario}, {"output": resposta})
    return resposta


# === LOOP PRINCIPAL ===
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏠 CORRETORA ANA PAULA - Imóveis sob medida pra você")
    print("="*60 + "\n")

    # Inicializa índice FTS5 para busca rápida em descrições
    inicializar_fts()

    primeira = ana_paula_chat("oi")
    print(f"🏠 Ana Paula: {primeira}\n")

    while True:
        voce = input("👤 Você: ")
        if voce.strip().lower() in ['sair', 'parar', 'tchau']:
            print("\n🏠 Ana Paula: Foi um prazer te atender! Quando quiser voltar a conversar sobre imóveis, é só me chamar. Até logo! 👋\n")
            break

        print(f"\n   [💬 Você disse: '{voce}']")
        print(f"   [⏳ Buscando as melhores opções...]\n")

        resposta = ana_paula_chat(voce)
        print(f"🏠 Ana Paula: {resposta}\n")
