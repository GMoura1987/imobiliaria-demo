import streamlit as st
import sqlite3
import pandas as pd
import unicodedata
import ollama
import os
from vanna.chromadb import ChromaDB_VectorStore
from vanna.ollama import Ollama

# ==========================================
# MÓDULO DE MEMÓRIA: REESCRITOR CONTEXTUAL
# ==========================================
def reescrever_pergunta_com_contexto(nova_pergunta, historico, model='deepseek-r1:8b'):
    """Usa o histórico para reescrever a pergunta de forma independente (standalone)."""
    if not historico or len(historico) == 0:
        return nova_pergunta
    
    # Pega apenas as últimas 4 mensagens para dar contexto sem gastar muito token
    contexto_str = ""
    for msg in historico[-4:]:
        if msg["role"] == "user":
            contexto_str += f"Cliente: {msg['content']}\n"
        elif msg["role"] == "assistant":
            # Ignora os dados técnicos no histórico para não confundir o modelo
            contexto_str += f"Bia: {msg['content']}\n"
            
    system_prompt = """Você é um assistente interno de reescrita de texto em uma imobiliária.
    Sua ÚNICA função é ler o contexto da conversa e reescrever a 'Nova pergunta' do cliente para que ela faça sentido sozinha.
    Você deve incorporar o assunto implícito (ex: tipo de imóvel, quantidade de quartos, pets, etc.) que estava sendo discutido.
    
    REGRAS CRÍTICAS:
    - NÃO responda à pergunta do cliente.
    - NÃO adicione saudações, explicações ou confirmações.
    - Se a 'Nova pergunta' já for completa e não depender do contexto, apenas repita-a.
    - Retorne APENAS a frase reescrita, nada mais.
    
    Exemplo de Contexto:
    Cliente: Queria casas no Centro.
    Bia: Não temos casas lá.
    Nova pergunta: E no São Mateus?
    
    Sua Resposta Esperada:
    Tem casas no São Mateus?
    """
    
    prompt = f"Contexto recente:\n{contexto_str}\nNova pergunta: {nova_pergunta}\nSua Resposta Esperada:"
    
    try:
        response = ollama.generate(model=model, system=system_prompt, prompt=prompt, options={'temperature': 0.0})
        reescrita = response['response']
        
        # Limpeza severa das tags de raciocínio do DeepSeek (se houver)
        if "</thought>" in reescrita:
            reescrita = reescrita.split("</thought>")[-1]
            
        return reescrita.strip()
    except Exception as e:
        print(f"Erro no reescritor contextual: {e}")
        # Se falhar, retorna a pergunta original como fallback de segurança
        return nova_pergunta

# ==========================================
# AGENTE 1: ANALISTA SQL (Versão Final 5.0)
# ==========================================
class SQLAnalyst(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

    def preparar_agente(self, db_path):
        self.connect_to_sqlite(db_path)
        
        df_meta = self.run_sql("SELECT DISTINCT bairro, rua, especificacao FROM core_imovel")
        self.bairros = [str(x) for x in df_meta['bairro'].dropna().unique().tolist()]
        self.ruas = [str(x) for x in df_meta['rua'].dropna().unique().tolist()]
        self.tipos = [str(x) for x in df_meta['especificacao'].dropna().unique().tolist()]

        if self.get_training_data().empty:
            self.train(ddl="""
            CREATE TABLE core_imovel (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                titulo VARCHAR(200), 
                descricao TEXT,
                quartos INTEGER, 
                banheiros INTEGER, 
                garagem INTEGER, 
                area DECIMAL, 
                bairro VARCHAR(100), 
                rua VARCHAR(100), 
                preco_aluguel DECIMAL, 
                preco_iptu DECIMAL, 
                preco_condominio DECIMAL, 
                aceita_pets BOOLEAN, -- 1 para Sim, 0 para Não
                especificacao VARCHAR(100) -- apartamento, casa, kitnet, studio, loft, cobertura
            );
            """)

            self.train(documentation=f"""
            - Localização: Juiz de Fora, MG.
            - REGRA DE ID: O campo 'id' é um INTEIRO. Ex: 'imóvel 131' deve ser traduzido como WHERE id = 131.
            - REGRA DE PETS: Se o cliente citar 'gato', 'cachorro' ou 'pets', use 'aceita_pets = 1'. 
            - NUNCA use LOWER() ou LIKE em colunas booleanas (aceita_pets) ou numéricas (preços, quartos, id).
            - Use LOWER() apenas para colunas de texto: bairro, rua, especificacao.
            - Custo Total = (preco_aluguel + preco_condominio + preco_iptu).
            - Bairros válidos em JF: {", ".join(self.bairros)}.
            """)

            self.train(question="Qual o apartamento mais barato no Centro?", 
                       sql="SELECT * FROM core_imovel WHERE LOWER(especificacao) = 'apartamento' AND LOWER(bairro) = 'centro' ORDER BY preco_aluguel ASC LIMIT 1")
            
            self.train(question="Tem casa com 3 quartos que aceita cachorro?", 
                       sql="SELECT * FROM core_imovel WHERE LOWER(especificacao) = 'casa' AND quartos >= 3 AND aceita_pets = 1 LIMIT 5")
            
            self.train(question="Qual o custo total desse imóvel?", 
                       sql="SELECT id, titulo, (preco_aluguel + preco_condominio + preco_iptu) as custo_total FROM core_imovel LIMIT 5")
            
            self.train(question="Imóveis no São Mateus por menos de 2000 reais", 
                       sql="SELECT * FROM core_imovel WHERE LOWER(bairro) = 'são mateus' AND (preco_aluguel + preco_condominio + preco_iptu) < 2000 LIMIT 5")
            # Exemplos para ensinar a LLM a lidar com "teto" e "piso" de valores
            self.train(question="Quero um apartamento de até 1500 reais", 
                    sql="SELECT * FROM core_imovel WHERE LOWER(especificacao) = 'apartamento' AND (preco_aluguel + preco_condominio + preco_iptu) <= 1500 LIMIT 5")

            self.train(question="Tem casa mais barata que 2 mil?", 
                    sql="SELECT * FROM core_imovel WHERE LOWER(especificacao) = 'casa' AND (preco_aluguel + preco_condominio + preco_iptu) < 2000 ORDER BY preco_aluguel ASC LIMIT 5")

            self.train(question="Imóveis entre 1000 e 2000 reais no Centro", 
                    sql="SELECT * FROM core_imovel WHERE LOWER(bairro) = 'centro' AND (preco_aluguel + preco_condominio + preco_iptu) BETWEEN 1000 AND 2000 LIMIT 5")

            self.train(question="Imóveis entre 1000 e 2000 reais no Centro", 
                    sql="SELECT * FROM core_imovel WHERE LOWER(bairro) = 'centro' AND (preco_aluguel + preco_condominio + preco_iptu) BETWEEN 1000 AND 2000 LIMIT 5")

    def fuzzy_cleanup(self, pergunta):
        pergunta_limpa = str(pergunta).lower().strip()
        if any(x in pergunta_limpa for x in ["gato", "cachorro", "animal", "pet"]):
            pergunta_limpa += " que aceita pets"
        return pergunta_limpa

    def executar_consulta(self, pergunta):
        pergunta_limpa = self.fuzzy_cleanup(pergunta)
        
        try:
            sql = self.generate_sql(pergunta_limpa)
            if "LIMIT" not in sql.upper():
                sql = sql.strip().rstrip(";") + " LIMIT 10;"
            df = self.run_sql(sql)
            return df, sql
            
        except Exception as e:
            try:
                prompt_correcao = f"A pergunta era '{pergunta_limpa}'. O SQL gerado falhou com o erro: {str(e)}. Gere apenas o SQL corrigido, sem explicações."
                sql_corrigido = self.generate_sql(prompt_correcao)
                if "LIMIT" not in sql_corrigido.upper():
                    sql_corrigido = sql_corrigido.strip().rstrip(";") + " LIMIT 10;"
                df = self.run_sql(sql_corrigido)
                return df, sql_corrigido
            except Exception as e2:
                return None, f"Falha na consulta e na tentativa de correção. Erro: {str(e2)}"

# ==========================================
# AGENTE 2: BIA (Persona Geofenced)
# ==========================================
class BiaPersona:
    def __init__(self, bairros_validos, model_name='deepseek-r1:8b'):
        self.model = model_name
        self.bairros_validos = bairros_validos
        self.system_prompt = f"""
        Você é a Bia, secretária virtual de uma imobiliária em Juiz de Fora.
        REGRAS:
        1. Se o banco de dados retornar 'Vazio' ou 'Nenhum imóvel', não invente dados. Diga que não encontrou e sugira bairros como: {", ".join(self.bairros_validos[:5])}.
        2. Nunca use termos técnicos de programação ou mencione SQL/Banco de dados.
        3. Para cálculos, use os valores de aluguel, IPTU e condomínio fornecidos.
        4. Seja simpática, concisa e vá direto ao ponto.
        """

    def responder(self, pergunta_original, df):
        # PROTEÇÃO MÁXIMA: Se não tem dado, nem chama a LLM. Retorna texto fixo.
        if df is None or isinstance(df, str) or df.empty:
            bairros_sugestao = ", ".join(self.bairros_validos[:3])
            return f"Poxa, infelizmente não encontrei nenhum imóvel com essas características no banco de dados. Que tal tentarmos em outros bairros como {bairros_sugestao}?"
            
        # Se tem dado, aí sim passa para a LLM formatar
        contexto = df.head(5).to_dict(orient='records') 
        prompt = f"Pergunta do Cliente: {pergunta_original}\nDados Reais do Banco: {contexto}\nBia, responda:"
        
        try:
            response = ollama.generate(model=self.model, system=self.system_prompt, prompt=prompt, options={'temperature': 0.1})
            clean_response = response['response']
            if "</thought>" in clean_response:
                clean_response = clean_response.split("</thought>")[-1]
            return clean_response.strip()
        except Exception:
            return "Tive uma falha técnica rápida, mas posso pesquisar outro bairro para você em JF!"
        if df is not None and not isinstance(df, str) and not df.empty:
            contexto = df.head(5).to_dict(orient='records') 
        else:
            contexto = "Nenhum imóvel encontrado no banco de dados com essas características."
            
        prompt = f"Pergunta do Cliente: {pergunta_original}\nDados Reais do Banco: {contexto}\nBia, responda:"
        
        try:
            response = ollama.generate(model=self.model, system=self.system_prompt, prompt=prompt, options={'temperature': 0.1})
            clean_response = response['response']
            if "</thought>" in clean_response:
                clean_response = clean_response.split("</thought>")[-1]
            return clean_response.strip()
        except Exception:
            return "Tive uma falha técnica rápida, mas posso pesquisar outro bairro para você em JF!"

# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Imobiliária Chatbot - Bia", page_icon="🏠", layout="centered")

st.title("🏠 Sistema de Atendimento - Bia")
st.markdown("Faça perguntas sobre imóveis em Juiz de Fora!")

@st.cache_resource(show_spinner="Carregando modelos e treinando banco de dados...")
def inicializar_agentes():
    db_path = "db.sqlite3"
    if not os.path.exists(db_path):
        db_path = "../db.sqlite3" 
    
    # Path alterado para v5 para forçar um banco vetorial limpo
    config_sql = {"model": "qwen2.5-coder:7b", "path": "./vanna_chroma_final_v5", "temperature": 0.0}
    analista = SQLAnalyst(config=config_sql)
    analista.preparar_agente(db_path)
    
    bia = BiaPersona(bairros_validos=analista.bairros)
    return analista, bia

try:
    analista, bia = inicializar_agentes()
except Exception as e:
    st.error(f"Erro crítico ao iniciar agentes: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant" and "sql" in msg:
            with st.expander("🔍 Detalhes Técnicos (SQL & Dados)"):
                if "pergunta_traduzida" in msg:
                    st.markdown(f"**Reescrita de Contexto:** `{msg['pergunta_traduzida']}`")
                st.code(msg["sql"], language="sql")
                if msg["df"] is not None and isinstance(msg["df"], pd.DataFrame) and not msg["df"].empty:
                    st.dataframe(msg["df"])
                else:
                    st.info("Nenhum registro retornado ou erro na consulta.")

if prompt := st.chat_input("Ex: Qual o apartamento mais barato no Centro?"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Bia está consultando o banco de dados..."):
            
            # 1. Recupera histórico e reescreve a pergunta
            historico = st.session_state.messages[:-1] # Exclui a pergunta atual que acabou de ser adicionada
            pergunta_enriquecida = reescrever_pergunta_com_contexto(prompt, historico)
            
            # 2. Executa a busca com a pergunta enriquecida
            df, sql = analista.executar_consulta(pergunta_enriquecida)
            
            # 3. Responde com base na pergunta original para manter naturalidade
            resposta = bia.responder(prompt, df)
            
            st.markdown(resposta)
            
            with st.expander("🔍 Detalhes Técnicos (SQL & Dados)"):
                st.markdown(f"**Reescrita de Contexto:** `{pergunta_enriquecida}`")
                st.code(sql, language="sql")
                if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                    st.dataframe(df)
                else:
                    st.info("Nenhum registro retornado ou erro na consulta.")
            
    st.session_state.messages.append({
        "role": "assistant",
        "content": resposta,
        "sql": sql,
        "df": df,
        "pergunta_traduzida": pergunta_enriquecida
    })